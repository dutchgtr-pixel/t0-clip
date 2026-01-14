-- Public Release: Certification, guardrails, and access control

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.t0_viewdef_baseline (
  entrypoint     text NOT NULL,
  object_fqn     text NOT NULL,
  relkind        char NOT NULL,
  viewdef_sha256 text NOT NULL,
  captured_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(entrypoint, object_fqn)
);


WITH RECURSIVE rels(oid) AS (
  SELECT 'ml.survival_feature_store_t0_v1_v'::regclass
  UNION
  SELECT c2.oid
  FROM rels r
  JOIN pg_rewrite w ON w.ev_class = r.oid
  JOIN pg_depend  d ON d.objid = w.oid
  JOIN pg_class   c2 ON c2.oid = d.refobjid
  WHERE c2.relkind IN ('v','m')
),
objs AS (
  SELECT DISTINCT r.oid, n.nspname, c.relname, c.relkind
  FROM rels r
  JOIN pg_class c ON c.oid = r.oid
  JOIN pg_namespace n ON n.oid = c.relnamespace
),
defs AS (
  SELECT
    'ml.survival_feature_store_t0_v1_v'::text AS entrypoint,
    nspname||'.'||relname AS object_fqn,
    relkind,
    encode(digest(convert_to(pg_get_viewdef(oid,true),'utf8'),'sha256'),'hex') AS viewdef_sha256
  FROM objs
)
INSERT INTO audit.t0_viewdef_baseline(entrypoint, object_fqn, relkind, viewdef_sha256)
SELECT entrypoint, object_fqn, relkind, viewdef_sha256
FROM defs
ON CONFLICT (entrypoint, object_fqn)
DO UPDATE SET
  viewdef_sha256 = EXCLUDED.viewdef_sha256,
  captured_at = now();


CREATE TABLE IF NOT EXISTS audit.t0_dataset_hash_baseline (
  entrypoint     text NOT NULL,
  t0_day         date NOT NULL,
  sample_limit   int  NOT NULL,
  dataset_sha256 text NOT NULL,
  computed_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(entrypoint, t0_day, sample_limit)
);


CREATE OR REPLACE FUNCTION audit.dataset_sha256(
  p_entrypoint regclass,
  p_t0_day date,
  p_sample_limit int
) RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
  out_sha text;
  sql text;
BEGIN
  sql := format($q$
    WITH sample AS (
      SELECT *
      FROM %s
      WHERE edited_date::date = %L::date
      ORDER BY listing_id
      LIMIT %s
    ),
    row_hashes AS (
      SELECT
        listing_id,
        encode(digest(convert_to(to_jsonb(sample)::text,'utf8'),'sha256'),'hex') AS row_h
      FROM sample
    )
    SELECT encode(
      digest(convert_to(string_agg(row_h, '' ORDER BY listing_id),'utf8'),'sha256'),
      'hex'
    )
    FROM row_hashes
  $q$, p_entrypoint, p_t0_day, p_sample_limit);

  EXECUTE sql INTO out_sha;
  RETURN out_sha;
END $$;


WITH picked AS (
  WITH bounds AS (
    SELECT max(edited_date::date) AS max_day
    FROM ml.survival_feature_store_t0_v1_v
  ),
  candidates AS (
    SELECT (max_day - d)::date AS t0_day
    FROM bounds, (VALUES (30),(90),(180),(365),(540)) v(d)
  )
  SELECT c.t0_day
  FROM candidates c
  WHERE EXISTS (
    SELECT 1
    FROM ml.survival_feature_store_t0_v1_v s
    WHERE s.edited_date::date = c.t0_day
  )
)
INSERT INTO audit.t0_dataset_hash_baseline(entrypoint, t0_day, sample_limit, dataset_sha256)
SELECT
  'ml.survival_feature_store_t0_v1_v',
  t0_day,
  2000,
  audit.dataset_sha256('ml.survival_feature_store_t0_v1_v'::regclass, t0_day, 2000)
FROM picked
ON CONFLICT (entrypoint, t0_day, sample_limit)
DO UPDATE SET dataset_sha256 = EXCLUDED.dataset_sha256,
              computed_at = now();


WITH base AS (
  SELECT *
  FROM audit.t0_dataset_hash_baseline
  WHERE entrypoint = 'ml.survival_feature_store_t0_v1_v'
    AND sample_limit = 2000
),
cur AS (
  SELECT
    b.t0_day,
    audit.dataset_sha256('ml.survival_feature_store_t0_v1_v'::regclass, b.t0_day, b.sample_limit) AS current_sha
  FROM base b
)
SELECT
  b.entrypoint,
  b.t0_day,
  b.sample_limit,
  b.dataset_sha256 AS baseline_sha,
  c.current_sha,
  (b.dataset_sha256 = c.current_sha) AS matches
FROM base b
JOIN cur c USING (t0_day)
ORDER BY b.t0_day;


CREATE TABLE IF NOT EXISTS audit.t0_cert_registry (
  entrypoint      text PRIMARY KEY,
  status          text NOT NULL,
  certified_at    timestamptz NOT NULL DEFAULT now(),
  viewdef_objects int NOT NULL,
  dataset_days    int NOT NULL,
  notes           text
);


CREATE OR REPLACE FUNCTION audit.assert_viewdef_baseline(p_entrypoint text)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  mismatch_ct int;
BEGIN
  WITH RECURSIVE rels(oid) AS (
    SELECT p_entrypoint::regclass
    UNION
    SELECT c2.oid
    FROM rels r
    JOIN pg_rewrite w ON w.ev_class = r.oid
    JOIN pg_depend  d ON d.objid = w.oid
    JOIN pg_class   c2 ON c2.oid = d.refobjid
    WHERE c2.relkind IN ('v','m')
  ),
  objs AS (
    SELECT DISTINCT
      r.oid,
      n.nspname||'.'||c.relname AS object_fqn,
      c.relkind::text AS relkind_txt
    FROM rels r
    JOIN pg_class c ON c.oid = r.oid
    JOIN pg_namespace n ON n.oid = c.relnamespace
  ),
  cur AS (
    SELECT
      p_entrypoint AS entrypoint,
      object_fqn,
      relkind_txt,
      encode(digest(convert_to(pg_get_viewdef(oid,true),'utf8'),'sha256'),'hex') AS current_sha
    FROM objs
  ),
  base AS (
    SELECT
      entrypoint,
      object_fqn,
      relkind::text AS relkind_txt,
      viewdef_sha256 AS baseline_sha
    FROM audit.t0_viewdef_baseline
    WHERE entrypoint = p_entrypoint
  )
  SELECT COUNT(*) INTO mismatch_ct
  FROM base b
  FULL OUTER JOIN cur c USING (entrypoint, object_fqn, relkind_txt)
  WHERE b.baseline_sha IS DISTINCT FROM c.current_sha;

  IF mismatch_ct > 0 THEN
    RAISE EXCEPTION 'CERT GUARD FAIL: viewdef drift detected for % object(s) under %', mismatch_ct, p_entrypoint;
  END IF;
END $$;


CREATE OR REPLACE FUNCTION audit.require_certified(entrypoint text, max_age interval DEFAULT interval '24 hours')
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  st text;
  ts timestamptz;
BEGIN
  SELECT status, certified_at INTO st, ts
  FROM audit.t0_cert_registry
  WHERE audit.t0_cert_registry.entrypoint = require_certified.entrypoint;

  IF st IS DISTINCT FROM 'CERTIFIED' THEN
    RAISE EXCEPTION 'CERT GUARD: % is not CERTIFIED (status=%)', entrypoint, st;
  END IF;

  IF ts < now() - max_age THEN
    RAISE EXCEPTION 'CERT GUARD: % certification is stale (certified_at=%)', entrypoint, ts;
  END IF;
END $$;


CREATE OR REPLACE FUNCTION audit.require_certified_strict(
  entrypoint text,
  max_age interval DEFAULT interval '24 hours'
) RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM audit.require_certified(entrypoint, max_age);
  PERFORM audit.assert_viewdef_baseline(entrypoint);
END $$;


CREATE OR REPLACE FUNCTION audit.cert_guard(entrypoint text, max_age interval DEFAULT interval '24 hours')
RETURNS boolean
LANGUAGE plpgsql VOLATILE
AS $$
BEGIN
  PERFORM audit.require_certified_strict(entrypoint, max_age);
  RETURN true;
END $$;


CREATE OR REPLACE FUNCTION audit.assert_t0_certified_survival_v1()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  ep regclass := 'ml.survival_feature_store_t0_v1_v'::regclass;

  forbid_ct int;
  token_ct int;

  img_violation_ct bigint;
  dmg_violation_ct bigint;

  drift_ct int;
BEGIN
  /* 0) Forbidden dependency objects in closure */
  WITH RECURSIVE rels(oid) AS (
    SELECT ep
    UNION
    SELECT c2.oid
    FROM rels r
    JOIN pg_rewrite w ON w.ev_class = r.oid
    JOIN pg_depend  d ON d.objid = w.oid
    JOIN pg_class   c2 ON c2.oid = d.refobjid
    WHERE c2.relkind IN ('r','p','v','m','f')
  ),
  objs AS (
    SELECT DISTINCT n.nspname||'.'||c.relname AS fqn
    FROM rels
    JOIN pg_class c ON c.oid=rels.oid
    JOIN pg_namespace n ON n.oid=c.relnamespace
  )
  SELECT COUNT(*) INTO forbid_ct
  FROM objs
  WHERE fqn IN (
    'ml.tom_labels_v1_mv',
    'ref.geo_mapping_current',
    'ml.geo_dim_super_metro_v4_current_v',
    'ml.tom_speed_anchor_v1_mv',
    'ml.tom_features_v1_enriched_speed_mv',
    'ml.tom_features_v1_enriched_ai_clean_mv',
    'ml.v_anchor_speed_train_base_rich_v1'
  );

  IF forbid_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL: forbidden dependency objects present in closure: %', forbid_ct;
  END IF;

  /* 1) Time-of-query tokens anywhere in view/matview closure */
  WITH RECURSIVE rels(oid) AS (
    SELECT ep
    UNION
    SELECT c2.oid
    FROM rels r
    JOIN pg_rewrite w ON w.ev_class = r.oid
    JOIN pg_depend  d ON d.objid = w.oid
    JOIN pg_class   c2 ON c2.oid = d.refobjid
    WHERE c2.relkind IN ('v','m')
  )
  SELECT COUNT(*) INTO token_ct
  FROM rels
  WHERE lower(pg_get_viewdef(oid,true)) ~
    '\m(current_date|current_timestamp|now\(|clock_timestamp\(|transaction_timestamp\(|statement_timestamp\(|localtimestamp)\M';

  IF token_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL: time-of-query tokens detected in closure: %', token_ct;
  END IF;

  /* 2) Image/damage SLA leakage checks (must be 0) */
  SELECT COUNT(*) INTO img_violation_ct
  FROM ml.survival_feature_store_t0_v1_v s
  WHERE NOT s.img_within_sla
    AND EXISTS (
      SELECT 1
      FROM jsonb_each(to_jsonb(s)) kv
      WHERE kv.key ~ '^img__'
        AND kv.value <> 'null'::jsonb
    );

  IF img_violation_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL: image leakage: % rows have img__* populated while img_within_sla=false', img_violation_ct;
  END IF;

  SELECT COUNT(*) INTO dmg_violation_ct
  FROM ml.survival_feature_store_t0_v1_v s
  WHERE NOT s.img_within_sla
    AND EXISTS (
      SELECT 1
      FROM jsonb_each(to_jsonb(s)) kv
      WHERE kv.key ~ '^dmg__'
        AND kv.value <> 'null'::jsonb
    );

  IF dmg_violation_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL: damage leakage: % rows have dmg__* populated while img_within_sla=false', dmg_violation_ct;
  END IF;

  /* 3) Dataset hash drift vs baseline (must be 0 mismatches) */
  SELECT COUNT(*) INTO drift_ct
  FROM (
    WITH base AS (
      SELECT *
      FROM audit.t0_dataset_hash_baseline
      WHERE entrypoint = 'ml.survival_feature_store_t0_v1_v'
        AND sample_limit = 2000
    ),
    cur AS (
      SELECT
        b.t0_day,
        audit.dataset_sha256('ml.survival_feature_store_t0_v1_v'::regclass, b.t0_day, b.sample_limit) AS current_sha
      FROM base b
    )
    SELECT 1
    FROM base b
    JOIN cur c USING (t0_day)
    WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha
  ) x;

  IF drift_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL: dataset hash drift vs baseline: % day(s)', drift_ct;
  END IF;
END $$;


CREATE OR REPLACE PROCEDURE audit.run_t0_cert_survival_v1()
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.survival_feature_store_t0_v1_v';
  v_viewdef_objects int;
  v_dataset_days int;
  msg text;
BEGIN
  v_viewdef_objects := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days    := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep);

  BEGIN
    PERFORM audit.assert_t0_certified_survival_v1();
    msg := 'T0 CERT PASS: invariants + baselines validated';

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status=EXCLUDED.status,
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;

  EXCEPTION WHEN others THEN
    msg := SQLERRM;

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'FAILED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status=EXCLUDED.status,
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;
  END;

  RAISE NOTICE '%', msg;
END $$;


CREATE OR REPLACE VIEW ml.survival_feature_store_train_v AS
SELECT fs.*
FROM ml.survival_feature_store_t0_v1_v fs
CROSS JOIN LATERAL (
  SELECT audit.cert_guard('ml.survival_feature_store_t0_v1_v', interval '24 hours') AS ok
) g
WHERE g.ok;


INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
SELECT
  'ml.survival_feature_store_train_v',
  'CERTIFIED',
  (SELECT count(*) FROM audit.t0_viewdef_baseline WHERE entrypoint='ml.survival_feature_store_t0_v1_v'),
  (SELECT count(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint='ml.survival_feature_store_t0_v1_v'),
  'Alias points to ml.survival_feature_store_t0_v1_v; certification inherited.'
ON CONFLICT (entrypoint) DO UPDATE SET
  status='CERTIFIED',
  certified_at=now(),
  viewdef_objects=EXCLUDED.viewdef_objects,
  dataset_days=EXCLUDED.dataset_days,
  notes=EXCLUDED.notes;


-- one-time role
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='ml_training') THEN
    CREATE ROLE ml_training;
  END IF;
END $$;

GRANT USAGE ON SCHEMA ml TO ml_training;
GRANT USAGE ON SCHEMA audit TO ml_training;

-- allow reading ONLY the guarded training view
GRANT SELECT ON ml.survival_feature_store_train_v TO ml_training;

-- allow guard functions to execute
GRANT EXECUTE ON FUNCTION audit.cert_guard(text, interval) TO ml_training;
GRANT EXECUTE ON FUNCTION audit.require_certified_strict(text, interval) TO ml_training;

-- revoke legacy objects from training role (defense in depth)
REVOKE SELECT ON ml.tom_speed_anchor_v1_mv FROM ml_training;
REVOKE SELECT ON ml.tom_features_v1_enriched_speed_mv FROM ml_training;
REVOKE SELECT ON ml.tom_features_v1_enriched_ai_clean_mv FROM ml_training;


BEGIN;

UPDATE audit.t0_cert_registry
SET certified_at = now() - interval '2 days'
WHERE entrypoint='ml.survival_feature_store_t0_v1_v';

-- should FAIL because certification is stale
SELECT count(*) FROM ml.survival_feature_store_train_v;

ROLLBACK;
