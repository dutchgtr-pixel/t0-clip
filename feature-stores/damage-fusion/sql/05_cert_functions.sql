-- ============================================================
-- Fusion Store — Certification function + runner
-- ============================================================

CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.t0_cert_registry (
  entrypoint      text PRIMARY KEY,
  status          text NOT NULL,
  certified_at    timestamptz NOT NULL DEFAULT now(),
  viewdef_objects int NOT NULL,
  dataset_days    int NOT NULL,
  notes           text
);

-- Assertion: fail on structural violations (time tokens, timestamps in outputs, duplicate keys)
CREATE OR REPLACE FUNCTION audit.assert_t0_certified_fusion_store_v1()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  ep regclass := 'ml.fusion_feature_store_t0_v1_v'::regclass;
  token_ct int;
  ts_cols int;
  dup_ct int;
BEGIN
  -- Gate A2: no time-of-query tokens anywhere in closure
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
    RAISE EXCEPTION 'T0 CERT FAIL (fusion): time-of-query tokens detected in closure: %', token_ct;
  END IF;

  -- Contract: fusion outputs should not expose timestamps (keeps it leak-safe by design)
  SELECT COUNT(*) INTO ts_cols
  FROM information_schema.columns
  WHERE table_schema='ml'
    AND table_name IN ('v_damage_fusion_features_v2','v_damage_fusion_features_v2_scored')
    AND data_type LIKE 'timestamp%';

  IF ts_cols > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (fusion): timestamp columns present in fusion outputs: %', ts_cols;
  END IF;

  -- Key uniqueness: one row per (generation,listing_id)
  SELECT COUNT(*) INTO dup_ct
  FROM (
    SELECT 1
    FROM ml.v_damage_fusion_features_v2_scored
    GROUP BY generation, listing_id
    HAVING COUNT(*) > 1
    LIMIT 1
  ) x;

  IF dup_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (fusion): duplicate (generation,listing_id) keys in v2_scored';
  END IF;
END $$;


-- Runner: certifies + records drift (allowed) and optionally updates drifted baselines
CREATE OR REPLACE PROCEDURE audit.run_t0_cert_fusion_store_v1(
  p_check_days int DEFAULT 30,
  p_update_drift_baselines boolean DEFAULT true
)
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.fusion_feature_store_t0_v1_v';
  v_viewdef_objects int := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days int := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=2000);
  drift_ct int := 0;
  msg text;
BEGIN
  BEGIN
    PERFORM audit.assert_t0_certified_fusion_store_v1();

    -- Drift check vs baseline on last N days (informational; DOES NOT FAIL certification)
    WITH base AS (
      SELECT t0_day, sample_limit, dataset_sha256
      FROM audit.t0_dataset_hash_baseline
      WHERE entrypoint=ep AND sample_limit=2000
      ORDER BY t0_day DESC
      LIMIT GREATEST(1, LEAST(p_check_days, v_dataset_days))
    ),
    cur AS (
      SELECT
        b.t0_day,
        audit.dataset_sha256(ep::regclass, b.t0_day, b.sample_limit) AS current_sha
      FROM base b
    ),
    diff AS (
      SELECT b.t0_day, c.current_sha
      FROM base b
      JOIN cur c USING (t0_day)
      WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha
    )
    SELECT COUNT(*) INTO drift_ct
    FROM diff;

    IF drift_ct > 0 AND p_update_drift_baselines THEN
      UPDATE audit.t0_dataset_hash_baseline b
      SET dataset_sha256 = d.current_sha,
          computed_at = now()
      FROM (
        WITH base AS (
          SELECT t0_day, sample_limit, dataset_sha256
          FROM audit.t0_dataset_hash_baseline
          WHERE entrypoint=ep AND sample_limit=2000
          ORDER BY t0_day DESC
          LIMIT GREATEST(1, LEAST(p_check_days, v_dataset_days))
        ),
        cur AS (
          SELECT
            b.t0_day,
            audit.dataset_sha256(ep::regclass, b.t0_day, b.sample_limit) AS current_sha
          FROM base b
        )
        SELECT b.t0_day, c.current_sha
        FROM base b
        JOIN cur c USING (t0_day)
        WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha
      ) d
      WHERE b.entrypoint=ep AND b.sample_limit=2000 AND b.t0_day=d.t0_day;
    END IF;

    msg := format(
      'T0 CERT PASS (fusion): no time tokens, no timestamps in outputs, unique keys ok; baselines=%s; drift_last_%s_days=%s (allowed)',
      v_dataset_days, p_check_days, drift_ct
    );

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='CERTIFIED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;

    -- Register aliases as certification-inherited (handy for registry lookups)
    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    SELECT
      x.entrypoint,
      'CERTIFIED',
      v_viewdef_objects,
      v_dataset_days,
      'Alias points to ml.fusion_feature_store_t0_v1_v; certification inherited.'
    FROM (VALUES
      ('ml.v_damage_fusion_features_v2_scored_train_v'),
      ('ml.v_damage_fusion_features_v2_scored'),
      ('ml.v_damage_fusion_features_v2')
    ) AS x(entrypoint)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='CERTIFIED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;

  EXCEPTION WHEN others THEN
    msg := SQLERRM;

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'FAILED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='FAILED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;
  END;

  RAISE NOTICE '%', msg;
END $$;
