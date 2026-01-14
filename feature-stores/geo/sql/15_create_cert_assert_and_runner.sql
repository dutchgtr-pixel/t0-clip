-- 15_create_cert_assert_and_runner.sql
-- Geo certification assertion + procedure.

CREATE OR REPLACE FUNCTION audit.assert_t0_certified_geo_store_v1(p_check_days int DEFAULT 10)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  ep regclass := 'ml.geo_feature_store_t0_v1_v'::regclass;
  token_ct int;
  drift_ct int;
  dup_ct int;
  pin_ct int;
BEGIN
  -- pinned release must exist (1 row)
  SELECT COUNT(*) INTO pin_ct FROM ref.geo_mapping_pinned_super_metro_v4_v1;
  IF pin_ct <> 1 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (geo): pinned release view must return exactly 1 row; got %', pin_ct;
  END IF;

  -- Gate A2: no time-of-query tokens anywhere in view/matview closure
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
    RAISE EXCEPTION 'T0 CERT FAIL (geo): time-of-query tokens detected in closure: %', token_ct;
  END IF;

  -- Unique key in geo dim view: 1 row per listing_id
  SELECT COUNT(*) INTO dup_ct
  FROM (
    SELECT 1
    FROM ml.geo_dim_super_metro_v4_t0_v1
    GROUP BY listing_id
    HAVING COUNT(*) > 1
    LIMIT 1
  ) x;

  IF dup_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (geo): duplicate listing_id keys in geo dim view';
  END IF;

  -- Drift check vs last N baseline days
  WITH base AS (
    SELECT t0_day, sample_limit, dataset_sha256
    FROM audit.t0_dataset_hash_baseline
    WHERE entrypoint='ml.geo_feature_store_t0_v1_v'
      AND sample_limit=2000
    ORDER BY t0_day DESC
    LIMIT GREATEST(1, p_check_days)
  ),
  cur AS (
    SELECT
      b.t0_day,
      audit.dataset_sha256('ml.geo_feature_store_t0_v1_v'::regclass, b.t0_day, b.sample_limit) AS current_sha
    FROM base b
  )
  SELECT COUNT(*) INTO drift_ct
  FROM base b
  JOIN cur c USING (t0_day)
  WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha;

  IF drift_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (geo): dataset hash drift vs baseline: % day(s)', drift_ct;
  END IF;
END $$;


CREATE OR REPLACE PROCEDURE audit.run_t0_cert_geo_store_v1(p_check_days int DEFAULT 10)
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.geo_feature_store_t0_v1_v';
  v_viewdef_objects int := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days int := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=2000);
  msg text;
BEGIN
  PERFORM audit.assert_t0_certified_geo_store_v1(p_check_days);

  msg := format('T0 CERT PASS (geo): pinned release + no time tokens + drift_last_%s_days=0; baselines=%s', p_check_days, v_dataset_days);

  INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
  VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
  ON CONFLICT (entrypoint) DO UPDATE SET
    status='CERTIFIED',
    certified_at=now(),
    viewdef_objects=EXCLUDED.viewdef_objects,
    dataset_days=EXCLUDED.dataset_days,
    notes=EXCLUDED.notes;

  INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
  SELECT
    x.entrypoint,
    'CERTIFIED',
    v_viewdef_objects,
    v_dataset_days,
    'Alias points to ml.geo_feature_store_t0_v1_v; certification inherited.'
  FROM (VALUES
    ('ml.geo_dim_super_metro_v4_t0_train_v'),
    ('ml.geo_dim_super_metro_v4_t0_v1')
  ) AS x(entrypoint)
  ON CONFLICT (entrypoint) DO UPDATE SET
    status='CERTIFIED',
    certified_at=now(),
    viewdef_objects=EXCLUDED.viewdef_objects,
    dataset_days=EXCLUDED.dataset_days,
    notes=EXCLUDED.notes;

  RAISE NOTICE '%', msg;
END $$;
