-- 03_device_meta_drift_allowed_cert.sql
-- Drift-allowed certification for device_meta.

CREATE OR REPLACE PROCEDURE audit.run_t0_cert_device_meta_v3(
  p_check_days int DEFAULT 10,
  p_update_baselines boolean DEFAULT true,
  p_sample_limit int DEFAULT 2000
)
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.device_meta_store_t0_v1_v';
  token_ct int;
  dup_ct int;
  drift_ct int;
  v_viewdef_objects int := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days int := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=p_sample_limit);
  msg text;
BEGIN
  -- Gate A2: forbid time-of-query tokens in closure
  WITH RECURSIVE rels(oid) AS (
    SELECT ep::regclass
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
    RAISE EXCEPTION 'T0 CERT FAIL (device_meta): time tokens in closure: %', token_ct;
  END IF;

  -- Key uniqueness
  SELECT COUNT(*) INTO dup_ct
  FROM (
    SELECT 1
    FROM ml.device_meta_store_t0_v1_v
    GROUP BY generation, listing_id, edited_date
    HAVING COUNT(*) > 1
    LIMIT 1
  ) x;

  IF dup_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (device_meta): duplicate keys (generation,listing_id,t0)';
  END IF;

  -- Drift computation on last N days
  WITH picked AS (
    SELECT t0_day
    FROM (
      SELECT DISTINCT edited_date::date AS t0_day
      FROM ml.device_meta_store_t0_v1_v
      WHERE edited_date IS NOT NULL
    ) d
    ORDER BY t0_day DESC
    LIMIT GREATEST(1, p_check_days)
  ),
  cur AS (
    SELECT p.t0_day,
           audit.dataset_sha256(ep::regclass, p.t0_day, p_sample_limit) AS current_sha
    FROM picked p
  ),
  base AS (
    SELECT b.t0_day,
           b.dataset_sha256 AS baseline_sha
    FROM audit.t0_dataset_hash_baseline b
    JOIN picked p USING (t0_day)
    WHERE b.entrypoint = ep
      AND b.sample_limit = p_sample_limit
  ),
  drift AS (
    SELECT c.t0_day, b.baseline_sha, c.current_sha
    FROM cur c
    LEFT JOIN base b USING (t0_day)
    WHERE b.baseline_sha IS DISTINCT FROM c.current_sha
  )
  SELECT COUNT(*) INTO drift_ct FROM drift;

  IF drift_ct > 0 THEN
    IF p_update_baselines THEN
      INSERT INTO audit.t0_dataset_hash_baseline(entrypoint, t0_day, sample_limit, dataset_sha256, computed_at)
      SELECT ep, d.t0_day, p_sample_limit, d.current_sha, now()
      FROM drift d
      ON CONFLICT (entrypoint, t0_day, sample_limit)
      DO UPDATE SET dataset_sha256 = EXCLUDED.dataset_sha256,
                    computed_at    = EXCLUDED.computed_at;
    ELSE
      RAISE EXCEPTION 'T0 CERT FAIL (device_meta): drift_last_%_days=%', p_check_days, drift_ct;
    END IF;
  END IF;

  v_dataset_days := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=p_sample_limit);

  msg := format(
    'T0 CERT PASS (device_meta): drift_last_%s_days=%s %s; baselines=%s',
    p_check_days,
    drift_ct,
    CASE WHEN drift_ct>0 THEN '(allowed+rebaselined)' ELSE '' END,
    v_dataset_days
  );

  INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
  VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
  ON CONFLICT (entrypoint) DO UPDATE SET
    status='CERTIFIED',
    certified_at=now(),
    viewdef_objects=EXCLUDED.viewdef_objects,
    dataset_days=EXCLUDED.dataset_days,
    notes=EXCLUDED.notes;

  RAISE NOTICE '%', msg;
END $$;
