-- 05_drift_diagnostics_queries.sql
-- Find which days drifted (last N baseline days), example for device_meta.

WITH base AS (
  SELECT t0_day, sample_limit, dataset_sha256
  FROM audit.t0_dataset_hash_baseline
  WHERE entrypoint='ml.device_meta_store_t0_v1_v'
    AND sample_limit=2000
  ORDER BY t0_day DESC
  LIMIT 10
),
cur AS (
  SELECT
    b.t0_day,
    audit.dataset_sha256('ml.device_meta_store_t0_v1_v'::regclass, b.t0_day, b.sample_limit) AS current_sha
  FROM base b
)
SELECT
  b.t0_day,
  b.dataset_sha256 AS baseline_sha,
  c.current_sha
FROM base b
JOIN cur c USING (t0_day)
WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha
ORDER BY b.t0_day DESC;
