-- 07_baseline_dataset_hashes_last10.sql
-- Computes dataset hash baselines for the last 10 days (policy choice for speed).

DELETE FROM audit.t0_dataset_hash_baseline
WHERE entrypoint = 'ml.woe_anchor_feature_store_t0_v1_v'
  AND sample_limit = 2000;

WITH picked AS (
  SELECT t0_day
  FROM (
    SELECT DISTINCT edited_date::date AS t0_day
    FROM ml.woe_anchor_feature_store_t0_v1_v
    WHERE edited_date IS NOT NULL
  ) d
  ORDER BY t0_day DESC
  LIMIT 10
)
INSERT INTO audit.t0_dataset_hash_baseline (entrypoint, t0_day, sample_limit, dataset_sha256)
SELECT
  'ml.woe_anchor_feature_store_t0_v1_v',
  p.t0_day,
  2000,
  audit.dataset_sha256('ml.woe_anchor_feature_store_t0_v1_v'::regclass, p.t0_day, 2000)
FROM picked p
ON CONFLICT (entrypoint, t0_day, sample_limit)
DO UPDATE SET dataset_sha256 = EXCLUDED.dataset_sha256,
              computed_at    = now();
