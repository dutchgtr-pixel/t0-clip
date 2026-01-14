-- ============================================================
-- Fusion Store — Baseline dataset hashes (Gate C)
--
-- This will insert up to 365 distinct edited_date::date days.
-- If fewer days exist in the store, fewer baselines will be written.
-- ============================================================

DELETE FROM audit.t0_dataset_hash_baseline
WHERE entrypoint = 'ml.fusion_feature_store_t0_v1_v'
  AND sample_limit = 2000;

WITH picked AS (
  SELECT t0_day
  FROM (
    SELECT DISTINCT edited_date::date AS t0_day
    FROM ml.fusion_feature_store_t0_v1_v
    WHERE edited_date IS NOT NULL
  ) d
  ORDER BY t0_day DESC
  LIMIT 365
)
INSERT INTO audit.t0_dataset_hash_baseline (entrypoint, t0_day, sample_limit, dataset_sha256)
SELECT
  'ml.fusion_feature_store_t0_v1_v',
  p.t0_day,
  2000,
  audit.dataset_sha256('ml.fusion_feature_store_t0_v1_v'::regclass, p.t0_day, 2000)
FROM picked p
ON CONFLICT (entrypoint, t0_day, sample_limit)
DO UPDATE SET
  dataset_sha256 = EXCLUDED.dataset_sha256,
  computed_at    = now();

-- Sanity
SELECT COUNT(*) AS baseline_days, MIN(t0_day) AS oldest, MAX(t0_day) AS newest
FROM audit.t0_dataset_hash_baseline
WHERE entrypoint='ml.fusion_feature_store_t0_v1_v'
  AND sample_limit=2000;
