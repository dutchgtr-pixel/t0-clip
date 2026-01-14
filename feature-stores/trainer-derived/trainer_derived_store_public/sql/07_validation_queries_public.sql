-- 07_validation_queries.sql

-- Coverage of derived store relative to training base keys:
WITH base AS (
  SELECT generation, listing_id, edited_date AS t0
  FROM ml.anchor_features_v1_mv
  WHERE edited_date IS NOT NULL
),
store AS (
  SELECT generation, listing_id, t0
  FROM ml.trainer_derived_features_train_v
)
SELECT
  COUNT(*) AS base_rows,
  COUNT(s.listing_id) AS covered_rows,
  ROUND(100.0 * COUNT(s.listing_id) / NULLIF(COUNT(*),0), 3) AS coverage_pct
FROM base b
LEFT JOIN store s
  ON s.generation=b.generation AND s.listing_id=b.listing_id AND s.t0=b.t0;

-- Fast single-key lookup plan should use the MV PK:
EXPLAIN (ANALYZE, BUFFERS)
SELECT d.generation, d.listing_id, d.t0
FROM ml.trainer_derived_features_train_v d
JOIN (
  SELECT * FROM unnest(
    ARRAY[13]::int[],
    ARRAY[123456789]::bigint[],
    ARRAY[timestamptz '2026-01-01 00:00:00+00']::timestamptz[]
  ) AS t(generation, listing_id, t0)
) k
  ON k.generation=d.generation AND k.listing_id=d.listing_id AND k.t0=d.t0;

-- Certification registry check:
SELECT entrypoint, status, certified_at, dataset_days, notes
FROM audit.t0_cert_registry
WHERE entrypoint IN (
  'ml.trainer_derived_feature_store_t0_v1_v',
  'ml.trainer_derived_features_train_v'
)
ORDER BY entrypoint;
