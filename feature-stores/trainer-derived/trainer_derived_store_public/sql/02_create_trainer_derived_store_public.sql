-- 02_create_trainer_derived_store.sql
-- Creates:
--   - ml.trainer_derived_features_v1 (logical view)
--   - ml.trainer_derived_features_v1_mv (materialized view for fast training lookup)
--   - ml.trainer_derived_feature_store_t0_v1_v (cert entrypoint)
--   - ml.trainer_derived_features_train_v (guarded consumer view)
--
-- Preconditions:
--   - ml.market_feature_store_train_v exists + is certified
--   - ml.image_features_unified_v1_train_v exists + is certified
--   - audit.require_certified_strict exists
--
-- IMPORTANT:
--   - The MV is keyed by (generation, listing_id, t0) and must have a unique index.

BEGIN;

-- 1) Logical view: compute derived features deterministically at T0.
CREATE OR REPLACE VIEW ml.trainer_derived_features_v1 AS
WITH base AS (
  -- Anchor universe + t0 to the same place training uses
  SELECT
    b.generation,
    b.listing_id,
    b.edited_date AS t0
  FROM ml.anchor_features_v1_mv b
  WHERE b.edited_date IS NOT NULL
),
fs AS (
  -- Pull required fields from the certified training surface (T0-safe)
  SELECT
    generation,
    listing_id,
    edited_date AS t0,
    price,
    condition_score,
    battery_pct_effective,
    delta_vs_sold_median_30d,
    delta_vs_ask_median_day,
    ptv_final,
    COALESCE(damage_severity_ai, 0)::int AS sev,
    COALESCE(damage_binary_ai, 0)::int   AS binary
  FROM ml.market_feature_store_train_v
  WHERE edited_date IS NOT NULL
),
f AS (
  SELECT
    b.generation,
    b.listing_id,
    b.t0,
    fs.price,
    fs.condition_score,
    fs.battery_pct_effective,
    fs.delta_vs_sold_median_30d,
    fs.delta_vs_ask_median_day,
    fs.ptv_final,
    fs.sev,
    fs.binary
  FROM base b
  LEFT JOIN fs
    ON fs.generation = b.generation
   AND fs.listing_id    = b.listing_id
   AND fs.t0         = b.t0
),
w AS (
  SELECT
    f.*,
    ((EXTRACT(ISODOW FROM f.t0)::int) - 1)::int AS dow,
    (CASE WHEN EXTRACT(ISODOW FROM f.t0)::int IN (6,7) THEN 1 ELSE 0 END)::int AS is_weekend,

    -- strict T0: count strictly before t0 (exclude same-t0)
    COUNT(*) OVER (
      PARTITION BY f.generation
      ORDER BY f.t0
      RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '00:00:00.000001' PRECEDING
    )::int AS gen_30d_post_count,

    COUNT(*) OVER (
      ORDER BY f.t0
      RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '00:00:00.000001' PRECEDING
    )::int AS allgen_30d_post_count
  FROM f
),
img AS (
  SELECT generation, listing_id, battery_pct_img
  FROM ml.image_features_unified_v1_train_v
),
j AS (
  SELECT
    w.*,
    i.battery_pct_img::float8 AS battery_pct_img,
    NULLIF(w.battery_pct_effective, 0)::float8 AS battery_pct_text,
    COALESCE(i.battery_pct_img::float8, NULLIF(w.battery_pct_effective, 0)::float8) AS battery_pct_effective_fused,
    CASE
      WHEN i.battery_pct_img IS NOT NULL AND NULLIF(w.battery_pct_effective,0) IS NOT NULL
      THEN (i.battery_pct_img::float8 - NULLIF(w.battery_pct_effective,0)::float8)
      ELSE NULL::float8
    END AS battery_img_minus_text,
    CASE
      WHEN i.battery_pct_img IS NOT NULL AND NULLIF(w.battery_pct_effective,0) IS NOT NULL
       AND ABS(i.battery_pct_img::float8 - NULLIF(w.battery_pct_effective,0)::float8) >= 7.0
      THEN 1 ELSE 0
    END AS battery_img_conflict
  FROM w
  LEFT JOIN img i
    ON i.generation = w.generation
   AND i.listing_id    = w.listing_id
)
SELECT
  generation,
  listing_id,
  t0,

  dow,
  is_weekend,
  gen_30d_post_count,
  allgen_30d_post_count,

  battery_img_conflict,
  battery_img_minus_text,
  battery_pct_effective_fused,

  -- rule features (placeholders; align with your current policy)
  (CASE WHEN condition_score >= 0.90 AND sev = 0 THEN 1 ELSE 0 END)::int AS rocket_clean,
  (CASE WHEN condition_score >= 0.70 AND sev <= 1 THEN 1 ELSE 0 END)::int AS rocket_heavy,

  (CASE WHEN ptv_final IS NOT NULL AND ptv_final < 0.95 THEN 1 ELSE 0 END)::int AS fast_pattern_v2,
  (CASE WHEN COALESCE(delta_vs_sold_median_30d,0) > 500 THEN 1 ELSE 0 END)::int AS is_zombie_pattern

FROM j;

-- 2) Materialized view for fast, indexed key lookups during training.
DROP MATERIALIZED VIEW IF EXISTS ml.trainer_derived_features_v1_mv;

CREATE MATERIALIZED VIEW ml.trainer_derived_features_v1_mv AS
SELECT * FROM ml.trainer_derived_features_v1;

-- Unique index required (fast joins + REFRESH CONCURRENTLY)
CREATE UNIQUE INDEX trainer_derived_features_v1_mv_pk
  ON ml.trainer_derived_features_v1_mv (generation, listing_id, t0);

ANALYZE ml.trainer_derived_features_v1_mv;

-- 3) Certification entrypoint (adds edited_date for dataset hashing)
CREATE OR REPLACE VIEW ml.trainer_derived_feature_store_t0_v1_v AS
SELECT
  t0 AS edited_date,
  *
FROM ml.trainer_derived_features_v1_mv;

-- 4) Guarded consumer view (fail-closed)
CREATE OR REPLACE VIEW ml.trainer_derived_features_train_v AS
SELECT d.*
FROM ml.trainer_derived_features_v1_mv d
CROSS JOIN (
  SELECT audit.require_certified_strict('ml.trainer_derived_feature_store_t0_v1_v', interval '24 hours') AS ok
) g;

COMMIT;
