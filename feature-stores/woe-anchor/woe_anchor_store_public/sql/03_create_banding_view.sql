-- 03_create_banding_view.sql
-- Creates ml.v_woe_anchor_bands_live_v1 (banding view) used by scoring.
-- NOTE: This view expects a currently-active model_key + final cuts row.

CREATE OR REPLACE VIEW ml.v_woe_anchor_bands_live_v1 AS
WITH active AS (
  SELECT
    model_key,
    COALESCE(dsold_t1, -0.15)::float8 AS dsold_t1,
    COALESCE(dsold_t2, -0.05)::float8 AS dsold_t2,
    COALESCE(dsold_t3,  0.05)::float8 AS dsold_t3,
    COALESCE(dsold_t4,  0.15)::float8 AS dsold_t4,
    COALESCE(
      base_logit,
      LN(base_rate / NULLIF(1.0 - base_rate, 0.0))
    )::float8 AS base_logit
  FROM ml.woe_anchor_model_registry_v1
  WHERE is_active = true
  ORDER BY created_at DESC
  LIMIT 1
),
cuts AS (
  SELECT c1::float8, c2::float8
  FROM ml.woe_anchor_cuts_v1 c
  JOIN active a ON a.model_key = c.model_key
  WHERE c.fold_id IS NULL
),
b0 AS (
  SELECT
    f.generation,
    f.listing_id,
    COALESCE(f.t0, f.edited_date) AS t0,

    f.delta_vs_sold_median_30d,
    f.condition_score,
    f.damage_severity_ai,
    f.battery_pct_effective,

    f.seller_rating,
    f.review_count,
    f.member_since_year,

    f.ai_ship_can,
    f.ai_ship_pickup,
    f.ai_pickup_only_bin,

    f.ai_sale_mode_firm,
    f.ai_sale_mode_bids,
    f.ai_sale_mode_obo,

    u.image_count,
    u.caption_share,
    u.stock_photo_share,
    u.photo_quality_avg,
    u.bg_clean_avg,
    u.dmg_band_hq_struct,
    u.charger_bundle_level,
    u.box_present,
    u.receipt_present,

    (u.listing_id IS NOT NULL)::int AS has_assets

  FROM ml.socio_market_feature_store_train_v f
  LEFT JOIN ml.iphone_image_features_unified_v1_train_v u
    ON u.generation = f.generation
   AND u.listing_id    = f.listing_id
),
b1 AS (
  SELECT
    b0.*,
    CASE
      WHEN b0.has_assets = 0 THEN NULL::float8
      ELSE (
        0.30*LN(1.0 + COALESCE(b0.image_count,0)) +
        0.25*COALESCE(b0.photo_quality_avg,0) +
        0.20*COALESCE(b0.bg_clean_avg,0) +
        0.15*(1.0 - COALESCE(b0.stock_photo_share,0)) +
        0.10*COALESCE(b0.caption_share,0)
      )::float8
    END AS presentation_score,

    CASE
      WHEN b0.has_assets = 0 THEN NULL::int
      ELSE (
        COALESCE(b0.charger_bundle_level,0)::int +
        (CASE WHEN COALESCE(b0.box_present,false) THEN 1 ELSE 0 END) +
        (CASE WHEN COALESCE(b0.receipt_present,false) THEN 1 ELSE 0 END)
      )
    END AS acc_score
  FROM b0
)
SELECT
  a.model_key,
  a.base_logit,

  b1.generation,
  b1.listing_id,
  b1.t0,

  CASE
    WHEN b1.delta_vs_sold_median_30d IS NULL THEN 'dSold_missing'
    WHEN b1.delta_vs_sold_median_30d <= a.dsold_t1 THEN 'dSold_cheap'
    WHEN b1.delta_vs_sold_median_30d <= a.dsold_t2 THEN 'dSold_under'
    WHEN b1.delta_vs_sold_median_30d <= a.dsold_t3 THEN 'dSold_fair'
    WHEN b1.delta_vs_sold_median_30d <= a.dsold_t4 THEN 'dSold_over'
    ELSE 'dSold_overpriced'
  END AS dsold_band,

  CASE
    WHEN b1.seller_rating >= 9.7 AND b1.review_count >= 50
     AND b1.member_since_year IS NOT NULL
     AND b1.member_since_year <= EXTRACT(YEAR FROM b1.t0) - 3 THEN 'HIGH'
    WHEN b1.seller_rating >= 9.0 AND b1.review_count >= 10 THEN 'MED'
    ELSE 'LOW'
  END AS trust_tier,

  CASE
    WHEN b1.ai_pickup_only_bin = 1 OR b1.ai_ship_pickup = 1 THEN 'pickup_heavy'
    WHEN b1.ai_ship_can = 1 THEN 'shipping_ok'
    ELSE 'ship_unknown'
  END AS ship_band,

  CASE
    WHEN b1.ai_sale_mode_firm = 1 THEN 'firm'
    WHEN b1.ai_sale_mode_bids = 1 THEN 'bids'
    WHEN b1.ai_sale_mode_obo  = 1 THEN 'obo'
    ELSE 'sale_unspec'
  END AS sale_band,

  CASE
    WHEN b1.condition_score IS NULL THEN 'cond_missing'
    WHEN b1.condition_score >= 0.90 THEN 'cond_hi'
    WHEN b1.condition_score >= 0.70 THEN 'cond_mid'
    ELSE 'cond_lo'
  END AS cond_band,

  CASE
    WHEN b1.damage_severity_ai IS NULL THEN 'dmg_missing'
    WHEN b1.damage_severity_ai <= 0 THEN 'dmg_0'
    WHEN b1.damage_severity_ai = 1 THEN 'dmg_1'
    WHEN b1.damage_severity_ai = 2 THEN 'dmg_2'
    ELSE 'dmg_3p'
  END AS dmg_ai_band,

  CASE
    WHEN b1.battery_pct_effective IS NULL THEN 'bat_missing'
    WHEN b1.battery_pct_effective >= 95 THEN 'bat_hi'
    WHEN b1.battery_pct_effective >= 88 THEN 'bat_mid'
    ELSE 'bat_lo'
  END AS bat_band,

  b1.has_assets,
  b1.presentation_score,

  CASE
    WHEN b1.presentation_score IS NULL THEN NULL
    WHEN b1.presentation_score < (SELECT c1 FROM cuts) THEN 'present_lo'
    WHEN b1.presentation_score < (SELECT c2 FROM cuts) THEN 'present_mid'
    ELSE 'present_hi'
  END AS presentation_band,

  CASE
    WHEN b1.has_assets = 0 THEN NULL
    WHEN b1.dmg_band_hq_struct IS NULL THEN 'vdmg_missing'
    WHEN b1.dmg_band_hq_struct <= 0 THEN 'vdmg_0'
    WHEN b1.dmg_band_hq_struct = 1 THEN 'vdmg_1'
    WHEN b1.dmg_band_hq_struct = 2 THEN 'vdmg_2'
    ELSE 'vdmg_3p'
  END AS vdmg_band,

  CASE
    WHEN b1.acc_score IS NULL THEN NULL
    WHEN b1.acc_score <= 0 THEN 'acc_lo'
    WHEN b1.acc_score <  2 THEN 'acc_mid'
    ELSE 'acc_hi'
  END AS accessories_band

FROM b1
CROSS JOIN active a;
