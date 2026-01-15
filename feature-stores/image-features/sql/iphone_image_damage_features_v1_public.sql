CREATE VIEW ml.iphone_image_damage_features_v1 AS
WITH assets AS (
  SELECT
    generation,
    listing_id,
    COUNT(*)::bigint AS n_assets
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
),
per_image AS (
  SELECT
    generation,
    listing_id,
    image_index,
    photo_quality_level,
    background_clean_level,
    is_stock_photo,
    has_screen_protector,
    damage_on_protector_only,
    visible_damage_level,
    damage_summary_text
  FROM ml.iphone_image_features_v1
  WHERE feature_version = 1
),
per_listing AS (
  SELECT
    generation,
    listing_id,

    -- counts / coverage inside features table
    COUNT(*) AS n_feat_rows,
    COUNT(*) FILTER (WHERE visible_damage_level IS NOT NULL) AS n_dmg_labeled,
    MIN(image_index) FILTER (WHERE visible_damage_level IS NOT NULL) AS min_labeled_idx,
    MAX(image_index) FILTER (WHERE visible_damage_level IS NOT NULL) AS max_labeled_idx,

    -- quality / background (labeled images only)
    AVG(photo_quality_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS photo_quality_avg,
    MAX(photo_quality_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS photo_quality_max,
    AVG(background_clean_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS bg_clean_avg,

    -- stock
    BOOL_OR(is_stock_photo) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_stock_labeled,
    AVG((is_stock_photo)::int) FILTER (WHERE visible_damage_level IS NOT NULL) AS stock_share_labeled,

    -- protectors
    BOOL_OR(has_screen_protector) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_screen_protector_any,
    AVG((damage_on_protector_only IS TRUE)::int) FILTER (WHERE visible_damage_level IS NOT NULL) AS protector_only_share_labeled,

    -- damage aggregates (ALL labeled)
    MIN(visible_damage_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS dmg_min_all,
    MAX(visible_damage_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS dmg_max_all,
    AVG(visible_damage_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS dmg_mean_all,
    STDDEV_SAMP(visible_damage_level) FILTER (WHERE visible_damage_level IS NOT NULL) AS dmg_sd_all,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY visible_damage_level)
      FILTER (WHERE visible_damage_level IS NOT NULL) AS dmg_p90_all,

    -- damage aggregates (NO stock)
    MAX(visible_damage_level) FILTER (
      WHERE visible_damage_level IS NOT NULL AND NOT is_stock_photo
    ) AS dmg_max_nostock,

    -- evidence: HQ non-stock
    COUNT(*) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
    ) AS n_hq_nostock,

    MAX(visible_damage_level) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
    ) AS dmg_max_hq,

    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY visible_damage_level)
      FILTER (
        WHERE visible_damage_level IS NOT NULL
          AND NOT is_stock_photo
          AND photo_quality_level >= 3
      ) AS dmg_p90_hq,

    AVG(visible_damage_level) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
    ) AS dmg_mean_hq,

    -- evidence: HQ non-stock structural (exclude protector-only)
    COUNT(*) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS n_hq_struct,

    MAX(visible_damage_level) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS dmg_max_hq_struct,

    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY visible_damage_level)
      FILTER (
        WHERE visible_damage_level IS NOT NULL
          AND NOT is_stock_photo
          AND photo_quality_level >= 3
          AND damage_on_protector_only IS NOT TRUE
      ) AS dmg_p90_hq_struct,

    AVG((visible_damage_level >= 3)::int) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS share_3plus_hq_struct,

    AVG((visible_damage_level >= 5)::int) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS share_5plus_hq_struct,

    AVG((visible_damage_level >= 8)::int) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND NOT is_stock_photo
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS share_8plus_hq_struct,

    -- damage type flags (simple keywording)
    BOOL_OR(
      (damage_summary_text ILIKE '%lcd%' OR damage_summary_text ILIKE '%oled%'
       OR damage_summary_text ILIKE '%stripe%' OR damage_summary_text ILIKE '%vertical line%'
       OR damage_summary_text ILIKE '%horizontal line%' OR damage_summary_text ILIKE '%screen line%')
      AND visible_damage_level >= 5
    ) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_display_fault_5plus,

    BOOL_OR(
      (damage_summary_text ILIKE '%back%' AND (damage_summary_text ILIKE '%crack%' OR damage_summary_text ILIKE '%shatter%'))
      AND visible_damage_level >= 5
    ) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_back_glass_struct_5plus,

    BOOL_OR(
      (damage_summary_text ILIKE '%front%' AND (damage_summary_text ILIKE '%crack%' OR damage_summary_text ILIKE '%shatter%'))
      AND visible_damage_level >= 5
    ) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_front_glass_struct_5plus,

    BOOL_OR(
      ((damage_summary_text ILIKE '%camera%' OR damage_summary_text ILIKE '%lens%')
       AND (damage_summary_text ILIKE '%broken%' OR damage_summary_text ILIKE '%missing%' OR damage_summary_text ILIKE '%crack%'))
      AND visible_damage_level >= 5
    ) FILTER (WHERE visible_damage_level IS NOT NULL) AS has_camera_lens_issue_5plus

  FROM per_image
  GROUP BY 1,2
),
final AS (
  SELECT
    a.generation,
    a.listing_id,
    a.n_assets,

    p.n_feat_rows,
    p.n_dmg_labeled,
    p.min_labeled_idx,
    p.max_labeled_idx,

    (p.n_dmg_labeled::numeric / NULLIF(a.n_assets,0)) AS dmg_coverage_assets,
    (a.n_assets - COALESCE(p.n_dmg_labeled,0)) AS unlabeled_tail_assets,
    (COALESCE(p.n_dmg_labeled,0) < a.n_assets) AS is_incomplete,

    (p.max_labeled_idx = 7)  AS hit_cap_8,
    (p.max_labeled_idx = 15) AS hit_cap_16,

    p.photo_quality_avg,
    p.photo_quality_max,
    p.bg_clean_avg,

    p.has_stock_labeled,
    p.stock_share_labeled,

    p.has_screen_protector_any,
    p.protector_only_share_labeled,

    p.dmg_min_all,
    p.dmg_max_all,
    p.dmg_mean_all,
    p.dmg_sd_all,
    p.dmg_p90_all,

    p.dmg_max_nostock,

    p.n_hq_nostock,
    p.dmg_max_hq,
    p.dmg_p90_hq,
    p.dmg_mean_hq,

    p.n_hq_struct,
    p.dmg_max_hq_struct,
    p.dmg_p90_hq_struct,
    p.share_3plus_hq_struct,
    p.share_5plus_hq_struct,
    p.share_8plus_hq_struct,

    CASE
      WHEN p.dmg_max_hq_struct IS NULL THEN NULL
      WHEN p.dmg_max_hq_struct <= 2 THEN 0
      WHEN p.dmg_max_hq_struct <= 4 THEN 1
      WHEN p.dmg_max_hq_struct <= 7 THEN 2
      ELSE 3
    END AS dmg_band_hq_struct,

    p.has_display_fault_5plus,
    p.has_back_glass_struct_5plus,
    p.has_front_glass_struct_5plus,
    p.has_camera_lens_issue_5plus

  FROM assets a
  LEFT JOIN per_listing p
    ON p.generation = a.generation
   AND p.listing_id    = a.listing_id
)
SELECT * FROM final;
