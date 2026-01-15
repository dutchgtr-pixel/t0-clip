DROP VIEW IF EXISTS ml.iphone_image_features_unified_v1;

CREATE VIEW ml.iphone_image_features_unified_v1 AS
WITH assets AS (
  SELECT
    generation,
    listing_id,
    COUNT(*)::bigint AS image_count,
    COUNT(*) FILTER (WHERE caption_text IS NOT NULL AND caption_text <> '')::bigint AS caption_count_raw
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
),

/* Generic per-listing aggregates from the raw per-image features table
   (quality/stock/battery). */
generic AS (
  SELECT
    generation,
    listing_id,

    COUNT(*)::bigint AS n_feat_rows_all,

    AVG(photo_quality_level) AS photo_quality_avg,
    MAX(photo_quality_level) AS photo_quality_max,

    AVG(background_clean_level) AS bg_clean_avg,

    BOOL_OR(is_stock_photo IS TRUE) AS has_stock_photo,
    AVG((is_stock_photo IS TRUE)::int)::double precision AS stock_photo_share,

    COUNT(*) FILTER (WHERE battery_screenshot IS TRUE)::bigint AS n_battery_screenshot_imgs,
    MAX(battery_health_pct_img)
      FILTER (WHERE battery_screenshot IS TRUE AND battery_health_pct_img IS NOT NULL)
      AS battery_pct_img

  FROM ml.iphone_image_features_v1
  WHERE feature_version = 1
  GROUP BY 1,2
),

/* Fix for your error:
   ml.iphone_image_damage_features_v1 does NOT have dmg_mean_hq_struct,
   so we compute it here directly from per-image rows using the same HQ+struct filters. */
dmg_struct_mean AS (
  SELECT
    generation,
    listing_id,
    AVG(visible_damage_level) FILTER (
      WHERE visible_damage_level IS NOT NULL
        AND is_stock_photo IS FALSE
        AND photo_quality_level >= 3
        AND damage_on_protector_only IS NOT TRUE
    ) AS dmg_mean_hq_struct
  FROM ml.iphone_image_features_v1
  WHERE feature_version = 1
  GROUP BY 1,2
)

SELECT
  /* ---------- keys ---------- */
  a.generation,
  a.listing_id,

  /* ---------- image inventory ---------- */
  a.image_count,

  /* legacy-compatible caption_count (NULL when 0) */
  CASE WHEN a.caption_count_raw = 0 THEN NULL::bigint ELSE a.caption_count_raw END AS caption_count,

  /* recommended “true” count (0 when none) */
  a.caption_count_raw,

  (a.caption_count_raw > 0) AS has_any_caption,
  (a.caption_count_raw::numeric / NULLIF(a.image_count,0)::numeric) AS caption_share,

  /* ---------- generic image quality/meta ---------- */
  COALESCE(g.n_feat_rows_all, 0::bigint) AS n_feat_rows_all,
  (COALESCE(g.n_feat_rows_all,0)::numeric / NULLIF(a.image_count,0)::numeric) AS feat_row_coverage_assets,

  g.photo_quality_avg,
  g.photo_quality_max,
  g.bg_clean_avg,

  g.has_stock_photo,
  g.stock_photo_share,

  (COALESCE(g.n_battery_screenshot_imgs,0) > 0) AS has_battery_screenshot,
  COALESCE(g.n_battery_screenshot_imgs,0) AS n_battery_screenshot_imgs,
  g.battery_pct_img,

  /* ============================================================
     ACCESSORIES (ml.iphone_image_accessory_features_v1)
     ============================================================ */
  acc.n_feat_rows              AS acc_n_feat_rows,
  acc.n_acc_done_imgs,
  acc.acc_done_coverage_assets,
  acc.is_incomplete            AS acc_is_incomplete,
  acc.hit_cap_16               AS acc_hit_cap_16,

  acc.any_acc_output_observed,
  acc.acc_done_but_all_outputs_null,

  acc.box_present,
  acc.box_known,
  acc.box_state_max,
  acc.box_state_known,

  acc.charger_present,
  acc.charger_known,
  acc.charger_bundle_level,
  acc.cable_present,
  acc.cable_known,
  acc.brick_present,
  acc.brick_known,

  acc.earbuds_present,
  acc.earbuds_known,

  acc.case_present,
  acc.case_known,
  acc.case_count_max,
  acc.case_count_p90,
  acc.case_count_known,

  acc.receipt_present,
  acc.receipt_known,

  acc.other_accessory_present,
  acc.other_accessory_known,

  /* ============================================================
     COLOR (ml.iphone_image_color_features_v1)
     ============================================================ */
  col.n_feat_rows              AS color_n_feat_rows,
  col.n_color_done_imgs,
  col.feat_row_coverage_assets AS color_feat_row_coverage_assets,
  col.color_done_coverage_assets,
  col.is_incomplete            AS color_is_incomplete,
  col.hit_cap_16               AS color_hit_cap_16,

  col.has_feature_rows         AS color_has_feature_rows,
  col.backlog_not_done         AS color_backlog_not_done,
  col.color_processed,
  col.color_known,
  col.done_but_no_votes        AS color_done_but_no_votes,
  col.color_conflict,

  col.n_color_votes,
  col.n_distinct_color_keys,

  col.color_key_primary,
  col.color_name_primary,
  col.color_primary_vote_share,

  col.color_key_secondary,
  col.color_name_secondary,
  col.color_secondary_vote_share,

  col.color_vote_margin_votes,
  col.color_vote_margin_share,

  col.color_conf_max,
  col.color_conf_avg,
  col.color_conf_p10,
  col.color_conf_p50,
  col.color_conf_p90,

  col.p_votes_from_case,
  col.p_votes_stock,
  col.p_votes_hq,

  col.color_key_primary_strict,

  /* convenient alias for older codepaths */
  col.color_key_primary        AS body_color_key_main,

  /* model correction + spam (computed in color store) */
  col.model_effective,
  col.model_effective_type,
  col.model_effective_generation,

  /* IMPORTANT: do not use for joins; use for modeling logic only */
  COALESCE(col.model_effective_generation, a.generation) AS generation_effective_for_modeling,

  col.model_fixed_any,
  col.model_fix_new_model,
  col.model_generation_mismatch,
  col.is_spam_below13,

  /* ============================================================
     DAMAGE (ml.iphone_image_damage_features_v1)
     ============================================================ */
  dmg.n_feat_rows              AS dmg_n_feat_rows,
  dmg.n_dmg_labeled,
  dmg.dmg_coverage_assets,
  dmg.is_incomplete            AS dmg_is_incomplete,
  dmg.hit_cap_16               AS dmg_hit_cap_16,

  /* summary damage signals */
  dmg.dmg_max_all              AS dmg_max,
  dmg.dmg_mean_all             AS dmg_mean,
  dmg.dmg_p90_all              AS dmg_p90,

  /* NULL when no labeled images */
  CASE WHEN dmg.n_dmg_labeled > 0 THEN (dmg.dmg_max_all >= 3) END AS dmg_any_3plus,
  CASE WHEN dmg.n_dmg_labeled > 0 THEN (dmg.dmg_max_all >= 5) END AS dmg_any_5plus,
  CASE WHEN dmg.n_dmg_labeled > 0 THEN (dmg.dmg_max_all >= 8) END AS dmg_any_8plus,

  dmg.has_screen_protector_any,
  dmg.protector_only_share_labeled,

  /* higher-precision structural HQ subset */
  dmg.n_hq_struct,
  dmg.dmg_max_hq_struct,
  dmg.dmg_p90_hq_struct,
  ds.dmg_mean_hq_struct,          -- <-- FIXED: computed here (was missing in dmg view)
  dmg.share_3plus_hq_struct,
  dmg.share_5plus_hq_struct,
  dmg.share_8plus_hq_struct,
  dmg.dmg_band_hq_struct,

  /* targeted severe issues */
  dmg.has_display_fault_5plus,
  dmg.has_back_glass_struct_5plus,
  dmg.has_front_glass_struct_5plus,
  dmg.has_camera_lens_issue_5plus

FROM assets a
LEFT JOIN generic g
  ON g.generation = a.generation AND g.listing_id = a.listing_id
LEFT JOIN ml.iphone_image_accessory_features_v1 acc
  ON acc.generation = a.generation AND acc.listing_id = a.listing_id
LEFT JOIN ml.iphone_image_color_features_v1 col
  ON col.generation = a.generation AND col.listing_id = a.listing_id
LEFT JOIN ml.iphone_image_damage_features_v1 dmg
  ON dmg.generation = a.generation AND dmg.listing_id = a.listing_id
LEFT JOIN dmg_struct_mean ds
  ON ds.generation = a.generation AND ds.listing_id = a.listing_id
;
