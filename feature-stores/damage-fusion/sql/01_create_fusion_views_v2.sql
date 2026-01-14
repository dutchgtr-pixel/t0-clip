-- ============================================================
-- Fusion Store (v2) — Create views
--   - ml.v_damage_fusion_features_v2
--   - ml.v_damage_fusion_features_v2_scored
--
-- Source of truth: 'docs/REFERENCE__Create_Improved_Fusion_View_v2.txt'
-- ============================================================

CREATE OR REPLACE VIEW ml.v_damage_fusion_features_v2 AS
WITH img_prof AS (
  SELECT
    generation,
    listing_id,

    COUNT(*) FILTER (WHERE feature_version=1) AS img_rows_v1,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE) AS img_rows_damage_done,

    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND is_stock_photo IS TRUE) AS img_stock_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE) AS img_real_cnt,

    -- photo quality counts (0-4) on real images
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level=0) AS img_pq0_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level=1) AS img_pq1_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level=2) AS img_pq2_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level=3) AS img_pq3_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level=4) AS img_pq4_cnt,
    AVG(photo_quality_level::numeric) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND photo_quality_level IS NOT NULL
    ) AS img_pq_avg_real,

    -- background cleanliness counts (0-2) on real images
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND background_clean_level=0) AS img_bg0_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND background_clean_level=1) AS img_bg1_cnt,
    COUNT(*) FILTER (WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND background_clean_level=2) AS img_bg2_cnt,
    AVG(background_clean_level::numeric) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE AND COALESCE(is_stock_photo,false)=FALSE AND background_clean_level IS NOT NULL
    ) AS img_bg_avg_real,

    -- usable-image counts
    COUNT(*) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE
        AND COALESCE(is_stock_photo,false)=FALSE
        AND COALESCE(photo_quality_level,0) >= 2
    ) AS img_real_q2_cnt,

    COUNT(*) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE
        AND COALESCE(is_stock_photo,false)=FALSE
        AND COALESCE(photo_quality_level,0) >= 2
        AND visible_damage_level IS NOT NULL
    ) AS img_real_q2_vis_cnt,

    COUNT(*) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE
        AND COALESCE(is_stock_photo,false)=FALSE
        AND COALESCE(photo_quality_level,0) >= 1
        AND visible_damage_level IS NOT NULL
    ) AS img_real_q1_vis_cnt,

    -- damage maxima: q2 primary
    MAX(
      CASE
        WHEN feature_version=1 AND damage_done=TRUE
          AND COALESCE(is_stock_photo,false)=FALSE
          AND COALESCE(photo_quality_level,0) >= 2
          AND visible_damage_level IS NOT NULL
        THEN CASE WHEN COALESCE(damage_on_protector_only,false) THEN 0 ELSE visible_damage_level END
      END
    ) AS img_max_eff_q2,

    -- damage maxima: q1 fallback
    MAX(
      CASE
        WHEN feature_version=1 AND damage_done=TRUE
          AND COALESCE(is_stock_photo,false)=FALSE
          AND COALESCE(photo_quality_level,0) >= 1
          AND visible_damage_level IS NOT NULL
        THEN CASE WHEN COALESCE(damage_on_protector_only,false) THEN 0 ELSE visible_damage_level END
      END
    ) AS img_max_eff_q1,

    -- non-protector structural max (q2)
    MAX(
      CASE
        WHEN feature_version=1 AND damage_done=TRUE
          AND COALESCE(is_stock_photo,false)=FALSE
          AND COALESCE(photo_quality_level,0) >= 2
          AND COALESCE(damage_on_protector_only,false)=FALSE
          AND visible_damage_level IS NOT NULL
        THEN visible_damage_level
      END
    ) AS img_max_noprotector_q2,

    -- evidence strength (q2)
    COUNT(*) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE
        AND COALESCE(is_stock_photo,false)=FALSE
        AND COALESCE(photo_quality_level,0) >= 2
        AND COALESCE(damage_on_protector_only,false)=FALSE
        AND visible_damage_level >= 3
    ) AS img_n_ge3_real_q2,

    COUNT(*) FILTER (
      WHERE feature_version=1 AND damage_done=TRUE
        AND COALESCE(is_stock_photo,false)=FALSE
        AND COALESCE(photo_quality_level,0) >= 2
        AND COALESCE(damage_on_protector_only,false)=FALSE
        AND visible_damage_level >= 8
    ) AS img_n_ge8_real_q2

  FROM ml.iphone_image_features_v1
  GROUP BY 1,2
),
batt_prof AS (
  SELECT
    generation,
    listing_id,
    MIN(battery_health_pct_img) FILTER (
      WHERE feature_version=1 AND battery_screenshot=TRUE AND battery_health_pct_img BETWEEN 50 AND 100
    ) AS batt_img_min,
    MAX(battery_health_pct_img) FILTER (
      WHERE feature_version=1 AND battery_screenshot=TRUE AND battery_health_pct_img BETWEEN 50 AND 100
    ) AS batt_img_max,
    COUNT(*) FILTER (
      WHERE feature_version=1 AND battery_screenshot=TRUE AND battery_health_pct_img BETWEEN 50 AND 100
    ) AS batt_img_hits
  FROM ml.iphone_image_features_v1
  GROUP BY 1,2
),
base AS (
  SELECT
    l.generation,
    l.listing_id,

    COALESCE(i.img_rows_v1,0) AS img_rows_v1,
    COALESCE(i.img_rows_damage_done,0) AS img_rows_damage_done,
    COALESCE(i.img_stock_cnt,0) AS img_stock_cnt,
    COALESCE(i.img_real_cnt,0) AS img_real_cnt,
    CASE
      WHEN COALESCE(i.img_rows_damage_done,0)=0 THEN NULL
      ELSE (i.img_stock_cnt::numeric / NULLIF(i.img_rows_damage_done,0))
    END AS img_stock_ratio,

    COALESCE(i.img_pq0_cnt,0) AS img_pq0_cnt,
    COALESCE(i.img_pq1_cnt,0) AS img_pq1_cnt,
    COALESCE(i.img_pq2_cnt,0) AS img_pq2_cnt,
    COALESCE(i.img_pq3_cnt,0) AS img_pq3_cnt,
    COALESCE(i.img_pq4_cnt,0) AS img_pq4_cnt,
    i.img_pq_avg_real,

    COALESCE(i.img_bg0_cnt,0) AS img_bg0_cnt,
    COALESCE(i.img_bg1_cnt,0) AS img_bg1_cnt,
    COALESCE(i.img_bg2_cnt,0) AS img_bg2_cnt,
    i.img_bg_avg_real,

    COALESCE(i.img_real_q2_cnt,0) AS img_real_q2_cnt,
    COALESCE(i.img_real_q2_vis_cnt,0) AS img_real_q2_vis_cnt,
    COALESCE(i.img_real_q1_vis_cnt,0) AS img_real_q1_vis_cnt,

    i.img_max_eff_q2,
    i.img_max_eff_q1,
    i.img_max_noprotector_q2,
    COALESCE(i.img_n_ge3_real_q2,0) AS img_n_ge3_real_q2,
    COALESCE(i.img_n_ge8_real_q2,0) AS img_n_ge8_real_q2,

    b.batt_img_min,
    b.batt_img_max,
    CASE WHEN b.batt_img_min IS NOT NULL AND b.batt_img_max IS NOT NULL THEN (b.batt_img_max - b.batt_img_min) END AS batt_img_spread,
    COALESCE(b.batt_img_hits,0) AS batt_img_hits,

    -- robust numeric extraction
    CASE WHEN (l.damage_ai_json->>'battery_effective') ~ '^[0-9]+$'
         THEN (l.damage_ai_json->>'battery_effective')::int END AS batt_eff_text,

    (l.damage_ai_json IS NOT NULL AND l.damage_ai_json->'decision' IS NOT NULL) AS text_damage_scored,
    CASE WHEN (l.damage_ai_json->'decision'->>'sev') ~ '^[0-9]+$'
         THEN (l.damage_ai_json->'decision'->>'sev')::int END AS text_sev_decision,
    CASE WHEN (l.damage_ai_json->'decision'->>'bin') ~ '^[0-9]+$'
         THEN (l.damage_ai_json->'decision'->>'bin')::int END AS text_bin_decision,

    COALESCE((l.damage_ai_json->'decision'->'meta'->>'no_wear_global')::boolean,false)  AS txt_no_wear_global,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'protector_only')::boolean,false) AS txt_protector_only,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'glass')::boolean,false)          AS txt_glass,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'back_glass')::boolean,false)     AS txt_back_glass,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'lens_glass')::boolean,false)     AS txt_lens_glass,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'panel_severe')::boolean,false)   AS txt_panel_severe,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'light_panel')::boolean,false)    AS txt_light_panel,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'charging')::boolean,false)       AS txt_charging,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'non_oem')::boolean,false)        AS txt_non_oem,
    COALESCE((l.damage_ai_json->'decision'->'meta'->>'battery_clamp')::boolean,false)  AS txt_battery_clamp

  FROM "iPhone".iphone_listings l
  LEFT JOIN img_prof  i ON i.generation=l.generation AND i.listing_id=l.listing_id
  LEFT JOIN batt_prof b ON b.generation=l.generation AND b.listing_id=l.listing_id
  WHERE l.spam IS NULL
),
derived AS (
  SELECT
    base.*,

    COALESCE(batt_img_min, batt_eff_text) AS batt_fused,
    CASE
      WHEN batt_img_min IS NOT NULL AND batt_eff_text IS NOT NULL THEN 'both'
      WHEN batt_img_min IS NOT NULL THEN 'img_only'
      WHEN batt_eff_text IS NOT NULL THEN 'text_only'
      ELSE 'none'
    END AS batt_source,

    CASE
      WHEN (img_real_q2_vis_cnt >= 2 AND img_max_eff_q2 IS NOT NULL) THEN 2
      WHEN (img_real_q1_vis_cnt >= 2 AND img_max_eff_q1 IS NOT NULL) THEN 1
      ELSE 0
    END AS img_damage_source_level,

    CASE
      WHEN (img_real_q2_vis_cnt >= 2 AND img_max_eff_q2 IS NOT NULL) THEN img_max_eff_q2
      WHEN (img_real_q1_vis_cnt >= 2 AND img_max_eff_q1 IS NOT NULL) THEN img_max_eff_q1
      ELSE NULL
    END AS img_damage_score,

    CASE
      WHEN img_rows_v1 = 0 THEN 1
      WHEN img_rows_damage_done = 0 THEN 2
      WHEN img_real_cnt = 0 THEN 3
      WHEN img_real_q2_cnt = 0 THEN 4
      WHEN img_real_q2_vis_cnt = 0 THEN 5
      WHEN img_real_q2_vis_cnt < 2 AND img_real_q1_vis_cnt >= 2 THEN 7
      WHEN img_real_q2_vis_cnt < 2 THEN 6
      ELSE 0
    END AS img_score_reason_code,

    GREATEST(
      COALESCE(text_sev_decision, 0),
      CASE
        WHEN (COALESCE(batt_img_min, batt_eff_text) IS NOT NULL AND COALESCE(batt_img_min, batt_eff_text) < 80)
          OR txt_battery_clamp OR txt_non_oem OR txt_charging
        THEN 2 ELSE 0 END,
      CASE
        WHEN txt_glass OR txt_back_glass OR txt_lens_glass OR txt_panel_severe
        THEN 3 ELSE 0 END
    ) AS text_sev_corr,

    CASE
      WHEN COALESCE(batt_img_min, batt_eff_text) IS NULL THEN NULL
      WHEN COALESCE(batt_img_min, batt_eff_text) < 80 THEN 'batt:<80'
      WHEN COALESCE(batt_img_min, batt_eff_text) < 85 THEN 'batt:80-84'
      WHEN COALESCE(batt_img_min, batt_eff_text) < 90 THEN 'batt:85-89'
      WHEN COALESCE(batt_img_min, batt_eff_text) < 95 THEN 'batt:90-94'
      ELSE 'batt:95-100'
    END AS batt_bucket,

    (COALESCE(batt_img_min, batt_eff_text) IS NOT NULL AND COALESCE(batt_img_min, batt_eff_text) < 80) AS batt_lt80,
    (COALESCE(batt_img_min, batt_eff_text) IS NOT NULL AND COALESCE(batt_img_min, batt_eff_text) >= 90) AS batt_ge90,

    CASE
      WHEN (COALESCE(img_max_noprotector_q2, 0) >= 8) OR (GREATEST(
            COALESCE(text_sev_decision, 0),
            CASE
              WHEN (COALESCE(batt_img_min, batt_eff_text) IS NOT NULL AND COALESCE(batt_img_min, batt_eff_text) < 80)
                OR txt_battery_clamp OR txt_non_oem OR txt_charging
              THEN 2 ELSE 0 END,
            CASE WHEN txt_glass OR txt_back_glass OR txt_lens_glass OR txt_panel_severe THEN 3 ELSE 0 END
          ) >= 3)
      THEN 3
      WHEN COALESCE(
             CASE WHEN (img_real_q2_vis_cnt >= 2) THEN img_max_eff_q2 END,
             CASE WHEN (img_real_q1_vis_cnt >= 2) THEN img_max_eff_q1 END,
             0
           ) >= 3
      THEN 2
      WHEN COALESCE(
             CASE WHEN (img_real_q2_vis_cnt >= 2) THEN img_max_eff_q2 END,
             CASE WHEN (img_real_q1_vis_cnt >= 2) THEN img_max_eff_q1 END
           ) = 2
        OR (GREATEST(
             COALESCE(text_sev_decision, 0),
             CASE
               WHEN (COALESCE(batt_img_min, batt_eff_text) IS NOT NULL AND COALESCE(batt_img_min, batt_eff_text) < 80)
                 OR txt_battery_clamp OR txt_non_oem OR txt_charging
               THEN 2 ELSE 0 END,
             CASE WHEN txt_glass OR txt_back_glass OR txt_lens_glass OR txt_panel_severe THEN 3 ELSE 0 END
           ) >= 2)
      THEN 1
      ELSE 0
    END AS damage_fused_tier

  FROM base
)
SELECT * FROM derived;

CREATE OR REPLACE VIEW ml.v_damage_fusion_features_v2_scored AS
SELECT
  v.*,

  (v.img_damage_source_level >= 1) AS img_damage_scored,
  (v.batt_fused IS NOT NULL) AS batt_scored,

  ((v.img_damage_source_level >= 1) OR v.text_damage_scored OR (v.batt_fused IS NOT NULL)) AS damage_any_scored,

  CASE WHEN v.text_damage_scored THEN v.text_sev_decision END AS text_sev_raw_known,
  CASE WHEN (v.text_damage_scored OR (v.batt_fused IS NOT NULL)) THEN v.text_sev_corr END AS text_sev_corr_known,

  CASE WHEN ((v.img_damage_source_level >= 1) OR v.text_damage_scored OR (v.batt_fused IS NOT NULL))
       THEN v.damage_fused_tier END AS damage_fused_tier_known

FROM ml.v_damage_fusion_features_v2 v;
