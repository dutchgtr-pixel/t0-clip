DROP VIEW IF EXISTS ml.iphone_image_color_features_v1;

CREATE VIEW ml.iphone_image_color_features_v1 AS
WITH assets AS (
  SELECT
    generation,
    listing_id,
    COUNT(*)::bigint AS n_assets
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
),
listing_ctx AS (
  SELECT
    generation,
    listing_id,
    model AS db_model,
    spam  AS db_spam
  FROM "iPhone".iphone_listings
),
per_image AS (
  SELECT
    generation,
    listing_id,
    image_index,

    -- processed marker
    color_done,

    -- color outputs
    body_color_name,
    body_color_key,
    body_color_confidence,
    body_color_from_case,

    -- optional quality/meta (may be NULL for some generations)
    is_stock_photo,
    photo_quality_level,
    background_clean_level,

    -- model correction fields (do NOT expose timestamps/evidence)
    model_fix_new_model,
    model_fix_reason
  FROM ml.iphone_image_features_v1
  WHERE feature_version = 1
),
per_listing AS (
  SELECT
    generation,
    listing_id,

    COUNT(*)::bigint AS n_feat_rows,

    COUNT(*) FILTER (WHERE color_done IS TRUE)::bigint AS n_color_done_imgs,
    MIN(image_index) FILTER (WHERE color_done IS TRUE) AS min_color_done_idx,
    MAX(image_index) FILTER (WHERE color_done IS TRUE) AS max_color_done_idx,

    COUNT(*) FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL)::bigint AS n_color_votes,
    COUNT(DISTINCT body_color_key) FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL)::int AS n_distinct_color_keys,

    MAX(body_color_confidence) FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL) AS color_conf_max,
    AVG(body_color_confidence) FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL) AS color_conf_avg,

    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY body_color_confidence::double precision)
      FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL) AS color_conf_p10,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY body_color_confidence::double precision)
      FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL) AS color_conf_p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY body_color_confidence::double precision)
      FILTER (WHERE color_done IS TRUE AND body_color_key IS NOT NULL) AS color_conf_p90,

    COUNT(*) FILTER (
      WHERE color_done IS TRUE
        AND body_color_key IS NOT NULL
        AND body_color_from_case IS TRUE
    )::bigint AS n_votes_from_case,

    COUNT(*) FILTER (
      WHERE color_done IS TRUE
        AND body_color_key IS NOT NULL
        AND is_stock_photo IS TRUE
    )::bigint AS n_votes_stock,

    COUNT(*) FILTER (
      WHERE color_done IS TRUE
        AND body_color_key IS NOT NULL
        AND photo_quality_level >= 3
    )::bigint AS n_votes_hq,

    BOOL_OR(model_fix_new_model IS NOT NULL) AS model_fixed_any,
    MAX(model_fix_new_model) FILTER (WHERE model_fix_new_model IS NOT NULL) AS model_fix_new_model,
    BOOL_OR(model_fix_reason = 'spam_below13') AS model_fix_below13_any

  FROM per_image
  GROUP BY 1,2
),
per_color AS (
  SELECT
    generation,
    listing_id,
    body_color_key,
    MAX(body_color_name) AS body_color_name,

    COUNT(*)::bigint AS votes,

    MAX(body_color_confidence) AS max_conf,
    AVG(body_color_confidence) AS avg_conf,

    AVG((body_color_from_case IS TRUE)::int)::double precision AS p_from_case,

    MIN(image_index) AS first_vote_idx
  FROM per_image
  WHERE color_done IS TRUE
    AND body_color_key IS NOT NULL
  GROUP BY 1,2,3
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY generation, listing_id
      ORDER BY
        votes DESC,
        p_from_case ASC,
        max_conf DESC,
        avg_conf DESC,
        first_vote_idx ASC,
        body_color_key ASC
    ) AS rn
  FROM per_color
),
top2 AS (
  SELECT
    generation,
    listing_id,

    MAX(body_color_key)  FILTER (WHERE rn = 1) AS color_key_primary,
    MAX(body_color_name) FILTER (WHERE rn = 1) AS color_name_primary,
    MAX(votes)           FILTER (WHERE rn = 1) AS color_votes_primary,
    MAX(max_conf)        FILTER (WHERE rn = 1) AS color_conf_primary_max,
    MAX(avg_conf)        FILTER (WHERE rn = 1) AS color_conf_primary_avg,
    MAX(p_from_case)     FILTER (WHERE rn = 1) AS color_primary_p_from_case,
    MAX(first_vote_idx)  FILTER (WHERE rn = 1) AS color_primary_first_vote_idx,

    MAX(body_color_key)  FILTER (WHERE rn = 2) AS color_key_secondary,
    MAX(body_color_name) FILTER (WHERE rn = 2) AS color_name_secondary,
    MAX(votes)           FILTER (WHERE rn = 2) AS color_votes_secondary,
    MAX(max_conf)        FILTER (WHERE rn = 2) AS color_conf_secondary_max,
    MAX(avg_conf)        FILTER (WHERE rn = 2) AS color_conf_secondary_avg,
    MAX(p_from_case)     FILTER (WHERE rn = 2) AS color_secondary_p_from_case

  FROM ranked
  WHERE rn <= 2
  GROUP BY 1,2
),
final AS (
  SELECT
    a.generation,
    a.listing_id,
    a.n_assets,

    /* ---------- row coverage ---------- */
    COALESCE(p.n_feat_rows, 0::bigint) AS n_feat_rows,
    COALESCE(p.n_color_done_imgs, 0::bigint) AS n_color_done_imgs,
    p.min_color_done_idx,
    p.max_color_done_idx,

    COALESCE(p.n_feat_rows, 0)::numeric / NULLIF(a.n_assets, 0)::numeric AS feat_row_coverage_assets,
    COALESCE(p.n_color_done_imgs, 0)::numeric / NULLIF(a.n_assets, 0)::numeric AS color_done_coverage_assets,

    (a.n_assets - COALESCE(p.n_color_done_imgs, 0))::bigint AS undone_assets,
    (COALESCE(p.n_color_done_imgs, 0) < a.n_assets) AS is_incomplete,
    (a.n_assets > 16 AND COALESCE(p.max_color_done_idx::int, -1) = 15) AS hit_cap_16,

    (COALESCE(p.n_feat_rows, 0) > 0) AS has_feature_rows,
    (COALESCE(p.n_feat_rows, 0) > 0 AND COALESCE(p.n_color_done_imgs, 0) = 0) AS backlog_not_done,
    (COALESCE(p.n_color_done_imgs, 0) > 0) AS color_processed,

    /* ---------- vote / consistency ---------- */
    COALESCE(p.n_color_votes, 0::bigint) AS n_color_votes,
    COALESCE(p.n_distinct_color_keys, 0) AS n_distinct_color_keys,

    (COALESCE(p.n_color_votes, 0) > 0) AS color_known,
    (COALESCE(p.n_color_done_imgs, 0) > 0 AND COALESCE(p.n_color_votes, 0) = 0) AS done_but_no_votes,
    (COALESCE(p.n_color_votes, 0) > 0 AND COALESCE(p.n_distinct_color_keys, 0) >= 2) AS color_conflict,

    p.color_conf_max,
    p.color_conf_avg,
    p.color_conf_p10,
    p.color_conf_p50,
    p.color_conf_p90,

    COALESCE(p.n_votes_from_case, 0::bigint) AS n_votes_from_case,
    CASE WHEN COALESCE(p.n_color_votes, 0) > 0
         THEN COALESCE(p.n_votes_from_case, 0)::numeric / NULLIF(p.n_color_votes, 0)::numeric
    END AS p_votes_from_case,

    COALESCE(p.n_votes_stock, 0::bigint) AS n_votes_stock,
    CASE WHEN COALESCE(p.n_color_votes, 0) > 0
         THEN COALESCE(p.n_votes_stock, 0)::numeric / NULLIF(p.n_color_votes, 0)::numeric
    END AS p_votes_stock,

    COALESCE(p.n_votes_hq, 0::bigint) AS n_votes_hq,
    CASE WHEN COALESCE(p.n_color_votes, 0) > 0
         THEN COALESCE(p.n_votes_hq, 0)::numeric / NULLIF(p.n_color_votes, 0)::numeric
    END AS p_votes_hq,

    /* ---------- selected colors (primary + runner-up) ---------- */
    t.color_key_primary,
    t.color_name_primary,
    t.color_votes_primary,
    CASE WHEN COALESCE(p.n_color_votes, 0) > 0 AND t.color_votes_primary IS NOT NULL
         THEN t.color_votes_primary::numeric / NULLIF(p.n_color_votes, 0)::numeric
    END AS color_primary_vote_share,

    t.color_key_secondary,
    t.color_name_secondary,
    t.color_votes_secondary,
    CASE WHEN COALESCE(p.n_color_votes, 0) > 0 AND t.color_votes_secondary IS NOT NULL
         THEN t.color_votes_secondary::numeric / NULLIF(p.n_color_votes, 0)::numeric
    END AS color_secondary_vote_share,

    CASE WHEN t.color_votes_secondary IS NOT NULL
         THEN (t.color_votes_primary - t.color_votes_secondary)::bigint
    END AS color_vote_margin_votes,

    CASE
      WHEN COALESCE(p.n_color_votes, 0) > 0
       AND t.color_votes_primary IS NOT NULL
       AND t.color_votes_secondary IS NOT NULL
      THEN (t.color_votes_primary::numeric - t.color_votes_secondary::numeric)
           / NULLIF(p.n_color_votes, 0)::numeric
    END AS color_vote_margin_share,

    t.color_conf_primary_max,
    t.color_conf_primary_avg,
    t.color_primary_p_from_case,
    t.color_primary_first_vote_idx,

    /* Optional “strict” color you can use when you want higher precision */
    CASE
      WHEN COALESCE(p.n_color_votes, 0) = 0 THEN NULL
      WHEN COALESCE(p.n_distinct_color_keys, 0) <= 1 THEN t.color_key_primary
      WHEN t.color_votes_secondary IS NULL THEN t.color_key_primary
      WHEN (t.color_votes_primary::numeric / NULLIF(p.n_color_votes, 0)::numeric) >= 0.67
        THEN t.color_key_primary
      ELSE NULL
    END AS color_key_primary_strict,

    /* ---------- model correction integration ---------- */
    COALESCE(p.model_fixed_any, FALSE) AS model_fixed_any,
    p.model_fix_new_model,

    COALESCE(p.model_fix_new_model, lc.db_model) AS model_effective,

    CASE
      WHEN COALESCE(p.model_fix_new_model, lc.db_model) ILIKE '%pro max%' THEN 'pro_max'
      WHEN COALESCE(p.model_fix_new_model, lc.db_model) ILIKE '%pro%'     THEN 'pro'
      WHEN COALESCE(p.model_fix_new_model, lc.db_model) ILIKE '%plus%'    THEN 'plus'
      WHEN COALESCE(p.model_fix_new_model, lc.db_model) ILIKE '%mini%'    THEN 'mini'
      WHEN COALESCE(p.model_fix_new_model, lc.db_model) IS NULL
        OR COALESCE(p.model_fix_new_model, lc.db_model) = ''              THEN NULL
      ELSE 'base'
    END AS model_effective_type,

    NULLIF(
      (regexp_match(COALESCE(p.model_fix_new_model, lc.db_model), '([0-9]+)'))[1],
      ''
    )::int AS model_effective_generation,

    (
      NULLIF((regexp_match(COALESCE(p.model_fix_new_model, lc.db_model), '([0-9]+)'))[1], '')::int IS NOT NULL
      AND
      NULLIF((regexp_match(COALESCE(p.model_fix_new_model, lc.db_model), '([0-9]+)'))[1], '')::int <> a.generation
    ) AS model_generation_mismatch,

    (COALESCE(p.model_fix_below13_any, FALSE) OR lc.db_spam = 'below13') AS is_spam_below13

  FROM assets a
  LEFT JOIN per_listing p
    ON p.generation = a.generation AND p.listing_id = a.listing_id
  LEFT JOIN top2 t
    ON t.generation = a.generation AND t.listing_id = a.listing_id
  LEFT JOIN listing_ctx lc
    ON lc.generation = a.generation AND lc.listing_id = a.listing_id
)
SELECT * FROM final;
