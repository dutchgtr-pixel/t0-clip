-- Public release: T0-anchored Device Meta Feature Store (PostgreSQL)
-- 
-- Goals:
--   * Platform-agnostic: uses `listing_id` as the primary listing identifier.
--   * Time-safe: population statistics are computed "as-of" the anchor day (t0_day),
--     so historical feature rows do not drift as new listings arrive.
--   * No secrets: this file contains no credentials, tokens, or endpoints.
--
-- Prerequisites (upstream relations):
--   feature_store.listing_device_features_clean_mv
--     Columns expected (minimum):
--       - generation        (int or text)
--       - listing_id        (bigint or text)
--       - edited_date       (timestamp or date)   -- anchor timestamp (T0)
--       - model             (text)                -- raw model string
--
--   feature_store.device_image_features_unified_v1
--     Columns expected (minimum):
--       - generation                   (int or text)
--       - listing_id                   (bigint or text)
--       - image_count                  (int)
--       - body_color_key_main          (text)
--       - color_known                  (bool)
--       - color_conflict               (bool)
--       - n_color_votes                (int)
--       - color_primary_vote_share     (float)
--       - color_vote_margin_share      (float)
--       - model_effective_generation   (int or text)
--       - model_fixed_any              (bool)
--       - model_generation_mismatch    (bool)
--       - is_spam_below13              (bool)
--
-- Notes:
--   * If you use different column names upstream, create a thin compatibility view
--     with these expected names, or edit the SELECTs below.
--   * For CONCURRENT refresh, a UNIQUE index is required on the MV.
--
-- ------------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS feature_store;

-- ------------------------------------------------------------------------------
-- 0) Base per-listing device meta (no population aggregates)
-- ------------------------------------------------------------------------------

CREATE OR REPLACE VIEW feature_store.v_device_meta_base_t0_v1 AS
WITH base AS (
  SELECT
    b.generation,
    b.listing_id,
    b.edited_date AS t0,
    (b.edited_date::date) AS t0_day,
    lower(regexp_replace(b.model, '\s+', ' ', 'g')) AS model_norm
  FROM feature_store.listing_device_features_clean_mv b
),
img AS (
  SELECT
    u.generation,
    u.listing_id,
    u.image_count::int AS image_count,
    COALESCE(NULLIF(u.body_color_key_main, ''), '_MISSING') AS color_key,
    (u.color_known IS TRUE)::int AS color_known,
    (u.color_conflict IS TRUE)::int AS color_conflict,
    u.n_color_votes::int AS n_color_votes,
    u.color_primary_vote_share::float8 AS color_vote_share,
    u.color_vote_margin_share::float8 AS color_vote_margin,
    u.model_effective_generation,
    (u.model_fixed_any IS TRUE)::int AS model_fixed_any,
    (u.model_generation_mismatch IS TRUE)::int AS model_generation_mismatch,
    (u.is_spam_below13 IS TRUE)::int AS is_spam_below13
  FROM feature_store.device_image_features_unified_v1 u
),
joined AS (
  SELECT
    b.generation,
    b.listing_id,
    b.t0,
    b.t0_day,
    COALESCE(i.model_effective_generation, b.generation) AS gen_key,
    CASE
      WHEN b.model_norm IS NULL OR b.model_norm = '' THEN 'other'
      WHEN b.model_norm LIKE '%pro max%' THEN 'pro_max'
      WHEN b.model_norm LIKE '%pro%' THEN 'pro'
      WHEN b.model_norm LIKE '%plus%' THEN 'plus'
      WHEN b.model_norm LIKE '%mini%' THEN 'mini'
      WHEN b.model_norm LIKE '%air%' THEN 'air'
      ELSE 'base'
    END AS model_variant,
    CASE
      WHEN COALESCE(i.color_key, '_MISSING') = '_MISSING' THEN '_MISSING'
      WHEN COALESCE(i.color_key, '_MISSING') = ANY (
        ARRAY[
          'graphite','black_titanium','silver','space_black','midnight','sierra_blue','blue',
          'natural_titanium','gold','deep_purple','white_titanium','starlight','desert_titanium',
          'blue_titanium','black','alpine_green','pink','green','deep_blue','cosmic_orange'
        ]
      ) THEN COALESCE(i.color_key, '_MISSING')
      ELSE '_OTHER'
    END AS color_bucket,

    i.image_count,
    CASE WHEN i.image_count IS NOT NULL AND i.image_count > 0 THEN 1 ELSE 0 END AS dev_has_images,
    CASE WHEN COALESCE(i.color_key, '_MISSING') = '_MISSING' THEN 1 ELSE 0 END AS dev_color_is_missing,
    COALESCE(i.color_known, 0) AS dev_color_known,
    COALESCE(i.color_conflict, 0) AS dev_color_conflict,
    i.n_color_votes AS dev_n_color_votes,
    i.color_vote_share AS dev_color_vote_share,
    i.color_vote_margin AS dev_color_vote_margin,
    COALESCE(i.model_fixed_any, 0) AS dev_model_fixed_any,
    COALESCE(i.model_generation_mismatch, 0) AS dev_model_generation_mismatch,
    COALESCE(i.is_spam_below13, 0) AS dev_is_spam_below13
  FROM base b
  LEFT JOIN img i
    ON i.generation = b.generation
   AND i.listing_id = b.listing_id
)
SELECT * FROM joined;

-- ------------------------------------------------------------------------------
-- 1) T0-anchored population stats MV (as-of GMC-style features)
-- ------------------------------------------------------------------------------

CREATE MATERIALIZED VIEW feature_store.device_meta_gmc_stats_t0_v1_mv AS
WITH
params AS (
  SELECT 0.5::float8 AS alpha, 22::float8 AS k_color
),
color_domain AS (
  SELECT unnest(ARRAY[
    'graphite','black_titanium','silver','space_black','midnight','sierra_blue','blue',
    'natural_titanium','gold','deep_purple','white_titanium','starlight','desert_titanium',
    'blue_titanium','black','alpine_green','pink','green','deep_blue','cosmic_orange',
    '_MISSING','_OTHER'
  ]) AS color_bucket
),
days AS (
  SELECT DISTINCT t0_day
  FROM feature_store.v_device_meta_base_t0_v1
),
segments AS (
  SELECT DISTINCT gen_key, model_variant
  FROM feature_store.v_device_meta_base_t0_v1
),
seg_first AS (
  SELECT gen_key, model_variant, MIN(t0_day) AS first_day
  FROM feature_store.v_device_meta_base_t0_v1
  GROUP BY 1,2
),
n_segs_prior AS (
  SELECT
    d.t0_day,
    COUNT(*)::float8 AS n_segs_prior
  FROM days d
  JOIN seg_first sf
    ON sf.first_day < d.t0_day
  GROUP BY 1
),
grid AS (
  SELECT
    d.t0_day,
    s.gen_key,
    s.model_variant,
    c.color_bucket
  FROM days d
  CROSS JOIN segments s
  CROSS JOIN color_domain c
),
daily_counts AS (
  SELECT
    t0_day,
    gen_key,
    model_variant,
    color_bucket,
    COUNT(*)::float8 AS n_day
  FROM feature_store.v_device_meta_base_t0_v1
  GROUP BY 1,2,3,4
),
grid_filled AS (
  SELECT
    g.t0_day,
    g.gen_key,
    g.model_variant,
    g.color_bucket,
    COALESCE(dc.n_day, 0.0) AS n_day
  FROM grid g
  LEFT JOIN daily_counts dc
    ON dc.t0_day = g.t0_day
   AND dc.gen_key = g.gen_key
   AND dc.model_variant = g.model_variant
   AND dc.color_bucket = g.color_bucket
),
-- cumulative counts PRIOR to the anchor day (exclude current day)
cum_color_seg AS (
  SELECT
    t0_day,
    gen_key,
    model_variant,
    color_bucket,
    SUM(n_day) OVER (
      PARTITION BY gen_key, model_variant, color_bucket
      ORDER BY t0_day
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS n_color_seg_prior
  FROM grid_filled
),
seg_day AS (
  SELECT
    t0_day,
    gen_key,
    model_variant,
    SUM(n_day) AS n_seg_day
  FROM grid_filled
  GROUP BY 1,2,3
),
cum_seg AS (
  SELECT
    t0_day,
    gen_key,
    model_variant,
    SUM(n_seg_day) OVER (
      PARTITION BY gen_key, model_variant
      ORDER BY t0_day
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS n_seg_prior
  FROM seg_day
),
glob_day AS (
  SELECT
    t0_day,
    color_bucket,
    SUM(n_day) AS n_color_glob_day
  FROM grid_filled
  GROUP BY 1,2
),
cum_color_glob AS (
  SELECT
    t0_day,
    color_bucket,
    SUM(n_color_glob_day) OVER (
      PARTITION BY color_bucket
      ORDER BY t0_day
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS n_color_glob_prior
  FROM glob_day
),
tot_day AS (
  SELECT t0_day, SUM(n_seg_day) AS n_tot_day
  FROM seg_day
  GROUP BY 1
),
cum_tot AS (
  SELECT
    t0_day,
    SUM(n_tot_day) OVER (
      ORDER BY t0_day
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS n_tot_prior
  FROM tot_day
),
sc AS (
  SELECT
    c.t0_day,
    c.gen_key,
    c.model_variant,
    c.color_bucket,
    COALESCE(c.n_color_seg_prior, 0.0) AS n_color_seg_prior,
    COALESCE(s.n_seg_prior, 0.0) AS n_seg_prior,
    COALESCE(g.n_color_glob_prior, 0.0) AS n_color_glob_prior,
    COALESCE(t.n_tot_prior, 0.0) AS n_tot_prior,
    COALESCE(ns.n_segs_prior, 0.0) AS n_segs_prior
  FROM cum_color_seg c
  JOIN cum_seg s
    ON s.t0_day = c.t0_day
   AND s.gen_key = c.gen_key
   AND s.model_variant = c.model_variant
  JOIN cum_color_glob g
    ON g.t0_day = c.t0_day
   AND g.color_bucket = c.color_bucket
  JOIN cum_tot t
    ON t.t0_day = c.t0_day
  LEFT JOIN n_segs_prior ns
    ON ns.t0_day = c.t0_day
),
sc_rank AS (
  SELECT
    *,
    dense_rank() OVER (
      PARTITION BY t0_day, gen_key, model_variant
      ORDER BY n_color_seg_prior DESC
    ) AS color_rank_in_seg
  FROM sc
),
seg_div AS (
  SELECT
    t0_day,
    gen_key,
    model_variant,
    COUNT(*) FILTER (WHERE n_color_seg_prior > 0.0)::float8 AS n_distinct_colors,
    (-1.0)::float8 * SUM(
      CASE
        WHEN n_seg_prior > 0.0 AND n_color_seg_prior > 0.0 THEN
          (n_color_seg_prior / n_seg_prior) * LN(n_color_seg_prior / n_seg_prior)
        ELSE 0.0
      END
    ) AS color_entropy,
    MAX(CASE WHEN n_seg_prior > 0.0 THEN n_color_seg_prior / n_seg_prior ELSE NULL END) AS top_color_share,
    SUM(CASE WHEN color_bucket = '_MISSING' AND n_seg_prior > 0.0 THEN n_color_seg_prior / n_seg_prior ELSE 0.0 END) AS missing_color_share
  FROM sc_rank
  GROUP BY 1,2,3
)
SELECT
  r.t0_day,
  r.gen_key,
  r.model_variant,
  r.color_bucket,

  -- Segment counts/shares (as-of)
  r.n_seg_prior::int AS gmc_seg_n,
  (r.n_seg_prior + p.alpha) / NULLIF(r.n_tot_prior + p.alpha * GREATEST(r.n_segs_prior, 1.0), 0.0) AS gmc_seg_share,

  -- Segment×color
  r.n_color_seg_prior::int AS gmc_color_n_in_seg,
  (r.n_color_seg_prior + p.alpha) / NULLIF(r.n_seg_prior + p.alpha * p.k_color, 0.0) AS gmc_color_share_in_seg,
  -LN(GREATEST((r.n_color_seg_prior + p.alpha) / NULLIF(r.n_seg_prior + p.alpha * p.k_color, 0.0), 1e-12)) AS gmc_color_rarity_score,

  -- Global color
  r.n_color_glob_prior::int AS gmc_color_n_global,
  (r.n_color_glob_prior + p.alpha) / NULLIF(r.n_tot_prior + p.alpha * p.k_color, 0.0) AS gmc_color_share_global,
  -LN(GREATEST((r.n_color_glob_prior + p.alpha) / NULLIF(r.n_tot_prior + p.alpha * p.k_color, 0.0), 1e-12)) AS gmc_color_rarity_global,

  r.color_rank_in_seg::int AS gmc_color_rank_in_seg,

  -- Diversity
  sd.n_distinct_colors::int AS gmc_seg_n_distinct_colors,
  sd.color_entropy           AS gmc_seg_color_entropy,
  sd.top_color_share         AS gmc_seg_top_color_share,
  sd.missing_color_share     AS gmc_seg_missing_color_share

FROM sc_rank r
JOIN seg_div sd
  ON sd.t0_day = r.t0_day
 AND sd.gen_key = r.gen_key
 AND sd.model_variant = r.model_variant
CROSS JOIN params p;

-- Required unique index so REFRESH MATERIALIZED VIEW CONCURRENTLY is allowed
CREATE UNIQUE INDEX IF NOT EXISTS ux_device_meta_gmc_stats_t0_v1
  ON feature_store.device_meta_gmc_stats_t0_v1_mv (t0_day, gen_key, model_variant, color_bucket);

-- ------------------------------------------------------------------------------
-- 2) Final encoded device meta view (contract-preserving)
-- ------------------------------------------------------------------------------

CREATE OR REPLACE VIEW feature_store.device_meta_encoded_t0_v1 AS
WITH b AS (
  SELECT * FROM feature_store.v_device_meta_base_t0_v1
),
g AS (
  SELECT * FROM feature_store.device_meta_gmc_stats_t0_v1_mv
)
SELECT
  b.generation,
  b.listing_id,

  (b.gen_key = 13)::int AS dev_gen_13,
  (b.gen_key = 14)::int AS dev_gen_14,
  (b.gen_key = 15)::int AS dev_gen_15,
  (b.gen_key = 16)::int AS dev_gen_16,
  (b.gen_key = 17)::int AS dev_gen_17,
  (b.gen_key IS NULL OR (b.gen_key <> ALL (ARRAY[13,14,15,16,17])))::int AS dev_gen_other,

  (b.model_variant = 'base')::int    AS dev_var_base,
  (b.model_variant = 'pro')::int     AS dev_var_pro,
  (b.model_variant = 'pro_max')::int AS dev_var_pro_max,
  (b.model_variant = 'plus')::int    AS dev_var_plus,
  (b.model_variant = 'mini')::int    AS dev_var_mini,
  (b.model_variant = 'air')::int     AS dev_var_air,
  (b.model_variant = 'other')::int   AS dev_var_other,

  (b.color_bucket = 'graphite')::int         AS dev_color_graphite,
  (b.color_bucket = 'black_titanium')::int   AS dev_color_black_titanium,
  (b.color_bucket = 'silver')::int           AS dev_color_silver,
  (b.color_bucket = 'space_black')::int      AS dev_color_space_black,
  (b.color_bucket = 'midnight')::int         AS dev_color_midnight,
  (b.color_bucket = 'sierra_blue')::int      AS dev_color_sierra_blue,
  (b.color_bucket = 'blue')::int             AS dev_color_blue,
  (b.color_bucket = 'natural_titanium')::int AS dev_color_natural_titanium,
  (b.color_bucket = 'gold')::int             AS dev_color_gold,
  (b.color_bucket = 'deep_purple')::int      AS dev_color_deep_purple,
  (b.color_bucket = 'white_titanium')::int   AS dev_color_white_titanium,
  (b.color_bucket = 'starlight')::int        AS dev_color_starlight,
  (b.color_bucket = 'desert_titanium')::int  AS dev_color_desert_titanium,
  (b.color_bucket = 'blue_titanium')::int    AS dev_color_blue_titanium,
  (b.color_bucket = 'black')::int            AS dev_color_black,
  (b.color_bucket = 'alpine_green')::int     AS dev_color_alpine_green,
  (b.color_bucket = 'pink')::int             AS dev_color_pink,
  (b.color_bucket = 'green')::int            AS dev_color_green,
  (b.color_bucket = 'deep_blue')::int        AS dev_color_deep_blue,
  (b.color_bucket = 'cosmic_orange')::int    AS dev_color_cosmic_orange,
  (b.color_bucket = '_MISSING')::int         AS dev_color_missing,
  (b.color_bucket = '_OTHER')::int           AS dev_color_other,

  b.image_count AS dev_image_count,
  b.dev_has_images,
  b.dev_color_is_missing,
  b.dev_color_known,
  b.dev_color_conflict,
  b.dev_n_color_votes,
  b.dev_color_vote_share,
  b.dev_color_vote_margin,
  b.dev_model_fixed_any,
  b.dev_model_generation_mismatch,
  b.dev_is_spam_below13,

  -- As-of GMC features joined by (t0_day, segment, color_bucket)
  COALESCE(g.gmc_seg_n, 0) AS gmc_seg_n,
  g.gmc_seg_share,

  COALESCE(g.gmc_color_n_in_seg, 0) AS gmc_color_n_in_seg,
  g.gmc_color_share_in_seg,
  g.gmc_color_rarity_score,

  COALESCE(g.gmc_color_n_global, 0) AS gmc_color_n_global,
  g.gmc_color_share_global,
  g.gmc_color_rarity_global,

  g.gmc_color_rank_in_seg,

  COALESCE(g.gmc_seg_n_distinct_colors, 0) AS gmc_seg_n_distinct_colors,
  g.gmc_seg_color_entropy,
  g.gmc_seg_top_color_share,
  g.gmc_seg_missing_color_share

FROM b
LEFT JOIN g
  ON g.t0_day = b.t0_day
 AND g.gen_key = b.gen_key
 AND g.model_variant = b.model_variant
 AND g.color_bucket = b.color_bucket;

-- ------------------------------------------------------------------------------
-- Runbook (operations)
-- ------------------------------------------------------------------------------
-- Refresh and analyze the MV when new source data is loaded:
--
--   REFRESH MATERIALIZED VIEW CONCURRENTLY feature_store.device_meta_gmc_stats_t0_v1_mv;
--   ANALYZE feature_store.device_meta_gmc_stats_t0_v1_mv;
--
-- The final encoded object is a VIEW and therefore does not refresh.
-- ------------------------------------------------------------------------------

