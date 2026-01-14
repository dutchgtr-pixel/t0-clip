-- 01_create_socio_t0_mv.sql
-- SOCIO (T0) materialized view with affordability features.
--
-- Inputs assumed to exist:
--   - ml.tom_features_v2_enriched_ai_ob_clean_mv
--   - ml.iphone_listings_geo_current
--   - ref.postal_code_to_kommune_history
--   - ref.kommune_socio_history
--
-- This MV is T0-safe because:
--   - socio joins are AS-OF: snapshot_date <= t0::date
--   - no time-of-query tokens
--   - affordability uses pinned winsor bounds

DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv AS
WITH base AS (
  SELECT
    f.*,
    f.edited_date AS t0,
    lower(
      regexp_replace(
        COALESCE(
          NULLIF(f.model,''),
          NULLIF(f.title,''),
          NULLIF(f.description,'')
        ),
        '\s+',' ','g'
      )
    ) AS _model_str
  FROM ml.tom_features_v2_enriched_ai_ob_clean_mv f
  WHERE f.edited_date IS NOT NULL
),
geo AS (
  SELECT
    g.generation,
    g.listing_id,
    g.postal_code,
    g.super_metro_v4_geo
  FROM ml.iphone_listings_geo_current g
),
socio_joined AS (
  SELECT
    b.*,

    -- geo surface used for segmentation
    g.postal_code AS geo_postal_code,
    g.super_metro_v4_geo,

    -- normalized postal for reference joins (per layer documentation)
    lpad(regexp_replace(COALESCE(g.postal_code, b.postal_code)::text, '\D','','g'), 4, '0') AS postal_code4,

    -- as-of postal->kommune mapping
    pk.kommune_code4,
    pk.snapshot_date AS postal_kommune_snapshot_date,

    -- as-of socio record
    ks.centrality_class,
    ks.income_median_after_tax_nok,
    ks.snapshot_date AS socio_snapshot_date,

    -- missing flags
    CASE WHEN pk.kommune_code4 IS NULL THEN 1 ELSE 0 END AS miss_kommune,
    CASE WHEN ks.kommune_code4 IS NULL THEN 1 ELSE 0 END AS miss_socio,

    -- segmentation key for within-segment normalization
    CASE
      WHEN (b._model_str IS NULL OR b._model_str = '') THEN 'unknown'
      WHEN (b._model_str LIKE '%pro max%') THEN 'pro_max'
      WHEN (b._model_str LIKE '%pro%')     THEN 'pro'
      WHEN (b._model_str LIKE '%mini%')    THEN 'mini'
      WHEN (b._model_str LIKE '%plus%')    THEN 'plus'
      ELSE 'base'
    END AS model_tier_socio

  FROM base b
  LEFT JOIN geo g
    ON g.generation = b.generation AND g.listing_id = b.listing_id

  LEFT JOIN LATERAL (
    SELECT
      pk1.postal_code,
      pk1.kommune_code4,
      pk1.snapshot_date
    FROM ref.postal_code_to_kommune_history pk1
    WHERE pk1.postal_code = lpad(regexp_replace(COALESCE(g.postal_code, b.postal_code)::text, '\D','','g'), 4, '0')
      AND pk1.snapshot_date <= (b.t0)::date
    ORDER BY pk1.snapshot_date DESC
    LIMIT 1
  ) pk ON true

  LEFT JOIN LATERAL (
    SELECT
      ks1.kommune_code4,
      ks1.centrality_class,
      ks1.income_median_after_tax_nok,
      ks1.snapshot_date
    FROM ref.kommune_socio_history ks1
    WHERE ks1.kommune_code4 = pk.kommune_code4
      AND ks1.snapshot_date <= (b.t0)::date
    ORDER BY ks1.snapshot_date DESC
    LIMIT 1
  ) ks ON true
),
aff_base AS (
  SELECT
    s.*,

    -- (1) price_months_income = price / (income/12) = 12*price/income
    CASE
      WHEN s.price IS NOT NULL AND s.price > 0
       AND s.income_median_after_tax_nok IS NOT NULL AND s.income_median_after_tax_nok > 0
      THEN (12.0 * s.price::float8) / (s.income_median_after_tax_nok::float8)
      ELSE NULL::float8
    END AS price_months_income,

    -- (2) log_price_to_income_cap = ln((price+1)/income) winsorized to p01/p99 constants
    CASE
      WHEN s.price IS NOT NULL AND s.price >= 0
       AND s.income_median_after_tax_nok IS NOT NULL AND s.income_median_after_tax_nok > 0
      THEN GREATEST(
             (-5.964859)::float8,
             LEAST(
               (-3.610855)::float8,
               LN((s.price::float8 + 1.0) / (s.income_median_after_tax_nok::float8))
             )
           )
      ELSE NULL::float8
    END AS log_price_to_income_cap
  FROM socio_joined s
),
aff_with_resid AS (
  SELECT
    a.*,
    -- (3) residual within (generation × model_tier_socio)
    (a.log_price_to_income_cap
      - AVG(a.log_price_to_income_cap) OVER (PARTITION BY a.generation, a.model_tier_socio)
    )::float8 AS aff_resid_in_seg
  FROM aff_base a
),
deciles AS (
  -- (4) deciles only for non-null rows
  SELECT
    generation, listing_id, t0,
    NTILE(10) OVER (PARTITION BY generation, model_tier_socio ORDER BY log_price_to_income_cap) AS aff_decile_in_seg
  FROM aff_with_resid
  WHERE log_price_to_income_cap IS NOT NULL
),
final AS (
  SELECT
    a.*,
    d.aff_decile_in_seg
  FROM aff_with_resid a
  LEFT JOIN deciles d USING (generation, listing_id, t0)
)
SELECT * FROM final;

-- Unique index required for REFRESH ... CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_uq
  ON ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv (listing_id);

ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv;
