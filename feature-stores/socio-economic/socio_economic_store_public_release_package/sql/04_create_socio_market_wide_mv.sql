-- 04_create_socio_market_wide_mv.sql
-- Wide SOCIO+MARKET MV used as the features_view for Slow21 training.
-- Uses `s.*` so SOCIO columns automatically propagate (including affordability).

DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv AS
SELECT
  s.*,

  -- market comp stats
  m.comp_n_30d_super_metro,
  m.comp_log_price_mean_30d_super_metro,
  m.comp_max_t0_super_metro,
  m.comp_n_30d_kommune,
  m.comp_log_price_mean_30d_kommune,
  m.comp_max_t0_kommune,

  -- relative pricing (requires coverage threshold)
  CASE
    WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_super_metro >= 20)
    THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_super_metro)
    ELSE NULL::float8
  END AS rel_log_price_to_comp_mean30_super,

  CASE
    WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_kommune >= 20)
    THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_kommune)
    ELSE NULL::float8
  END AS rel_log_price_to_comp_mean30_kommune,

  -- canonical fallback (coverage-aware)
  COALESCE(
    CASE
      WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_super_metro >= 20)
      THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_super_metro)
      ELSE NULL::float8
    END,
    CASE
      WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_kommune >= 20)
      THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_kommune)
      ELSE NULL::float8
    END
  ) AS rel_price_best,

  CASE
    WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_super_metro >= 20) THEN 1
    WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_kommune >= 20)     THEN 2
    ELSE NULL::int
  END AS rel_price_source,

  CASE
    WHEN COALESCE(
      CASE
        WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_super_metro >= 20)
        THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_super_metro)
        ELSE NULL::float8
      END,
      CASE
        WHEN (s.price IS NOT NULL AND s.price > 0 AND m.comp_n_30d_kommune >= 20)
        THEN (LN(s.price::float8) - m.comp_log_price_mean_30d_kommune)
        ELSE NULL::float8
      END
    ) IS NULL
    THEN 1 ELSE 0
  END AS miss_rel_price

FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv s
JOIN ml.market_relative_socio_t0_v1_mv m
  ON m.generation = s.generation
 AND m.listing_id     = s.listing_id
 AND m.t0          = s.t0;

CREATE UNIQUE INDEX IF NOT EXISTS socio_market_t0_v1_uq
  ON ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv (generation, listing_id, t0);

ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv;
