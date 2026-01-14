-- 12_daily_refresh_runbook.sql
-- Daily refresh + certify for the SOCIO+MARKET certified feature store (socio_economic_store layer).
--
-- Recommended training surfaces:
--   --features_view ml.socio_market_feature_store_train_v

-- 0) Optional: snapshot ref tables (if you maintain history snapshots)
-- NOTE: these use CURRENT_DATE/now() but are outside the certified closure because they populate reference history tables.
-- Run only as part of reference maintenance.
INSERT INTO ref.postal_code_to_kommune_history
SELECT l.*, CURRENT_DATE::date AS snapshot_date, now() AS loaded_at
FROM ref.postal_code_to_kommune_latest l
ON CONFLICT DO NOTHING;

INSERT INTO ref.kommune_socio_history
SELECT l.*, CURRENT_DATE::date AS snapshot_date, now() AS loaded_at
FROM ref.kommune_socio_latest l
ON CONFLICT DO NOTHING;

ANALYZE ref.postal_code_to_kommune_history;
ANALYZE ref.kommune_socio_history;

-- 1) Refresh upstream base MV used by SOCIO layer (must be refreshed before SOCIO)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v2_enriched_ai_ob_clean_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_mv;

-- 2) SOCIO T0 MV (includes affordability)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv;

-- 3) Rebuild market base table
TRUNCATE TABLE ml.market_base_socio_t0_v1;

INSERT INTO ml.market_base_socio_t0_v1 (
  generation, listing_id, t0, super_metro_v4_geo, kommune_code4, price, log_price_mkt
)
SELECT
  generation,
  listing_id,
  t0,
  super_metro_v4_geo,
  kommune_code4,
  price,
  LN(price)::float8 AS log_price_mkt
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
WHERE t0 IS NOT NULL
  AND price IS NOT NULL
  AND price > 0;

ANALYZE ml.market_base_socio_t0_v1;

-- 4) Refresh market comps MV
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.market_relative_socio_t0_v1_mv;
ANALYZE ml.market_relative_socio_t0_v1_mv;

-- 5) Refresh wide SOCIO+MARKET MV
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv;

-- 6) Quick invariants (must be 0)
SELECT COUNT(*) AS n_bad_socio
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
WHERE (postal_kommune_snapshot_date IS NOT NULL AND postal_kommune_snapshot_date > t0::date)
   OR (socio_snapshot_date IS NOT NULL AND socio_snapshot_date > t0::date);

SELECT COUNT(*) AS n_bad_market
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
WHERE (comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0)
   OR (comp_max_t0_kommune     IS NOT NULL AND comp_max_t0_kommune     >= t0);

-- 7) Registry certification (bounded drift check) + strict enforcement
CALL audit.run_t0_cert_socio_market_v1(30);
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');
