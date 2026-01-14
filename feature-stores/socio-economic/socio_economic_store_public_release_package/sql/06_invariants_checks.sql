-- 06_invariants_checks.sql
-- Post-refresh structural invariants for SOCIO and MARKET layers.

-- 6.1 SOCIO snapshot as-of correctness (must be 0)
SELECT COUNT(*) AS n_bad_socio
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
WHERE (postal_kommune_snapshot_date IS NOT NULL AND postal_kommune_snapshot_date > t0::date)
   OR (socio_snapshot_date IS NOT NULL AND socio_snapshot_date > t0::date);

-- 6.2 MARKET strictness (no future comps) (must be 0)
SELECT COUNT(*) AS n_bad_market
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
WHERE (comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0)
   OR (comp_max_t0_kommune     IS NOT NULL AND comp_max_t0_kommune     >= t0);

-- 6.3 Presence check for the 12 modeling columns (should all be present)
SELECT
  COUNT(*) FILTER (WHERE c.attname IN (
    'centrality_class','miss_kommune','miss_socio',
    'price_months_income','log_price_to_income_cap','aff_resid_in_seg','aff_decile_in_seg',
    'rel_log_price_to_comp_mean30_super','rel_log_price_to_comp_mean30_kommune',
    'rel_price_best','rel_price_source','miss_rel_price'
  )) AS n_present_of_12
FROM pg_attribute c
WHERE c.attrelid = 'ml.socio_market_feature_store_t0_v1_v'::regclass
  AND c.attnum > 0 AND NOT c.attisdropped;
