# 6. Refresh runbook (operational)

This store is refreshed and certified daily in dependency order:

1) refresh upstream base features (not defined in this package)
2) refresh `ml.tom_features_v2_enriched_ai_ob_clean_mv`
3) refresh `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv`
4) reload `ml.market_base_socio_t0_v1` (truncate + insert)
5) refresh `ml.market_relative_socio_t0_v1_mv`
6) refresh `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv`
7) run invariants checks
8) call `audit.run_t0_cert_socio_market_v1(p_check_days=30)`
9) enforce: `audit.require_certified_strict(entrypoint, interval '24 hours')`

The full SQL runbook is provided in `sql/12_daily_refresh_runbook.sql`.

