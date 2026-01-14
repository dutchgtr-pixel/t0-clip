-- 05_create_entrypoint_and_guard_views.sql
-- Certified entrypoint and guarded training view for the SOCIO+MARKET store.

CREATE OR REPLACE VIEW ml.socio_market_feature_store_t0_v1_v AS
SELECT *
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv;

-- Guarded training view: fails closed if entrypoint is uncertified or drifted.
CREATE OR REPLACE VIEW ml.socio_market_feature_store_train_v AS
SELECT f.*
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv f
CROSS JOIN LATERAL (
  SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours')
) guard;
