-- 04_certify_all_stores_dev_10d.sql
-- Pre-train drift-tolerant certification block (last 10 days).

-- SOCIO/MARKET
CALL audit.rebaseline_last_n_days('ml.socio_market_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_socio_market_v1(10);
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');

-- GEO
CALL audit.rebaseline_last_n_days('ml.geo_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_geo_store_v1(10);
SELECT audit.require_certified_strict('ml.geo_feature_store_t0_v1_v', interval '24 hours');

-- VISION
CALL audit.rebaseline_last_n_days('ml.vision_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_vision_store_v1(10, true);
SELECT audit.require_certified_strict('ml.vision_feature_store_t0_v1_v', interval '24 hours');

-- FUSION
CALL audit.rebaseline_last_n_days('ml.fusion_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_fusion_store_v1(10, true);
SELECT audit.require_certified_strict('ml.fusion_feature_store_t0_v1_v', interval '24 hours');

-- TRAINER_DERIVED
CALL audit.rebaseline_last_n_days('ml.trainer_derived_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_trainer_derived_store_v1(10);
SELECT audit.require_certified_strict('ml.trainer_derived_feature_store_t0_v1_v', interval '24 hours');

-- DEVICE_META (refresh MV + drift-allowed cert)
REFRESH MATERIALIZED VIEW ml.device_meta_gmc_stats_t0_v1_mv;
ANALYZE ml.device_meta_gmc_stats_t0_v1_mv;

CALL audit.run_t0_cert_device_meta_v3(10, true, 2000);
SELECT audit.require_certified_strict('ml.device_meta_store_t0_v1_v', interval '24 hours');
