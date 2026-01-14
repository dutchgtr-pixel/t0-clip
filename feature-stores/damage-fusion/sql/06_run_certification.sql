-- ============================================================
-- Fusion Store — Run certification + verify enforcement
-- ============================================================

CALL audit.run_t0_cert_fusion_store_v1(30, true);

SELECT *
FROM audit.t0_cert_registry
WHERE entrypoint='ml.fusion_feature_store_t0_v1_v';

SELECT audit.require_certified_strict('ml.fusion_feature_store_t0_v1_v', interval '24 hours');

-- Should return a row when certified
SELECT 1
FROM ml.v_damage_fusion_features_v2_scored_train_v
LIMIT 1;
