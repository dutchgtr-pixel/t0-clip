-- 09_run_certification.sql
-- Runs certification and proves guard behavior.

CALL audit.run_t0_cert_woe_anchor_store_v1(10);

SELECT *
FROM audit.t0_cert_registry
WHERE entrypoint = 'ml.woe_anchor_feature_store_t0_v1_v';

SELECT entrypoint, status, certified_at, dataset_days, notes
FROM audit.t0_cert_registry
WHERE entrypoint IN ('ml.woe_anchor_scores_live_train_v', 'ml.woe_anchor_scores_live_v1')
ORDER BY entrypoint;

SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours');

SELECT * FROM ml.woe_anchor_scores_live_train_v LIMIT 10;
