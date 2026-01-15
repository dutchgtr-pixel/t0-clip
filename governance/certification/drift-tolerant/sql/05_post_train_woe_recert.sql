-- 05_post_train_woe_recert.sql
-- Run after training completes (active model_key changes).

CALL audit.rebaseline_last_n_days('ml.woe_anchor_feature_store_t0_v1_v'::regclass, 10, 2000);
CALL audit.run_t0_cert_woe_anchor_store_v1(10);
SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours');
