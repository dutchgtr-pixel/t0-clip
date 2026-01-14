-- 10_run_certification.sql
-- Run certification and enforce strict guard.

CALL audit.run_t0_cert_socio_market_v1(30);

-- Must pass (fail-closed enforcement):
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');

-- Registry record:
SELECT *
FROM audit.t0_cert_registry
WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v';
