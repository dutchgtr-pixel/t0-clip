-- 16_run_certification_365.sql
-- Run a deep 365-day certification and show registry entries.

CALL audit.run_t0_cert_geo_store_v1(365);
SELECT audit.require_certified_strict('ml.geo_feature_store_t0_v1_v', interval '24 hours');

SELECT entrypoint, status, certified_at, viewdef_objects, dataset_days, notes
FROM audit.t0_cert_registry
WHERE entrypoint IN (
  'ml.geo_feature_store_t0_v1_v',
  'ml.geo_dim_super_metro_v4_t0_train_v',
  'ml.geo_dim_super_metro_v4_t0_v1'
)
ORDER BY entrypoint;

SELECT * FROM ml.geo_dim_super_metro_v4_t0_train_v LIMIT 5;
