-- 99_introspection_helpers.sql
-- Useful commands to inspect the store and certification state.

-- Show MV definitions
SELECT schemaname, matviewname, definition
FROM pg_matviews
WHERE schemaname='ml'
  AND matviewname IN (
    'tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv',
    'market_relative_socio_t0_v1_mv',
    'tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv'
  );

-- Show entrypoint view definitions
SELECT pg_get_viewdef('ml.socio_market_feature_store_t0_v1_v'::regclass, true);
SELECT pg_get_viewdef('ml.socio_market_feature_store_train_v'::regclass, true);

-- Column presence checks
SELECT attname
FROM pg_attribute
WHERE attrelid = 'ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv'::regclass
  AND attnum > 0 AND NOT attisdropped
ORDER BY attnum;

-- Registry status
SELECT *
FROM audit.t0_cert_registry
WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v';

-- Baseline counts
SELECT COUNT(*) AS viewdef_objects
FROM audit.t0_viewdef_baseline
WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v';

SELECT COUNT(*) AS baseline_days, MIN(t0_day), MAX(t0_day)
FROM audit.t0_dataset_hash_baseline
WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v'
  AND sample_limit=2000;

-- Guard check
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');
SELECT COUNT(*) FROM ml.socio_market_feature_store_train_v;
