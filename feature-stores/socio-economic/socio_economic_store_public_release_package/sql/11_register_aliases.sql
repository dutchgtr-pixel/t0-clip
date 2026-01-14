-- 11_register_aliases.sql
-- Optional: explicitly register alias objects as certified/inherited from the entrypoint.

INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
SELECT
  x.entrypoint,
  'CERTIFIED',
  (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v'),
  (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v' AND sample_limit=2000),
  'Alias points to ml.socio_market_feature_store_t0_v1_v; certification inherited.'
FROM (VALUES
  ('ml.socio_market_feature_store_train_v'),
  ('ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv'),
  ('ml.market_relative_socio_t0_v1_mv'),
  ('ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv')
) AS x(entrypoint)
ON CONFLICT (entrypoint) DO UPDATE SET
  status='CERTIFIED',
  certified_at=now(),
  viewdef_objects=EXCLUDED.viewdef_objects,
  dataset_days=EXCLUDED.dataset_days,
  notes=EXCLUDED.notes;
