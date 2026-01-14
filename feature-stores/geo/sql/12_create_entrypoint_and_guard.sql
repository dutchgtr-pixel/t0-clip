-- 12_create_entrypoint_and_guard.sql
-- Certified entrypoint + guarded geo dim view for training.

CREATE OR REPLACE VIEW ml.geo_feature_store_t0_v1_v AS
SELECT
  f.edited_date,
  f.generation,
  f.listing_id,
  g.geo_release_id,
  g.region_geo,
  g.pickup_metro_30_200_geo,
  g.super_metro_v4_geo,
  g.geo_match_method
FROM ml.tom_features_v1_mv f
LEFT JOIN ml.geo_dim_super_metro_v4_t0_v1 g
  ON g.listing_id = f.listing_id
WHERE f.edited_date IS NOT NULL;

CREATE OR REPLACE VIEW ml.geo_dim_super_metro_v4_t0_train_v AS
SELECT g.*
FROM ml.geo_dim_super_metro_v4_t0_v1 g
CROSS JOIN LATERAL (
  SELECT audit.require_certified_strict('ml.geo_feature_store_t0_v1_v', interval '24 hours')
) guard;
