-- 11_create_pinned_geo_dim.sql
-- Create the pinned geo dim view using a unique base surface (ml.tom_features_v1_mv).

CREATE OR REPLACE VIEW ml.geo_dim_super_metro_v4_pinned_v1 AS
WITH base AS (
  SELECT
    f.listing_id,
    ref.norm_postal_code(f.postal_code::text) AS postal_code_norm,
    ref.norm_city(f.location_city)            AS location_city_norm
  FROM ml.tom_features_v1_mv f
),
pin AS (
  SELECT release_id FROM ref.geo_mapping_pinned_super_metro_v4_v1
)
SELECT
  b.listing_id,
  pin.release_id AS geo_release_id,

  COALESCE(pc.region, cc.region, 'unknown') AS region_geo,

  COALESCE(pc.pickup_metro_30_200, cc.pickup_metro_30_200,
           'other_' || lower(COALESCE(pc.region, cc.region, 'unknown'))) AS pickup_metro_30_200_geo,

  COALESCE(pc.super_metro_v4, cc.super_metro_v4,
           'other_' || lower(COALESCE(pc.region, cc.region, 'unknown'))) AS super_metro_v4_geo,

  CASE
    WHEN pc.postal_code IS NOT NULL THEN 'postal'
    WHEN cc.location_city_norm IS NOT NULL THEN 'city'
    WHEN b.postal_code_norm IS NULL AND b.location_city_norm IS NULL THEN 'missing_keys'
    ELSE 'unmapped'
  END AS geo_match_method

FROM base b
CROSS JOIN pin
LEFT JOIN ref.postal_code_to_super_metro pc
  ON pc.release_id = pin.release_id AND pc.postal_code = b.postal_code_norm
LEFT JOIN ref.city_to_super_metro cc
  ON cc.release_id = pin.release_id AND cc.location_city_norm = b.location_city_norm;

CREATE OR REPLACE VIEW ml.geo_dim_super_metro_v4_t0_v1 AS
SELECT * FROM ml.geo_dim_super_metro_v4_pinned_v1;
