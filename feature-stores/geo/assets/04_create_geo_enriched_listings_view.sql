-- 04_create_geo_enriched_listings_view.sql
-- View that adds region/metro/super_metro columns to raw listings using the current release.

CREATE OR REPLACE VIEW ml.listings_geo_current AS
SELECT
  l.*,
  c.release_id AS geo_release_id,

  COALESCE(pc.region, cc.region, 'unknown') AS region_geo,
  COALESCE(pc.pickup_metro_30_200, cc.pickup_metro_30_200,
           'other_' || lower(COALESCE(pc.region, cc.region, 'unknown'))) AS pickup_metro_30_200_geo,
  COALESCE(pc.super_metro_v4, cc.super_metro_v4,
           'other_' || lower(COALESCE(pc.region, cc.region, 'unknown'))) AS super_metro_v4_geo

FROM marketplace.listings_raw l
CROSS JOIN ref.geo_mapping_current c
LEFT JOIN ref.postal_code_to_super_metro pc
  ON pc.release_id = c.release_id
 AND pc.postal_code = ref.norm_postal_code(l.postal_code::text)
LEFT JOIN ref.city_to_super_metro cc
  ON cc.release_id = c.release_id
 AND cc.location_city_norm = ref.norm_city(l.location_city);
