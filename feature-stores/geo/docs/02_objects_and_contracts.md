# 2. Data Model and Objects

## 2.1 Release model (reference layer)

The geo mapping system uses a release table + release-scoped mapping tables.

Typical objects (see `assets/01_create_versioned_reference_tables.sql` and friends):
- `ref.geo_mapping_release` (release_id, label, loaded_at, is_current, etc.)
- `ref.postal_code_to_super_metro` (release_id, postal_code, region, pickup_metro_30_200, super_metro_v4, ...)
- `ref.city_to_super_metro` (release_id, location_city_norm, region, pickup_metro_30_200, super_metro_v4, ...)

Normalization helpers:
- `ref.norm_postal_code(text)`
- `ref.norm_city(text)`

## 2.2 Current release selector views

`ref.geo_mapping_current` returns the currently active release_id.
Other “current” views join through it.

This is operationally useful but not deterministic for training unless pinned.

## 2.3 Certified pinned release

`ref.geo_mapping_pinned_super_metro_v4_v1` is a 1-row view: `{ release_id }`

It is created using `CREATE OR REPLACE VIEW` (not DROP) to avoid breaking dependencies.

## 2.4 Certified geo dimension view (T0)

`ml.geo_dim_super_metro_v4_t0_v1` is a 1-row-per-listing_id dimension:
- keys: `listing_id`
- outputs: `geo_release_id`, `region_geo`, `pickup_metro_30_200_geo`, `super_metro_v4_geo`, `geo_match_method`

Critical detail: it uses a unique base surface (`ml.tom_features_v1_mv`) to guarantee uniqueness.

## 2.5 Certified entrypoint and guarded training view

- Entrypoint: `ml.geo_feature_store_t0_v1_v`
  Includes `edited_date`, `generation`, and the geo fields.

- Guarded view: `ml.geo_dim_super_metro_v4_t0_train_v`
  Fails closed if `ml.geo_feature_store_t0_v1_v` is not certified or stale.
