-- 01_create_versioned_reference_tables.sql
-- Creates versioned geo mapping reference tables (no impact to raw listings).

CREATE SCHEMA IF NOT EXISTS ref;
CREATE SCHEMA IF NOT EXISTS ml;

-- Control plane for mapping releases
CREATE TABLE IF NOT EXISTS ref.geo_mapping_release (
  release_id   bigserial PRIMARY KEY,
  label        text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  is_current   boolean NOT NULL DEFAULT false,
  notes        text
);

-- Enforce: at most 1 current release
CREATE UNIQUE INDEX IF NOT EXISTS ux_geo_mapping_release_one_current
  ON ref.geo_mapping_release ((1))
  WHERE is_current;

-- Postal code backbone (release_id-scoped)
CREATE TABLE IF NOT EXISTS ref.postal_code_to_super_metro (
  release_id         bigint NOT NULL REFERENCES ref.geo_mapping_release(release_id),
  postal_code        text   NOT NULL CHECK (postal_code ~ '^[0-9]{4}$'),
  region             text   NOT NULL,
  pickup_metro_30_200 text,
  super_metro_v4      text   NOT NULL,
  source              text,
  PRIMARY KEY (release_id, postal_code)
);

CREATE INDEX IF NOT EXISTS idx_pc_map_release_postal
  ON ref.postal_code_to_super_metro(release_id, postal_code);

CREATE INDEX IF NOT EXISTS idx_pc_map_release_super
  ON ref.postal_code_to_super_metro(release_id, super_metro_v4);

-- City fallback (release_id-scoped)
CREATE TABLE IF NOT EXISTS ref.city_to_super_metro (
  release_id          bigint NOT NULL REFERENCES ref.geo_mapping_release(release_id),
  location_city_norm  text   NOT NULL,
  region              text   NOT NULL,
  pickup_metro_30_200  text,
  super_metro_v4       text   NOT NULL,
  source               text,
  PRIMARY KEY (release_id, location_city_norm)
);

CREATE INDEX IF NOT EXISTS idx_city_map_release_city
  ON ref.city_to_super_metro(release_id, location_city_norm);
