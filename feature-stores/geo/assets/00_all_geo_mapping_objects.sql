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


-- 02_create_current_views.sql
-- Views that expose only the mapping rows for the current release.

CREATE OR REPLACE VIEW ref.geo_mapping_current AS
SELECT release_id
FROM ref.geo_mapping_release
WHERE is_current
ORDER BY release_id DESC
LIMIT 1;

CREATE OR REPLACE VIEW ref.postal_code_to_super_metro_current AS
SELECT m.*
FROM ref.postal_code_to_super_metro m
JOIN ref.geo_mapping_current c ON c.release_id = m.release_id;

CREATE OR REPLACE VIEW ref.city_to_super_metro_current AS
SELECT m.*
FROM ref.city_to_super_metro m
JOIN ref.geo_mapping_current c ON c.release_id = m.release_id;


-- 03_create_normalization_functions.sql
-- Canonical normalization helpers for join keys.

CREATE OR REPLACE FUNCTION ref.norm_city(x text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT NULLIF(lower(regexp_replace(btrim(x), '\s+', ' ', 'g')), '');
$$;

CREATE OR REPLACE FUNCTION ref.norm_postal_code(x text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
  WITH cleaned AS (
    SELECT regexp_replace(btrim(coalesce(x,'')), '\D', '', 'g') AS pc
  )
  SELECT CASE WHEN pc ~ '^[0-9]{4}$' THEN pc ELSE NULL END
  FROM cleaned;
$$;


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


-- 05_post_load_qa_checks.sql
-- Post-load QA / proof queries. Run after every mapping release load.

-- 1) Ensure exactly one current release
SELECT COUNT(*) AS current_cnt
FROM ref.geo_mapping_release
WHERE is_current;

-- 2) Show current release
SELECT release_id, label, created_at, is_current, notes
FROM ref.geo_mapping_release
WHERE is_current;

-- 3) Mapping row counts
SELECT COUNT(*) AS pc_rows
FROM ref.postal_code_to_super_metro_current;

SELECT COUNT(*) AS city_rows
FROM ref.city_to_super_metro_current;

-- 4) Postal code integrity (must be 4 digits)
SELECT COUNT(*) AS bad_postcodes
FROM ref.postal_code_to_super_metro_current
WHERE postal_code !~ '^[0-9]{4}$';

-- 5) Listing coverage (geo view should match raw)
SELECT COUNT(*) AS listings_geo
FROM ml.listings_geo_current;

SELECT COUNT(*) AS listings_raw
FROM marketplace.listings_raw;

-- 6) Unknown coverage
SELECT
  COUNT(*) FILTER (WHERE region_geo='unknown') AS unknown_region,
  COUNT(*) FILTER (WHERE super_metro_v4_geo LIKE 'other_%' OR super_metro_v4_geo='unknown') AS other_or_unknown_super,
  COUNT(*) AS total
FROM ml.listings_geo_current;


-- 06_switch_current_release.sql
-- Switch the active mapping release (atomic flip).
-- Replace :target_release_id with the release id you want to activate.

BEGIN;

UPDATE ref.geo_mapping_release
SET is_current = false
WHERE is_current = true;

UPDATE ref.geo_mapping_release
SET is_current = true
WHERE release_id = :target_release_id;

COMMIT;


-- 07_optional_indexes.sql
-- Optional performance indexes (non-breaking).
-- Only apply after measuring query plans and considering lock impact.

-- Example: accelerate joins on raw listing postal_code
-- NOTE: Use CONCURRENTLY in production to reduce blocking.
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_listings_raw_postal_code
--   ON marketplace.listings_raw (postal_code);

-- Example: city normalization is expensive; consider storing a normalized city column
-- in your ingestion pipeline OR use an expression index.
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_listings_raw_city_norm
--   ON marketplace.listings_raw (ref.norm_city(location_city));

-- If you create an index referencing ref.norm_city, ensure the function is IMMUTABLE (it is).
