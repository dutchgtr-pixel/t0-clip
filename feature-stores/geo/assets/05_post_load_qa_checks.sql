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
