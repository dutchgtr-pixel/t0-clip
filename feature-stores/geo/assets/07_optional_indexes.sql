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
