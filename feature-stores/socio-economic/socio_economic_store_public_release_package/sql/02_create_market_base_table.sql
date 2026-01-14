-- 02_create_market_base_table.sql
-- Thin market base table used for strict trailing 30-day comp statistics.

CREATE TABLE IF NOT EXISTS ml.market_base_socio_t0_v1 (
  generation         int4        NOT NULL,
  listing_id            int8        NOT NULL,
  t0                 timestamptz NOT NULL,
  super_metro_v4_geo  text,
  kommune_code4       text,
  price              float8,
  log_price_mkt       float8
);

-- Recommended uniqueness (enables stable joins)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname='ml'
      AND indexname='market_base_socio_t0_v1_uq'
  ) THEN
    EXECUTE 'CREATE UNIQUE INDEX market_base_socio_t0_v1_uq
             ON ml.market_base_socio_t0_v1 (generation, listing_id, t0)';
  END IF;
END $$;

-- Performance indexes for window partitions
CREATE INDEX IF NOT EXISTS market_base_socio_t0_v1_super_t0_idx
  ON ml.market_base_socio_t0_v1 (super_metro_v4_geo, t0);

CREATE INDEX IF NOT EXISTS market_base_socio_t0_v1_kommune_t0_idx
  ON ml.market_base_socio_t0_v1 (kommune_code4, t0);

ANALYZE ml.market_base_socio_t0_v1;
