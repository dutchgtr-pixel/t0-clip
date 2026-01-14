-- 03_market_relative_mv.sql
-- Strict 30-day trailing market comp statistics (leave-one-out) keyed by (generation,listing_id,t0).
--
-- Depends on:
--   ml.market_base_socio_t0_v1

DROP MATERIALIZED VIEW IF EXISTS ml.market_relative_socio_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.market_relative_socio_t0_v1_mv AS
WITH b AS (
  SELECT
    generation,
    listing_id,
    t0,
    super_metro_v4_geo,
    kommune_code4,
    price,
    log_price_mkt
  FROM ml.market_base_socio_t0_v1
),
sup AS (
  SELECT
    b1.generation,
    b1.listing_id,
    b1.t0,
    CASE WHEN b1.super_metro_v4_geo IS NULL THEN NULL::bigint
         ELSE COUNT(*) OVER w END AS comp_n_30d_super_metro,
    CASE WHEN b1.super_metro_v4_geo IS NULL THEN NULL::double precision
         ELSE AVG(b1.log_price_mkt) OVER w END AS comp_log_price_mean_30d_super_metro,
    CASE WHEN b1.super_metro_v4_geo IS NULL THEN NULL::timestamptz
         ELSE MAX(b1.t0) OVER w END AS comp_max_t0_super_metro
  FROM b b1
  WINDOW w AS (
    PARTITION BY b1.super_metro_v4_geo
    ORDER BY b1.t0
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '00:00:00.000001' PRECEDING
  )
),
kom AS (
  SELECT
    b1.generation,
    b1.listing_id,
    b1.t0,
    CASE WHEN b1.kommune_code4 IS NULL THEN NULL::bigint
         ELSE COUNT(*) OVER w END AS comp_n_30d_kommune,
    CASE WHEN b1.kommune_code4 IS NULL THEN NULL::double precision
         ELSE AVG(b1.log_price_mkt) OVER w END AS comp_log_price_mean_30d_kommune,
    CASE WHEN b1.kommune_code4 IS NULL THEN NULL::timestamptz
         ELSE MAX(b1.t0) OVER w END AS comp_max_t0_kommune
  FROM b b1
  WINDOW w AS (
    PARTITION BY b1.kommune_code4
    ORDER BY b1.t0
    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '00:00:00.000001' PRECEDING
  )
)
SELECT
  b.generation,
  b.listing_id,
  b.t0,
  sup.comp_n_30d_super_metro,
  sup.comp_log_price_mean_30d_super_metro,
  sup.comp_max_t0_super_metro,
  kom.comp_n_30d_kommune,
  kom.comp_log_price_mean_30d_kommune,
  kom.comp_max_t0_kommune
FROM b
JOIN sup USING (generation, listing_id, t0)
JOIN kom USING (generation, listing_id, t0);

-- Unique index required for REFRESH ... CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS market_relative_socio_t0_v1_uq
  ON ml.market_relative_socio_t0_v1_mv (generation, listing_id, t0);

CREATE INDEX IF NOT EXISTS market_relative_socio_t0_v1_t0_idx
  ON ml.market_relative_socio_t0_v1_mv (t0);

ANALYZE ml.market_relative_socio_t0_v1_mv;

-- Proof checks (should be 0)
SELECT COUNT(*) AS n_bad_market_super
FROM ml.market_relative_socio_t0_v1_mv
WHERE comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0;

SELECT COUNT(*) AS n_bad_market_kommune
FROM ml.market_relative_socio_t0_v1_mv
WHERE comp_max_t0_kommune IS NOT NULL AND comp_max_t0_kommune >= t0;
