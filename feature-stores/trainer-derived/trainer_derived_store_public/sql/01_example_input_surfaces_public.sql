-- 01_example_input_surfaces_public.sql
-- OPTIONAL: Creates synthetic example input surfaces so the derived store
-- can be built and certified in a clean Postgres database.
--
-- In a real deployment, you would replace these with your own:
--   - ml.anchor_features_v1_mv
--   - ml.market_feature_store_train_v (must be certified in your framework)
--   - ml.image_features_unified_v1_train_v (must be certified in your framework)

BEGIN;

CREATE SCHEMA IF NOT EXISTS ml;

-- Anchor universe (what training uses as the key set)
DROP TABLE IF EXISTS ml.anchor_events;
CREATE TABLE ml.anchor_events (
  generation  int         NOT NULL,
  listing_id  bigint      NOT NULL,
  edited_date timestamptz NOT NULL
);

INSERT INTO ml.anchor_events(generation, listing_id, edited_date) VALUES
  (1, 100000001, '2026-01-01 00:00:00+00'),
  (1, 100000002, '2026-01-01 12:00:00+00'),
  (1, 100000003, '2026-01-02 00:00:00+00'),
  (2, 200000001, '2026-01-02 09:00:00+00'),
  (2, 200000002, '2026-01-03 15:00:00+00');

DROP MATERIALIZED VIEW IF EXISTS ml.anchor_features_v1_mv;
CREATE MATERIALIZED VIEW ml.anchor_features_v1_mv AS
SELECT generation, listing_id, edited_date
FROM ml.anchor_events;

CREATE UNIQUE INDEX IF NOT EXISTS anchor_features_v1_mv_pk
  ON ml.anchor_features_v1_mv (generation, listing_id, edited_date);

ANALYZE ml.anchor_features_v1_mv;

-- Certified base feature surface (training-safe)
DROP TABLE IF EXISTS ml.market_feature_store;
CREATE TABLE ml.market_feature_store (
  generation                int         NOT NULL,
  listing_id                bigint      NOT NULL,
  edited_date               timestamptz NOT NULL,
  price                     numeric     NULL,
  condition_score           float8      NULL,
  battery_pct_effective     int         NULL,
  delta_vs_sold_median_30d  numeric     NULL,
  delta_vs_ask_median_day   numeric     NULL,
  ptv_final                 float8      NULL,
  damage_severity_ai        int         NULL,
  damage_binary_ai          int         NULL
);

INSERT INTO ml.market_feature_store VALUES
  (1, 100000001, '2026-01-01 00:00:00+00', 1000, 0.92, 88,  50,  10, 0.98, 0, 0),
  (1, 100000002, '2026-01-01 12:00:00+00',  900, 0.75, 85, 120, -20, 0.94, 1, 1),
  (1, 100000003, '2026-01-02 00:00:00+00', 1100, 0.65,  0, -30,   5, 1.02, 0, 0),
  (2, 200000001, '2026-01-02 09:00:00+00', 1500, 0.88, 92, 400,  80, 0.90, 0, 0),
  (2, 200000002, '2026-01-03 15:00:00+00', 1400, 0.72, 90, 600, 100, 0.93, 2, 1);

CREATE OR REPLACE VIEW ml.market_feature_store_train_v AS
SELECT
  generation,
  listing_id,
  edited_date,
  price,
  condition_score,
  battery_pct_effective,
  delta_vs_sold_median_30d,
  delta_vs_ask_median_day,
  ptv_final,
  damage_severity_ai,
  damage_binary_ai
FROM ml.market_feature_store;

-- Certified image/vision surface (training-safe)
DROP TABLE IF EXISTS ml.image_features_unified;
CREATE TABLE ml.image_features_unified (
  generation      int     NOT NULL,
  listing_id      bigint  NOT NULL,
  battery_pct_img int     NULL
);

INSERT INTO ml.image_features_unified VALUES
  (1, 100000001, 87),
  (1, 100000002, NULL),
  (1, 100000003, 84),
  (2, 200000001, 93),
  (2, 200000002, 91);

CREATE OR REPLACE VIEW ml.image_features_unified_v1_train_v AS
SELECT generation, listing_id, battery_pct_img
FROM ml.image_features_unified;

COMMIT;
