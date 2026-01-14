## 1) Speed anchors (as-of) — replaces time-of-query speed anchors

### 1.1 Create `ml.speed_anchor_asof_v1_mv`
This materialized view anchors all “recency weighting” and embargo logic to an explicit `anchor_day`
rather than `CURRENT_DATE`.

> The SQL below is identical to `sql/01_speed_anchor_asof.sql`, provided here for readability.

```sql
-- Speed anchors (as-of) — database materialized view
--
-- Creates:  ml.speed_anchor_asof_v1_mv
-- Keyed by: (anchor_day, generation, sbucket, ptv_bucket)
--
-- Leak-safety:
--   - recency weighting is anchored to explicit anchor_day
--   - SOLD rows must satisfy: sold_day < anchor_day
--   - no CURRENT_DATE / now() tokens are used
--
-- Required upstream objects (rename if your schema differs):
--   - public.listings (or another table named "listings") with edited_date
--   - ml.sold_durations_v1_mv(listing_id, sold_day, duration_hours, ...)
--   - ml.listing_features_enriched_v1_mv(listing_id, generation, storage_gb, ptv_final, ...)

CREATE SCHEMA IF NOT EXISTS ml;

DO $$
DECLARE
  min_day date;
  max_day date;
  lookback_days int := 730;  -- tune as needed
  listings_tbl regclass;
BEGIN
  -- Locate base listings table (prefer schema 'public' if multiple exist).
  SELECT c.oid::regclass INTO listings_tbl
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE c.relname = 'listings'
    AND c.relkind IN ('r','p','f')
  ORDER BY (n.nspname = 'public') DESC,
           (n.nspname = 'marketplace') DESC,
           n.nspname
  LIMIT 1;

  IF listings_tbl IS NULL THEN
    RAISE EXCEPTION 'Could not locate a table named "listings" (expected e.g. public.listings)';
  END IF;

  EXECUTE format(
    'SELECT min(edited_date::date), max(edited_date::date) FROM %s WHERE edited_date IS NOT NULL',
    listings_tbl
  ) INTO min_day, max_day;

  IF max_day IS NULL THEN
    RAISE EXCEPTION 'No edited_date found in %s', listings_tbl;
  END IF;

  min_day := GREATEST(min_day, (max_day - lookback_days));

  RAISE NOTICE 'Using listings table: %', listings_tbl::text;
  RAISE NOTICE 'anchor_day range: % -> %', min_day, max_day;

  DROP MATERIALIZED VIEW IF EXISTS ml.speed_anchor_asof_v1_mv CASCADE;

  CREATE MATERIALIZED VIEW ml.speed_anchor_asof_v1_mv AS
  WITH anchors AS (
    SELECT gs::date AS anchor_day
    FROM generate_series(min_day::date, max_day::date, interval '1 day') gs
  ),
  sold AS (
    SELECT
      f.generation,
      CASE
        WHEN f.storage_gb >= 900 THEN 1024
        WHEN f.storage_gb >= 500 THEN 512
        WHEN f.storage_gb >= 250 THEN 256
        WHEN f.storage_gb >= 120 THEN 128
        ELSE f.storage_gb
      END AS sbucket,
      CASE
        WHEN f.ptv_final < 0.50 THEN NULL::int
        WHEN f.ptv_final < 0.80 THEN 1
        WHEN f.ptv_final < 0.90 THEN 2
        WHEN f.ptv_final < 1.00 THEN 3
        WHEN f.ptv_final < 1.10 THEN 4
        WHEN f.ptv_final < 1.20 THEN 5
        WHEN f.ptv_final < 1.40 THEN 6
        ELSE 7
      END AS ptv_bucket,
      d.sold_day,
      d.duration_hours
    FROM ml.sold_durations_v1_mv d
    JOIN ml.listing_features_enriched_v1_mv f USING (listing_id)
    WHERE f.ptv_final IS NOT NULL
      AND f.ptv_final BETWEEN 0.50 AND 2.50
  ),
  recent AS (
    SELECT
      a.anchor_day,
      s.generation,
      s.sbucket,
      s.ptv_bucket,
      s.sold_day,
      s.duration_hours,

      -- Recency weight is anchored to anchor_day (NOT CURRENT_DATE)
      (0.5 ^ ((a.anchor_day - s.sold_day)::numeric / 90.0)) AS w,

      CASE WHEN s.duration_hours <=  24    THEN 1::numeric ELSE 0::numeric END AS is_fast24,
      CASE WHEN s.duration_hours <= 168    THEN 1::numeric ELSE 0::numeric END AS is_fast7,
      CASE WHEN s.duration_hours >  21*24 THEN 1::numeric ELSE 0::numeric END AS is_slow21
    FROM anchors a
    JOIN sold s
      ON s.ptv_bucket IS NOT NULL
     -- Embargo: never use sales on/after anchor_day
     AND s.sold_day < a.anchor_day
     AND s.sold_day >= (a.anchor_day - 90)
  ),
  agg AS (
    SELECT
      anchor_day,
      generation,
      sbucket,
      ptv_bucket,

      SUM(w) AS sum_w,
      SUM(w * is_fast7)  / NULLIF(SUM(w),0) AS speed_fast7_anchor,
      SUM(w * is_fast24) / NULLIF(SUM(w),0) AS speed_fast24_anchor,
      SUM(w * is_slow21) / NULLIF(SUM(w),0) AS speed_slow21_anchor,

      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_hours) AS speed_median_hours_ptv,

      CASE WHEN SUM(w*w) > 0 THEN (SUM(w)*SUM(w)) / SUM(w*w) ELSE 0 END AS speed_n_eff_ptv,

      -- Proof columns (useful for leak-safety checks)
      MIN(sold_day) AS min_sold_day_used,
      MAX(sold_day) AS max_sold_day_used
    FROM recent
    GROUP BY anchor_day, generation, sbucket, ptv_bucket
  )
  SELECT * FROM agg;

  CREATE UNIQUE INDEX IF NOT EXISTS speed_anchor_asof_v1_uq
    ON ml.speed_anchor_asof_v1_mv (anchor_day, generation, sbucket, ptv_bucket);

  CREATE INDEX IF NOT EXISTS speed_anchor_asof_v1_anchor_idx
    ON ml.speed_anchor_asof_v1_mv (anchor_day);

  ANALYZE ml.speed_anchor_asof_v1_mv;
END $$;
```

### 1.2 Proof checks
```sql
-- No SOLD rows on/after anchor_day (must be 0)
SELECT COUNT(*) AS violations_future_sales
FROM ml.speed_anchor_asof_v1_mv
WHERE max_sold_day_used >= anchor_day;

-- No time-of-query tokens in MV definition (must be false/false)
SELECT
  strpos(lower(pg_get_viewdef('ml.speed_anchor_asof_v1_mv'::regclass,true)),'current_date')>0 AS has_current_date,
  strpos(lower(pg_get_viewdef('ml.speed_anchor_asof_v1_mv'::regclass,true)),'now(')>0          AS has_now;
```
