# Appendix — Original Documentation (Part 1 of 2)

This appendix preserves the original merged documentation. The only edits applied are **credential redactions** (e.g., passwords).

-- =====================================================================
-- PHASE 1 — LEAK-SAFE FEATURE STORE (Postgres MVs) + EMBARGOED ANCHORS
-- All blocks are independent. Run top→bottom. Uses only real columns.
-- Spam rows are excluded at source. No outcomes in features.
-- Raw title/description are stored for future NLP (not fed to model).
-- =====================================================================


-- ============================================================
-- A) BASE t₀ FEATURE SNAPSHOT (one row per listing_id at edited_date)
--    Keeps only non-leaky, original columns + raw text for later NLP
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_mv AS
WITH t0 AS (
  SELECT DISTINCT ON (l.listing_id)
         l.listing_id,                 -- join key (NOT a feature)
         l.edited_date,             -- t₀ (use for time features only)
         l.generation,
         l.model,                   -- kept for future NLP/variant mapping
         l.storage_gb,
         l.price,
         l.condition_score,
         l.battery_pct_fixed_ai,    -- AI-cleaned battery proxy (non-leak)
         l.damage_severity_ai,      -- 0..3 (non-leak)
         l.damage_binary_ai,        -- 0/1  (non-leak)
         l.seller_rating,
         l.review_count,
         l.member_since_year,
         l.location_city,
         l.postal_code,
         l.title,                   -- raw text (store-only; not a model feature yet)
         l.description              -- raw text (store-only; not a model feature yet)
  FROM "iPhone".iphone_listings l
  WHERE l.spam IS NULL
  ORDER BY l.listing_id, l.edited_date ASC NULLS LAST
)
SELECT
  listing_id,
  edited_date,
  generation,
  model,
  storage_gb,
  price,
  condition_score,
  battery_pct_fixed_ai,
  damage_severity_ai,
  damage_binary_ai,
  seller_rating,
  review_count,
  member_since_year,
  location_city,
  postal_code,
  title,
  description,
  char_length(COALESCE(title,''))       AS title_length,
  char_length(COALESCE(description,'')) AS description_length
FROM t0;

CREATE UNIQUE INDEX IF NOT EXISTS tfeat_v1_listing_uq ON ml.tom_features_v1_mv (listing_id);
CREATE INDEX        IF NOT EXISTS tfeat_v1_dt_idx  ON ml.tom_features_v1_mv (edited_date, generation, storage_gb);
ANALYZE ml.tom_features_v1_mv;



-- ============================================
-- B) LABELS (leak-safe duration & sold flag)
--    Uses sold_date + last_seen only (no leakage)
-- ============================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_labels_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_labels_v1_mv AS
WITH base AS (
  SELECT listing_id, edited_date
  FROM   ml.tom_features_v1_mv           -- already 1 row per listing_id
),
sold AS (
  SELECT listing_id, MIN(sold_date) AS sold_date
  FROM   "iPhone".iphone_listings
  WHERE  spam IS NULL AND sold_date IS NOT NULL
  GROUP BY listing_id
),
last_seen AS (
  SELECT listing_id, MAX(last_seen) AS last_seen
  FROM   "iPhone".iphone_listings
  WHERE  spam IS NULL
  GROUP BY listing_id
)
SELECT
  b.listing_id,
  b.edited_date,
  s.sold_date,
  ls.last_seen,
  (s.sold_date IS NOT NULL) AS sold_event,
  GREATEST(
    0.0,
    EXTRACT(EPOCH FROM (COALESCE(s.sold_date, ls.last_seen) - b.edited_date))/3600.0
  ) AS duration_hours
FROM base b
LEFT JOIN sold      s  USING (listing_id)
LEFT JOIN last_seen ls USING (listing_id);

CREATE UNIQUE INDEX IF NOT EXISTS tlabels_v1_listing_uq ON ml.tom_labels_v1_mv (listing_id);
CREATE INDEX        IF NOT EXISTS tlabels_v1_evt_idx ON ml.tom_labels_v1_mv (sold_event);
ANALYZE ml.tom_labels_v1_mv;



-- ==========================================
-- C) TRAINING BASE (features + labels join)
-- ==========================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_train_base_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_train_base_v1_mv AS
SELECT
  f.*,
  y.sold_event,
  y.duration_hours
FROM   ml.tom_features_v1_mv f
JOIN   ml.tom_labels_v1_mv   y USING (listing_id);

CREATE UNIQUE INDEX IF NOT EXISTS ttrain_v1_listing_uq ON ml.tom_train_base_v1_mv (listing_id);
CREATE INDEX        IF NOT EXISTS ttrain_v1_evt_idx ON ml.tom_train_base_v1_mv (sold_event);
ANALYZE ml.tom_train_base_v1_mv;



-- ======================================================
-- D1) SOLD DURATION HISTORY (for market-speed anchors)
--      (history only; no leak)
-- ======================================================
DROP MATERIALIZED VIEW IF EXISTS ml.sold_durations_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.sold_durations_v1_mv AS
SELECT
  listing_id,
  generation,
  storage_gb,
  sold_date::date AS sold_day,
  EXTRACT(EPOCH FROM (sold_date - edited_date))/3600.0 AS duration_hours
FROM "iPhone".iphone_listings
WHERE spam IS NULL
  AND sold_date IS NOT NULL
  AND edited_date IS NOT NULL;

CREATE INDEX IF NOT EXISTS sold_dur_v1_idx
  ON ml.sold_durations_v1_mv (generation, storage_gb, sold_day);
ANALYZE ml.sold_durations_v1_mv;



-- =====================================================
-- D2) ASK-PRICE DAY MEDIANS (as-of; supply anchor)
--       (no outcomes; no leak)
-- =====================================================
DROP MATERIALIZED VIEW IF EXISTS ml.ask_daily_median_v1 CASCADE;

CREATE MATERIALIZED VIEW ml.ask_daily_median_v1 AS
SELECT
  generation,
  storage_gb,
  edited_date::date AS ask_day,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS ask_median_price,
  COUNT(*) AS ask_cnt
FROM "iPhone".iphone_listings
WHERE spam IS NULL
  AND edited_date IS NOT NULL
GROUP BY generation, storage_gb, edited_date::date;

CREATE INDEX IF NOT EXISTS ask_daily_med_v1_idx
  ON ml.ask_daily_median_v1 (generation, storage_gb, ask_day);
ANALYZE ml.ask_daily_median_v1;



-- =====================================================
-- D3) SOLD-PRICE HISTORY (value anchor; embargo source)
--       (uses sold_price / real_sold_price, no leak)
-- =====================================================
DROP MATERIALIZED VIEW IF EXISTS ml.sold_prices_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.sold_prices_v1_mv AS
SELECT
  generation,
  storage_gb,
  sold_date::date AS sold_day,
  COALESCE(real_sold_price, sold_price) AS sold_price
FROM "iPhone".iphone_listings
WHERE spam IS NULL
  AND sold_date IS NOT NULL
  AND storage_gb IS NOT NULL
  AND COALESCE(real_sold_price, sold_price) IS NOT NULL;

CREATE INDEX IF NOT EXISTS sold_prices_v1_idx
  ON ml.sold_prices_v1_mv (generation, storage_gb, sold_day);
ANALYZE ml.sold_prices_v1_mv;



-- =================================================================
-- E) ENRICHED t₀ FEATURES (attach embargoed medians via LATERAL)
--     - anchor_median_hours_14d (market speed; sold_day < t₀)
--     - ask_median_price (as-of)
--     - sold_median_price_30d (value; sold_day < t₀)
--     - PTV & Δ features (relative to anchors)
-- =================================================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_mv AS
SELECT
  f.*,

  -- 14d historical median time-to-sell (embargoed)
  d14.anchor_median_hours_14d,
  d14.anchor_14d_n,

  -- same-day cohort ask median (as-of)
  a.ask_median_price,
  a.ask_cnt,

  -- 30d embargoed sold-price median
  sp.sold_median_price_30d,
  sp.sold_cnt_30d,

  -- relative price features (feed these to the model)
  CASE WHEN sp.sold_median_price_30d > 0
       THEN f.price::numeric / sp.sold_median_price_30d END AS ptv_sold_30d,
  CASE WHEN a.ask_median_price > 0
       THEN f.price::numeric / a.ask_median_price END       AS ptv_ask_day,

  CASE WHEN sp.sold_median_price_30d IS NOT NULL
       THEN f.price - sp.sold_median_price_30d END          AS delta_vs_sold_median_30d,
  CASE WHEN a.ask_median_price IS NOT NULL
       THEN f.price - a.ask_median_price END                AS delta_vs_ask_median_day,

  -- optional single “final” PTV with reliability thresholds
  CASE
    WHEN sp.sold_cnt_30d >= 10 AND sp.sold_median_price_30d > 0
      THEN f.price::numeric / sp.sold_median_price_30d
    WHEN a.ask_cnt >= 5 AND a.ask_median_price > 0
      THEN f.price::numeric / a.ask_median_price
    ELSE NULL
  END AS ptv_final

FROM ml.tom_features_v1_mv AS f

-- 14d duration anchor, using only sells strictly before t₀ day
LEFT JOIN LATERAL (
  SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.duration_hours) AS anchor_median_hours_14d,
    COUNT(*) AS anchor_14d_n
  FROM ml.sold_durations_v1_mv s
  WHERE s.generation = f.generation
    AND s.storage_gb = f.storage_gb
    AND s.sold_day  <  f.edited_date::date
    AND s.sold_day >=  f.edited_date::date - INTERVAL '14 days'
) d14 ON TRUE

-- as-of ask median for the same cohort/day
LEFT JOIN LATERAL (
  SELECT am.ask_median_price, am.ask_cnt
  FROM ml.ask_daily_median_v1 am
  WHERE am.generation = f.generation
    AND am.storage_gb = f.storage_gb
    AND am.ask_day    = f.edited_date::date
  LIMIT 1
) a ON TRUE

-- embargoed 30d sold-price median (value anchor)
LEFT JOIN LATERAL (
  SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p.sold_price) AS sold_median_price_30d,
    COUNT(*) AS sold_cnt_30d
  FROM ml.sold_prices_v1_mv p
  WHERE p.generation = f.generation
    AND p.storage_gb = f.storage_gb
    AND p.sold_day  <  f.edited_date::date
    AND p.sold_day >=  f.edited_date::date - INTERVAL '30 days'
) sp ON TRUE
;

CREATE UNIQUE INDEX IF NOT EXISTS tfeat_enr_v1_uq ON ml.tom_features_v1_enriched_mv (listing_id);
CREATE INDEX        IF NOT EXISTS tfeat_enr_v1_dt ON ml.tom_features_v1_enriched_mv (edited_date, generation, storage_gb);
ANALYZE ml.tom_features_v1_enriched_mv;
-- =====================================================================
-- PHASE-1 GOVERNANCE: FEATURE–CONTRACT HASH (PURE SQL, NO PYTHON)
-- Run in psql. This file contains:
--   1) Compute & SET ml.feature_set.features_hash for 'tom_features_v1'
--   2) One-shot assertion (recompute hash and compare → shows OK/MISMATCH)
--   3) Reusable assertion function ml.assert_feature_contract(text)
-- =====================================================================


-- =========================================================
-- (1) COMPUTE & SET features_hash FOR 'tom_features_v1'
-- • Requires: ml.feature_set + ml.feature_contract populated
-- • Uses pgcrypto.digest( … , 'sha256' )
-- • Updates ml.feature_set.features_hash and n_features
-- =========================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

WITH fs AS (
  SELECT set_id FROM ml.feature_set WHERE name = 'tom_features_v1'
),
rows AS (
  SELECT
      ordinal,
      feature_name,
      dtype,
      COALESCE(expr_sql,'')      AS expr_sql,
      COALESCE(leakage_rule,'')  AS leakage_rule,
      CASE WHEN is_nullable THEN '1' ELSE '0' END AS is_nullable
  FROM ml.feature_contract
  WHERE set_id = (SELECT set_id FROM fs)
  ORDER BY ordinal
),
canon AS (
  SELECT
      string_agg(
        ordinal::text || '|' || feature_name || '|' || dtype || '|' ||
        expr_sql || '|' || leakage_rule || '|' || is_nullable,
        '||' ORDER BY ordinal
      ) AS payload,
      COUNT(*) AS n_rows
  FROM rows
)
UPDATE ml.feature_set f
SET    features_hash = encode( digest( convert_to(c.payload,'utf8'), 'sha256' ), 'hex' ),
       n_features    = c.n_rows
FROM   canon c, fs
WHERE  f.set_id = fs.set_id;

-- Show what was written
SELECT name, n_features, features_hash
FROM   ml.feature_set
WHERE  name = 'tom_features_v1';



-- ===============================================================
-- (2) ONE-SHOT ASSERTION (READ-ONLY)
-- • Recomputes canonical hash from ml.feature_contract and compares
-- • Prints expected vs computed and 'OK' or 'MISMATCH'
-- ===============================================================
WITH fs AS (
  SELECT set_id, features_hash AS expected_hash, n_features AS expected_n
  FROM   ml.feature_set
  WHERE  name = 'tom_features_v1'
),
rows AS (
  SELECT
      ordinal,
      feature_name,
      dtype,
      COALESCE(expr_sql,'')      AS expr_sql,
      COALESCE(leakage_rule,'')  AS leakage_rule,
      CASE WHEN is_nullable THEN '1' ELSE '0' END AS is_nullable
  FROM   ml.feature_contract
  WHERE  set_id = (SELECT set_id FROM fs)
  ORDER  BY ordinal
),
canon AS (
  SELECT
      string_agg(
        ordinal::text || '|' || feature_name || '|' || dtype || '|' ||
        expr_sql || '|' || leakage_rule || '|' || is_nullable,
        '||' ORDER BY ordinal
      ) AS payload,
      COUNT(*) AS n_rows
  FROM rows
)
SELECT
  fs.expected_n,
  fs.expected_hash,
  c.n_rows                                             AS computed_n,
  encode( digest( convert_to(c.payload,'utf8'),'sha256'),'hex') AS computed_hash,
  CASE
    WHEN fs.expected_n = c.n_rows
     AND fs.expected_hash = encode(digest(convert_to(c.payload,'utf8'),'sha256'),'hex')
    THEN 'OK' ELSE 'MISMATCH'
  END AS contract_check
FROM fs, canon c;



-- =====================================================================
-- (3) REUSABLE ASSERTION FUNCTION
-- • Create once.
-- • Call: SELECT ml.assert_feature_contract('tom_features_v1');
-- • Raises EXCEPTION if contract rows or hash differ (fail-fast in jobs)
-- =====================================================================
CREATE OR REPLACE FUNCTION ml.assert_feature_contract(p_set text)
RETURNS void
LANGUAGE plpgsql AS
$$
DECLARE
  v_set_id        bigint;
  v_expected_hash text;
  v_expected_n    int;
  v_now_hash      text;
  v_now_n         int;
BEGIN
  SELECT set_id, features_hash, n_features
    INTO v_set_id, v_expected_hash, v_expected_n
  FROM ml.feature_set
  WHERE name = p_set;

  IF v_set_id IS NULL THEN
    RAISE EXCEPTION 'feature_set % not found', p_set;
  END IF;

  WITH rows AS (
    SELECT
        ordinal,
        feature_name,
        dtype,
        COALESCE(expr_sql,'')      AS expr_sql,
        COALESCE(leakage_rule,'')  AS leakage_rule,
        CASE WHEN is_nullable THEN '1' ELSE '0' END AS is_nullable
    FROM   ml.feature_contract
    WHERE  set_id = v_set_id
    ORDER  BY ordinal
  ),
  canon AS (
    SELECT
        string_agg(
          ordinal::text || '|' || feature_name || '|' || dtype || '|' ||
          expr_sql || '|' || leakage_rule || '|' || is_nullable,
          '||' ORDER BY ordinal
        ) AS payload,
        COUNT(*) AS n_rows
    FROM rows
  )
  SELECT encode( digest( convert_to(payload,'utf8'),'sha256' ), 'hex' ),
         n_rows
    INTO v_now_hash, v_now_n
  FROM canon;

  IF v_now_hash IS DISTINCT FROM v_expected_hash OR v_now_n IS DISTINCT FROM v_expected_n THEN
    RAISE EXCEPTION 'FEATURE CONTRACT DRIFT: expected n=% hash=%, got n=% hash=%',
                    v_expected_n, v_expected_hash, v_now_n, v_now_hash;
  END IF;

  RAISE NOTICE 'Feature contract OK (n=%, hash=%)', v_now_n, v_now_hash;
END;
$$;

-- Example usage (will raise on mismatch):
-- SELECT ml.assert_feature_contract('tom_features_v1');
Phase-1 Feature Contracts & Governance — End-to-End Guide
This doc explains what a feature contract is, how it ties into your leak-safe feature store (Phase-1 MVs), how to lock it with a hash, and how to operate & evolve it safely. All examples match your running schema (ml.*) and the views you’ve just built.

1) What is a “feature contract”?
A feature contract is the governed, explicit whitelist of columns the model is allowed to read at training/inference time. It defines:


Which fields are model inputs (names & SQL expressions),


In what order (ordinal → affects hashing),


Types (dtype), nullability, and a leakage rule (e.g., clip_at_edited_date).


The contract gives you:


Leak safety by construction (only expressions valid at t₀ are allowed),


Reproducibility via a canonical hash stored in ml.feature_set.features_hash,


Drift protection with a runtime assert that fails fast if anyone changes the spec.



Important: The contract governs model inputs, not the entire store. You are storing raw title/description for future NLP, but they are not listed in the contract until you add derived numeric features.


2) Where the contract lives (tables & fields)


ml.feature_set
set_id (PK), name, features_hash, n_features, created_at, author, notes
One row per feature set (e.g., tom_features_v1). features_hash + n_features record the canonical spec.


ml.feature_contract (the spec itself)
set_id (FK), ordinal, feature_name, dtype, expr_sql, leakage_rule, is_nullable, description
Each row is one feature entry. Order matters (used in hashing).


ml.feature_set_current
Holds the active set (singleton_id=true, set_id). Lets you “flip” active sets without code changes.


Key semantics


expr_sql: The exact SQL expression used to source the feature at t₀. In Phase-1 we always use clip_at_edited_date semantics; the source for model features is ml.tom_features_v1_enriched_mv which is already constructed to be leak-safe (no sold_date/last_seen etc., anchors use <= edited_date::date).


leakage_rule: Metadata describing how the expression is guaranteed non-leaky. For Phase-1 we use clip_at_edited_date. If later you add cohort anchors with different windows, keep the metadata accurate (e.g., embargo_14d_sold).


is_nullable: Whether the pipeline must tolerate nulls. You may keep the raw feature nullable and let the model handle it, or add SQL defaults.



3) How the contract connects to your MV “store”
You built these leak-safe layers (all spam-filtered, all aligned to edited_date as t₀):


A: ml.tom_features_v1_mv — base snapshot (one row per listing_id at earliest edited_date), raw text stored, no outcomes.


B: ml.tom_labels_v1_mv — label view using only sold_date and last_seen → sold_event, duration_hours (right-censored).


C: ml.tom_train_base_v1_mv — features + labels join for training.


D1: ml.sold_durations_v1_mv — sold history → sold_day, duration_hours.


D2: ml.ask_daily_median_v1 — as-of day ask medians.


D3: ml.sold_prices_v1_mv — sold price history (value anchor source).


E: ml.tom_features_v1_enriched_mv — embargoed anchors and relative price features, all evaluated at edited_date::date:


anchor_median_hours_14d, anchor_14d_n


ask_median_price, ask_cnt


sold_median_price_30d, sold_cnt_30d


ptv_sold_30d, ptv_ask_day, delta_vs_*, ptv_final




The contract references columns from E (and a few from A that are non-leak), not outcomes. This is how you ensure the model always reads the same leak-safe spec at t₀.

4) Locking the contract with a canonical hash (no Python)
You already ran these two blocks:


Block 1 (SET hash): concatenates the ordered contract rows (ordinal|feature|dtype|expr|leakage|nullable) and writes a SHA-256 to ml.feature_set.features_hash along with n_features.
Result:
name=tom_features_v1  n_features=26  
features_hash=376417e5653f05f5953e4bbbed2f562bb54329314e705c292d46bd1e3fbf4088



Block 2 (ASSERT): recomputes the same canonical hash from the live ml.feature_contract rows and compares it to ml.feature_set.features_hash. You got contract_check = OK.


You also created ml.assert_feature_contract(text), a reusable function that throws EXCEPTION if any row or ordering changes. Call it at the top of train/infer to fail fast:
SELECT ml.assert_feature_contract('tom_features_v1');  -- prints “Feature contract OK …” or raises error

Why this works:
By hashing the ordered row materialization, any drift — add/remove/reorder/rename, dtype change, SQL change, leakage rule change — flips the hash. You’ll catch it before training/inference proceeds.

5) What should be in the contract (Phase-1)
Do include (non-leak, t₀-safe, original names, plus anchors):


Cohort & price: generation, model (if you plan to encode it), storage_gb, price


Condition & quality: condition_score, battery_pct_fixed_ai, damage_severity_ai, damage_binary_ai


Seller: seller_rating, review_count, member_since_year (or derived age_years)


Location & text signals: location_city, postal_code (for region bins), title_length, description_length


Anchors & relatives: anchor_median_hours_14d, anchor_14d_n,
ask_median_price, ask_cnt,
sold_median_price_30d, sold_cnt_30d,
ptv_sold_30d, ptv_ask_day, delta_vs_sold_median_30d, delta_vs_ask_median_day, ptv_final


Do not include:


Any outcomes: sold_date, sold_price, real_sold_price, last_seen, status


Ops/audit: battery_pct_*_at/by/ctx/source, quality_ai_*, damage_ai_*, seller_stats_fetched, url, storage_gb_raw, storage (text)


IDs: listing_id (join key only; not a feature)


Raw text: title, description (kept in store, not in contract until you derive numeric NLP features)



6) Operating the contract (day-to-day)
Trainer & Infer should both:


Call SELECT ml.assert_feature_contract('tom_features_v1');


Read only the contracted columns from ml.tom_train_base_v1_mv (plus anchors from ml.tom_features_v1_enriched_mv).


For inference, read ml.tom_features_v1_enriched_mv at the current edited_date::date, compute expected_hours, p_24h, p_48h, and UPSERT into ml.predictions_v1 (Phase-2).


Refreshing MVs (order):
A → B → C → D1 → D2 → D3 → E
Where possible, add UNIQUE indexes so you can REFRESH MATERIALIZED VIEW CONCURRENTLY … without blocking readers.

7) Evolving the contract (vNext)
When you add/change features:


Prototype in a new MV (e.g., ml.tom_features_v2_enriched_mv) or add new columns to existing MV.


Insert new rows into ml.feature_contract for tom_features_v2 with the exact expr_sql and ordinal ordering.


Stamp the hash for tom_features_v2 (Block 1).


Update ml.feature_set_current to point at the new set only after the trainer & inference pass assert_feature_contract('tom_features_v2') and you’ve validated metrics.


Keep tom_features_v1 around for rollback (or archive it).


Never mutate an active contract in place without bumping its set name or re-stamping the hash and revalidating — the assert will catch it, but you don’t want to break running jobs.

8) Troubleshooting


“MISMATCH” / ASSERT failure: Someone changed ml.feature_contract (feature added/removed/renamed/reordered, type/expr/leakage changed). Re-stamp the hash only after you confirm the change is intended and you’ve retrained/validated.


Unique index errors on labels: Ensure B reads from A (already deduped by listing_id), not raw listings.


Anchor columns NULL: sold_cnt_30d or ask_cnt is low for that cohort/day. Either use ptv_final (gated fallback) or add generation-only/global fallbacks.


“Leakage” concerns: Verify E lateral joins use s.sold_day < edited_date::date for sold anchors; verify A contains no sold_*/last_seen/status/audit.



9) Quick references (what you ran for governance)
Set hash (writes features_hash + n_features):
CREATE EXTENSION IF NOT EXISTS pgcrypto;
WITH fs AS (...), rows AS (...), canon AS (...)
UPDATE ml.feature_set f
SET features_hash = encode(digest(convert_to(c.payload,'utf8'),'sha256'),'hex'),
    n_features    = c.n_rows
FROM canon c, fs
WHERE f.set_id = fs.set_id;
SELECT name, n_features, features_hash FROM ml.feature_set WHERE name='tom_features_v1';

One-shot check (read-only):
WITH fs AS (...), rows AS (...), canon AS (...)
SELECT expected_n, expected_hash,
       computed_n, computed_hash,
       CASE WHEN expected_n=computed_n AND expected_hash=computed_hash THEN 'OK' ELSE 'MISMATCH' END AS contract_check
FROM fs, canon;

Reusable assert:
SELECT ml.assert_feature_contract('tom_features_v1');
-- raises EXCEPTION on drift; prints NOTICE on success


10) Summary


Your feature store (MVs A–E) is leak-safe and indexed.


The feature contract defines the exact inputs, frozen by a hash and enforced by a runtime assert.


Anchors are embargoed; PTV/Δ features are computed strictly at t₀; no ops/outcomes/noise slip into the model.


You can now train against ml.tom_train_base_v1_mv (+ enriched), assert the contract in every job, and move on to Phase-2 (predictions table + UPSERT) with confidence.

-- Create a spam-clean materialized view for training/inference.
-- It pulls rows from your enriched read layer and excludes any listing
-- whose raw "spam" flag (text) in "iPhone".iphone_listings resolves to TRUE
-- under your semantics (wtb/wanted/duplicate-junk/junk/bundled/below13, etc.).

CREATE MATERIALIZED VIEW IF NOT EXISTS ml.tom_features_v1_enriched_clean_mv AS
SELECT e.*
FROM ml.tom_features_v1_enriched_mv e
WHERE NOT EXISTS (
  SELECT 1
  FROM "iPhone".iphone_listings l
  WHERE l.listing_id = e.listing_id
    AND (
      CASE
        WHEN l.spam IS NULL THEN FALSE
        -- Mark these tokens as spam/should-exclude (extend as needed).
        WHEN trim(lower(l.spam)) IN (
          't','true','1','yes','y','spam',
          'wtb','wanted',
          'duplicate-junk','junk',
          'bundled','below13'
        )
        THEN TRUE
        WHEN trim(lower(l.spam)) IN ('f','false','0','no','n') THEN FALSE
        ELSE FALSE
      END
    )
);

-- Ensure the spam-clean view is refreshable CONCURRENTLY:
-- it must be truly unique on listing_id. If you see an error here,
-- it means duplicates exist; fix the view definition or upstream first.
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS tom_features_v1_enriched_clean_mv_uq
  ON ml.tom_features_v1_enriched_clean_mv (listing_id);

-- Optional: quick sanity checks (run manually if you want)
-- Expect 0 duplicates:
-- SELECT COUNT(*) FROM (
--   SELECT listing_id, COUNT(*) FROM ml.tom_features_v1_enriched_clean_mv
--   GROUP BY listing_id HAVING COUNT(*) > 1
-- ) d;
--
-- Expect 0 spam-present rows:
-- WITH spam_raw AS (
--   SELECT listing_id FROM "iPhone".iphone_listings l
--   WHERE CASE
--           WHEN l.spam IS NULL THEN FALSE
--           WHEN trim(lower(l.spam)) IN ('t','true','1','yes','y','spam','wtb','wanted','duplicate-junk','junk','bundled','below13') THEN TRUE
--           WHEN trim(lower(l.spam)) IN ('f','false','0','no','n') THEN FALSE
--           ELSE FALSE
--         END
-- )
-- SELECT COUNT(*) FROM spam_raw r
-- JOIN ml.tom_features_v1_enriched_clean_mv e USING (listing_id);
Feature Store Refresh — Ops Guide (with Spam-Clean Read Layer)

This doc explains what we changed and how to refresh your TOM feature store end-to-end so your training/inference reads a spam-clean, unique view that can be refreshed concurrently (non-blocking).

What changed (TL;DR)

Added a spam-clean materialized view:

ml.tom_features_v1_enriched_clean_mv

It selects from your enriched read layer and excludes spam/wtb/duplicate/bundled/below13 rows using your raw "iPhone".iphone_listings.spam text flag.

Enforced uniqueness on (listing_id) so we can REFRESH CONCURRENTLY.

Standardized a safe refresh sequence:

Blocking refresh for base + history layers (A/B/C/D1/D2/D3).

Concurrent refresh for read layers (E enriched + E-clean spam-cleaned).

ANALYZE the read layers for good plans.

Objects you care about

Base / labels / training base / history (blocking refresh)

ml.tom_features_v1_mv

ml.tom_labels_v1_mv

ml.tom_train_base_v1_mv

ml.sold_durations_v1_mv (D1)

ml.ask_daily_median_v1 (D2)

ml.sold_prices_v1_mv (D3)

Read layers (concurrent refresh)

ml.tom_features_v1_enriched_mv (unique by listing_id)

ml.tom_features_v1_enriched_clean_mv (new, spam-clean, unique by listing_id)

Model input should point to: ml.tom_features_v1_enriched_clean_mv.

One-time setup

Create the spam-clean MV and its unique index:

CREATE MATERIALIZED VIEW IF NOT EXISTS ml.tom_features_v1_enriched_clean_mv AS
SELECT e.*
FROM ml.tom_features_v1_enriched_mv e
WHERE NOT EXISTS (
  SELECT 1
  FROM "iPhone".iphone_listings l
  WHERE l.listing_id = e.listing_id
    AND (
      CASE
        WHEN l.spam IS NULL THEN FALSE
        WHEN trim(lower(l.spam)) IN (
          't','true','1','yes','y','spam',
          'wtb','wanted',
          'duplicate-junk','junk',
          'bundled','below13'
        ) THEN TRUE
        WHEN trim(lower(l.spam)) IN ('f','false','0','no','n') THEN FALSE
        ELSE FALSE
      END
    )
);

-- Must be unique to allow CONCURRENT refresh:
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS tom_features_v1_enriched_clean_mv_uq
  ON ml.tom_features_v1_enriched_clean_mv (listing_id);


(If not done already): enriched view also needs a unique index:

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS tom_features_v1_enriched_mv_uq
  ON ml.tom_features_v1_enriched_mv (listing_id);

Regular refresh procedure (manual)

Run this in psql. Order matters.

-- A/B/C/D1/D2/D3: blocking (base + history layers)
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;   -- D1
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;    -- D2
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;      -- D3

-- E: enriched (unique by listing_id) -> concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;

-- E-clean: spam-free enriched (unique by listing_id) -> concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

-- stats (planner)
ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;


Result: the model will read fresh, spam-clean, unique rows from ml.tom_features_v1_enriched_clean_mv without blocking writers.

Switch the model to the clean read view

In your training/inference script, use:

From: ml.tom_features_v1_enriched_mv

To: ml.tom_features_v1_enriched_clean_mv

No other pipeline changes required.

Quick verification checks

After a refresh:

-- Unique guarantee (should be 0):
SELECT COUNT(*) FROM (
  SELECT listing_id, COUNT(*) 
  FROM ml.tom_features_v1_enriched_clean_mv
  GROUP BY listing_id
  HAVING COUNT(*) > 1
) d;

-- Freshness window:
SELECT MIN(edited_date), MAX(edited_date)
FROM ml.tom_features_v1_enriched_clean_mv;

-- Spot-check: spam rows excluded (should be 0)
WITH spam_raw AS (
  SELECT listing_id
  FROM "iPhone".iphone_listings l
  WHERE CASE
          WHEN l.spam IS NULL THEN FALSE
          WHEN trim(lower(l.spam)) IN (
            't','true','1','yes','y','spam',
            'wtb','wanted','duplicate-junk','junk','bundled','below13'
          ) THEN TRUE
          WHEN trim(lower(l.spam)) IN ('f','false','0','no','n') THEN FALSE
          ELSE FALSE
        END
)
SELECT COUNT(*) AS spam_present_in_clean
FROM spam_raw r
JOIN ml.tom_features_v1_enriched_clean_mv e USING (listing_id);

Common pitfalls & fixes

“cannot refresh concurrently” → The MV must be unique by listing_id. Create the unique index (no WHERE clause) first, then refresh concurrently.

Duplicate key when creating unique index → Fix upstream duplication in the MV definition (the clean MV’s WHERE NOT EXISTS approach already avoids join-duplication issues).

Long locks → Only the base/history layers block writers. The enriched read layers can be refreshed CONCURRENTLY (non-blocking).

Spam semantics change → Update the token list in the CASE and refresh the clean MV.

Automate it (nightly)

Bash/cron example:

#!/usr/bin/env bash
set -euo pipefail

PSQL="psql -v ON_ERROR_STOP=1 ${PG_DSN}"

$PSQL <<'SQL'
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;

REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;
SQL


Crontab (example, nightly @ 03:20):

20 3 * * * PG_DSN='${PG_DSN}' /opt/refresh_feature_store.sh >> /var/log/feature_refresh.log 2>&1

Operational checklist

 Both read MVs have unique indexes on (listing_id)

tom_features_v1_enriched_mv_uq

tom_features_v1_enriched_clean_mv_uq

 Refresh order follows blocking → concurrent pattern

 Model reads from ml.tom_features_v1_enriched_clean_mv

 Post-refresh ANALYZE executed on read layers

 Optional: verify spam exclusion & uniqueness with the sanity SQL

That’s it. You now have a robust, non-blocking refresh path, and the model consumes a clean feature store that excludes noisy spam/WTB/duplicate/bundled/below13 rows.
Absolutely—here’s a single, “write-once, read-forever” doc that captures exactly what we built, the full SQL you can paste into psql, and the why/when/how behind each piece so you (or future you) can edit/change it safely.

0) What we added (high level)

AI‑augmented read view ml.tom_features_v1_enriched_ai_clean_mv

Built on top of your spam‑clean, unique base read view ml.tom_features_v1_enriched_clean_mv.

Joins to "iPhone".iphone_ai_enrich and derives 25 AI features (one‑hots, 0/1 bins, normalized numeric, and a ratio).

Unique on listing_id so you can REFRESH CONCURRENTLY like your other read layers.

Keeps 0/1 encoding (no NULLs) except for the ai_opening_offer_ratio which is NULL when price is not usable.

Feature contract entries for those 25 AI features under the existing set tom_features_v1, plus a hash stamp to lock the spec.

This is the governance layer that tells the trainer/inference exactly which inputs are valid, in which order, and with what types.

Your store/governance pattern (sets, contracts, hash) is described in the Phase‑1 docs 

Phase-1 Feature Contracts & Gov…

 

Phase-1 Feature Contracts & Gov…

.

Refresh procedure for the whole store (base, labels, anchors, read layers) with CONCURRENT refresh for read layers and “spam‑clean” path, exactly as your Ops Guide prescribes 

Feature Store Refresh — Ops Gui…

 

Feature Store Refresh — Ops Gui…

 

Refresh the feature store end-t…

.

1) Full SQL — AI‑augmented, spam‑clean read view

Paste this block as‑is. It drops & recreates the MV, adds a unique index on listing_id, and ANALYZEs for good plans.

-- =====================================================================
-- AI-augmented read view (built on the spam-clean, unique-by-listing_id view)
-- Produces 25 AI features from "iPhone".iphone_ai_enrich and base price.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_ai_clean_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_ai_clean_mv AS
SELECT
  b.*,

  -- sale_mode one-hots (text → 0/1)
  (CASE WHEN lower(ai.sale_mode) = 'obo'  THEN 1 ELSE 0 END)::int AS ai_sale_mode_obo,
  (CASE WHEN lower(ai.sale_mode) = 'firm' THEN 1 ELSE 0 END)::int AS ai_sale_mode_firm,
  (CASE WHEN lower(ai.sale_mode) = 'bids' THEN 1 ELSE 0 END)::int AS ai_sale_mode_bids,
  (CASE WHEN COALESCE(lower(ai.sale_mode),'') NOT IN ('obo','firm','bids')
        THEN 1 ELSE 0 END)::int                                   AS ai_sale_mode_unspecified,

  -- owner_type one-hots
  (CASE WHEN lower(ai.owner_type) = 'private'    THEN 1 ELSE 0 END)::int AS ai_owner_private,
  (CASE WHEN lower(ai.owner_type) = 'work_phone' THEN 1 ELSE 0 END)::int AS ai_owner_work,
  (CASE WHEN ai.owner_type IS NULL OR lower(ai.owner_type) NOT IN ('private','work_phone')
        THEN 1 ELSE 0 END)::int                                           AS ai_owner_unknown,

  -- shipping one-hots (from booleans can_ship / pickup_only)
  (CASE WHEN COALESCE(ai.can_ship,false) AND NOT COALESCE(ai.pickup_only,false)
        THEN 1 ELSE 0 END)::int                                           AS ai_ship_can,
  (CASE WHEN COALESCE(ai.pickup_only,false) THEN 1 ELSE 0 END)::int       AS ai_ship_pickup,
  (CASE WHEN NOT COALESCE(ai.can_ship,false) AND NOT COALESCE(ai.pickup_only,false)
        THEN 1 ELSE 0 END)::int                                           AS ai_ship_unspecified,

  -- repair_provider one-hots
  (CASE WHEN lower(ai.repair_provider) = 'apple'       THEN 1 ELSE 0 END)::int AS ai_rep_apple,
  (CASE WHEN lower(ai.repair_provider) = 'authorized'  THEN 1 ELSE 0 END)::int AS ai_rep_authorized,
  (CASE WHEN lower(ai.repair_provider) = 'independent' THEN 1 ELSE 0 END)::int AS ai_rep_independent,
  (CASE WHEN ai.repair_provider IS NULL OR lower(ai.repair_provider) NOT IN ('apple','authorized','independent')
        THEN 1 ELSE 0 END)::int                                                AS ai_rep_unknown,

  -- booleans → ints
  (COALESCE(ai.can_ship,false))::int             AS ai_can_ship_bin,
  (COALESCE(ai.pickup_only,false))::int          AS ai_pickup_only_bin,
  (COALESCE(ai.vat_invoice,false))::int          AS ai_vat_invoice_bin,
  (COALESCE(ai.first_owner,false))::int          AS ai_first_owner_bin,
  (COALESCE(ai.used_with_case_claim,false))::int AS ai_used_with_case_bin,

  -- numeric & ratios (clipped)
  GREATEST(0, LEAST(1, COALESCE(ai.negotiability_ai, 0)))::float8 AS ai_negotiability_f,
  GREATEST(0, LEAST(1, COALESCE(ai.urgency_ai,       0)))::float8 AS ai_urgency_f,
  GREATEST(0, LEAST(1, COALESCE(ai.lqs_textonly,     0)))::float8 AS ai_lqs_textonly_f,
  GREATEST(0,        COALESCE(ai.opening_offer_nok,  0))::float8  AS ai_opening_offer_nok_f,

  CASE WHEN b.price IS NOT NULL AND b.price > 0
       THEN GREATEST(0, LEAST(5.0, COALESCE(ai.opening_offer_nok,0)::float8 / b.price::float8))
       ELSE NULL::float8 END                                      AS ai_opening_offer_ratio,

  -- storage hint (int)
  ai.storage_gb_fixed_ai::int                                     AS ai_storage_gb_fixed

FROM ml.tom_features_v1_enriched_clean_mv AS b
LEFT JOIN "iPhone".iphone_ai_enrich AS ai
  USING (listing_id);

-- Unique index to allow CONCURRENT refresh on this read layer
CREATE UNIQUE INDEX IF NOT EXISTS tfeat_enr_ai_clean_uq
  ON ml.tom_features_v1_enriched_ai_clean_mv (listing_id);

-- Planner stats for better plans
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;


Why it looks like this

We consume the spam‑clean + unique ml.tom_features_v1_enriched_clean_mv (per your Ops Guide) and only then attach AI features so all downstream model reads remain spam‑free and safe to refresh concurrently 

Feature Store Refresh — Ops Gui…

 

Feature Store Refresh — Ops Gui…

.

All one‑hots/booleans are 0/1 ints (easier for most trainers).

Ratios are clipped (0..5) for sanity; undefined denominators produce NULL (not 0).

"iPhone".iphone_ai_enrich column names match the catalog you listed (sale_mode, owner_type, can_ship, pickup_only, repair_provider, negotiability_ai, urgency_ai, lqs_textonly, opening_offer_nok, storage_gb_fixed_ai, …).

Optional time‑guard (commented design)
If/when you want “as‑of” semantics (e.g., only use AI rows with ai.updated_at <= edited_date), add the predicate to the JOIN:

LEFT JOIN "iPhone".iphone_ai_enrich ai
  ON ai.listing_id = b.listing_id
 AND (ai.updated_at IS NULL OR ai.updated_at <= b.edited_date)


In your snapshot, almost all AI rows were after edited_date (you verified n_ai_at_or_before_edit = 0), so enabling a hard guard would null out nearly all features. Leave it off unless your upstream enrich flow is aligned first.

2) Column dictionary (what each feature means)
Feature	Type	Source	Semantics
ai_sale_mode_obo / firm / bids / unspecified	int4	ai.sale_mode	1 for that sale mode; unspecified=1 when not in {obo, firm, bids}.
ai_owner_private / work / unknown	int4	ai.owner_type	1 for that owner type; unknown=1 when not in {private, work_phone} or NULL.
ai_ship_can / pickup / unspecified	int4	ai.can_ship / pickup_only	Mutually exclusive 0/1 flags derived from the two booleans.
ai_rep_apple / authorized / independent / unknown	int4	ai.repair_provider	1 for that provider; unknown=1 for NULL / anything else.
ai_can_ship_bin	int4	ai.can_ship	Boolean → int (0/1).
ai_pickup_only_bin	int4	ai.pickup_only	Boolean → int (0/1).
ai_vat_invoice_bin	int4	ai.vat_invoice	Boolean → int (0/1).
ai_first_owner_bin	int4	ai.first_owner	Boolean → int (0/1).
ai_used_with_case_bin	int4	ai.used_with_case_claim	Boolean → int (0/1).
ai_negotiability_f	float8	ai.negotiability_ai	Clipped to [0,1].
ai_urgency_f	float8	ai.urgency_ai	Clipped to [0,1].
ai_lqs_textonly_f	float8	ai.lqs_textonly	Clipped to [0,1].
ai_opening_offer_nok_f	float8	ai.opening_offer_nok	Non‑negative.
ai_opening_offer_ratio	float8	ai.opening_offer_nok / b.price	NULL if price ≤ 0; clipped to [0,5].
ai_storage_gb_fixed	int4	ai.storage_gb_fixed_ai	Integerized storage hint.
3) Feature contract — add the 25 AI features (idempotent)

This block adds the AI features to your contract only if they’re not already present, sets leakage_rule='clip_at_edited_date', and keeps window_def=NULL (Phase‑1 convention). The contract/governance approach and hash are from the Phase‑1 guide 

Phase-1 Feature Contracts & Gov…

.

-- ============================================================
-- Append AI features to the feature contract (idempotent upsert)
-- Requires: ml.feature_set (with 'tom_features_v1'), ml.feature_contract
-- ============================================================

WITH fs AS (
  SELECT set_id FROM ml.feature_set WHERE name = 'tom_features_v1'
),
max_ord AS (
  SELECT set_id, COALESCE(MAX(ordinal), 0) AS base_ord
  FROM   ml.feature_contract
  WHERE  set_id = (SELECT set_id FROM fs)
  GROUP  BY set_id
),
to_add (feature_name, dtype, expr_sql, is_nullable, description) AS (
  VALUES
  -- binaries
  ('ai_can_ship_bin','int4','ai_can_ship_bin',TRUE,'AI: can ship'),
  ('ai_pickup_only_bin','int4','ai_pickup_only_bin',TRUE,'AI: pickup only'),
  ('ai_vat_invoice_bin','int4','ai_vat_invoice_bin',TRUE,'AI: VAT invoice'),
  ('ai_first_owner_bin','int4','ai_first_owner_bin',TRUE,'AI: first owner'),
  ('ai_used_with_case_bin','int4','ai_used_with_case_bin',TRUE,'AI: used with case'),

  -- sale_mode one-hots
  ('ai_sale_mode_obo','int4','ai_sale_mode_obo',TRUE,'AI: sale mode OBO'),
  ('ai_sale_mode_firm','int4','ai_sale_mode_firm',TRUE,'AI: sale mode FIRM'),
  ('ai_sale_mode_bids','int4','ai_sale_mode_bids',TRUE,'AI: sale mode BIDS'),
  ('ai_sale_mode_unspecified','int4','ai_sale_mode_unspecified',TRUE,'AI: sale mode UNSPEC'),

  -- owner one-hots
  ('ai_owner_private','int4','ai_owner_private',TRUE,'AI: owner private'),
  ('ai_owner_work','int4','ai_owner_work',TRUE,'AI: owner work phone'),
  ('ai_owner_unknown','int4','ai_owner_unknown',TRUE,'AI: owner unknown'),

  -- shipping one-hots
  ('ai_ship_can','int4','ai_ship_can',TRUE,'AI: shipping can'),
  ('ai_ship_pickup','int4','ai_ship_pickup',TRUE,'AI: shipping pickup'),
  ('ai_ship_unspecified','int4','ai_ship_unspecified',TRUE,'AI: shipping unspecified'),

  -- repair provider one-hots
  ('ai_rep_apple','int4','ai_rep_apple',TRUE,'AI: repair Apple'),
  ('ai_rep_authorized','int4','ai_rep_authorized',TRUE,'AI: repair authorized'),
  ('ai_rep_independent','int4','ai_rep_independent',TRUE,'AI: repair independent'),
  ('ai_rep_unknown','int4','ai_rep_unknown',TRUE,'AI: repair unknown'),

  -- numeric
  ('ai_negotiability_f','float8','ai_negotiability_f',TRUE,'AI: negotiability 0..1'),
  ('ai_urgency_f','float8','ai_urgency_f',TRUE,'AI: urgency 0..1'),
  ('ai_lqs_textonly_f','float8','ai_lqs_textonly_f',TRUE,'AI: text quality 0..1'),
  ('ai_opening_offer_nok_f','float8','ai_opening_offer_nok_f',TRUE,'AI: opening offer NOK'),
  ('ai_opening_offer_ratio','float8','ai_opening_offer_ratio',TRUE,'AI: opening offer / ask'),
  ('ai_storage_gb_fixed','int4','ai_storage_gb_fixed',TRUE,'AI: storage GB fixed')
)
INSERT INTO ml.feature_contract
  (set_id, ordinal, feature_name, dtype, expr_sql, leakage_rule, window_def, is_nullable, description)
SELECT
  fs.set_id,
  max_ord.base_ord + ROW_NUMBER() OVER (ORDER BY to_add.feature_name)    AS ordinal,
  to_add.feature_name,
  to_add.dtype,
  to_add.expr_sql,
  'clip_at_edited_date'                                                  AS leakage_rule,
  NULL                                                                   AS window_def,
  to_add.is_nullable,
  to_add.description
FROM fs
JOIN max_ord USING (set_id)
JOIN to_add
LEFT JOIN ml.feature_contract c
  ON c.set_id = fs.set_id AND c.feature_name = to_add.feature_name
WHERE c.set_id IS NULL;  -- only insert if not already present


In your session you saw n_features = 51 after stamping the hash (the 25 AI features appended after your existing 26), consistent with the governance doc’s contract model 

Phase-1 Feature Contracts & Gov…

.

4) Governance — compute & stamp the canonical hash for tom_features_v1

This writes features_hash + n_features to ml.feature_set so jobs can assert the contract hasn’t drifted (exact SQL per your governance file) 

Feature contract SQL

 

Feature contract SQL

.

-- =========================================================
-- Stamp the 'tom_features_v1' contract hash (Phase-1 pattern)
-- =========================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;

WITH fs AS (
  SELECT set_id FROM ml.feature_set WHERE name = 'tom_features_v1'
),
rows AS (
  SELECT ordinal, feature_name, dtype,
         COALESCE(expr_sql,'')     AS expr_sql,
         COALESCE(leakage_rule,'') AS leakage_rule,
         CASE WHEN is_nullable THEN '1' ELSE '0' END AS is_nullable
  FROM   ml.feature_contract
  WHERE  set_id = (SELECT set_id FROM fs)
  ORDER  BY ordinal
),
canon AS (
  SELECT string_agg(
           ordinal::text || '|' || feature_name || '|' || dtype || '|' ||
           expr_sql || '|' || leakage_rule || '|' || is_nullable,
           '||' ORDER BY ordinal
         ) AS payload,
         COUNT(*) AS n_rows
  FROM rows
)
UPDATE ml.feature_set f
SET    features_hash = encode(digest(convert_to(c.payload,'utf8'),'sha256'),'hex'),
       n_features    = c.n_rows
FROM   canon c, fs
WHERE  f.set_id = fs.set_id;

-- Optional: check what we wrote
SELECT name, n_features, features_hash
FROM   ml.feature_set
WHERE  name = 'tom_features_v1';


And the quick, read‑only one‑shot contract check (handy in CI): 

Phase-1 Feature Contracts & Gov…

WITH fs AS (
  SELECT set_id, features_hash AS expected_hash, n_features AS expected_n
  FROM   ml.feature_set
  WHERE  name = 'tom_features_v1'
),
rows AS (
  SELECT ordinal, feature_name, dtype,
         COALESCE(expr_sql,'')     AS expr_sql,
         COALESCE(leakage_rule,'') AS leakage_rule,
         CASE WHEN is_nullable THEN '1' ELSE '0' END AS is_nullable
  FROM   ml.feature_contract
  WHERE  set_id = (SELECT set_id FROM fs)
  ORDER  BY ordinal
),
canon AS (
  SELECT string_agg(
           ordinal::text || '|' || feature_name || '|' || dtype || '|' ||
           expr_sql || '|' || leakage_rule || '|' || is_nullable,
           '||' ORDER BY ordinal
         ) AS payload,
         COUNT(*) AS n_rows
  FROM rows
)
SELECT
  fs.expected_n,
  fs.expected_hash,
  c.n_rows                                              AS computed_n,
  encode(digest(convert_to(c.payload,'utf8'),'sha256'),'hex') AS computed_hash,
  CASE WHEN fs.expected_n = c.n_rows
        AND fs.expected_hash = encode(digest(convert_to(c.payload,'utf8'),'sha256'),'hex')
       THEN 'OK' ELSE 'MISMATCH' END AS contract_check
FROM fs, canon c;

5) Refresh procedure (safe order)

Use your standard Ops Guide sequence. Only base/history layers block; both read layers refresh concurrently (and are unique by listing_id) 

Feature Store Refresh — Ops Gui…

 

Feature Store Refresh — Ops Gui…

 

Refresh the feature store end-t…

:

-- ===========================
-- FEATURE STORE REFRESH (SAFE)
-- ===========================

-- A/B/C: base + labels + train
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

-- D1/D2/D3: anchors (normal refresh)
REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;

-- E: enriched (unique by listing_id) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;

-- E-clean: spam-clean enriched (unique by listing_id) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

-- E-clean+AI: the view created above (unique by listing_id)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

-- Stats for the planner
ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;

6) Sanity checks you can run anytime

Uniqueness (should be 0):

SELECT COUNT(*) FROM (
  SELECT listing_id, COUNT(*) 
  FROM ml.tom_features_v1_enriched_ai_clean_mv
  GROUP BY listing_id
  HAVING COUNT(*) > 1
) d;


Non‑zero AI signals present:

SELECT COUNT(*) AS n_rows,
       COUNT(*) FILTER (WHERE ai_sale_mode_firm=1 OR ai_sale_mode_obo=1) AS n_sale_mode_flag,
       COUNT(*) FILTER (WHERE ai_can_ship_bin=1 OR ai_pickup_only_bin=1) AS n_shipping_flag,
       COUNT(*) FILTER (WHERE ai_negotiability_f > 0)                    AS n_negotiability_pos
FROM ml.tom_features_v1_enriched_ai_clean_mv;


Spot‑check parity vs AI table (sample ids):

SELECT e.listing_id,
       ai.sale_mode, ai.owner_type, ai.can_ship, ai.pickup_only,
       ai.repair_provider, ai.negotiability_ai, ai.lqs_textonly, ai.opening_offer_nok,
       ai.storage_gb_fixed_ai, ai.updated_at
FROM ml.tom_features_v1_enriched_ai_clean_mv e
JOIN "iPhone".iphone_ai_enrich ai USING (listing_id)
WHERE e.listing_id IN (432361614, 432364948, 432099498);


(If experimenting with a time guard) distribution check:
You previously observed n_ai_at_or_before_edit = 0 and n_ai_after_edit ≈ all:

SELECT
  COUNT(*)                                           AS n_rows,
  COUNT(*) FILTER (WHERE ai.updated_at <= b.edited_date) AS n_ai_at_or_before_edit,
  COUNT(*) FILTER (WHERE ai.updated_at  > b.edited_date) AS n_ai_after_edit,
  COUNT(*) FILTER (WHERE ai.updated_at IS NULL)          AS n_ai_null
FROM ml.tom_features_v1_enriched_clean_mv b
LEFT JOIN "iPhone".iphone_ai_enrich ai USING (listing_id);

7) How to change things later (safe edits)

Add a new AI feature

Add its expression to the MV SELECT list with a stable name and type.

REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

Add a new row to to_add(...) in the contract block (above) and rerun the contract insert.

Re‑stamp the hash with the governance block, and (optionally) run the one‑shot assert.
Guidance & hash flow: 

Phase-1 Feature Contracts & Gov…

 

Phase-1 Feature Contracts & Gov…

.

Change mapping semantics (e.g., redefine “unspecified”, adjust clipping, etc.)
Update the relevant CASE/GREATEST/LEAST in the MV, refresh the MV, and consider bumping your feature set (e.g., tom_features_v2) if it impacts training compatibility; then stamp/validate per the governance guide 

Phase-1 Feature Contracts & Gov…

.

Spam semantics live in the clean view (not shown above, already created per Ops Guide). If you change spam tokens, refresh the clean view; the AI MV automatically stays aligned because it reads from the clean view 

1) Create the spam-clean read v…

.

8) Common pitfalls (and fixes)

“cannot refresh concurrently” → Ensure the MV has a plain unique index on (listing_id) (no WHERE clause). The create‑index line in section 1 covers this. See Ops Guide notes 

Feature Store Refresh — Ops Gui…

.

Dedup errors on the index → Fix upstream duplication (the clean view and enriched view are already designed to be unique per listing_id per the Ops Guide) 

Feature Store Refresh — Ops Gui…

.

Contract drift / failed assert → Someone changed ml.feature_contract. Re‑stamp only if you intend the change (and after retrain/validate) per governance doc 

Phase-1 Feature Contracts & Gov…

.

9) Why this wiring matches Phase‑1 architecture

Data reads are spam‑clean, unique, and refreshable concurrently at the read layers, with leak‑safe anchors and governed contracts on top, exactly as laid out in the Phase‑1 docs and Ops Guide 

SQL -- PHASE 1 — DATA CONTRACTS…

 

Feature Store Refresh — Ops Gui…

 

Refresh the feature store end-t…

.

TL;DR: copy/paste checklist

Create ml.tom_features_v1_enriched_ai_clean_mv (section 1).

Create unique index on (listing_id) + ANALYZE (section 1).

(Idempotent) insert AI features into ml.feature_contract (section 3).

Stamp features_hash + n_features (section 4).

Refresh sequence A/B/C → D1/D2/D3 → E/E‑clean/E‑clean+AI (section 5).

Run checks (section 6).

If you want, I can bundle this into a single .sql file with \echo headers and guard comments so your future self has one artifact to run end‑to‑end.
0) What this file defines (high level)

AI + Speed read view: ml.tom_features_v1_enriched_ai_clean_mv

Base: ml.tom_features_v1_enriched_speed_mv

This base view is already spam-clean, unique per listing_id, and includes:

All your original enriched features (ptv_final, anchors, etc.)

Speed anchors: speed_fast7_anchor, speed_fast24_anchor, speed_slow21_anchor, speed_median_hours_ptv, speed_n_eff_ptv from ml.tom_speed_anchor_v1_mv.

AI layer: joins "iPhone".iphone_ai_enrich and derives ~25 AI features:

Sale mode / owner type / shipping / repair provider one-hots

Binary flags for can_ship, pickup_only, vat_invoice, first_owner, used_with_case

Numeric scores (negotiability, urgency, text quality, opening offer)

Opening offer ratio vs current price

Storage hint (ai_storage_gb_fixed)

Key properties:

Unique on listing_id ⇒ safe to REFRESH MATERIALIZED VIEW CONCURRENTLY.

All one-hots/bools are 0/1 ints (no NULL), except ai_opening_offer_ratio which can be NULL when price is missing/invalid.

This read view is what your trainer should use (directly or via the existing _ai_clean_mv path).

Speed anchors themselves are defined in ml.tom_speed_anchor_v1_mv and wired into ml.tom_features_v1_enriched_speed_mv (see your Phase-1 / speed-anchor SQL). This file only adds the AI layer on top.

1) Full SQL — AI-augmented, spam-clean + speed read view

Paste this whole block into psql (or your schema migration) to recreate the AI+speed read view.

-- =====================================================================
-- AI+Speed augmented read view (spam-clean, unique-by-listing_id)
-- Base: ml.tom_features_v1_enriched_speed_mv
--  • includes all enriched + speed_* anchors
-- AI: joins "iPhone".iphone_ai_enrich and adds ~25 AI features.
-- =====================================================================

DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_ai_clean_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_ai_clean_mv AS
SELECT
  b.*,

  -- sale_mode one-hots (text → 0/1)
  (CASE WHEN lower(ai.sale_mode) = 'obo'  THEN 1 ELSE 0 END)::int AS ai_sale_mode_obo,
  (CASE WHEN lower(ai.sale_mode) = 'firm' THEN 1 ELSE 0 END)::int AS ai_sale_mode_firm,
  (CASE WHEN lower(ai.sale_mode) = 'bids' THEN 1 ELSE 0 END)::int AS ai_sale_mode_bids,
  (CASE WHEN COALESCE(lower(ai.sale_mode),'') NOT IN ('obo','firm','bids')
        THEN 1 ELSE 0 END)::int                                   AS ai_sale_mode_unspecified,

  -- owner_type one-hots
  (CASE WHEN lower(ai.owner_type) = 'private'    THEN 1 ELSE 0 END)::int AS ai_owner_private,
  (CASE WHEN lower(ai.owner_type) = 'work_phone' THEN 1 ELSE 0 END)::int AS ai_owner_work,
  (CASE WHEN ai.owner_type IS NULL OR lower(ai.owner_type) NOT IN ('private','work_phone')
        THEN 1 ELSE 0 END)::int                                           AS ai_owner_unknown,

  -- shipping one-hots (from booleans can_ship / pickup_only)
  (CASE WHEN COALESCE(ai.can_ship,false) AND NOT COALESCE(ai.pickup_only,false)
        THEN 1 ELSE 0 END)::int                                           AS ai_ship_can,
  (CASE WHEN COALESCE(ai.pickup_only,false) THEN 1 ELSE 0 END)::int       AS ai_ship_pickup,
  (CASE WHEN NOT COALESCE(ai.can_ship,false) AND NOT COALESCE(ai.pickup_only,false)
        THEN 1 ELSE 0 END)::int                                           AS ai_ship_unspecified,

  -- repair_provider one-hots
  (CASE WHEN lower(ai.repair_provider) = 'apple'       THEN 1 ELSE 0 END)::int AS ai_rep_apple,
  (CASE WHEN lower(ai.repair_provider) = 'authorized'  THEN 1 ELSE 0 END)::int AS ai_rep_authorized,
  (CASE WHEN lower(ai.repair_provider) = 'independent' THEN 1 ELSE 0 END)::int AS ai_rep_independent,
  (CASE WHEN ai.repair_provider IS NULL OR lower(ai.repair_provider) NOT IN ('apple','authorized','independent')
        THEN 1 ELSE 0 END)::int                                                AS ai_rep_unknown,

  -- booleans → ints
  (COALESCE(ai.can_ship,false))::int             AS ai_can_ship_bin,
  (COALESCE(ai.pickup_only,false))::int          AS ai_pickup_only_bin,
  (COALESCE(ai.vat_invoice,false))::int          AS ai_vat_invoice_bin,
  (COALESCE(ai.first_owner,false))::int          AS ai_first_owner_bin,
  (COALESCE(ai.used_with_case_claim,false))::int AS ai_used_with_case_bin,

  -- numeric & ratios (clipped)
  GREATEST(0, LEAST(1, COALESCE(ai.negotiability_ai, 0)))::float8 AS ai_negotiability_f,
  GREATEST(0, LEAST(1, COALESCE(ai.urgency_ai,       0)))::float8 AS ai_urgency_f,
  GREATEST(0, LEAST(1, COALESCE(ai.lqs_textonly,     0)))::float8 AS ai_lqs_textonly_f,
  GREATEST(0,        COALESCE(ai.opening_offer_nok,  0))::float8  AS ai_opening_offer_nok_f,

