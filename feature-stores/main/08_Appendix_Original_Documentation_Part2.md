# Appendix — Original Documentation (Part 2 of 2)

Continuation of the original merged documentation. The only edits applied are **credential redactions** (e.g., passwords).

  CASE WHEN b.price IS NOT NULL AND b.price > 0
       THEN GREATEST(0, LEAST(5.0, COALESCE(ai.opening_offer_nok,0)::float8 / b.price::float8))
       ELSE NULL::float8 END                                      AS ai_opening_offer_ratio,

  -- storage hint (int)
  ai.storage_gb_fixed_ai::int                                     AS ai_storage_gb_fixed

FROM ml.tom_features_v1_enriched_speed_mv AS b
LEFT JOIN "iPhone".iphone_ai_enrich AS ai
  USING (listing_id);

-- Unique index to allow CONCURRENT refresh on this read layer
CREATE UNIQUE INDEX IF NOT EXISTS tfeat_enr_ai_clean_uq
  ON ml.tom_features_v1_enriched_ai_clean_mv (listing_id);

-- Planner stats for better plans
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;


Key changes vs old version:

FROM ml.tom_features_v1_enriched_clean_mv AS b ➜
now FROM ml.tom_features_v1_enriched_speed_mv AS b.

That means all the speed_* columns are available as plain numeric features to your trainer, alongside the AI stuff.

2) Column dictionary (what the AI features mean)

(unchanged conceptually — speed_* features are already defined in the base view; here we focus on AI columns this MV adds.)

Binary 0/1 ints:

ai_sale_mode_obo / ai_sale_mode_firm / ai_sale_mode_bids / ai_sale_mode_unspecified
– From ai.sale_mode ('obo', 'firm', 'bids', other/NULL → unspecified).

ai_owner_private / ai_owner_work / ai_owner_unknown
– From ai.owner_type ('private', 'work_phone', other/NULL → unknown).

ai_ship_can / ai_ship_pickup / ai_ship_unspecified
– From ai.can_ship and ai.pickup_only booleans (mutually exclusive).

ai_rep_apple / ai_rep_authorized / ai_rep_independent / ai_rep_unknown
– From ai.repair_provider (apple, authorized, independent, other/NULL → unknown).

ai_can_ship_bin (ai.can_ship),

ai_pickup_only_bin (ai.pickup_only),

ai_vat_invoice_bin (ai.vat_invoice),

ai_first_owner_bin (ai.first_owner),

ai_used_with_case_bin (ai.used_with_case_claim).

Numeric features:

ai_negotiability_f – ai.negotiability_ai clipped to [0,1].

ai_urgency_f – ai.urgency_ai clipped to [0,1].

ai_lqs_textonly_f – ai.lqs_textonly clipped to [0,1].

ai_opening_offer_nok_f – ai.opening_offer_nok, floored at 0.

ai_opening_offer_ratio – ai.opening_offer_nok / price, NULL if price ≤ 0, clipped to [0,5].

ai_storage_gb_fixed – ai.storage_gb_fixed_ai cast to int.

Speed anchors (already in base view):

From ml.tom_features_v1_enriched_speed_mv (via ml.tom_speed_anchor_v1_mv):

speed_fast7_anchor – recency-weighted P(TOM ≤ 7d) per (generation, sbucket, ptv_bucket).

speed_fast24_anchor – recency-weighted P(TOM ≤ 24h) per bucket.

speed_slow21_anchor – recency-weighted P(TOM > 21d) per bucket.

speed_median_hours_ptv– median duration_hours for that bucket.

speed_n_eff_ptv – effective sample size for the speed prior.

They come “for free” because we’re selecting b.* from the speed-enriched base.

3) Feature contract — the AI features (optional but recommended)

If you’re using ml.feature_set / ml.feature_contract to lock the feature spec for tom_features_v1, you can append the AI features as before.

Important: The speed features (speed_fast7_anchor, etc.) are part of the base feature store and should be defined/hashed in your Phase-1 contract SQL. The block below is only for the AI columns.

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


Then stamp the hash as usual (unchanged from your existing governance):

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
SET    features_hash = encode(digest(convert_to(c.payload,'utf8'),'hex'),'hex'),
       n_features    = c.n_rows
FROM   canon c, fs
WHERE  f.set_id = fs.set_id;

-- Optional: check what we wrote
SELECT name, n_features, features_hash
FROM   ml.feature_set
WHERE  name = 'tom_features_v1';

4) Refresh order (including speed + AI)

In your “refresh everything” script, make sure you:

Refresh base & labels (A/B/C)

Refresh anchor/history (D1/D2/D3 + D4 speed)

Refresh enriched, enriched-speed, and AI views in the right order

Example:

-- ===========================
-- FEATURE STORE REFRESH (SAFE)
-- ===========================

-- A/B/C: base + labels + train
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

-- D1/D2/D3: core anchors (normal refresh)
REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;

-- D4: speed anchor (depends on D1 + enriched PTV)
REFRESH MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv;

-- E: enriched (unique by listing_id) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;

-- E+speed: enriched + speed_* (unique by listing_id) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_mv;

-- E-clean: spam-clean enriched (if you have a clean layer) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

-- E-clean+AI: this view (unique by listing_id) → concurrent
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

-- Stats for the planner
ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_speed_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;


(Adjust the exact order depending on how you’ve chained the clean vs speed view; key point is: speed MV and speed-enriched view must be fresh before the AI view refreshes.)

5) Sanity checks

Basic uniqueness:

SELECT COUNT(*) FROM (
  SELECT listing_id, COUNT(*) 
  FROM ml.tom_features_v1_enriched_ai_clean_mv
  GROUP BY listing_id
  HAVING COUNT(*) > 1
) d;


Check that AI flags aren’t all zero:

SELECT
  COUNT(*) AS n_rows,
  COUNT(*) FILTER (WHERE ai_sale_mode_firm=1 OR ai_sale_mode_obo=1) AS n_sale_mode_flag,
  COUNT(*) FILTER (WHERE ai_can_ship_bin=1 OR ai_pickup_only_bin=1) AS n_shipping_flag,
  COUNT(*) FILTER (WHERE ai_negotiability_f > 0)                    AS n_negotiability_pos
FROM ml.tom_features_v1_enriched_ai_clean_mv;


Spot-check AI vs raw table:

SELECT e.listing_id,
       ai.sale_mode, ai.owner_type, ai.can_ship, ai.pickup_only,
       ai.repair_provider, ai.negotiability_ai, ai.lqs_textonly, ai.opening_offer_nok,
       ai.storage_gb_fixed_ai, ai.updated_at
FROM ml.tom_features_v1_enriched_ai_clean_mv e
JOIN "iPhone".iphone_ai_enrich ai USING (listing_id)
LIMIT 50;


Speed anchors look sane (from the base view):

SELECT
  generation,
  CASE
    WHEN storage_gb >= 900 THEN 1024
    WHEN storage_gb >= 500 THEN 512
    WHEN storage_gb >= 250 THEN 256
    WHEN storage_gb >= 120 THEN 128
    ELSE storage_gb
  END AS sbucket,
  ptv_final,
  speed_fast7_anchor,
  speed_median_hours_ptv,
  speed_n_eff_ptv
FROM ml.tom_features_v1_enriched_ai_clean_mv
WHERE speed_fast7_anchor IS NOT NULL
ORDER BY generation, sbucket, ptv_final
LIMIT 50;


You should see speed_fast7_anchor decrease and speed_median_hours_ptv increase as ptv_final grows in each (gen,sbucket).

That’s the updated doc: you can save this as your new
“Full SQL — AI-augmented, spam-clean read view” file and your trainer will automatically see both:

all your existing enriched + speed features (from the base view), and

the AI-derived features from iphone_ai_enrich.
Phase-1 Architecture — Leak-Safe Feature Store (Postgres)
Design goals (what this does)

Leak-safe by construction: all features reflect only what existed at t₀ = edited_date; outcomes (sold_date, sold_price, last_seen, etc.) are not features.

Single source of truth: indexed materialized views for features, labels, training base, and embargoed anchors. Spam rows are filtered at the source.

PTV/Δ features computed relative to embargoed sold-price medians (value anchor) with ask-median fallback (supply anchor).

Raw text stored (title, description) for future NLP (SVD/SBERT), but not part of the current feature contract.

Fast to refresh, easy to reason about, easy to extend.

Visual wireframe (dataflow)
              ┌───────────────────────────────┐
              │  "iPhone".iphone_listings     │   (raw source; spam filtered everywhere)
              └───────────────┬───────────────┘
                              │
                              │  t₀ = edited_date, spam IS NULL
                              ▼
      ┌────────────────────────────────────────────────┐
      │  A) ml.tom_features_v1_mv                      │  ← base feature snapshot @ t₀
      │    • one row per listing_id (DISTINCT ON)         │    (raw text kept: title/description)
      └───────────────┬────────────────────────────────┘
                      │ (join on listing_id)
                      ▼
      ┌────────────────────────────────────────────────┐
      │  B) ml.tom_labels_v1_mv                        │  ← leak-safe labels
      │    • sold_event, duration_hours                │    (uses sold_date / last_seen only)
      └───────────────┬────────────────────────────────┘
                      │ (join on listing_id)
                      ▼
      ┌────────────────────────────────────────────────┐
      │  C) ml.tom_train_base_v1_mv                    │  ← features + labels (trainer SELECT)
      └────────────────────────────────────────────────┘

   (embargoed anchors, built from history)
      ┌────────────────────────────────────────────────┐
      │  D1) ml.sold_durations_v1_mv                   │  ← sold durations by day & cohort
      └────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────┐
      │  D2) ml.ask_daily_median_v1                    │  ← ask medians as-of day & cohort
      └────────────────────────────────────────────────┘
      ┌────────────────────────────────────────────────┐
      │  D3) ml.sold_prices_v1_mv                      │  ← sold prices (value anchor source)
      └────────────────────────────────────────────────┘
                              │
                              │  LATERAL (≤ t₀ date) lookups with embargo
                              ▼
      ┌────────────────────────────────────────────────┐
      │  E) ml.tom_features_v1_enriched_mv             │  ← features + anchors @ t₀
      │    • anchor_median_hours_14d, anchor_14d_n     │
      │    • ask_median_price, ask_cnt                 │
      │    • sold_median_price_30d, sold_cnt_30d       │
      │    • ptv_sold_30d, ptv_ask_day, deltas, ptv_final
      └────────────────────────────────────────────────┘

What each MV/view contains (and why)
A) ml.tom_features_v1_mv — base feature snapshot @ t₀

Input: "iPhone".iphone_listings filtered with spam IS NULL.

Logic: DISTINCT ON (listing_id) by earliest edited_date (t₀).

Columns kept:
listing_id (key), edited_date, generation, model, storage_gb, price,
condition_score, battery_pct_fixed_ai,
damage_severity_ai, damage_binary_ai,
seller_rating, review_count, member_since_year,
location_city, postal_code,
title, description, title_length, description_length.
(No outcomes. No ops/audit. Raw text kept for future NLP but not modeled now.)

Indexes: (listing_id) unique, (edited_date, generation, storage_gb) for PIT joins.

B) ml.tom_labels_v1_mv — leak-safe labels

Input: A’s listing_id/edited_date + "iPhone".iphone_listings outcomes.

Logic:
sold_event = sold_date IS NOT NULL
duration_hours = max(0, (coalesce(sold_date, last_seen) - edited_date))

Index: (listing_id) unique; (sold_event) for filtering.

C) ml.tom_train_base_v1_mv — features + labels

Join: A + B on listing_id. One row per listing with target attached.

Trainer SELECT reads from here (or from E if you include anchors).

D1) ml.sold_durations_v1_mv — history for market speed

Rows: sold listings only (sold_date NOT NULL) → sold_day + duration_hours.

Index: (generation, storage_gb, sold_day) for lateral lookups.

D2) ml.ask_daily_median_v1 — as-of day “supply” anchor

Rows: ask price medians per (generation, storage_gb, ask_day = edited_date::date).

Index: (generation, storage_gb, ask_day).

D3) ml.sold_prices_v1_mv — history for “value” anchor

Rows: COALESCE(real_sold_price, sold_price) per (generation, storage_gb, sold_day).

Index: (generation, storage_gb, sold_day).

E) ml.tom_features_v1_enriched_mv — features + embargoed anchors

Joins: LATERAL to D1, D2, D3 using only days ≤ t₀ day (embargo; no lookahead).

Adds:
anchor_median_hours_14d (market speed), anchor_14d_n (support),
ask_median_price, ask_cnt,
sold_median_price_30d, sold_cnt_30d,
relative price features:
ptv_sold_30d, ptv_ask_day,
delta_vs_sold_median_30d, delta_vs_ask_median_day,
ptv_final (reliability-gated PTV: sold anchor if sold_cnt_30d ≥ 10, else ask anchor if ask_cnt ≥ 5).

Contract & governance (what the model actually uses)

Contract tables: ml.feature_set, ml.feature_contract, ml.feature_set_current
(You already created them earlier; just populate with the keep columns.)

Recommended “keep” columns for model (no outcomes; no raw text):
generation, model (if you plan to category-encode; else drop),
storage_gb, price, condition_score, battery_pct_fixed_ai,
damage_severity_ai, damage_binary_ai,
seller_rating, review_count, member_since_year (or add SQL-derived age_years),
location_city / postal_code (if you map to region),
title_length, description_length (optional),
anchors: anchor_median_hours_14d, anchor_14d_n, ask_median_price, ask_cnt,
sold_median_price_30d, sold_cnt_30d,
ptv_sold_30d, ptv_ask_day, delta_vs_sold_median_30d, delta_vs_ask_median_day, ptv_final.

Do not add title/description to the contract until you create real SVD/SBERT features.
After inserting the rows in ml.feature_contract, run your tiny Python helper to compute and update ml.feature_set.features_hash, and call assert_contract('tom_features_v1') in both train & infer jobs to fail fast on any drift.

Refresh order & cadence

Initial build / full refresh (in this order):

A) tom_features_v1_mv

B) tom_labels_v1_mv

C) tom_train_base_v1_mv

D1) sold_durations_v1_mv

D2) ask_daily_median_v1

D3) sold_prices_v1_mv

E) tom_features_v1_enriched_mv

Ongoing cadence (examples):

A/B/C/E: hourly (or on new listing/price edits)

D1/D3: hourly or daily depending on volume

D2: hourly or daily (cheap)

Use REFRESH MATERIALIZED VIEW CONCURRENTLY … where you have a unique index (A, B, C, E) to avoid reader blocking.

Non-leakage guarantees (what to keep sacred)

No outcomes in A (features): no sold_date, sold_price, last_seen, status, etc.

Anchors in E use only history strictly before edited_date::date.

Spam rows are excluded everywhere at source.

listing_id is key only (never a model feature).

Troubleshooting & gotchas (you hit most of these)

Typos: table is "iPhone".iphone_listings (not iphone_lis etc.).

EXTRACT not EXPLICIT for epoch; no inline comments on SQL lines (psql will choke).

Unique indexes: build labels (B) from A to keep listing_id unique; don’t query raw listings for base/labels.

PERCENTILE_CONT: available; you’re on PG 15 (ok).

Performance: all anchor MVs have (generation, storage_gb, day) indexes; lateral lookups in E will use them.

How to replicate end-to-end (short checklist)

Run A → B → C → D1 → D2 → D3 → E in that order (they’re idempotent; drop/create).

Insert feature_contract rows for the keep set; run hash helper; add assert_contract to trainer/infer.

Trainer SELECT: from ml.tom_train_base_v1_mv (or join ml.tom_features_v1_enriched_mv on listing_id to include anchors); train survival/AFT/LightGBM with calibration & produce expected_hours, P(≤24h/≤48h) calibrators.

Inference SELECT: from ml.tom_features_v1_enriched_mv at t₀; compute expected_hours, P(≤24h/≤48h); UPSERT to ml.predictions_v1 (Phase-2).

Why these anchors matter

PTV relative to sold medians (embargoed) captures value vs market willingness-to-pay → strongest TOM driver.

Ask-median is the supply anchor (as-of), used as fallback and complementary signal.

Market speed anchor (anchor_median_hours_14d) encodes how fast similar items are selling right now.

What to edit later (safe extensions)

Add 30/60-day anchors (copy the lateral block; change interval).

Add generation-only fallback if sold_cnt_30d is small; wrap with COALESCE.

Add enrichment fields from iphone_ai_enrich with time guard (e.updated_at <= edited_date) to avoid leakage.

Add NLP (SVD/SBERT) on title/description to create real numeric features; then add those derived columns to the contract.

This is the complete Phase-1 SQL architecture and ops doc. It’s the exact feature store you’ve got running now — leak-safe, indexed, versionable, and with embargoed anchors ready to drive PTV and market-speed features in your survival model.


































Got it. Here’s the clear split: what is SQL vs what is Python (plus the few infra bits). I’m mapping your entire roadmap line-by-line to [SQL], [PY], [INFRA], or [BOTH] so there’s zero ambiguity.

Ownership legend

[SQL] Postgres DDL/DML, views/MVs, roles, indices, server settings

[PY] Python jobs (training, inference, explainability, eval), Airflow tasks

[INFRA] Postgres config/restart, read replica, dashboards/alerts, storage

[BOTH] Python drives it, but executes/updates SQL artifacts

Phase-by-phase mapping
Phase 0 — Scope & SLOs

One-pager/SLOs/artifact names → [INFRA/process] (docs/config)

Phase 1 — Data contracts & leakage safety

Feature contract (table for spec + hash) → [SQL] (store contract) + [PY] (hash/check in jobs) → [BOTH]

Leakage-safe feature MV (clipped at listed_at) → [SQL]

Labels/censoring policy (duration/event) defined in MV or view → [SQL]

Cohort definitions (dims tables/views) → [SQL]

Phase 2 — Serving schema (Postgres)

ml.predictions_vX table (+ indexes) → [SQL]

ml.feature_importance_vX table → [SQL]

ml.mae_summary_vX table → [SQL]

ml.model_registry table → [SQL]

ops.job_status table → [SQL]

ml.predictions_current view (flip on publish) → [SQL] (flipped by [PY] during publish → [BOTH])

Phase 3 — Observability (DB & jobs)

Enable pg_stat_statements, auto_explain (ALTER SYSTEM, restart) → [INFRA] + [SQL]

Standardize application_name per stage → [PY]

Batch heartbeats → write to ops.job_status → [PY] (UPSERT via SQL → [BOTH])

ml.run_metrics log table (optional) → [SQL], populated by [PY]

Dashboards (version, progress, drift) → [INFRA] (Grafana/Metabase etc.)

Phase 4 — ML training (AFT/Cox) & calibration

Training container reading features via SELECT → [PY]

Rolling time CV, metrics, isotonic calibration → [PY]

Write model_registry row → [PY] (INSERT) → [BOTH]

Optional Parquet snapshot to object storage → [INFRA] + [PY]

Phase 5 — Inference (bulk + incremental)

Bulk inference (≤15 min cadence), upsert to predictions_vX → [PY] (UPSERT SQL) → [BOTH]

Incremental inference on change events → [PY] (consumer) + event source ([INFRA/SQL] trigger/CDC) → [BOTH]

Quality gates check & publish (flip predictions_current) → [PY] (gating) + [SQL] (VIEW flip) → [BOTH]

Publish audit row → [PY] (INSERT to registry/log) → [BOTH]

Phase 6 — Explainability (fast) & audit (exact)

Hazard approx cube (Δprob ≈ h·(1–h)·Δη·100) aggregation → [PY] (fast vectorized build) → write hazard_cube_vX/feature_importance_vX → [BOTH]

Nightly exact audit on sampled cohort → [PY] (compute) → store comparison deltas in small table → [BOTH]

Phase 7 — Evaluation & gates

Compute MAE/median/p90, C-index, IBS, ECE per cohort → [PY]

Publish mae_summary_vX + run_metrics → [PY] (INSERT/UPSERT) → [BOTH]

Enforce gates & decide promotion → [PY]

Phase 8 — Orchestration

Airflow DAGs (features_refresh → train → infer → explain → eval → publish → drift_watch) → [PY]

Retries/backoff/SLA + alerts → [INFRA] + [PY]

Phase 9 — Reliability & guardrails

Roles/privs (RO training, RW inference, admin publish) → [SQL]

Circuit-breaker for interactive monsters (RLS/statement_timeout/denylist schema) → [SQL/INFRA]

Read replica setup → [INFRA]

Idempotent upserts → [PY] (logic) + [SQL] (constraints) → [BOTH]

Backups/retention → [INFRA]

Phase 10 — Dashboards & runbook

Ops/Analytics dashboards → [INFRA] (queries hit PG small tables → [SQL])

On-call runbook → [INFRA/process]

Phase 11 — Cutover

Read-only preview, SLO verify, flip consumers → [INFRA] + [PY] (toggle view)

Post-cutover watch & rollback lever → [INFRA] + [SQL]

Phase 12 — Continuous improvement

Drift triggers → [PY]

Per-segment calibrators → [PY]

Performance tuning (Arrow/Polars extract; shard inference) → [PY/INFRA]

Deliverables by stack (checklist)
✅ SQL (Postgres)

 Tables: ml.predictions_vX, ml.feature_importance_vX, ml.mae_summary_vX, ml.model_registry, ops.job_status, (ml.run_metrics optional)

 Views/MVs: leakage-safe feature MV, ml.predictions_current

 Indices/PKs on all serving artifacts

 Roles/GRANTs (RO train, RW inference, admin publish)

 Server settings (ALTER SYSTEM for pg_stat_statements, auto_explain; restart)

 Cohort dims/views

 (Optional) Read replica logical slots or publication if using CDC

✅ Python (ML & Orchestration)

 Train job: data loader (SELECT), AFT/Cox, rolling CV, isotonic calibration, metrics, write model_registry

 Infer bulk job (≤15 min): batch score live rows, UPSERT into predictions_vX, update job_status

 Infer incremental job: consume change events, score touched rows, UPSERT, update job_status

 Explainability (approx) job: build hazard approx cube, publish feature_importance_vX

 Eval job: compute MAE/median/p90, C-index, IBS, ECE, write mae_summary_vX + run_metrics, enforce gates

 Publish step: flip predictions_current if gates pass; else keep last version

 Drift watch: input/output/calibration drift metrics + alerts

 Airflow DAGs: wire all tasks; set application_name; write job_status heartbeats

 (Optional) Snapshot writer to Parquet/S3

✅ Infra / Monitoring

 Read replica (optional), connection pooling

 Dashboards (serving version, freshness, progress/ETA, drift, calibration)

 Alerts: SLO miss, stalled job, gate failure, drift

 Backups/restore runbook; retention for model artifacts & snapshots

Which tables are written by who

ml.predictions_vX: [PY] writes (UPSERT), [SQL] serves

ml.predictions_current: [PY] flips view via [SQL]

ml.feature_importance_vX: [PY] computes & inserts, [SQL] serves

ml.mae_summary_vX: [PY] computes & inserts, [SQL] serves

ml.model_registry: [PY] inserts (train/publish metadata), [SQL] serves

ops.job_status: [PY] updates; [SQL] schema/indexes

Repo layout (so teams don’t step on each other)

/sql/migrations/*.sql — all DDL (tables, views/MVs, roles, indices)

/ml/train/ — training pipeline (AFT/Cox + calibration)

/ml/infer/ — bulk + incremental inference

/ml/explain/ — approx cube builder (+ nightly exact audit)

/ml/eval/ — MAE/C-index/IBS/ECE + gates

/orchestration/airflow/dags/*.py — DAGs

/observability/ — dashboards, alert rules, runbook

/artifacts/ — model files, (optional) Parquet snapshots

Bottom line

If it’s schema, views, indices, roles, or serving reads → SQL.

If it’s training, inference, explainability, evaluation, gating, or orchestration → Python (writing results back to SQL).

Infra (logs, dashboards, replica, alerts) sits beside both.

If you want, I’ll turn this mapping into a one-page RACI (who owns what) you can hand to the team.
-- ===========================================================
-- D4) SPEED ANCHOR BY (GENERATION, STORAGE BUCKET, PTV BUCKET)
--      -> fast7/fast24/slow21 rates + median duration + n_eff
--      Using last 90 days of sold history, with recency decay.
-- ===========================================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_speed_anchor_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv AS
WITH sold AS (
  SELECT
    f.listing_id,
    f.generation,
    -- reuse the same storage bucketing as strict anchors
    CASE
      WHEN f.storage_gb >= 900 THEN 1024
      WHEN f.storage_gb >= 500 THEN 512
      WHEN f.storage_gb >= 250 THEN 256
      WHEN f.storage_gb >= 120 THEN 128
      ELSE f.storage_gb
    END AS sbucket,
    f.ptv_final,
    d.duration_hours,
    d.sold_day
  FROM ml.sold_durations_v1_mv d
  JOIN ml.tom_features_v1_enriched_mv f
    USING (listing_id)
  WHERE f.ptv_final IS NOT NULL
),
ptv_binned AS (
  SELECT
    generation,
    sbucket,
    CASE
      WHEN ptv_final < 0.50 THEN NULL         -- extreme underpricing, ignore
      WHEN ptv_final < 0.80 THEN 1           -- [0.50, 0.80)
      WHEN ptv_final < 0.90 THEN 2           -- [0.80, 0.90)
      WHEN ptv_final < 1.00 THEN 3           -- [0.90, 1.00)
      WHEN ptv_final < 1.10 THEN 4           -- [1.00, 1.10)
      WHEN ptv_final < 1.20 THEN 5           -- [1.10, 1.20)
      WHEN ptv_final < 1.40 THEN 6           -- [1.20, 1.40)
      ELSE 7                                 -- ≥ 1.40
    END AS ptv_bucket,
    sold_day,
    duration_hours
  FROM sold
  WHERE ptv_final BETWEEN 0.50 AND 2.50      -- clamp crazy outliers
),
recent AS (
  SELECT
    generation,
    sbucket,
    ptv_bucket,
    sold_day,
    duration_hours,
    -- Recency weight: half-life = 90 days (tuneable)
    (0.5 ^ (GREATEST(0, (CURRENT_DATE - sold_day))::numeric / 90.0)) AS w,
    CASE WHEN duration_hours <=  24    THEN 1::numeric ELSE 0::numeric END AS is_fast24,
    CASE WHEN duration_hours <= 168    THEN 1::numeric ELSE 0::numeric END AS is_fast7,
    CASE WHEN duration_hours >  21*24 THEN 1::numeric ELSE 0::numeric END AS is_slow21
  FROM ptv_binned
  WHERE ptv_bucket IS NOT NULL
    AND sold_day >= CURRENT_DATE - INTERVAL '90 days'
),
agg AS (
  SELECT
    generation,
    sbucket,
    ptv_bucket,
    SUM(w) AS sum_w,
    -- recency-weighted fast/slow probabilities
    SUM(w * is_fast7)  / NULLIF(SUM(w),0) AS speed_fast7_anchor,
    SUM(w * is_fast24) / NULLIF(SUM(w),0) AS speed_fast24_anchor,
    SUM(w * is_slow21) / NULLIF(SUM(w),0) AS speed_slow21_anchor,
    -- unweighted median duration in the bucket
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_hours)
      AS speed_median_hours_ptv,
    -- effective sample size: (Σw)^2 / Σ(w^2)
    CASE
      WHEN SUM(w*w) > 0 THEN (SUM(w)*SUM(w)) / SUM(w*w)
      ELSE 0
    END AS speed_n_eff_ptv
  FROM recent
  GROUP BY generation, sbucket, ptv_bucket
)
SELECT
  generation,
  sbucket,
  ptv_bucket,
  sum_w,
  speed_fast7_anchor,
  speed_fast24_anchor,
  speed_slow21_anchor,
  speed_median_hours_ptv,
  speed_n_eff_ptv
FROM agg;

CREATE INDEX IF NOT EXISTS tom_speed_anchor_v1_idx
  ON ml.tom_speed_anchor_v1_mv (generation, sbucket, ptv_bucket);

ANALYZE ml.tom_speed_anchor_v1_mv;



















speed anchors  


db=# -- ===========================================================
-- E+SPEED) ENRICHED + SPEED ANCHORS (spam-clean, unique by listing_id)
-- Base: ml.tom_features_v1_enriched_clean_mv
-- Joins: ml.tom_speed_anchor_v1_mv on (generation, sbucket, ptv_bucket)
-- ===========================================================
DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_speed_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_speed_mv AS
WITH f AS (
  SELECT
    f.*,
    -- storage bucket (same as strict anchors + speed MV)
    CASE
      WHEN f.storage_gb >= 900 THEN 1024
      WHEN f.storage_gb >= 500 THEN 512
      WHEN f.storage_gb >= 250 THEN 256
      WHEN f.storage_gb >= 120 THEN 128
      ELSE f.storage_gb
    END AS sbucket,
    -- PTV bucket for joining speed anchors (same thresholds as tom_speed_anchor_v1_mv)
    CASE
      WHEN f.ptv_final IS NULL
           OR f.ptv_final < 0.50
           OR f.ptv_final > 2.50
        THEN NULL::int
      WHEN f.ptv_final < 0.80 THEN 1     -- [0.50, 0.80)
      WHEN f.ptv_final < 0.90 THEN 2     -- [0.80, 0.90)
      WHEN f.ptv_final < 1.00 THEN 3     -- [0.90, 1.00)
      WHEN f.ptv_final < 1.10 THEN 4     -- [1.00, 1.10)
      WHEN f.ptv_final < 1.20 THEN 5     -- [1.10, 1.20)
      WHEN f.ptv_final < 1.40 THEN 6     -- [1.20, 1.40)
      ELSE 7                             -- ≥ 1.40
    END AS ptv_bucket
  FROM ml.tom_features_v1_enriched_clean_mv f
  -- this view is already spam-clean + unique-by-listing_id from Phase-1
)
SELECT
  f.*,
  s.speed_fast7_anchor,
  s.speed_fast24_anchor,
  s.speed_slow21_anchor,
  s.speed_median_hours_ptv,
  s.speed_n_eff_ptv
FROM f
LEFT JOIN ml.tom_speed_anchor_v1_mv s
  ON s.generation = f.generation
 AND s.sbucket    = f.sbucket
 AND s.ptv_bucket = f.ptv_bucket
;

-- Unique index to allow CONCURRENT refresh and make it a serving layer
CREATE UNIQUE INDEX IF NOT EXISTS tom_features_v1_enriched_speed_uq
ANALYZE ml.tom_features_v1_enriched_speed_mv;nn_id);
NOTICE:  materialized view "tom_features_v1_enriched_speed_mv" does not exist, skipping
DROP MATERIALIZED VIEW
SELECT 21867
CREATE INDEX
ANALYZE
db=#
ALTER TABLE ml.predictions_v1
  ADD COLUMN p_72h   double precision,
  ADD COLUMN p_7days double precision;
DROP COLUMN p_48h;





SELECT
  listing_id,
  edited_date,
  expected_hours,
  p_24h,
  p_72h,
  p_7days
FROM ml.predictions_current
WHERE edited_date >= now() - interval '100 day'
ORDER BY expected_hours ASC
LIMIT 50;



-- Live + unsold rows for the ACTIVE model, last 1 day by t₀, fastest first
SELECT
  p.listing_id,
  p.edited_date,
  p.expected_hours,
  p.p_24h,
  p.p_72h,
  p.p_7days
FROM ml.predictions_current AS p
WHERE p.edited_date >= now() - interval '1 day'
ORDER BY p.expected_hours ASC
LIMIT 1550;






SELECT
  p.listing_id,
  p.edited_date,
  p.expected_hours,
  p.p_24h,
  p.p_72h,
  p.p_7days
FROM ml.predictions_v1 AS p
WHERE p.model_key = 'tom_aft_v1_20251114T0231Z'
  AND p.edited_date::date = current_date
ORDER BY p.expected_hours ASC
LIMIT 5550;

















SELECT
  p.listing_id,

  -- from feature store
  f.model,
  f.storage_gb,
  f.condition_score,
  f.damage_severity_ai     AS sev,
  f.battery_pct_effective  AS battery_effective
  p.edited_date,
  p.expected_hours,
  p.p_24h,
  p.p_72h,
  p.p_7days,


FROM ml.predictions_v1 AS p
JOIN ml.tom_features_v1_mv AS f
  USING (listing_id)
WHERE p.model_key = 'tom_aft_v1_20251114T0231Z'
  AND p.edited_date::date = current_date
ORDER BY p.expected_hours ASC
LIMIT 5550;





SELECT
  p.listing_id,
  p.edited_date,
  p.expected_hours,
  p.p_24h,
  p.p_72h,
  p.p_7days,

  -- core listing info
  e.model,
  e.storage_gb,
  e.condition_score,
  e.damage_severity_ai        AS sev,
  e.battery_pct_effective     AS battery_effective,

  -- seller trust
  e.seller_rating,
  e.review_count,
  e.member_since_year,

  -- main anchor / speed / ptv features
  e.anchor_median_hours_14d,      -- market-speed anchor from last 14d
  e.ptv_final,                    -- your final PTV feature
  e.speed_median_hours_ptv        -- strict speed anchor that dominates FI

FROM ml.predictions_v1 AS p
JOIN ml.tom_features_v1_enriched_ai_clean_mv AS e
  USING (listing_id)
WHERE p.model_key = 'tom_aft_v1_20251114T0231Z'
  AND p.edited_date::date = current_date
ORDER BY p.expected_hours ASC
LIMIT 5550;
quality control command 



# --- API + DB ---
$env:DEEPSEEK_API_KEY = '<REDACTED_API_KEY>'
$env:OPENAI_API_KEY   = $env:DEEPSEEK_API_KEY
$env:OPENAI_BASE_URL  = 'https://api.deepseek.com/v1'   # needed for deepseek-chat
$env:PG_DSN           = '${PG_DSN}'

# --- Model + run behavior ---
$env:MODEL_NAME       = 'deepseek-chat'
$env:VERSION_TAG      = 'llm-quality-v1.1'
$env:WORKERS          = '8'            # keep modest to avoid 429s
$env:CAP_PER_GEN      = '999999'       # effectively "no cap" per generation
$env:ONLY_LIVE_HOURS  = '0'            # no time filter → ALL LIVE rows
$env:LOG_LEVEL        = 'INFO'
$env:PYTHONUNBUFFERED = '1'

# real run
$env:DRY_RUN = 'false'
python .\quality_control_ai.py



Enricher 



$env:DEEPSEEK_API_KEY = '<REDACTED_API_KEY>'
$env:PG_DSN = '${PG_DSN}' 

# optional
$env:MODEL_NAME = 'deepseek-chat'
$env:VERSION_TAG = 'enrich-core-v1.2'
$env:WORKERS = '64'
$env:QPS = '1.5'
$env:LOG_LEVEL = 'INFO' 
$env:PYTHONUNBUFFERED = '1' 
python .\iphone_ai_enrich_upserter.py


post sold command 





.\post_sold_audit.exe `
  --dsn "${PG_DSN}" `
  --gens 13,14,15,16,17 `
  --per-gen-cap=0 `
  --concurrency=64 `
  --rps=3 --rps-max=12 --jitter-ms=120 `
  --retry-max=2 `
  --backoff-initial=2s `
  --backoff-max=20s `
  --max-conns-per-host=6



condition sync  


$env:PG_DSN = "${PG_DSN}"
$env:LOG_LEVEL = "INFO"
$env:PSA_COND_VERSION = "psa-cond-sync-v1"
$env:EPS = "0.001"
$env:APPLY = "true"   # <— APPLY FOR REAL

python .\psa_condition_sync_and_rescore.py


DEDUPLICATE  SAFTY 


python .\dedupe_ai_sql.py `
  --pg-dsn "${PG_DSN}" `
  --provider deepseek --model deepseek-chat `
  --api-key "<REDACTED_API_KEY>" `
  --limit 200 `
  --bundle-mode spam-bundled `
  --bundle-spam bundled `
  --version-tag dedupe-ai/v3





Battery AI  llm


$env:OPENAI_API_KEY  = '<REDACTED_API_KEY>'
$env:OPENAI_BASE_URL = 'https://api.deepseek.com/v1'
$env:MODEL_NAME      = 'deepseek-chat'
$env:PG_DSN          = '${PG_DSN}'

# Process everything eligible (no paging)
$env:DRY_RUN = '0'
$env:UPDATE_MAIN = '1'
$env:UPDATE_PSA  = '1'
$env:PSA_ONLY = '1'
$env:EXCLUDE_COND_SCORE_EQ_1 = '1'
$env:BATCH_SIZE = '0'   # no limit
$env:LIMIT      = '0'
Remove-Item Env:LISTING_IDS -ErrorAction SilentlyContinue

python .\battery_refit_llm_v2.py 




Damage scorer 






# --- DeepSeek creds (put your real key here if the var is empty) ---
if (-not $env:DEEPSEEK_API_KEY -or $env:DEEPSEEK_API_KEY.Trim() -eq "") {
  Write-Host "DEEPSEEK_API_KEY is empty — paste your key now:" -ForegroundColor Yellow
  $env:DEEPSEEK_API_KEY = Read-Host
}

# The script expects OPENAI_API_KEY, so mirror DeepSeek -> OpenAI
$env:OPENAI_API_KEY  = $env:DEEPSEEK_API_KEY

# Optional: quick sanity check (prints masked length)
Write-Host ("OPENAI_API_KEY len: {0}" -f $env:OPENAI_API_KEY.Length)
if ($env:OPENAI_API_KEY.Length -lt 20) { throw "OPENAI_API_KEY looks empty/invalid" }

# --- API endpoint/model ---
$env:OPENAI_BASE_URL = "https://api.deepseek.com"   # try without /v1 first
# If your client does NOT append /v1 automatically, use this instead:
# $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

$env:MODEL_NAME  = "deepseek-chat"
$env:VERSION_TAG = "llm-deepseek-v1.0"

# --- Postgres DSN (adjust if needed) ---
$env:PG_DSN = "${PG_DSN}"
$env:PGHOST = "<PGHOST>"; $env:PGPORT="<PGPORT>"; $env:PGUSER="<PGUSER>"; $env:PGPASSWORD="<PGPASSWORD>"; $env:PGDATABASE="<PGDATABASE>"

# --- Run ---
python gbt_damage.py













psay sold price sync main table 




$env:PG_DSN = "${PG_DSN}"
$env:APPLY  = "false"
python .\psa_sold_price_sync.py



$env:PG_DSN = "${PG_DSN}"
$env:APPLY  = "true"
python .\psa_sold_price_sync.py



































one big giant command    










# --- API + DB ---
$env:DEEPSEEK_API_KEY = '<REDACTED_API_KEY>'
$env:OPENAI_API_KEY   = $env:DEEPSEEK_API_KEY
$env:OPENAI_BASE_URL  = 'https://api.deepseek.com/v1'   # needed for deepseek-chat
$env:PG_DSN           = '${PG_DSN}'

# --- Model + run behavior ---
$env:MODEL_NAME       = 'deepseek-chat'
$env:VERSION_TAG      = 'llm-quality-v1.1'
$env:WORKERS          = '8'            # keep modest to avoid 429s
$env:CAP_PER_GEN      = '999999'       # effectively "no cap" per generation
$env:ONLY_LIVE_HOURS  = '0'            # no time filter → ALL LIVE rows
$env:LOG_LEVEL        = 'INFO'
$env:PYTHONUNBUFFERED = '1'

# real run
$env:DRY_RUN = 'false'
python .\quality_control_ai.py







$env:DEEPSEEK_API_KEY = '<REDACTED_API_KEY>'
$env:PG_DSN = '${PG_DSN}' 

# optional
$env:MODEL_NAME = 'deepseek-chat'
$env:VERSION_TAG = 'enrich-core-v1.2'
$env:WORKERS = '64'
$env:QPS = '1.5'
$env:LOG_LEVEL = 'INFO' 
$env:PYTHONUNBUFFERED = '1' 
python .\iphone_ai_enrich_upserter.py








.\post_sold_audit.exe `
  --dsn "${PG_DSN}" `
  --gens 13,14,15,16,17 `
  --per-gen-cap=0 `
  --concurrency=64 `
  --rps=3 --rps-max=12 --jitter-ms=120 `
  --retry-max=2 `
  --backoff-initial=2s `
  --backoff-max=20s `
  --max-conns-per-host=6





$env:PG_DSN = "${PG_DSN}"
$env:LOG_LEVEL = "INFO"
$env:PSA_COND_VERSION = "psa-cond-sync-v1"
$env:EPS = "0.001"
$env:APPLY = "true"   # <— APPLY FOR REAL

python .\psa_condition_sync_and_rescore.py





python .\dedupe_ai_sql.py `
  --pg-dsn "${PG_DSN}" `
  --provider deepseek --model deepseek-chat `
  --api-key "<REDACTED_API_KEY>" `
  --limit 200 `
  --bundle-mode spam-bundled `
  --bundle-spam bundled `
  --version-tag dedupe-ai/v3







$env:OPENAI_API_KEY  = '<REDACTED_API_KEY>'
$env:OPENAI_BASE_URL = 'https://api.deepseek.com/v1'
$env:MODEL_NAME      = 'deepseek-chat'
$env:PG_DSN          = '${PG_DSN}'

# Process everything eligible (no paging)
$env:DRY_RUN = '0'
$env:UPDATE_MAIN = '1'
$env:UPDATE_PSA  = '1'
$env:PSA_ONLY = '1'
$env:EXCLUDE_COND_SCORE_EQ_1 = '1'
$env:BATCH_SIZE = '0'   # no limit
$env:LIMIT      = '0'
Remove-Item Env:LISTING_IDS -ErrorAction SilentlyContinue

python .\battery_refit_llm_v2.py 








# --- DeepSeek creds (put your real key here if the var is empty) ---
if (-not $env:DEEPSEEK_API_KEY -or $env:DEEPSEEK_API_KEY.Trim() -eq "") {
  Write-Host "DEEPSEEK_API_KEY is empty — paste your key now:" -ForegroundColor Yellow
  $env:DEEPSEEK_API_KEY = Read-Host
}

# The script expects OPENAI_API_KEY, so mirror DeepSeek -> OpenAI
$env:OPENAI_API_KEY  = $env:DEEPSEEK_API_KEY

# Optional: quick sanity check (prints masked length)
Write-Host ("OPENAI_API_KEY len: {0}" -f $env:OPENAI_API_KEY.Length)
if ($env:OPENAI_API_KEY.Length -lt 20) { throw "OPENAI_API_KEY looks empty/invalid" }

# --- API endpoint/model ---
$env:OPENAI_BASE_URL = "https://api.deepseek.com"   # try without /v1 first
# If your client does NOT append /v1 automatically, use this instead:
# $env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"

$env:MODEL_NAME  = "deepseek-chat"
$env:VERSION_TAG = "llm-deepseek-v1.0"

# --- Postgres DSN (adjust if needed) ---
$env:PG_DSN = "${PG_DSN}"
$env:PGHOST = "<PGHOST>"; $env:PGPORT="<PGPORT>"; $env:PGUSER="<PGUSER>"; $env:PGPASSWORD="<PGPASSWORD>"; $env:PGDATABASE="<PGDATABASE>"

# --- Run ---
python gbt_damage.py














$env:PG_DSN = "${PG_DSN}"
$env:APPLY  = "true"
python .\psa_sold_price_sync.py

1. One-shot full refresh ✅

If all the MVs and indexes are already created, you can just paste this whole thing into psql:

-- A/B/C: core base + labels + training base
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

-- D1/D2/D3: anchor/history layers
REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;

-- E: enriched
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;

-- E-clean: spam-clean enriched
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

-- D4: speed anchors (depends on D1 + E)
REFRESH MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv;

-- E+speed: enriched + speed_* (depends on clean + D4)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_mv;

-- E-clean+AI: AI-augmented read layer (depends on E+speed)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

-- Stats for the planner
ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_speed_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;




ANALYZE ml.sold_durations_v1_mv;
ANALYZE ml.sold_prices_v1_mv;
ANALYZE ml.tom_speed_anchor_v1_mv;



ANALYZE ref.postal_code_to_super_metro;
ANALYZE ref.city_to_super_metro;
ANALYZE ref.geo_mapping_release;




That’s a perfectly fine “nightly” script. Order matters, but you are in the right order:

Base & labels ➜ anchors ➜ enriched ➜ speed ➜ speed+enriched ➜ clean ➜ AI.

So: yes, safe to run as one script.

2. But you don’t need all of it every time

For quick iteration while you’re just messing with speed anchors and AI features, you can run only the parts that depend on what you changed.

Examples:

If you just changed the speed MV:
REFRESH MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

ANALYZE ml.tom_features_v1_enriched_speed_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;


(no need to refresh A/B/C/D1/D2/D3/E if you didn’t touch them)

If you just changed the AI MV:
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;


That’s it.

3. One-time vs repeated stuff

The REFRESH MATERIALIZED VIEW ... lines:
✅ Run them every time you want fresh data.

The CREATE UNIQUE INDEX ... lines:
⚠️ One time only when you first create the MV. You don’t need to recreate them on every refresh.

So your “refresh” file should only have REFRESH + ANALYZE. The CREATE MATERIALIZED VIEW / CREATE INDEX live in your setup/DDL scripts, not in the everyday refresh.

TL;DR

You can run that block in one go as your full refresh.

For day-to-day hacking on speed/AI, you can just run the D4 + E+speed + AI + ANALYZE subset.

Just keep the order: base ➜ anchors ➜ enriched ➜ speed ➜ enriched+speed ➜ clean ➜ AI.
