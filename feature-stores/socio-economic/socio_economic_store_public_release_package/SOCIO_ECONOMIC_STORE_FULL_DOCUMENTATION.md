# Socio‑Economic Store (T0 + Leak‑Proof) — Certified Package

This package documents and recreates the **socio_economic_store** feature layer used by the Slow21 gate classifier
and the registry-based certification you now enforce in Postgres.

## What this package covers

### Feature-store layer (DB objects)
The socio_economic_store is implemented as a **SOCIO (T0) MV + MARKET (T0) MV + WIDE SOCIO+MARKET MV**:

1. `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv`
   - Adds **geo→postal→kommune→socio** context *as-of T0* and the **4 affordability features**:
     - `price_months_income`
     - `log_price_to_income_cap` (winsorized with pinned p01/p99 constants)
     - `aff_resid_in_seg`
     - `aff_decile_in_seg`

2. `ml.market_base_socio_t0_v1` (table)
   - A thin, fast “market base” table populated from the SOCIO T0 MV.

3. `ml.market_relative_socio_t0_v1_mv`
   - A strict trailing 30‑day **leave-one-out** comp statistics MV keyed by `(generation, listing_id, t0)`
     with explicit embargo: window ends at `t0 - 1 microsecond`.

4. `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv`
   - The wide “features_view” used for training. It selects `s.*` from SOCIO T0 and joins market stats,
     so SOCIO columns (including affordability) propagate automatically.

### Certified entrypoints (registry + guards)
The certified entrypoints that enforce leak-proof usage at training time:

- `ml.socio_market_feature_store_t0_v1_v` (certified entrypoint)
- `ml.socio_market_feature_store_train_v` (guarded training view; fails closed if not certified)

You also register these as **certification-inherited aliases** in `audit.t0_cert_registry`:

- `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv`
- `ml.market_relative_socio_t0_v1_mv`
- `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv`

## Quick start (recreate + certify)

1) Create/replace the store objects (SQL in order):

- `sql/00_prereqs.sql`
- `sql/01_create_socio_t0_mv.sql`
- `sql/02_create_market_base_table.sql`
- `sql/03_market_relative_mv.sql`
- `sql/04_create_socio_market_wide_mv.sql`
- `sql/05_create_entrypoint_and_guard_views.sql`

2) Run post-refresh invariants:

- `sql/06_invariants_checks.sql`

3) Install certification functions & procedures:

- `sql/07_create_cert_functions.sql`

4) Baseline viewdefs and dataset hashes, then certify:

- `sql/08_baseline_viewdefs.sql`
- `sql/09_baseline_dataset_hashes_365.sql`
- `sql/10_run_certification.sql`
- `sql/11_register_aliases.sql`

5) Use the operational runbook:

- `sql/12_daily_refresh_runbook.sql`

## Notes
- This public-release package is **self-contained**: it includes a minimal audit/certification helper layer in
  `sql/00a_minimal_audit_framework.sql`.

  If you already have an audit framework in your database, you can skip that file. The store expects the following
  functions to exist (names can be adapted, but the semantics should match):
  - `audit.dataset_sha256(regclass, date, int [, text])`
  - `audit.assert_viewdef_baseline(text)`
  - `audit.require_certified_strict(text, interval)`


- The winsorization constants for `log_price_to_income_cap` are pinned:
  - p01 = -5.964859
  - p99 = -3.610855
  Re-estimate if the price/income distribution shifts materially.

---

# 1. Overview

## 1.1 What is the socio_economic_store?

In the Slow21 gate classifier taxonomy, **socio_economic_store** is the feature-store block that provides:

1) **Socio-economic context** (as-of the listing’s T0):
- kommune mapping (`kommune_code4`) derived from postal code
- `centrality_class`
- `income_median_after_tax_nok`
- missingness flags: `miss_kommune`, `miss_socio`
- model segmentation key: `model_tier_socio`

2) **Affordability features** (listing-level, as-of T0):
- `price_months_income`
- `log_price_to_income_cap` (winsorized to pinned p01/p99 constants)
- `aff_resid_in_seg`
- `aff_decile_in_seg`

3) **Market-relative pricing features** (strict trailing 30d leave-one-out windows, as-of T0):
- `rel_log_price_to_comp_mean30_super`
- `rel_log_price_to_comp_mean30_kommune`
- `rel_price_best`
- `rel_price_source`
- `miss_rel_price`

This block is designed to provide high-signal, leak-safe “pricing context” for slow-tail risk.

## 1.2 Why does this layer exist?

Slow21 performance is dominated by *overpricing relative to the relevant reference set*.
This layer provides two complementary perspectives:

- **Affordability (income-normalized pricing)**: the listing price relative to local income, and residualization within model segments.
- **Market comps (recent supply pricing)**: the listing price relative to comparable listings in the same local market over a strict trailing window.

Together these supply a stable, interpretable “liquidity prior” that generalizes across geography.

## 1.3 Where this store lives in your pipeline

This store is implemented inside the **certified features_view** used for training:

- `ml.socio_market_feature_store_train_v` (guarded; fail closed)
  → `ml.socio_market_feature_store_t0_v1_v` (certified entrypoint)
  → `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv` (wide MV)
     - built from:
       - `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv` (SOCIO+AFF)
       - `ml.market_relative_socio_t0_v1_mv` (MARKET)

In your SHAP reporting taxonomy, the socio_economic_store block is populated by the 12 modeling columns:

`centrality_class, miss_kommune, miss_socio, price_months_income, log_price_to_income_cap,
 aff_resid_in_seg, aff_decile_in_seg, rel_log_price_to_comp_mean30_super,
 rel_log_price_to_comp_mean30_kommune, rel_price_best, rel_price_source, miss_rel_price`

(Other socio context columns may exist in the wide MV but are not necessarily selected into the trainer’s “attached cols” log.)

---

# 2. Object inventory and dependency graph

## 2.1 Core DB objects

### (A) SOCIO (T0) MV
- **Name:** `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv`
- **Type:** MATERIALIZED VIEW
- **Key:** `listing_id` (unique in this environment)
- **Inputs:**
  - `ml.tom_features_v2_enriched_ai_ob_clean_mv` (listing-level feature surface)
  - `ml.iphone_listings_geo_current` (geo mapping; provides postal_code and super_metro_v4_geo)
  - `ref.postal_code_to_kommune_history` (postal→kommune snapshots)
  - `ref.kommune_socio_history` (kommune socio snapshots)
- **Outputs:**
  - adds `t0` column (alias of edited_date)
  - `geo_postal_code`, `super_metro_v4_geo`
  - socio context (`kommune_code4`, `centrality_class`, `income_median_after_tax_nok`, snapshot dates)
  - missingness flags
  - model segmentation key `model_tier_socio`
  - affordability columns (4)

### (B) Market base table
- **Name:** `ml.market_base_socio_t0_v1`
- **Type:** TABLE
- **Key:** `(generation, listing_id, t0)` recommended
- **Populated from:** SOCIO T0 MV
- **Fields:** `generation, listing_id, t0, super_metro_v4_geo, kommune_code4, price, log_price_mkt`

### (C) Market stats MV (thin)
- **Name:** `ml.market_relative_socio_t0_v1_mv`
- **Type:** MATERIALIZED VIEW
- **Key:** `(generation, listing_id, t0)` unique
- **Inputs:** `ml.market_base_socio_t0_v1`
- **Outputs:** strict 30d trailing window comps:
  - `comp_n_30d_super_metro`, `comp_log_price_mean_30d_super_metro`, `comp_max_t0_super_metro`
  - `comp_n_30d_kommune`, `comp_log_price_mean_30d_kommune`, `comp_max_t0_kommune`

### (D) Wide SOCIO+MARKET MV (features_view)
- **Name:** `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv`
- **Type:** MATERIALIZED VIEW
- **Key:** `(generation, listing_id, t0)` unique
- **Inputs:**
  - `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv` as `s`
  - `ml.market_relative_socio_t0_v1_mv` as `m`
- **Outputs:**
  - `s.*` (includes the affordability columns)
  - comp stats from `m`
  - derived market-relative pricing features (`rel_*` and `miss_rel_price`)

### (E) Certified entrypoint + guarded training view
- **Entry point:** `ml.socio_market_feature_store_t0_v1_v` (VIEW → wide MV)
- **Guarded training view:** `ml.socio_market_feature_store_train_v`
  - uses `audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours')`

## 2.2 Certification registry objects

- `audit.t0_viewdef_baseline` — stores closure viewdef SHA256 baselines
- `audit.t0_dataset_hash_baseline` — stores per-day dataset hashes for drift tests
- `audit.t0_cert_registry` — certification status registry

- `audit.assert_t0_certified_socio_market_v1(p_check_days int)` — assertion function
- `audit.run_t0_cert_socio_market_v1(p_check_days int)` — procedure that runs assertion and updates registry

## 2.3 Dependency graph (simplified)

```
ml.socio_market_feature_store_train_v
  -> audit.require_certified_strict(...)
  -> ml.socio_market_feature_store_t0_v1_v
     -> ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
        -> ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
           -> ml.tom_features_v2_enriched_ai_ob_clean_mv
           -> ml.iphone_listings_geo_current
           -> ref.postal_code_to_kommune_history (as-of t0)
           -> ref.kommune_socio_history (as-of t0)
        -> ml.market_relative_socio_t0_v1_mv
           -> ml.market_base_socio_t0_v1 (populated from SOCIO T0 MV)
```

---

# 3. Data dictionary (socio_economic_store)

This section documents the socio_economic_store feature columns, grouped by sub-layer.

## 3.1 Socio context (as-of T0)

| Column | Type | Description | T0 contract |
|---|---:|---|---|
| kommune_code4 | text | Kommune code derived from postal code mapping snapshot | mapping snapshot_date <= t0::date |
| centrality_class | int | Kommune centrality class | socio snapshot_date <= t0::date |
| income_median_after_tax_nok | float8 | Kommune median after-tax income (NOK) | socio snapshot_date <= t0::date |
| postal_kommune_snapshot_date | date | snapshot date used for postal→kommune mapping | must be <= t0::date |
| socio_snapshot_date | date | snapshot date used for kommune socio | must be <= t0::date |
| miss_kommune | int | 1 if kommune mapping missing, else 0 | derived |
| miss_socio | int | 1 if socio record missing, else 0 | derived |
| model_tier_socio | text | model tier bucket used for within-segment normalization | derived from model string (T0) |

## 3.2 Affordability features (as-of T0)

### price_months_income
- Definition:
  `price_months_income = price / (income_median_after_tax_nok / 12) = 12*price/income`
- NULL if price <= 0 or income <= 0.

### log_price_to_income_cap
- Definition:
  `ln((price+1)/income_median_after_tax_nok)`
- Winsorized to pinned constants:
  - p01 = -5.964859
  - p99 = -3.610855

### aff_resid_in_seg
- Definition:
  `log_price_to_income_cap − mean(log_price_to_income_cap | generation × model_tier_socio)`
- Mean computed within the MV at refresh time.

### aff_decile_in_seg
- Definition:
  `NTILE(10)` within `(generation × model_tier_socio)` ordered by `log_price_to_income_cap`
- NULL if `log_price_to_income_cap` is NULL.

## 3.3 Market-relative pricing (strict trailing 30d leave-one-out)

All market stats are computed using:

- `log_price_mkt = ln(price)` in the market base table (your current implementation)
- strict trailing window:
  `RANGE BETWEEN '30 days' PRECEDING AND '1 microsecond' PRECEDING`
  which ensures:
  - no same-row contamination
  - no future-row contamination

Market stats fields:

| Column | Type | Description |
|---|---:|---|
| comp_n_30d_super_metro | int | # comps in same super_metro_v4_geo over trailing 30d |
| comp_log_price_mean_30d_super_metro | float8 | mean(log_price_mkt) over comps |
| comp_max_t0_super_metro | timestamptz | max comp t0 used (must be < row t0) |
| comp_n_30d_kommune | int | # comps in same kommune_code4 over trailing 30d |
| comp_log_price_mean_30d_kommune | float8 | mean(log_price_mkt) over comps |
| comp_max_t0_kommune | timestamptz | max comp t0 used (must be < row t0) |

Derived relative pricing fields (computed in the wide MV):

| Column | Type | Definition |
|---|---:|---|
| rel_log_price_to_comp_mean30_super | float8 | ln(price) − comp_log_price_mean_30d_super_metro (only if comp_n>=20) |
| rel_log_price_to_comp_mean30_kommune | float8 | ln(price) − comp_log_price_mean_30d_kommune (only if comp_n>=20) |
| rel_price_best | float8 | coverage-aware fallback (super first in current implementation) |
| rel_price_source | int | 1=super_metro, 2=kommune, NULL missing |
| miss_rel_price | int | 1 if rel_price_best is NULL else 0 |

Note: your implementation uses comp thresholds of 20 before computing rel scores.

---

# 4. T0 + leak-proof contract (why this store is certifiable)

## 4.1 T0 definition

For each listing row, `t0` is the listing’s `edited_date` timestamp.
All socio and market features must be computable using only information available at or before that timestamp.

## 4.2 Leakage categories guarded against

1) Time-of-query leakage
- banned tokens: `CURRENT_DATE`, `now()`, `clock_timestamp()`, etc.
- enforcement: closure token scan inside the certification assertion

2) Structural time leakage (“future join”)
- as-of socio joins enforce: snapshot_date <= t0::date
- market comps enforce: comp_max_t0_* < t0 (strict trailing windows)

3) Label leakage
- store does not reference outcomes (sold_date, duration, status_final, etc.)

4) Determinism / drift
- dynamic invariance tested via dataset hashes for fixed historical t0_day slices

## 4.3 Structural invariants

### SOCIO as-of invariant
Must hold:
- `postal_kommune_snapshot_date <= t0::date` when non-null
- `socio_snapshot_date <= t0::date` when non-null

### MARKET strict trailing window invariant
Must hold:
- `comp_max_t0_super_metro < t0` when non-null
- `comp_max_t0_kommune < t0` when non-null

These are asserted both in:
- runbook checks, and
- certification assertion function.

## 4.4 Why the affordability features are T0-safe

All affordability columns depend only on:
- listing price at T0
- income snapshot as-of T0
- within-segment statistics computed from the MV output at refresh time

The winsor bounds are pinned constants, not computed at query time, so they do not introduce time-of-query dependence.

---

# 5. Certification (registry + baselines + guards)

This section documents the complete certification mechanism you use for the socio/market entrypoint.

## 5.1 Certified entrypoint and guarded training view

- **Certified entrypoint (VIEW):** `ml.socio_market_feature_store_t0_v1_v`
- **Guarded training view (VIEW):** `ml.socio_market_feature_store_train_v`

The guarded view forces the training query to fail closed if certification is stale or drifted, by calling:

`audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours')`

## 5.2 Registry tables

- `audit.t0_cert_registry` — status= CERTIFIED/FAILED, timestamps, counts, notes
- `audit.t0_viewdef_baseline` — closure viewdef SHA256 baselines
- `audit.t0_dataset_hash_baseline` — per-day dataset hashes for drift tests

## 5.3 Certification assertion: what is proven

The `audit.assert_t0_certified_socio_market_v1(p_check_days)` function asserts:

1) Gate A2: no forbidden time tokens exist in the transitive closure of the entrypoint
2) SOCIO invariant: n_bad_socio == 0
3) MARKET invariant: n_bad_market == 0
4) Drift check: for the most recent `p_check_days` baseline days, current dataset hash equals baseline

## 5.4 Certification runner: how the registry is written

The procedure `audit.run_t0_cert_socio_market_v1(p_check_days)`:

- calls the assertion function
- writes/upserts a registry row for `ml.socio_market_feature_store_t0_v1_v`
- optionally registers dependent MVs and the guarded view as “certification inherited” aliases

## 5.5 Baselines

### Viewdef baselines
You store SHA256 hashes of all view/matview definitions in the closure in `audit.t0_viewdef_baseline`.

### Dataset hash baselines
You store 365 per-day dataset hashes (sample_limit=2000) in `audit.t0_dataset_hash_baseline`.

The drift check recomputes hashes for the last N days and compares.

## 5.6 Re-certification after changes

Any time you change the definition of a view/matview in the certified closure:

1) delete and repopulate viewdef baselines for the entrypoint
2) delete and repopulate dataset hash baselines
3) rerun the certification procedure
4) verify `audit.require_certified_strict(...)` passes

---

# 6. Refresh runbook (operational)

This store is refreshed and certified daily in dependency order:

1) refresh upstream base features (not defined in this package)
2) refresh `ml.tom_features_v2_enriched_ai_ob_clean_mv`
3) refresh `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv`
4) reload `ml.market_base_socio_t0_v1` (truncate + insert)
5) refresh `ml.market_relative_socio_t0_v1_mv`
6) refresh `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv`
7) run invariants checks
8) call `audit.run_t0_cert_socio_market_v1(p_check_days=30)`
9) enforce: `audit.require_certified_strict(entrypoint, interval '24 hours')`

The full SQL runbook is provided in `sql/12_daily_refresh_runbook.sql`.

---

# 7. Recreation playbook (from scratch)

This section is an “operator checklist” for rebuilding the socio_economic_store in a new environment.

## 7.1 Preconditions (must exist)

The socio_economic_store depends on:

- A listing-level base feature surface:
  - `ml.tom_features_v2_enriched_ai_ob_clean_mv`
- Geo mapping:
  - `ml.iphone_listings_geo_current` with `(generation, listing_id, postal_code, super_metro_v4_geo)`
- Socio reference history tables:
  - `ref.postal_code_to_kommune_history(postal_code, kommune_code4, snapshot_date, loaded_at, ...)`
  - `ref.kommune_socio_history(kommune_code4, centrality_class, income_median_after_tax_nok, snapshot_date, loaded_at, ...)`
- Audit framework functions:
  - `audit.dataset_sha256(regclass, date, int)`
  - `audit.require_certified_strict(text, interval)`
  - `audit.assert_viewdef_baseline(text)` (called internally by require_certified_strict)

## 7.2 Creation order (one-time)

1) Run `sql/00_prereqs.sql`
2) Run `sql/02_create_market_base_table.sql`
3) Run `sql/01_create_socio_t0_mv.sql`
4) Run `sql/03_market_relative_mv.sql`
5) Run `sql/04_create_socio_market_wide_mv.sql`
6) Run `sql/05_create_entrypoint_and_guard_views.sql`
7) Run `sql/07_create_cert_functions.sql`

## 7.3 Baseline + certify (first time)

1) Run `sql/08_baseline_viewdefs.sql`
2) Run `sql/09_baseline_dataset_hashes_365.sql`
3) Run `sql/10_run_certification.sql`
4) (Optional) Run `sql/11_register_aliases.sql`

## 7.4 Daily operations

Use `sql/12_daily_refresh_runbook.sql`.

---

# 8. Validation queries

## 8.1 Column presence (affordability must exist)

```sql
SELECT attname
FROM pg_attribute
WHERE attrelid = 'ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv'::regclass
  AND attnum > 0 AND NOT attisdropped
  AND attname IN ('price_months_income','log_price_to_income_cap','aff_resid_in_seg','aff_decile_in_seg')
ORDER BY attname;
```

```sql
SELECT attname
FROM pg_attribute
WHERE attrelid = 'ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv'::regclass
  AND attnum > 0 AND NOT attisdropped
  AND attname IN ('price_months_income','log_price_to_income_cap','aff_resid_in_seg','aff_decile_in_seg')
ORDER BY attname;
```

## 8.2 Invariants

```sql
SELECT COUNT(*) AS n_bad_socio
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
WHERE (postal_kommune_snapshot_date IS NOT NULL AND postal_kommune_snapshot_date > t0::date)
   OR (socio_snapshot_date IS NOT NULL AND socio_snapshot_date > t0::date);
```

```sql
SELECT COUNT(*) AS n_bad_market
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
WHERE (comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0)
   OR (comp_max_t0_kommune     IS NOT NULL AND comp_max_t0_kommune     >= t0);
```

## 8.3 Guard proof

```sql
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');
SELECT COUNT(*) FROM ml.socio_market_feature_store_train_v;
```

## 8.4 Trainer attachment (12 columns)

The trainer is expected to log:

`centrality_class, miss_kommune, miss_socio, price_months_income, log_price_to_income_cap,
 aff_resid_in_seg, aff_decile_in_seg, rel_log_price_to_comp_mean30_super,
 rel_log_price_to_comp_mean30_kommune, rel_price_best, rel_price_source, miss_rel_price`

---

# 9. Troubleshooting

## 9.1 “Column does not exist” for affordability columns
Cause:
- The wide MV enumerates columns instead of selecting `s.*`, so new SOCIO columns do not propagate.

Fix:
- Rebuild the wide MV using `SELECT s.*` (see `sql/04_create_socio_market_wide_mv.sql`).

## 9.2 “cannot drop matview because other objects depend on it”
Cause:
- Entry point views depend on the wide MV.

Fix:
- Drop the dependent views first:
  - `DROP VIEW ml.socio_market_feature_store_train_v;`
  - `DROP VIEW ml.socio_market_feature_store_t0_v1_v;`
- Then drop/recreate the matview.

## 9.3 “CERT GUARD FAIL: viewdef drift detected …”
Cause:
- You changed a view/matview definition after baselining.

Fix:
- Delete and repopulate `audit.t0_viewdef_baseline` for the entrypoint.
- Recompute dataset hash baselines.
- Rerun certification procedure.

## 9.4 “T0 CERT FAIL: time-of-query tokens detected in closure”
Cause:
- A dependency still references `CURRENT_DATE` / `now()` etc.

Fix:
- Run a closure token scan and rebuild/rewire the offending MV(s).

## 9.5 “baselines=0”
Cause:
- You ran the certification runner without populating dataset baselines.

Fix:
- Populate 365-day baselines with `sql/09_baseline_dataset_hashes_365.sql`.
