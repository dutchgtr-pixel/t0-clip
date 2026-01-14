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

