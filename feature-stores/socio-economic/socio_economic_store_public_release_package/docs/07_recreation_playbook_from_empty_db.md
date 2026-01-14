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

