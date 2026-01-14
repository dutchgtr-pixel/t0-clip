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
