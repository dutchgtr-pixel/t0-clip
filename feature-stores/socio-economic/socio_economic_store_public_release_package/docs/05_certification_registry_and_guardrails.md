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

