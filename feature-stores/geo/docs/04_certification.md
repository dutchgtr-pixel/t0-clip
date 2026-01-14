# 4. Certification End-to-End

## 4.1 Baseline components

Certification uses the same two baseline systems as your other stores:

1) Viewdef baseline (`audit.t0_viewdef_baseline`)
- Hashes the closure of view/matview definitions under the entrypoint.
- Prevents silent logic drift.

2) Dataset hash baselines (`audit.t0_dataset_hash_baseline`)
- Stores SHA256 of a deterministic sample of rows for specific `t0_day` values.
- You store 365 days, check drift on p_check_days.

## 4.2 Certification procedure

- `audit.assert_t0_certified_geo_store_v1(p_check_days)`
  - asserts pinned release exists (1 row)
  - asserts no time-of-query tokens in closure
  - asserts unique listing_id keys in `ml.geo_dim_super_metro_v4_t0_v1`
  - asserts drift count = 0 over last p_check_days baselines

- `audit.run_t0_cert_geo_store_v1(p_check_days)`
  - runs assertions
  - writes `audit.t0_cert_registry` entries for:
    - `ml.geo_feature_store_t0_v1_v`
    - inherited alias entries for `ml.geo_dim_super_metro_v4_t0_train_v` and `ml.geo_dim_super_metro_v4_t0_v1`

## 4.3 Operational runbook notes

- Pinning is change-managed (do not repin daily automatically unless you intend to change training geo).
- When you load a new release and decide to switch, do:
  1) switch current release
  2) pin the new release id (CREATE OR REPLACE)
  3) rebuild viewdef baselines + dataset baselines
  4) re-certify

