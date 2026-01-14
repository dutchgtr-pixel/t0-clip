# 3. T0 and Leak-Proof Strategy

## 3.1 What “T0” means for geo

T0 for your ML rows is the listing `edited_date` (event time / as-of time).
Geo mapping features must be computable at T0 and deterministic under certification.

Geo mapping itself is not label-derived, but it can drift because:
- a release changes,
- mapping tables change,
- normalization rules change,
- listing postal/city fields are backfilled.

## 3.2 Pinning policy

For training determinism, certification pins a release_id:
- Training uses `ref.geo_mapping_pinned_super_metro_v4_v1.release_id`
- This decouples training from `ref.geo_mapping_current` changes.

## 3.3 Why we used `ml.tom_features_v1_mv` as base

Earlier attempts used `marketplace.listings_raw` as base, which can contain duplicate listing_id rows.
That produced duplicate listing_id keys in the geo dim view and broke certification.

Using `ml.tom_features_v1_mv` ensures 1 row per listing_id and resolves duplicates.

## 3.4 Certification boundary and drift policy

You stored dataset hash baselines for **365 days** (sample_limit=2000) and certified with `p_check_days=365`.
This is strict: any backfill that changes postal/city history can fail drift.

You can choose to run daily drift checks on last 10 days for speed, while keeping 365 stored baselines.
