# Public Device Meta Feature Store (T0-anchored)

This package defines a **time-safe (as-of) device-meta feature store** in PostgreSQL.

The key design choice is that population statistics (segment/color frequencies, rarity, ranks, diversity metrics) are computed as **T0-anchored features**: for a given anchor day `D`, statistics are computed only from rows with `t0_day < D`. This prevents historical feature values from drifting as new data arrives.

## What is included

- A per-listing base view with normalized model and color buckets.
- A materialized view that computes as-of segment/color statistics.
- A final encoded view that joins per-listing features with as-of population stats.

## What is intentionally omitted

- Any marketplace-specific connector logic, scraping logic, endpoints, HTML selectors, cookies, or request fingerprints.
- Any real listing identifiers or example data derived from real listings.
- Any secrets, credentials, tokens, DSNs, or private endpoints.

## Objects created

### 0) `feature_store.v_device_meta_base_t0_v1` (view)

A **pure per-listing** device-meta view with:

- `listing_id`, `t0`, `t0_day`
- `gen_key` (effective generation key)
- `model_variant` (normalized variant bucket)
- `color_bucket` (bounded color domain with `_MISSING` and `_OTHER`)
- image evidence and voting fields used downstream

This layer contains **no population aggregates**.

### 1) `feature_store.device_meta_gmc_stats_t0_v1_mv` (materialized view)

Computes as-of population statistics keyed by:

- `t0_day`
- `gen_key`
- `model_variant`
- `color_bucket`

The MV uses window frames ending at `1 PRECEDING` to enforce the invariant:

> For anchor day `D`, statistics are based only on earlier anchor days (`t0_day < D`).

### 2) `feature_store.device_meta_encoded_t0_v1` (view)

Final “training-ready” view that:

- preserves a stable column contract for downstream consumers
- left-joins the as-of MV using `(t0_day, gen_key, model_variant, color_bucket)`

## Refresh runbook

Because the population statistics live in a materialized view, refresh it after loading new upstream data:

```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY feature_store.device_meta_gmc_stats_t0_v1_mv;
ANALYZE feature_store.device_meta_gmc_stats_t0_v1_mv;
```

The encoded feature store is a view and does not refresh.

## Upstream compatibility

The SQL expects two upstream relations:

- `feature_store.listing_device_features_clean_mv`
- `feature_store.device_image_features_unified_v1`

If your column names differ, create small compatibility views so the feature-store SQL remains unchanged.

