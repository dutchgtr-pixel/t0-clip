# Anchor Priors — Integration Notes

This repository includes two “query fragment” anchors implemented as correlated `LEFT JOIN LATERAL (...)` blocks:

- `sql/03_strict_anchor_lateral.sql`
- `sql/04_advanced_anchor_v1_lateral.sql`

## How to use the fragments
1) Build a base query that emits one row per listing at the reference time (T0).
   In the fragments, this base row is aliased as `b`.

2) Ensure the base row provides the required columns referenced by the fragments, for example:
   - `b.listing_id`
   - `b.edited_date`
   - cohort keys such as `b.generation`, `b.model_norm`, `b.storage_gb`, `b.sbucket`, etc.

3) Paste the fragment into your SELECT as a `LEFT JOIN LATERAL (...) <alias> ON TRUE`
   and select the resulting anchor columns.

## Optional: materialization
If you want strict/advanced anchors to be DB-materialized (instead of computed per row), you can:
- turn the fragments into SQL functions that accept `(listing_id, t0, cohort keys…)`, or
- generate an “as-of” MV keyed by `(anchor_day, cohort keys…)`.

The speed anchors already follow the MV approach (`sql/01_speed_anchor_asof.sql`).
