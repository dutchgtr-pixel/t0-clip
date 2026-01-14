# 3. How the Anchor Priors Are Generated

This package provides two implementation patterns:

## 3.1 Database materialization (speed anchors)
The “speed anchors” are produced as a Postgres **materialized view** that is keyed by
`anchor_day` and uses an explicit embargo rule:

- only SOLD rows with `sold_day < anchor_day` are eligible.

Because the view is precomputed for each day in a date range, it supports efficient joins during both
training and inference without relying on time-of-query functions such as `CURRENT_DATE`.

See:
- `sql/01_speed_anchor_asof.sql`
- `sql/01_speed_anchor_asof.md`

## 3.2 Per-row computation (strict + advanced anchors)
The strict and advanced anchors are written as correlated `LEFT JOIN LATERAL (...)` SQL fragments.
They are intended to run inside a dataset assembly query that iterates over “base rows” (one row per listing at T0).

Both fragments use an explicit embargo:

- `sold_date::date < t0 - interval '5 days'`

This makes them safe to compute at training time or inference time, provided the base row’s T0 is defined consistently.

See:
- `sql/03_strict_anchor_lateral.sql`
- `sql/04_advanced_anchor_v1_lateral.sql`

## 3.3 Why this is leak-safe even though SOLD outcomes are used
Leak safety is achieved by *anchoring to T0* and enforcing an embargo. The anchor computations:
- never reference future statuses/outcomes relative to the active row’s T0,
- do not rely on time-of-query tokens (`CURRENT_DATE`, `now()`, etc.),
- use deterministic aggregation (medians, counts) and deterministic fallback selection.
