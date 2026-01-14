# 4. Leak-Safety and Reproducibility Notes (anchor_priors_store)

This document captures the practical checks that make anchor priors safe to use for training and inference.

## 4.1 Core leak-safety conditions

### A) Explicit “as-of” anchoring
Every anchor prior must be computed **as-of** a reference time T0 for the active row.

- For per-row anchors, T0 is typically `b.edited_date::date`.
- For the speed anchor MV, T0 is represented as `anchor_day`.

### B) Embargo windows
All SOLD comparables must satisfy an embargo that prevents near-T0 and post-T0 contamination.

Examples in this repo:
- Strict/advanced anchors: `sold_date::date < t0 - interval '5 days'`
- Speed anchors: `sold_day < anchor_day`

### C) No time-of-query dependence
Anchor computations should not depend on tokens such as:
- `CURRENT_DATE`, `CURRENT_TIMESTAMP`
- `now()`, `clock_timestamp()`, `transaction_timestamp()`, etc.

## 4.2 Practical reproducibility guidance (recommended)
If you operate a regulated or highly reproducible pipeline, consider adding:

- **Token scans** for prohibited time-of-query functions on all dependent views/MVs.
- **View-definition baselines** (hash the viewdef and alert on drift).
- **Input dataset baselines** (hash key upstream tables/views used by the anchor computation).

## 4.3 Verification queries
See `sql/05_verification_queries.sql` for:
- embargo correctness spot-checks,
- token scans for time-of-query dependence.
