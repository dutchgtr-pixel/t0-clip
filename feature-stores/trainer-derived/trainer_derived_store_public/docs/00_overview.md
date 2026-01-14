# Trainer Derived Feature Store: Overview

## Purpose
The *trainer derived feature store* is a **T0-safe, reproducible, registry-certified** feature block that contains features which are often computed ad-hoc at training time in application code:

- calendar / seasonality flags
- trailing activity counts
- deterministic rule features
- cross-modal consistency signals (e.g., comparing text-extracted vs image-extracted attributes)

The store moves these computations into PostgreSQL where they can be:
- versioned as SQL objects
- materialized for fast training joins
- certified via definition hashes and dataset hashes
- **fail-closed** at consumption time if drift is detected

## What “T0-safe” means in this package
Each row is anchored to an event time `t0` (timestamptz). Features must be computed using only information available **at or before** that row’s `t0`.

Key invariants implemented in the SQL:
1. **No time-of-query tokens** in feature logic (avoid `now()`, `current_date`, etc.).
2. **No self-influence** in trailing windows:
   - trailing counts exclude the current row by using:
     `RANGE BETWEEN INTERVAL '30 days' PRECEDING AND INTERVAL '00:00:00.000001' PRECEDING`
3. **Certified dependency chain**:
   - the derived store depends on upstream *training-safe surfaces* (e.g., base features and image features) and is guarded by certification.

## Why materialize
The derived store is backed by a **materialized view** with a unique index on `(generation, listing_id, t0)` to support:
- fast point-lookups during training
- `REFRESH MATERIALIZED VIEW CONCURRENTLY`
- stable execution plans

## Certification and guard
The store is consumed via a guarded view that calls:

- `audit.require_certified_strict(entrypoint, max_age)`

If certification has not been performed, is stale, or failed, consumption errors out (fail-closed).

