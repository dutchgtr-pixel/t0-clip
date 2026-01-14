# 4. T0 + leak-proof contract (why this store is certifiable)

## 4.1 T0 definition

For each listing row, `t0` is the listing’s `edited_date` timestamp.
All socio and market features must be computable using only information available at or before that timestamp.

## 4.2 Leakage categories guarded against

1) Time-of-query leakage
- banned tokens: `CURRENT_DATE`, `now()`, `clock_timestamp()`, etc.
- enforcement: closure token scan inside the certification assertion

2) Structural time leakage (“future join”)
- as-of socio joins enforce: snapshot_date <= t0::date
- market comps enforce: comp_max_t0_* < t0 (strict trailing windows)

3) Label leakage
- store does not reference outcomes (sold_date, duration, status_final, etc.)

4) Determinism / drift
- dynamic invariance tested via dataset hashes for fixed historical t0_day slices

## 4.3 Structural invariants

### SOCIO as-of invariant
Must hold:
- `postal_kommune_snapshot_date <= t0::date` when non-null
- `socio_snapshot_date <= t0::date` when non-null

### MARKET strict trailing window invariant
Must hold:
- `comp_max_t0_super_metro < t0` when non-null
- `comp_max_t0_kommune < t0` when non-null

These are asserted both in:
- runbook checks, and
- certification assertion function.

## 4.4 Why the affordability features are T0-safe

All affordability columns depend only on:
- listing price at T0
- income snapshot as-of T0
- within-segment statistics computed from the MV output at refresh time

The winsor bounds are pinned constants, not computed at query time, so they do not introduce time-of-query dependence.

