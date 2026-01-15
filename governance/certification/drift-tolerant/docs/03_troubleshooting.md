# 03 — Troubleshooting

## Certification stale
If `require_certified_strict` fails with “stale”, rerun certification block.

## Drift failures
If a store is strict and drift exists:
- run sql/05_drift_diagnostics_queries.sql to see which days drifted
- decide: allow drift (rebaseline) or treat as bug/leakage

device_meta is expected to drift during image backfills; use drift-allowed runner.

## Viewdef drift
If you rebuild views/MVs, refresh the viewdef baseline for that entrypoint closure.

## Performance
Most time is MV refresh time. Consider:
- non-concurrent refresh in dev
- running heavy MVs (e.g., orderbook) less frequently

