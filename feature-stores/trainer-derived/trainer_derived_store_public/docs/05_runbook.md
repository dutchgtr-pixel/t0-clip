# Operational runbook

## Daily (fast / drift-tolerant)
Typical daily cycle before model training:

1) Refresh MV
```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.trainer_derived_features_v1_mv;
ANALYZE ml.trainer_derived_features_v1_mv;
```

2) Baseline + certify + enforce
```sql
\i sql/03_certify_trainer_derived_store_public.sql
```

This script:
- captures viewdef baselines for the entrypoint closure
- rebaselines the last 10 days (configurable)
- certifies the last 10 days (configurable)
- validates strict guard

## Weekly (deep)
Optional deeper audit:
- rebuild longer baseline history (e.g., 365 days)
- certify with a larger `p_check_days`

## Common failure modes

### A) Viewdef drift detected
Cause:
- feature logic changed without recapturing baselines

Fix:
- run the viewdef baseline capture step and re-certify

### B) Dataset hash drift detected
Cause:
- upstream backfill changed recent derived values

Fix:
- rebaseline last N days, then re-certify
- if drift extends beyond your allowed N-day window, investigate upstream changes

