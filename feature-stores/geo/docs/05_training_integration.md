# 5. Training Integration

## 5.1 What to use in training CLI

Replace:

`--geo_dim_view ml.geo_dim_super_metro_v4_current_v`

With:

`--geo_dim_view ml.geo_dim_super_metro_v4_t0_train_v`

This makes training fail closed if geo certification is stale/failed.

## 5.2 Refresh runbook integration

Geo store has no matviews to refresh by default (views only). The runbook needs:
- the geo certification call (and guard check)
- optionally, baseline rebuild if policy requires frequent baseline updates

Recommended:
- Keep 365 stored baselines (as you have)
- Daily: drift check on last 10 days (fast)
- Weekly/monthly: drift check on 365 days (deep audit)

