# 02 — Operating Procedure (Daily/Dev)

## Goal
Before training, guarantee all guarded entrypoints required by the trainer are:

- CERTIFIED in audit.t0_cert_registry
- within the guard window (usually 24h)
- drift policy applied (last N days allowed/rebaselined for backfill-prone stores)

## Pre-train
1) Run your refresh phase (MATVIEW refresh + thin table rebuilds).
2) Run the certification block (sql/04_certify_all_stores_dev_10d.sql).

## Post-train (WOE)
WOE is keyed by active model_key and changes after training.
After training completes, run sql/05_post_train_woe_recert.sql.

## Recommended defaults
- sample_limit = 2000
- N=10 for daily dev loops
- keep 365 baselines stored for audit

