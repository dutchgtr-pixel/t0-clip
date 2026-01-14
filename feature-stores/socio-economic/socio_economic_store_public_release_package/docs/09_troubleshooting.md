# 9. Troubleshooting

## 9.1 “Column does not exist” for affordability columns
Cause:
- The wide MV enumerates columns instead of selecting `s.*`, so new SOCIO columns do not propagate.

Fix:
- Rebuild the wide MV using `SELECT s.*` (see `sql/04_create_socio_market_wide_mv.sql`).

## 9.2 “cannot drop matview because other objects depend on it”
Cause:
- Entry point views depend on the wide MV.

Fix:
- Drop the dependent views first:
  - `DROP VIEW ml.socio_market_feature_store_train_v;`
  - `DROP VIEW ml.socio_market_feature_store_t0_v1_v;`
- Then drop/recreate the matview.

## 9.3 “CERT GUARD FAIL: viewdef drift detected …”
Cause:
- You changed a view/matview definition after baselining.

Fix:
- Delete and repopulate `audit.t0_viewdef_baseline` for the entrypoint.
- Recompute dataset hash baselines.
- Rerun certification procedure.

## 9.4 “T0 CERT FAIL: time-of-query tokens detected in closure”
Cause:
- A dependency still references `CURRENT_DATE` / `now()` etc.

Fix:
- Run a closure token scan and rebuild/rewire the offending MV(s).

## 9.5 “baselines=0”
Cause:
- You ran the certification runner without populating dataset baselines.

Fix:
- Populate 365-day baselines with `sql/09_baseline_dataset_hashes_365.sql`.

