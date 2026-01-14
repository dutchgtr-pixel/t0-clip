# Change management

This store is treated like any other certified store: **fail closed** on drift or definition changes.

## When you change feature logic
Examples:
- thresholds (e.g., conflict threshold)
- adding/removing derived columns
- changing upstream dependencies

Required steps:
1. Update the logical definition (`ml.trainer_derived_features_v1`)
2. Refresh the MV (`REFRESH MATERIALIZED VIEW CONCURRENTLY ml.trainer_derived_features_v1_mv;`)
3. Capture new viewdef baselines:
   - `CALL audit.capture_viewdef_baseline('ml.trainer_derived_feature_store_t0_v1_v'::regclass);`
4. Rebaseline last N days (if you allow drift in recent history):
   - `CALL audit.rebaseline_last_n_days('ml.trainer_derived_feature_store_t0_v1_v'::regclass, 10, 2000);`
5. Certify:
   - `CALL audit.run_t0_cert_trainer_derived_store_v1(10);`
6. Consume via guarded view only (`ml.trainer_derived_features_train_v`)

## Expected failure mode
If the view definition changes but baselines are not recaptured, the strict guard should fail. This is desired behavior.

