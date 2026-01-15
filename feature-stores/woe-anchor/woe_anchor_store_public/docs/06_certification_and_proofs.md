# 6. Certification and Proofs

## 6.1 What is certified?

Entrypoint (certified surface):
- `ml.woe_anchor_feature_store_t0_v1_v`

Guarded consumer view (fail-closed):
- `ml.woe_anchor_scores_live_train_v`

Alias / raw scoring view:
- `ml.woe_anchor_scores_live_v1`

## 6.2 Certification gates

### Gate A: Viewdef baselines
- `audit.t0_viewdef_baseline` stores SHA256 hashes for every object in the view/matview closure.
- Drift in viewdefs causes strict guard to fail.

### Gate C: Dataset hash baselines
- `audit.t0_dataset_hash_baseline` stores dataset hashes for a bounded set of `t0_day` values.
- Policy here: last **10** days only for speed.
- Drift check is limited to those 10 days.

### Artifact assertions
`audit.assert_t0_certified_woe_anchor_store_v1()` enforces:
- there is an active model_key
- exactly 1 final cuts row exists (fold_id IS NULL)
- final mapping rows exist
- at least 7 band groups exist in final mapping

## 6.3 Registry recording
`audit.run_t0_cert_woe_anchor_store_v1(10)` writes:

- `audit.t0_cert_registry` row for `ml.woe_anchor_feature_store_t0_v1_v` as `CERTIFIED`
- `audit.t0_cert_registry` alias rows for:
  - `ml.woe_anchor_scores_live_train_v`
  - `ml.woe_anchor_scores_live_v1`

## 6.4 Proof queries

See `proofs/` for copy/paste verification queries and expected results.

