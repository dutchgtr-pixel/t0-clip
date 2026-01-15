# 4. Python Implementation (OOF + Persistence)

This section describes what the training script does when `disable_woe_oof=false` and `disable_woe_persist=false`.

## 4.1 Training slice used for WOE

WOE is trained on **TRAIN SOLD** rows only.
Your training pipeline constructs these splits:

- `train_sold`: SOLD rows with sold_time <= eval_cutoff (older than eval window)
- `eval_sold_last_7d`: recent SOLD rows for evaluation
- `censored_ge_21d`: censored rows (active/inactive) that have survived >= 21 days

The WOE binary label is:
- `y_slow21 = 1` if duration_hours >= 504 (21d) (or final_status older21days)
- else `0`

## 4.2 OOF (out-of-fold) scoring: no self-influence leakage

To avoid target-encoding leakage on TRAIN SOLD rows:

1. Assign folds by **t0_day** (blocked by day), not by row.
2. For each fold k:
   - fit WOE tables on TRAIN folds (days not in fold k)
   - score rows in fold k
3. Replace `woe_anchor_p_slow21` and `woe_anchor_logit_slow21` for TRAIN SOLD rows with OOF scores.

The fold assignment is deterministic and day-blocked:

- take unique `t0_day` values sorted
- map them into approximately equal-size contiguous blocks:
  `fid = floor(i * n_folds / n_days)`

This ensures temporal locality and reduces leakage via near-duplicate rows on the same day.

## 4.3 Final mapping (for inference)

After OOF scoring, the script fits one final mapping on all TRAIN SOLD rows.
This mapping is what is persisted with `fold_id IS NULL` and used for inference scoring views.

## 4.4 What gets persisted to Postgres

Once final + fold mappings exist, the trainer writes:

1) Registry row (`ml.woe_anchor_model_registry_v1`):
- `model_key`
- `train_cutoff_ts`
- `half_life_days`, `eps`, `n_folds`, `band_schema_version`
- `base_rate`, `base_logit`
- `dsold thresholds`
- `is_active=true` (and clears older actives)

2) Presentation cuts (`ml.woe_anchor_cuts_v1`):
- final: `(model_key, fold_id NULL, c1, c2)`
- folds: `(model_key, fold_id=k, c1, c2)` (optional but persisted for audit)

3) WOE maps (`ml.woe_anchor_map_v1`):
- final: `(model_key, fold_id NULL, band_name, band_value) -> woe`
- folds: `(model_key, fold_id=k, ...) -> woe`

4) OOF row-level scores (`ml.woe_anchor_scores_v1`):
- keyed by `(model_key, generation, listing_id, t0)`
- includes `fold_id` and `is_oof=true`

## 4.5 How to confirm persistence happened

The training log prints:
- `[db] persisted WOE anchor artifacts v1: model_key=... (active=True)`
- `[woe] persisted WOE anchor artifacts to Postgres ...`

And in SQL:
- `SELECT * FROM ml.woe_anchor_model_registry_v1 WHERE is_active=true;`
- `SELECT COUNT(*) FROM ml.woe_anchor_map_v1 WHERE fold_id IS NULL;`

