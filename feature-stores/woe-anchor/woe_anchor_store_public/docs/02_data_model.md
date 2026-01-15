# 2. Data Model: DB Objects and Contracts

## 2.1 Artifact tables (persistent)

### 2.1.1 `ml.woe_anchor_model_registry_v1`

One row per trained WOE mapping package.

Key columns:
- `model_key` (PK): unique run ID, e.g. `tom_aft_v1_YYYYMMDD_HHMMSS`.
- `train_cutoff_ts`: training cutoff timestamp used for defining train/eval split (e.g. eval_cutoff).
- `half_life_days`, `eps`, `n_folds`: WOE learning hyperparameters.
- `band_schema_version`: integer schema/version for band definitions (for forward compatibility).
- `code_sha256`: SHA256 of the training script used to compute artifacts.
- `is_active`: boolean pointer; exactly one row should be active for live inference.
- `base_rate`, `base_logit`: base slow21 rate and log-odds.
- `dsold_t1..dsold_t4`: thresholds used to band `delta_vs_sold_median_30d`.

### 2.1.2 `ml.woe_anchor_cuts_v1`

Stores band cutpoints for presentation score.
- `fold_id IS NULL` denotes the **final** mapping cuts.
- `fold_id = 0..K-1` denotes per-fold cuts (optional; stored for audit).

Uniqueness is enforced via partial unique indexes (see SQL scripts).

### 2.1.3 `ml.woe_anchor_map_v1`

Stores the WOE mapping tables as categorical contributions.
- `band_name` identifies the band type (dsold/trust/ship/sale/cond/dmg/bat/present/vdmg/acc).
- `band_value` is the category key (e.g., `dSold_under`, `HIGH`, `shipping_ok`, `present_hi`, ...).
- `woe` is the learned log-odds contribution for that band value.
- `fold_id IS NULL` is the final mapping; fold IDs store OOF fold mappings.

### 2.1.4 `ml.woe_anchor_scores_v1`

Stores row-level scores (optional but useful):
- Used to persist OOF scores for TRAIN SOLD rows (`is_oof=true`).
- Primary key: `(model_key, generation, listing_id, t0)`.

## 2.2 Scoring views (SQL inference)

### 2.2.1 `ml.v_woe_anchor_bands_live_v1`

A deterministic banding view that produces the categorical “band_value” keys for each listing using
**certified feature stores** as inputs:
- `ml.socio_market_feature_store_train_v` (T0 safe, guarded)
- `ml.iphone_image_features_unified_v1_train_v` (vision store, guarded)

Outputs include:
- `dsold_band`, `trust_tier`, `ship_band`, `sale_band`, `cond_band`, `dmg_ai_band`, `bat_band`
- `presentation_score`, `presentation_band`, `vdmg_band`, `accessories_band`

It also loads the active model’s:
- `dsold thresholds` and `base_logit`
- `presentation cuts (c1/c2)` from `ml.woe_anchor_cuts_v1`

### 2.2.2 `ml.woe_anchor_scores_live_v1`

Computes:
- `woe_logit = base_logit + Σ map_woe(band_name, band_value)`
- `woe_anchor_p_slow21 = sigmoid(woe_logit)`

### 2.2.3 `ml.woe_anchor_scores_live_train_v`

Fail-closed guarded inference view.
It calls `audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours')`.

### 2.2.4 `ml.woe_anchor_feature_store_t0_v1_v` (cert entrypoint)

A certified entrypoint used for dataset hashing and certification registry.
It adds `edited_date` (alias of `t0`) to satisfy `audit.dataset_sha256` expectations.

