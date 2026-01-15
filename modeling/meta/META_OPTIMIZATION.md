# Two-Stage, Leak-Free Meta-Optimization (Public Release)

This document describes a **generic, marketplace-agnostic** “meta-learner” that tunes a two-stage policy for **fast-from-slow selection**:

- **Stage-1 (selector)**: calibrate a fast-within-24h probability, apply soft gates, and tune a joint objective to pick a *flagged* candidate set.
- **Stage-2 (promoter / “mask-23”)**: within the Stage-1 flagged set, learn a constrained flip policy that promotes a limited subset to a fixed **23h** prediction under caps and penalties, using a **time-split** to avoid leakage.
- **Freeze → Apply-only**: save the best knobs from tuning, then run an apply-only evaluation on an untouched slice.

This public release is intentionally **connector-free**: it does not include any target-specific ingestion, scraping, platform headers, cookies, or real listing data. You bring your own features and base model predictions.

---

## Repository artifacts

### Primary script

- `fast24_flagger3_public.py` — reference implementation of:
  - feature selection & leakage guards
  - calibration (Platt / Isotonic / none)
  - Stage-1 Optuna tuning over calibration + gates + threshold
  - Stage-2 Optuna tuning (“mask-23”) with caps + barriers + (optional) lightweight ranker
  - deterministic freeze-and-apply pipeline

### Outputs (written to `--outdir`)

Stage-1:
- `best_params.json` — best Stage-1 knobs
- `sweep.csv` / logs — trial metrics (depending on enabled logging)
- candidate exports (e.g., `flagged_min.csv`)

Stage-2:
- `mask23_best_params.json` — best Stage-2 knobs
- `mask23_trials_log.csv` — Stage-2 telemetry
- `preds_with_mask23.csv`, `best_mask.csv`, `mask23_summary.json` (if enabled)

---

## Data contract (public template)

The script expects CSV inputs that contain:
- `listing_id` (stable identifier for joining / reporting)
- a duration column (e.g., hours-to-outcome) for evaluation when available
- one or more base model prediction columns (probability and/or predicted duration)
- optional structured features (numeric/categorical)
- optional embeddings (columns like `embed32_*`, `embed64_*`, `embed64w_*`)

This template intentionally does not prescribe *how* you create the feature set.

---

## Configuration (portable defaults)

For public release, default file locations are **portable** and can be overridden by environment variables:

- `DATA_DIR` (default: `./data`)
- `TRAIN_CSV` (default: `./data/train.csv`)
- `SLOW_CSV` (default: `./data/val_slow.csv`)
- `VALTEXT_CSV` (default: `./data/val_text.csv`)
- `OOF_CSV` (default: `./data/oof_predictions.csv`)
- `OUT_DIR` (default: `./data/out`)

You can also override paths directly via CLI flags where supported.

---

## Operating guide

### 1) Stage-1 + Stage-2 tuning (example)

```bash
python fast24_flagger3_public.py \
  --emb all \
  --tune 500 \
  --tune_mask23 200 \
  --tune_mask23_per_trial 40 \
  --slow "./data/val_slow.csv" \
  --outdir "./out/S1S2_tune_run" \
  --no_current_threshold \
  --fixed_prob_threshold 0.75
```

Artifacts (example):
- `./out/S1S2_tune_run/best_params.json`
- `./out/S1S2_tune_run/mask23_best_params.json`

### 2) Frozen apply-only run (example)

```bash
python fast24_flagger3_public.py \
  --emb all \
  --tune 0 --tune_mask23 0 --tune_mask23_per_trial 0 \
  --outdir "./out/178_frozen_eval" \
  --no_current_threshold \
  --fixed_prob_threshold 0.75 \
  --load_params "./out/S1S2_tune_run/best_params.json" \
  --load_mask23_params "./out/S1S2_tune_run/mask23_best_params.json"
```

Outputs (apply-only):
- `preds_with_mask23.csv` (full rows + flip bits)
- `best_mask.csv` (only flipped rows)
- `mask23_summary.json` / `mask23_summary.csv` (high-level metrics, if enabled)

---

## Public release notes

Included:
- two-stage optimization pattern (selector → constrained promoter)
- leak-free discipline (time-split Stage-2 + freeze-and-apply)
- artifacts for auditability (JSON knobs, per-trial telemetry, row-level exports)

Intentionally omitted:
- any marketplace-specific connector, scraping logic, request fingerprints, cookies, headers, endpoints, or HTML/JSON parsing
- any real listing identifiers or example rows derived from production data
- any secrets, credentials, or hard-coded infrastructure endpoints

