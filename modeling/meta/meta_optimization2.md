**Two-Stage, Leak-Free Meta-Optimization for Fast-from-Slow Selection**

**Design, math, and operating guide**

**0) What this pipeline solves**

You have a mixed pool of "predicted-slow" listings and you want to
**pull truly fast rows out of that slow pool** while **avoiding slow
mistakes** and **improving duration regression**. The pipeline does this
with a **two-stage meta-optimizer**:

-   **Stage-1 (candidate scorer & gate + Optuna)**: builds a calibrated
    probability of "fast within 24h", applies soft business/economic
    gates, and tunes a **joint objective** that rewards fast flips and
    penalizes slow flips (plus optional regression deltas), producing
    the set of **flagged** candidates.

-   **Stage-2 (mask-23 policy + Optuna)**: on the flagged set, learns a
    **flip policy** that promotes a subset to a 23-hour prediction (the
    "mask-23" flip), under hard caps and penalties, using a
    **time-split** so it learns on earlier rows and is **audited on
    later rows** without peeking.

Then you **freeze the knobs** (best parameters) and run an
**apply-only** pass on an untouched validation slice (e.g., the 178
"slow bucket"), giving **unbiased results**.

**1) Data slices & leak-free discipline**

-   **Training pool** → provides the feature space, embeddings, anchors,
    and model calibration.

-   **Held-out 20% "tuning slice" (SOLD & predicted-slow)** → used by
    Optuna in both stages:

    -   **Stage-1** uses this slice to score many joint policies (no
        label-fit thresholding).

    -   **Stage-2** uses an inline **time split**: tune on **earlier**
        flagged rows (durations known), then **apply to later** flagged
        rows (durations not used to select), so it's leak-resistant on
        the same slice.

-   **Untouched validation set (e.g., 178 rows)** → **never** used in
    tuning. After freezing knobs, you run apply-only and score
    truthfully. This is your **trust** set.

**2) Stage-1: calibrated candidate scorer & joint objective**

**2.1 Calibration + embeddings + soft gates**

-   Probabilities are **calibrated** (Platt or Isotonic) over
    cross-validation splits.

-   Feature space includes the **embeddings and numeric anchors**
    (price-to-value, deltas vs model/time medians, PPGB, storage, seller
    quality, warranty/receipt, etc.).

-   **Soft gates** ("economic" & "quality" filters) are controlled by
    the G\_\* family (e.g., G_MIN_COND, G_MAX_DAMAGE, G_MIN_BATT, PTV
    limits, underpricing deltas). Optuna tunes these too.

**2.2 Label-free thresholding**

Stage-1 uses a **fixed label-free cutoff** (\--no_current_threshold +
\--fixed_prob_threshold τ) or a tuned, label-free threshold window. The
aim: **don't fit a threshold to labels**; keep it reproducible.

**2.3 The Stage-1 joint objective (what Optuna maximizes)**

Define:

-   For the **flagged** pool on the tuning slice:\
    TP = true fasts captured, FP = slow mistakes, MID = mid (40--72h).

-   Optional regression deltas are computed relative to a baseline
    prediction (MAE, RMSE, R²).

The objective is:

J_stage1 = w_tp_fp·(TP − FP)

\+ w_fast·(# fast flips)

− w_slow·(# slow flips)

− w_mid ·(# mid flips)

\+ w_MAE · ΔMAE

\+ w_RMSE· ΔRMSE

\+ w_R2 · ΔR2

-   The w\_\* weights come from CLI flags (\--joint_w\_\*) and let you
    bias search strongly against slow flips and in favor of fast
    improvements (and optionally reward regression gains).

-   There's also an optional **soft barrier** to down-score policies
    that exceed minimum gain/quality caps or violate guardrails.

Outcome: Stage-1 returns a best policy (best_params.json) and a set of
**flagged** rows for Stage-2 to audit.

**3) Stage-2: mask-23 flip policy (time-split + caps)**

**3.1 Policy surface (the "mask-23" knobs)**

A flip policy is parameterized by:

-   p_threshold --- minimum calibrated fast probability (on the flagged
    set).

-   delta_overall_max, ptv_overall_max --- price delta & price-to-value
    ceilings.

-   cond_min, damage_max, batt_min --- quality gates (e.g., condition
    score, damage severity, battery %).

-   dpm14_max --- model-anchor delta bound.

-   flip_share --- target share of flips among the flagged pool (subject
    to floors and caps).

**Caps / constraints**

-   slow_cap_abs (absolute max slow flips)

-   slow_cap_ratio (max slow flips as % of flips)

-   Floors like min_share, min_flips can be used to avoid degenerate
    tiny selections.

**3.2 Time-split audit (leak-free on the tuning slice)**

For each trial:

-   **Train side (earlier flagged rows)**: knobs are picked using their
    durations.

-   **Audit side (later flagged rows)**: the policy is **applied**, and
    only then durations are revealed to compute audit metrics. No
    peeking during selection.

**3.3 Stage-2 objective**

Let m_i ∈ {0,1} mark a flip for row i in the audit half. Define:

-   fast_i = 1{duration_i ≤ 40}, slow_i = 1{duration_i \> 72}, mid_i =
    1{40 \< duration_i ≤ 72}.

-   K = Σ m_i (number of flips).

-   Baseline prediction ŷ_i and flipped prediction ŷ_i\' = 23 if m_i = 1
    else ŷ_i.

The policy value is:

J_stage2 = W_FAST · Σ_i m_i·fast_i

− W_SLOW · Σ_i m_i·slow_i

− W_MID · Σ_i m_i·mid_i

\+ W_MAE · (MAE_baseline − MAE_mask23)

\+ W_RMSE · (RMSE_baseline − RMSE_mask23)

\+ W_R2 · (R2_mask23 − R2_baseline)

− Barrier(slow_count, slow_cap_abs, slow_cap_ratio; power)

The **Barrier** is a large penalty that grows super-linearly if slow
flips exceed caps:

Barrier = 0 if slow ≤ min(slow_cap_abs, slow_cap_ratio·K)

= λ · (excess)\^power otherwise

(λ absorbed into W_SLOW or a separate constant.)

Outcome: Stage-2 gives a best policy (mask23_best_params.json) and full
telemetry (mask23_trials_log.csv, audit metrics, the flipped rows csv,
etc.).

**4) Freezing & apply-only on the untouched 178**

**Freeze**:

-   Stage-1: best_params.json (from your 20% tuning slice)

-   Stage-2: mask23_best_params.json (from the same slice's time-split
    tuner)

**Apply-only run** on 178:

1.  Load Stage-1 knobs (\--load_params) → build features, calibrate,
    gate, **flag**.

2.  Load Stage-2 knobs (\--load_mask23_params) → **apply** mask-23 once
    to the flagged set (no tuning, no refit).

This produces preds_with_mask23.csv, best_mask.csv, and a summary
(mask23_summary.json and mask23_summary.csv if enabled).\
The reported \"train_frac\": 0.4 in summary is **metadata only**; **no
training occurs** in apply-only.

**Your observed result on the 178** (example):\
Flips ≈ 10, **fast ≈ 8**, **slow ≈ 1**, mid ≈ 1; MAE improved by ≈ 2.4
h.\
This is **clean**, **frozen**, **not selection-biased**, and **not
optimistic**---because the 178 records were never seen in tuning.

**5) Architecture (high level)**

**Ingestion & Features**

-   Structured features, anchors, deltas, embeddings

-   Calibration (Platt/Isotonic) over CV

**Stage-1 Optuna**

-   Search: calibration choice, class weights, CV geometry, soft gates,
    fixed threshold

-   Joint objective (J_stage1)

-   Artifacts: best_params.json, joint_best_mask_rows.csv, optional
    union/consensus report

**Stage-2 Optuna**

-   Input: flagged set from Stage-1

-   Time-split: early (train knobs) → late (audit knobs)

-   Search: p_threshold, economic/quality gates, flip_share, caps

-   Objective (J_stage2) + barrier

-   Artifacts: mask23_best_params.json, mask23_trials_log.csv,
    best_mask.csv, preds_with_mask23.csv, mask23_summary.json

**Frozen Apply-Only**

-   Load Stage-1 knobs → flag

-   Load Stage-2 knobs → flip once

-   Outputs for the untouched validation set

**6) Mathematical summary**

Let:

-   Dataset D with features x, durations y.

-   Calibrated model pθ(x) = P(y ≤ 24 \| x) with calibration θ
    (platt/isotonic).

-   Soft gate g(x; γ) with gate knobs γ → yields flags F = { i : pθ(x_i)
    ≥ τ and g(x_i; γ) }.

**Stage-1 optimization**\
Search over (θ, γ, τ) and CV geometry to maximize:

J_stage1(θ, γ, τ) =

w_tp_fp (TP − FP)

\+ w_fast Σ\_{i∈F} 1{y_i ≤ 40}

− w_slow Σ\_{i∈F} 1{y_i \> 72}

− w_mid Σ\_{i∈F} 1{40 \< y_i ≤ 72}

\+ w_MAE (MAE_base − MAE_stage1)

\+ w_RMSE (RMSE_base − RMSE_stage1)

\+ w_R2 (R2_stage1 − R2_base)

− barriers(...)

**Stage-2 optimization (time-split)**\
On audit half A ⊂ F, choose m ∈ {0,1} \\{\|A\|} via policy π(·; φ) with
knobs φ to maximize:

J_stage2(φ) =

W_FAST Σ\_{i∈A} m_i·1{y_i ≤ 40}

− W_SLOW Σ\_{i∈A} m_i·1{y_i \> 72}

− W_MID Σ\_{i∈A} m_i·1{40 \< y_i ≤ 72}

\+ W_MAE (MAE_base − MAE_mask23(m))

\+ W_RMSE (RMSE_base − RMSE_mask23(m))

\+ W_R2 (R2_mask23(m) − R2_base)

− Barrier(slow(m), caps)

where ŷ_i\' = 23 if m_i=1 else ŷ_i.

**Meta-learner**: Optuna searches the continuous/discrete knob space to
maximize J_stage1 and J_stage2 with cross-validation splits and audit
caps. The "learning" is the **selection policy learning** over
knobs---not refitting a new model on the 178.

**7) Operating guide (commands & artifacts)**

**7.1 Stage-1 + Stage-2 tuning on the 20% slice**

python fast24_flagger3_public.py \\

\--emb all \\

\--tune 500 \\

\--tune_mask23 200 \\

\--tune_mask23_per_trial 40 \\

\--slow \"./data/val_predictions.csv\" \\

\--outdir \"./out/S1S2_tune_run\" \\

\--no_current_threshold \\

\--fixed_prob_threshold 0.75

Artifacts:

-   Stage-1 best: ./out/S1S2_tune_run/best_params.json

-   Stage-2 best: ./out/S1S2_tune_run/mask23_best_params.json

-   Telemetry: mask23_trials_log.csv, joint_best_mask_rows.csv,
    sweep.csv, etc.

**7.2 Frozen apply-only on the untouched 178**

python .\\fast24_flagger3_public.py \\

\--emb all \\

\--tune 0 \--tune_mask23 0 \--tune_mask23_per_trial 0 \\

\--outdir \"./out/178_frozen_eval\" \\

\--no_current_threshold \\

\--fixed_prob_threshold 0.75 \\

\--load_params \"./out/S1S2_tune_run/best_params.json\" \\

\--load_mask23_params \"./out/S1S2_tune_run/mask23_best_params.json\"

Outputs (apply-only):

-   preds_with_mask23.csv (full rows + flip bits)

-   best_mask.csv (only flipped rows)

-   mask23_summary.json **and** mask23_summary.csv (high-level metrics)

-   Stage-1 standard artifacts:
    slow_bucket_pred_fast24_with_fastflag.csv, flagged_min.csv, etc.

The \"train_frac\" field in the Stage-2 summary reflects the template
used in tuning; it **does not** imply training during apply-only.

**8) Why it works / what you observed**

-   Stage-1 forces the search to **prefer fast gains and punish slow
    mistakes**, not just generic metrics.

-   Stage-2 converts policy intent into a **small set of flips** via
    robust gates and **time-split audit**, so it generalizes.

-   \*\*Freezing

**Two-Stage, Leak-Free Meta-Optimization for Fast-from-Slow Selection**

**Design, math, and operating guide**

**0) What this pipeline solves**

You have a mixed pool of "predicted-slow" listings and you want to
**pull truly fast rows out of that slow pool** while **avoiding slow
mistakes** and **improving duration regression**. The pipeline does this
with a **two-stage meta-optimizer**:

-   **Stage-1 (candidate scorer & gate + Optuna)**: builds a calibrated
    probability of "fast within 24h", applies soft business/economic
    gates, and tunes a **joint objective** that rewards fast flips and
    penalizes slow flips (plus optional regression deltas), producing
    the set of **flagged** candidates.

-   **Stage-2 (mask-23 policy + Optuna)**: on the flagged set, learns a
    **flip policy** that promotes a subset to a 23-hour prediction (the
    "mask-23" flip), under hard caps and penalties, using a
    **time-split** so it learns on earlier rows and is **audited on
    later rows** without peeking.

Then you **freeze the knobs** (best parameters) and run an
**apply-only** pass on an untouched validation slice (e.g., the 178
"slow bucket"), giving **unbiased results**.

**1) Data slices & leak-free discipline**

-   **Training pool** → provides the feature space, embeddings, anchors,
    and model calibration.

-   **Held-out 20% "tuning slice" (SOLD & predicted-slow)** → used by
    Optuna in both stages:

    -   **Stage-1** uses this slice to score many joint policies (no
        label-fit thresholding).

    -   **Stage-2** uses an inline **time split**: tune on **earlier**
        flagged rows (durations known), then **apply to later** flagged
        rows (durations not used to select), so it's leak-resistant on
        the same slice.

-   **Untouched validation set (e.g., 178 rows)** → **never** used in
    tuning. After freezing knobs, you run apply-only and score
    truthfully. This is your **trust** set.

**2) Stage-1: calibrated candidate scorer & joint objective**

**2.1 Calibration + embeddings + soft gates**

-   Probabilities are **calibrated** (Platt or Isotonic) over
    cross-validation splits.

-   Feature space includes the **embeddings and numeric anchors**
    (price-to-value, deltas vs model/time medians, PPGB, storage, seller
    quality, warranty/receipt, etc.).

-   **Soft gates** ("economic" & "quality" filters) are controlled by
    the G\_\* family (e.g., G_MIN_COND, G_MAX_DAMAGE, G_MIN_BATT, PTV
    limits, underpricing deltas). Optuna tunes these too.

**2.2 Label-free thresholding**

Stage-1 uses a **fixed label-free cutoff** (\--no_current_threshold +
\--fixed_prob_threshold τ) or a tuned, label-free threshold window. The
aim: **don't fit a threshold to labels**; keep it reproducible.

**2.3 The Stage-1 joint objective (what Optuna maximizes)**

Define:

-   For the **flagged** pool on the tuning slice:\
    TP = true fasts captured, FP = slow mistakes, MID = mid (40--72h).

-   Optional regression deltas are computed relative to a baseline
    prediction (MAE, RMSE, R²).

The objective is:

J_stage1 = w_tp_fp·(TP − FP)

\+ w_fast·(# fast flips)

− w_slow·(# slow flips)

− w_mid ·(# mid flips)

\+ w_MAE · ΔMAE

\+ w_RMSE· ΔRMSE

\+ w_R2 · ΔR2

-   The w\_\* weights come from CLI flags (\--joint_w\_\*) and let you
    bias search strongly against slow flips and in favor of fast
    improvements (and optionally reward regression gains).

-   There's also an optional **soft barrier** to down-score policies
    that exceed minimum gain/quality caps or violate guardrails.

Outcome: Stage-1 returns a best policy (best_params.json) and a set of
**flagged** rows for Stage-2 to audit.

**3) Stage-2: mask-23 flip policy (time-split + caps)**

**3.1 Policy surface (the "mask-23" knobs)**

A flip policy is parameterized by:

-   p_threshold --- minimum calibrated fast probability (on the flagged
    set).

-   delta_overall_max, ptv_overall_max --- price delta & price-to-value
    ceilings.

-   cond_min, damage_max, batt_min --- quality gates (e.g., condition
    score, damage severity, battery %).

-   dpm14_max --- model-anchor delta bound.

-   flip_share --- target share of flips among the flagged pool (subject
    to floors and caps).

**Caps / constraints**

-   slow_cap_abs (absolute max slow flips)

-   slow_cap_ratio (max slow flips as % of flips)

-   Floors like min_share, min_flips can be used to avoid degenerate
    tiny selections.

**3.2 Time-split audit (leak-free on the tuning slice)**

For each trial:

-   **Train side (earlier flagged rows)**: knobs are picked using their
    durations.

-   **Audit side (later flagged rows)**: the policy is **applied**, and
    only then durations are revealed to compute audit metrics. No
    peeking during selection.

**3.3 Stage-2 objective**

Let m_i ∈ {0,1} mark a flip for row i in the audit half. Define:

-   fast_i = 1{duration_i ≤ 40}, slow_i = 1{duration_i \> 72}, mid_i =
    1{40 \< duration_i ≤ 72}.

-   K = Σ m_i (number of flips).

-   Baseline prediction ŷ_i and flipped prediction ŷ_i\' = 23 if m_i = 1
    else ŷ_i.

The policy value is:

J_stage2 = W_FAST · Σ_i m_i·fast_i

− W_SLOW · Σ_i m_i·slow_i

− W_MID · Σ_i m_i·mid_i

\+ W_MAE · (MAE_baseline − MAE_mask23)

\+ W_RMSE · (RMSE_baseline − RMSE_mask23)

\+ W_R2 · (R2_mask23 − R2_baseline)

− Barrier(slow_count, slow_cap_abs, slow_cap_ratio; power)

The **Barrier** is a large penalty that grows super-linearly if slow
flips exceed caps:

Barrier = 0 if slow ≤ min(slow_cap_abs, slow_cap_ratio·K)

= λ · (excess)\^power otherwise

(λ absorbed into W_SLOW or a separate constant.)

Outcome: Stage-2 gives a best policy (mask23_best_params.json) and full
telemetry (mask23_trials_log.csv, audit metrics, the flipped rows csv,
etc.).

**4) Freezing & apply-only on the untouched 178**

**Freeze**:

-   Stage-1: best_params.json (from your 20% tuning slice)

-   Stage-2: mask23_best_params.json (from the same slice's time-split
    tuner)

**Apply-only run** on 178:

3.  Load Stage-1 knobs (\--load_params) → build features, calibrate,
    gate, **flag**.

4.  Load Stage-2 knobs (\--load_mask23_params) → **apply** mask-23 once
    to the flagged set (no tuning, no refit).

This produces preds_with_mask23.csv, best_mask.csv, and a summary
(mask23_summary.json and mask23_summary.csv if enabled).\
The reported \"train_frac\": 0.4 in summary is **metadata only**; **no
training occurs** in apply-only.

**Your observed result on the 178** (example):\
Flips ≈ 10, **fast ≈ 8**, **slow ≈ 1**, mid ≈ 1; MAE improved by ≈ 2.4
h.\
This is **clean**, **frozen**, **not selection-biased**, and **not
optimistic**---because the 178 records were never seen in tuning.

**5) Architecture (high level)**

**Ingestion & Features**

-   Structured features, anchors, deltas, embeddings

-   Calibration (Platt/Isotonic) over CV

**Stage-1 Optuna**

-   Search: calibration choice, class weights, CV geometry, soft gates,
    fixed threshold

-   Joint objective (J_stage1)

-   Artifacts: best_params.json, joint_best_mask_rows.csv, optional
    union/consensus report

**Stage-2 Optuna**

-   Input: flagged set from Stage-1

-   Time-split: early (train knobs) → late (audit knobs)

-   Search: p_threshold, economic/quality gates, flip_share, caps

-   Objective (J_stage2) + barrier

-   Artifacts: mask23_best_params.json, mask23_trials_log.csv,
    best_mask.csv, preds_with_mask23.csv, mask23_summary.json

**Frozen Apply-Only**

-   Load Stage-1 knobs → flag

-   Load Stage-2 knobs → flip once

-   Outputs for the untouched validation set

**6) Mathematical summary**

Let:

-   Dataset D with features x, durations y.

-   Calibrated model pθ(x) = P(y ≤ 24 \| x) with calibration θ
    (platt/isotonic).

-   Soft gate g(x; γ) with gate knobs γ → yields flags F = { i : pθ(x_i)
    ≥ τ and g(x_i; γ) }.

**Stage-1 optimization**\
Search over (θ, γ, τ) and CV geometry to maximize:

J_stage1(θ, γ, τ) =

w_tp_fp (TP − FP)

\+ w_fast Σ\_{i∈F} 1{y_i ≤ 40}

− w_slow Σ\_{i∈F} 1{y_i \> 72}

− w_mid Σ\_{i∈F} 1{40 \< y_i ≤ 72}

\+ w_MAE (MAE_base − MAE_stage1)

\+ w_RMSE (RMSE_base − RMSE_stage1)

\+ w_R2 (R2_stage1 − R2_base)

− barriers(...)

**Stage-2 optimization (time-split)**\
On audit half A ⊂ F, choose m ∈ {0,1} \\{\|A\|} via policy π(·; φ) with
knobs φ to maximize:

J_stage2(φ) =

W_FAST Σ\_{i∈A} m_i·1{y_i ≤ 40}

− W_SLOW Σ\_{i∈A} m_i·1{y_i \> 72}

− W_MID Σ\_{i∈A} m_i·1{40 \< y_i ≤ 72}

\+ W_MAE (MAE_base − MAE_mask23(m))

\+ W_RMSE (RMSE_base − RMSE_mask23(m))

\+ W_R2 (R2_mask23(m) − R2_base)

− Barrier(slow(m), caps)

where ŷ_i\' = 23 if m_i=1 else ŷ_i.

**Meta-learner**: Optuna searches the continuous/discrete knob space to
maximize J_stage1 and J_stage2 with cross-validation splits and audit
caps. The "learning" is the **selection policy learning** over
knobs---not refitting a new model on the 178.

**7) Operating guide (commands & artifacts)**

**7.1 Stage-1 + Stage-2 tuning on the 20% slice**

python fast24_flagger3_public.py \\

\--emb all \\

\--tune 500 \\

\--tune_mask23 200 \\

\--tune_mask23_per_trial 40 \\

\--slow \"./data/val_predictions.csv\" \\

\--outdir \"./out/S1S2_tune_run\" \\

\--no_current_threshold \\

\--fixed_prob_threshold 0.75

Artifacts:

-   Stage-1 best: ./out/S1S2_tune_run/best_params.json

-   Stage-2 best: ./out/S1S2_tune_run/mask23_best_params.json

-   Telemetry: mask23_trials_log.csv, joint_best_mask_rows.csv,
    sweep.csv, etc.

**7.2 Frozen apply-only on the untouched 178**

python .\\fast24_flagger3_public.py \\

\--emb all \\

\--tune 0 \--tune_mask23 0 \--tune_mask23_per_trial 0 \\

\--outdir \"./out/178_frozen_eval\" \\

\--no_current_threshold \\

\--fixed_prob_threshold 0.75 \\

\--load_params \"./out/S1S2_tune_run/best_params.json\" \\

\--load_mask23_params \"./out/S1S2_tune_run/mask23_best_params.json\"

Outputs (apply-only):

-   preds_with_mask23.csv (full rows + flip bits)

-   best_mask.csv (only flipped rows)

-   mask23_summary.json **and** mask23_summary.csv (high-level metrics)

-   Stage-1 standard artifacts:
    slow_bucket_pred_fast24_with_fastflag.csv, flagged_min.csv, etc.

The \"train_frac\" field in the Stage-2 summary reflects the template
used in tuning; it **does not** imply training during apply-only.

**8) Why it works / what you observed**

-   Stage-1 forces the search to **prefer fast gains and punish slow
    mistakes**, not just generic metrics.

-   Stage-2 converts policy intent into a **small set of flips** via
    robust gates and **time-split audit**, so it generalizes.

-   \*\*Freezing

# 9) Why this two-stage, freeze-and-apply design works

**9.1 Separation of concerns (classification vs. promotion)**

-   **Stage-1** is a *selector*: it decides which rows are even
    candidates for "fast." It optimizes a **joint objective** that
    rewards fast flips and punishes slow flips (plus small penalties for
    mid), while also keeping the base classifier honest via TP--FP
    balance and regression deltas.

-   **Stage-2** is a *promoter*: given Stage-1's candidates, it flips a
    **limited share** to a hard 23 h target using *rules* (probability
    gate + economic & physical gates). This modular split prevents one
    learner from overfitting to both tasks at once.

**9.2 Time-awareness and leak-free evaluation**

-   Stage-2 tuning is **early→late**: it learns gates on earlier flagged
    rows (where durations are known) and evaluates on later flagged rows
    **without peeking** at their durations.

-   When you freeze and apply on the unseen 178-row bucket, nothing is
    tuned; you only *apply* previously saved knobs. That's a proper
    out-of-sample evaluation.

**9.3 Fixed decision boundary = stable deployment**

-   Stage-1's threshold is **label-free** at run time (e.g.,
    \--no_current_threshold with a fixed cutoff or a tuned, label-free
    sweep range).

-   Because the cutoff isn't re-fit to labels each run, behavior is
    reproducible and not subject to "post-selection" optimism.

**9.4 Search regularization via the joint objective**

-   JOINT scoring weighs: **+fast**, **--slow**, **--mid**, small
    **(+TP--FP)**, and optional regression improvements (ΔMAE, ΔRMSE,
    ΔR²).

-   This structure nudges Optuna away from degenerate regimes (e.g.,
    harvesting a ton of flags that happen to score on one metric but
    ruin slow-flip cost).

**9.5 Share, caps, and barriers curb overreach**

-   Stage-2's **flip_share** and **slow caps** (absolute/ratio) are hard
    guardrails that physically prevent the promoter from "spraying" 23 h
    flips.

-   The **soft/hard barrier** in Stage-1 (and penalty multipliers in
    Stage-2) codify a **steep cost for slow flips**, so the search must
    "earn" each extra fast flip.

**9.6 Calibrated probabilities used as *ranking*, not as an oracle**

-   Platt/Isotonic calibration is used to **order** candidates (and gate
    minimum probability), but Stage-2 success *also* requires passing
    economic/physical gates (ptv, delta from anchors, condition, damage,
    battery, dpm14).

-   This avoids over-reliance on a single scalar and ties flips to
    interpretable constraints.

**9.7 Compact policy surface in Stage-2 = generalizes better**

-   Stage-2 is a small, structured policy (a dozenish knobs) instead of
    a dense model.

-   With caps + share + gate logic, it's robust to covariate shift
    across time. Fewer degrees of freedom = less variance and less
    overfitting.

**9.8 Freeze-then-apply eliminates post-selection bias on the 178**

-   You saved **two artifacts**:

    1.  best_params.json (Stage-1 knobs),

    2.  mask23_best_params.json (Stage-2 knobs).

-   The final run on the 178 uses \--tune 0 \--tune_mask23 0 **and**
    loads both JSONs. No parameters are re-tuned on the 178, so the
    reported 178-row metrics are **not optimistic**.

**9.9 Explicit costs aligned with business goal**

-   The objective mirrors the real target: *pull as many truly-fast rows
    as possible from a slow pool while almost never flipping slow*.

-   Because slow flips carry a much higher weight (e.g., 12--18× fast
    weight), the optimizer learns exactly the trade you want.

**9.10 Diagnostic stability tools**

-   Optional **union/consensus** across trials shows which flips are
    robust to many different policies --- a sanity check against brittle
    recipes during R&D.

**9.11 Deterministic artifacts = traceability**

-   Every stage writes structured outputs (best rows, mask CSVs, JSON
    summaries, telemetry) so you can audit exactly **which rows**
    flipped, **why** they flipped (gates), and **what** the deltas were
    (MAE/RMSE/R², fast/slow/mid counts).

**Mathematical representation of the algorithm for the meta leaner**

 **Stage-1** searches (calibration θ, gates γ, threshold τ) to maximize

Jstage1=wtpfp(TP ,− ,FP)+wfast ,∑F ,1\[y≤40\]−wslow ,∑F ,1\[y\>72\]−wmid ,∑F ,1\[40\<y≤72\]+wMAE ΔMAE+wRMSE ΔRMSE+wR2 ΔR2−barriersJ\_{\\text{stage1}}
= w\_{tpfp}(TP\\!-\\!FP) + w\_{fast}\\!\\sum\_{F}\\!1\[y\\le 40\] -
w\_{slow}\\!\\sum\_{F}\\!1\[y\>72\] -
w\_{mid}\\!\\sum\_{F}\\!1\[40\<y\\le 72\] + w\_{MAE}\\,\\Delta MAE +
w\_{RMSE}\\,\\Delta RMSE + w\_{R2}\\,\\Delta R\^2 -
\\text{barriers}Jstage1​=wtpfp​(TP−FP)+wfast​F∑​1\[y≤40\]−wslow​F∑​1\[y\>72\]−wmid​F∑​1\[40\<y≤72\]+wMAE​ΔMAE+wRMSE​ΔRMSE+wR2​ΔR2−barriers

with **label-free thresholding** when requested.

Meta-Optimization_documentation

fast24_flagger3_public

 **Stage-2** (audit half A only) learns a flip mask mmm to maximize

Jstage2=WFAST ,∑A ,mi 1\[y≤40\]−WSLOW ,∑A ,mi 1\[y\>72\]−WMID ,∑A ,mi 1\[40\<y≤72\]+WMAE ΔMAE+WRMSE ΔRMSE+WR2 ΔR2−Barrier(slow,caps)J\_{\\text{stage2}}
= W\_{FAST}\\!\\sum\_{A}\\!m_i\\,1\[y\\le 40\] -
W\_{SLOW}\\!\\sum\_{A}\\!m_i\\,1\[y\>72\] -
W\_{MID}\\!\\sum\_{A}\\!m_i\\,1\[40\<y\\le 72\] + W\_{MAE}\\,\\Delta
MAE + W\_{RMSE}\\,\\Delta RMSE + W\_{R2}\\,\\Delta R\^2 -
\\text{Barrier}(\\text{slow},\\text{caps})Jstage2​=WFAST​A∑​mi​1\[y≤40\]−WSLOW​A∑​mi​1\[y\>72\]−WMID​A∑​mi​1\[40\<y≤72\]+WMAE​ΔMAE+WRMSE​ΔRMSE+WR2​ΔR2−Barrier(slow,caps)

where y\^i′=23\\hat y\'\_i = 23y\^​i′​=23 if mi=1m_i=1mi​=1 else y\^i\\hat
y_iy\^​i​; **Barrier** grows super-linearly if slow flips exceed
absolute/ratio caps.
