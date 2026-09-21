# Stage0 implementation: original code and public entrypoints

The released Stage0 is an executable neural survival model. Its tokenizer,
attention, mixture-of-experts, survival-probability and loss definitions are
preserved from the original implementation. The public package also trains new
models, evaluates supplied cohorts, fits probability ensembles, selects policies,
routes predictions and saves or reloads locally created models. This document
maps those capabilities to the original module responsibilities.

The release is source-neutral. It contains no private observations, trained
weights, fitted category values, credentials, private paths or source-specific
database adapters. The preserved numerical model and the new portable controller
are identified separately throughout.

## What was recovered and checked

The inspected original folder contains CPU and GPU recovery trees, training
programs, ensemble search, checkpoint-pool management, live inference, schema
contracts, deployment verification and documentation. These are recovery
snapshots with subsequent differences; their directory names do not establish
that every file is an independently maintained model implementation.

The intact CPU trainer contains 57 top-level definitions. The public
[`legacy_core.py`](legacy_core.py) preserves 23 selected definitions from it,
including the actual neural network and differentiable survival mathematics.
All 23 match the recorded abstract-syntax-tree hashes. The remaining definitions
include data loading, feature selection, training orchestration, reports and
environment-bound artifact handling; a definition count is not a measure of
model completeness.

Retained trainer copies in both recovery trees are byte-identical to one another.
They independently match 22 of these 23 exported definitions; their older
`TrainConfig` differs. Retained ensemble copies match 11 of 12 exported ensemble
definitions; `calibrate_matrix` differs because the newer selected version also
returns fitted calibration state. All seven exported phase-search helpers match
the retained search copies. Four exported cascade probability/routing helpers
match both intact forwarding implementations.

The current top-level GPU copies of the trainer, phase runner and ensemble tuner
have their text collapsed onto one physical line beginning with a shebang. They
therefore expose no Python definitions when parsed. The public numerical export
uses the intact, hash-identified source, and does not redistribute or claim to run
those damaged copies. This is a source-recovery distinction, not evidence that
the operational model was absent.

[`provenance.json`](provenance.json) records original definition ranges, source
span hashes, AST hashes and public export hashes. The Stage0-related selection is:

| Source label | Public implementation | Preserved definitions |
|---|---|---:|
| `legacy_shared_architecture` | `legacy_core.py` | 23 |
| `stage0_ensemble` | `stage0_ensemble_core.py` | 12 |
| `stage0_three_phase_search` | `search_core.py` | 7 |
| `cascade_forward` | `routing_core.py` | 4 |

These 46 definitions are part of the 122-definition historical cascade export.
They are substantial implementation bodies, rather than interfaces that require
an unavailable private model module at runtime.

## Original responsibilities and released equivalents

| Original responsibility | Released entrypoint or implementation | Coverage and boundary |
|---|---|---|
| Multimodal trainer, invoked by `stage0_train_phaseabc.py` | `legacy_core.MultiModalSurvModel`; `StageEstimator.fit` | Exact architecture and loss primitives; new portable optimizer loop and explicit objective configuration |
| Three-phase runner behind `stage0_train_phaseabc.py` | `search_core._suggest_phase_A/B/C` and architecture/freeze helpers | Exact parameter suggestions; original subprocess restart, study-storage and resume controller is not copied |
| `stage0_merge_pool.py` | Lists of `StageEstimator` members passed to `CascadeEstimator` | Public caller supplies selected/fitted members; original directory discovery, file linking and pool registry are excluded |
| `stage0_ensemble_tune.py` and its numerical tuner | `stage0_ensemble_core`; `ProbabilityEnsemble`; `choose_ensemble` | Exact combination/calibration mathematics plus frozen-state fitting; new finite inner/outer candidate controller |
| `stage0_serving_replay.py`, `score_stage0_serving.py` | `StageEstimator.load/predict`; `CascadeEstimator.load/predict` | Complete public model persistence and forward path; does not interpret an original private bundle without an explicit conversion |
| Checkpoint and preprocessing compatibility | `FeatureBatch.validate`; `FrozenFeatureEncoder`; strict model-state load | Enforces public matrix dimensions and category bounds; original schema sidecars and database readiness certificates remain external |
| `verify_stage0_serving.py`, `prove_stage0_serving.py` and daemon parity checks | `tests/test_cascade.py`; saved synthetic smoke aggregate | Public analytical and round-trip checks; original private replay rows and database-backed certification are not distributed |
| `stage1_prepare_inputs.py`, `cascade_forward_nn.py` | `CascadeEstimator.fit` and `route_scores` | Later training duration cap and prediction-gated development population are explicit; private table joins and row exports are excluded |
| `stage0_live_score_worker.py`, `stage0_live_infer_daemon.py` and snapshot workers | Caller invokes fitted `predict` methods | Scoring is reusable; leases, retries, active windows, source queries and database writes require a caller-owned service |
| Container dispatch and artifact publication | `python -m research.cascade --smoke`; Python APIs | No private container image, registry credentials, deployment path or live service is needed |

The original serving runtime does more than load a tensor file: it validates a
bundle, orders selected members, prewarms checkpoints and preprocessing, applies
saved calibration/combination state, and records lineage. The public controller
retains fitted models and ensemble state in memory and reuses them for prediction.
It has its own documented bundle format. It does not claim binary compatibility
with every historical deployment artifact.

## Model architecture and input contract

Stage0 predicts persistence beyond 504 hours, or 21 days. Its legacy input is a
numerical matrix, a categorical-ID matrix, one 768-dimensional listing-text
vector and one 512-dimensional image vector per observation. Upstream encoders
are frozen/external to this model. Their original services and fitted
preprocessing artifacts are not necessary to instantiate the neural architecture.

Each numerical scalar receives a learned affine token representation. Each
categorical column has its own embedding table. The compact tabular path uses
learned attention queries to produce a fixed number of numerical and categorical
summary tokens. Text and image projections turn each supplied vector into several
learned tokens. Eight text tokens are eight projections of one vector, not eight
separately encoded text passages.

An inspected selected configuration uses width 256, 32 latents, nine fusion
blocks, eight attention heads, seven dense hazard experts and 128 linear time
bins over 504 hours. Eight numerical, eight categorical, eight text and eight
image tokens produce 32 input tokens. The inspected six-member bundle is not
architecturally uniform: later selected members use six blocks and fewer
modality/summary tokens. Exact selected numerical settings are recorded in
[`historical_evidence.json`](historical_evidence.json); `TrainConfig` defaults
should not be substituted for those saved settings.

Each fusion block cross-attends from learned latents to input tokens, applies
latent self-attention and feed-forward processing, and uses residual connections
and normalization. Pooled latents feed dense expert hazard heads, mixture weights
and a separate scalar tail head. All experts execute; this is not sparse routing
among separate devices.

For expert $m$, survival is $S_m(b_k)=\prod_{j=1}^{k}(1-h_{mj})$. The model mixes
complete expert distributions: $S(t)=\sum_m\pi_m S_m(t)$. It does not average
hazard logits and call their sigmoid the survival mixture. The separate scalar
head estimates a slow-positive score. Stage0 exposes curve, head and arithmetic
combined channels; the chosen channel belongs in the deployed policy contract.

The later direct eight-slot network under
[`../production_reference`](../production_reference/README.md) is a separate
architecture. Its eight image and eight report vectors, 40 input tokens and
17,145,736 parameters must not be used to describe every legacy Stage0 model.

## Training objective: exact primitives and portable choices

The original Stage0 loop uses row-weighted mixture survival negative log
likelihood, optional multi-horizon BCE and optional dedicated tail-head BCE.
The exported `mixture_survival_nll_discrete_vec`, `horizon_bce_loss` and
`masked_bce_vec` preserve that mathematics. Its horizon BCE sums losses over the
configured prefix of target horizons; it does not average them. For each horizon,
unresolved short follow-up contributes no binary label. Known slow observations
can receive the configured additional negative-class weight in that helper.

Events beyond modeled support must be recensored at the support boundary before
calling the likelihood primitive. Censored observations retain partial-bin
exposure through interpolation in log survival. The dedicated tail target is
slow-positive; an observed event exactly at 504 hours is fast, while a censor at
that horizon is treated as known slow under the archived convention.

The original sample-weight assembly combines recency, Gaussian boundary focus,
duration-region weights and a fast-case guard, then normalizes the weights to
mean one. Exact recency and boundary functions are exported. For observation age
$a$ and half-life $\tau$, the age factor is $2^{-a/\tau}$. Original assembly also
multiplies rows beyond 168 hours and beyond the tail horizon by their respective
configured factors. Its fast guard multiplies rows below its cutoff by
$1+\text{fast_guard_k}\max(\text{boundary_focus_k},0)$. Public callers pass their
explicit training weights to `fit`; configuration fields alone do not cause the
new portable controller to reconstruct private timestamp-based weighting.

`StageEstimator` intentionally exposes a new explicit `StageObjective`. Its
default trains survival, a gate-curve target and a scalar head with unit
coefficients. Those defaults are not claimed as the original selected Stage0
objective. Original multi-horizon and weighting primitives remain callable for a
caller reproducing an exact objective. The portable loop uses AdamW, an epoch
cosine schedule, finite-gradient checks and validation-only early stopping. The
historical warmup, mixed-precision and study controller are not silently implied.

Two historical details are retained rather than concealed. Some inspected
Stage0 members used a combined curve/head score with zero dedicated head-loss
weight, so their scalar head must not be described as supervised. The archived
expected-time helper is decorated with `no_grad`; an optional MAE term computed
through it does not become a differentiable training signal merely because its
configuration exists. The public fitting objective omits that unsupported claim.

## Run a new Stage0 model without private infrastructure

Install the repository's declared research dependencies. This complete example
uses only newly generated artificial tensors and executes the preserved backbone:

```python
import numpy as np
import torch
from research.cascade import FeatureBatch, StageEstimator, StageObjective
from research.cascade.legacy_core import TrainConfig

torch.set_num_threads(1)
rng = np.random.default_rng(42)
n = 32
x = FeatureBatch(
    rng.normal(size=(n, 3)).astype("float32"),
    rng.integers(0, 3, size=(n, 2)),
    rng.normal(size=(n, 768)).astype("float32"),
    rng.normal(size=(n, 512)).astype("float32"),
)
duration = np.tile([24., 72., 200., 480., 650., 40., 300., 600.], 4)
event = np.tile([1, 1, 1, 1, 0, 0, 0, 1], 4)
cfg = TrainConfig(
    d_model=16, n_latents=2, fusion_layers=1, n_heads=2,
    n_experts=2, n_bins=8, text_tokens=1, img_tokens=1,
    tab_num_tokens=2, tab_cat_tokens=2,
)
model = StageEstimator(
    0, cfg, StageObjective(nll=1, curve=1, head=1),
    seed=42, epochs=2, batch_size=8,
).fit(
    x.take(slice(0, 24)), duration[:24], event[:24],
    validation=(x.take(slice(24, 28)), duration[24:28], event[24:28]),
    cardinalities=[3, 3],
)
prediction = model.predict(x.take(slice(28, None)))
assert prediction["score"].shape == (4,)
assert np.isfinite(prediction["survival"]).all()
```

This is a small execution example, not historical model accuracy. For real
research, construct decision-time cohorts, freeze train-only transformations,
retain category ordering, specify the objective and supply independent
development/test populations. `FrozenFeatureEncoder` supplies a neutral
train-only numerical/category transformation; it does not recreate unavailable
original fitted vocabularies.

Run `python -m research.cascade --smoke` to exercise all three stages, or
`python -m pytest tests/test_cascade.py -q` to check source identity, analytical
censoring cases, fitting, frozen ensemble behavior, routing and persistence.
`StageEstimator.save/load` supports one new model; `CascadeEstimator.save/load`
supports member sets, ensembles and thresholds. Keep generated weights and row
outputs outside the public source release.

## What remains an integration responsibility

No additional private numerical module is required for the released backbone to
train or predict. The intentionally excluded pieces are the original data
adapters, fitted preprocessing, stored checkpoint pools, deployment registry,
source-specific feature/schema joins, queue workers and production artifact
publication. Historical grouped-attribution reports and full recovery/search
launchers are also outside this portable model package. Their absence must not
be disguised as a completed reproduction of the original environment.

The current release therefore provides the actual reusable model and a tested
portable fitting/scoring path. Reconstructing a particular historical run or
deploying it against a new source is separate integration work, with its own
input contracts, validation evidence and operational controls.
