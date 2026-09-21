# Three-stage survival cascade

This package preserves actual numerical implementations for the historical
Stage0 → Stage1 → Stage2 family and supplies new portable training, evaluation,
ensemble and routing adapters. The later direct slot-based Stage2 network remains
separate under [`../production_reference`](../production_reference/README.md).

No fitted private weights, observations, vocabularies or source-specific data
connectors are distributed. The package can train new models from caller-supplied
matrices; it does not claim to reproduce unavailable historical fit datasets.

## Run

Install the repository's research dependencies, then from its root:

```sh
python -m research.cascade --smoke
python -m pytest tests/test_cascade.py -q
```

The smoke command performs actual optimizer steps for all three archived-backbone
stages on artificial rows, applies fitted ensembles and predicts through the
ordered gates. Its deliberately small configuration and prespecified thresholds
exercise software behavior; output is not an accuracy result.

## API

`FeatureBatch(numeric, categorical, text, image)` accepts matrices with text width
768 and image width512. Categorical values are integer IDs with a frozen
vocabulary. `FrozenFeatureEncoder.fit` learns numerical imputation/scaling and
categorical IDs using only TRAIN; `transform` maps unknown categories to zero.

```python
from research.cascade import StageEstimator, StageObjective

stage = StageEstimator(1, objective=StageObjective(nll=1, curve=1, head=1))
stage.fit(train_features, train_duration, train_event,
          validation=(validation_features, validation_duration, validation_event),
          cardinalities=frozen_encoder.cardinalities_)
scores = stage.predict(test_features)
metrics = stage.evaluate(test_features, test_duration, test_event, threshold=0.6)
```

The caller supplies an explicit temporal/entity split and as-of outcome labels.
The estimator cannot infer whether supplied embeddings contain future content.
`predict` returns the survival curve, curve/head/combined slow probabilities and
the stage's selected policy score. Stage0's score is slow-positive; Stage1/2 scores
are fast-positive complements at168/72hours. `evaluate` does not fit anything.

`CascadeEstimator.fit(..., validation=(...), policy=RoutingPolicy(...))` trains all
three stages using a prespecified policy. Omit `policy` to select thresholds on
development data with explicit feasibility constraints. Both later TRAIN cohorts
use the declared duration cap (historically504hours); downstream development
cohorts are prediction-gated. This distinction is intentional and documented.

`ProbabilityEnsemble.fit` supports probability/logit averages, conservative logit
aggregation, minimum-variance and discriminant/logistic weights, and a gradient-
boosting stack. Temperature/isotonic calibration and learned weights use only the
passed fitting mask. `predict` reuses fitted state. `choose_ensemble` can fit
registered candidates on inner development data and select on outer development
data. No final-test argument exists in the selection interface.

`StageEstimator.save/load` preserves one locally trained model. The cascade's
`save/load` preserves all members, fitted ensemble rules and thresholds; loading
a caller-created bundle requires `trusted=True` because fitted external
estimators use joblib. Retain the caller's fitted feature encoder separately.
These generated bundles must not be included in the public source release.

## Preserved source and new adapters

| Module | Role |
|---|---|
| `legacy_core.py` | Exact legacy multimodal backbone, survival math and recency weighting |
| `stage1_core.py`, `stage2_core.py` | Exact stage-specific labels, binary losses, threshold/risk mathematics |
| `stage0_ensemble_core.py`, `stage1_ensemble_core.py`, `stage2_ensemble_core.py` | Exact calibration, ensemble and policy functions |
| `search_core.py` | Exact Stage0 PhaseA/B/C parameter suggestions and freeze helpers |
| `routing_core.py` | Exact ordered bucket assignment and mean-logit meta combination |
| `portable.py`, `ensemble.py`, `pipeline.py` | New platform-neutral training and inference controllers |

[`provenance.json`](provenance.json) records exact definition hashes and original
source ranges. [`historical_evidence.json`](historical_evidence.json) publishes
only numerical configuration and hashed provenance from selected manifests.
Archived functions remain unchanged, including documented limitations. The new
adapter uses strict input validation, rejects required missing scores, freezes
fitted meta parameters and reports infeasible policies explicitly. Its optimizer
defaults are demonstration choices, not recovered historical trial settings.

The retained functions include advanced historical searches, but original
infrastructure restart loops and study-storage launchers are outside the release.
Search seeds over one saved checkpoint pool must not be described as independent
neural retraining. The later K8 architecture, weights and population must not be
substituted silently for this legacy family.

Read [`METHODS.md`](METHODS.md) for targets, losses, conditional-population limits,
recency mechanisms, tuning, leakage investigations and evidence boundaries.
The [`Stage0 implementation guide`](STAGE0_IMPLEMENTATION.md) maps original
module responsibilities to released code, records the recovery/source checks,
and provides a standalone runnable Stage0 example.
