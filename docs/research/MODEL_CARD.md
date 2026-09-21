# MarketNeural model card

Version 0.3 - 2026-09-22. **Historical neural results and executable comparison software.**

## Model identity and scope

This release contains three distinct model resources. The compact `marketneural` package is a runnable comparison implementation. `research/cascade/` preserves the historical Stage 0/1/2 numerical definitions and supplies portable fitting, ensemble selection, routing and persistence. `research/production_reference/` preserves the later slot-based multimodal architecture with source hashes and a generic tensor training adapter. The manuscript documents their mathematical and operational context. Each resource has its own configuration and execution scope.

The compact benchmark consumes precomputed structured and vector inputs. It downloads no pretrained image or text model and includes no private model checkpoint. It is intended for research, debugging temporal evaluation, and comparing survival estimators on explicitly provided data. It is not a turnkey collection service, a live marketplace recommendation product, a causal pricing model, or a profit estimator.

## Historical neural improvement

Recorded 21-day tail-screening F1 increased from **0.8462 for XGBoost AFT** to **0.9209 for the neural meta-ensemble**, with precision increasing from 0.8314 to 0.9802 and recall from 0.8614 to 0.8684. The earlier tree evaluation contains 941 records and the neural holdout 964. The [empirical chapter](../../papers/marketneural/chapters/06_empirical_evidence.md) and [aggregate evidence](../../research/thesis_evidence/historical_neural_comparison.json) preserve those scopes, source hashes and completed experiment counts. The compact estimators below are a separate public comparison implementation.



## Included compact estimators

| CLI name | Model | Inputs and assumptions |
|---|---|---|
| `km` | Training-cohort Kaplan–Meier survival estimate | Ignores covariate values; provides a common survival prior |
| `coxph` | Penalized Cox proportional-hazards regression | Shared flattened features; proportional hazards with a linear log-risk function |
| `rsf` | Random survival forest | Shared flattened features; nonlinear survival trees |
| `gbsa` | Gradient-boosting survival analysis | Shared flattened features; not the historical boosted AFT model |
| `mlp` | Neural discrete hazard model | Shared flattened features; nonlinear single-expert hazard sequence |
| `perceiver_moe` | Compact latent-attention mixture survival model | Structured tokenization, text tokens, image/report slots, missing-slot representations, and mixture of expert survival curves |

For the conventional estimators and MLP, masked image/report slots are zeroed and their masks are included in flattened inputs. The attention model receives the same underlying evidence but processes it as tokens. This tests a limited architecture contrast; it is not the complete planned structured-only/pooled/token ablation program.

## Neural output semantics

Each expert emits interval hazards. Survival is the product of conditional interval survival probabilities, with log-survival interpolation within a bin. The expert gate forms a convex mixture of complete survival curves, not a simple average of hazard logits. The likelihood uses event-bin probability for an observed event and fractional-bin survival exposure for censoring. Events after the configured modeling horizon are treated as censored at that horizon.

Outputs are survival probabilities on requested time points. They refer to duration from the supplied `decision_time`. A 72-hour event probability is `1 - S(72)` under that origin. Applying the model to an older listing without a valid landmark construction does not make this a residual 72-hour forecast. Arbitrary time-varying covariates and competing terminal events are not handled by the compact single-event contract.

## Training and selection

The public data loader forms chronological train, validation, and test cohorts. Training outcomes are administratively censored at validation start, validation outcomes at test start, and test outcomes at its declared as-of date. One entity appears once and cannot cross cohorts. Numeric imputation/scaling and categorical vocabulary are learned on training rows only.

Configuration selection uses validation integrated Brier score, averaged across configured seeds. Neural early stopping uses validation data. Operating thresholds also use validation, maximizing eligible-row recall subject to registered empirical precision and support constraints. If no threshold satisfies the constraint, an explicit reject-all policy is recorded. Configuration and policy selection are frozen before test prediction; there is no post-selection refit in the compact lane.

The compact lane has no independently fitted calibration block. The full real-data protocol adds a separate calibration/policy partition. The code can reject obvious outcome-column names and impossible timestamps, but cannot prove that supplied vectors or aggregates contain no future information.

## Evaluation and example result

The primary implementation metric is integrated Brier score on a fixed time grid, using the training cohort's marginal censoring estimate. IPCW concordance and locked-threshold horizon confusion counts are reported. Short-censored cases remain unresolved for binary operating metrics. The censoring estimator assumes independent censoring and adequate transfer to later cohorts; a changing observation process can violate that assumption.

The included nonlinear synthetic smoke example has 593 training, 267 validation, and 340 test records, with 258 observed test events. Its grid spans 12–120 hours. Observed test IBS is 0.180544 for Cox, 0.193811 for MLP, and 0.193862 for compact Perceiver-MoE; Cox is best in this run. See the [complete aggregate summary](../../research/examples/synthetic_smoke/summary.json). This is an artificial 1,200-record execution fixture, not evidence of marketplace performance.

Paired intervals in the example resample test entities with fitted models and censor weights held fixed. They do not include training uncertainty, correct for multiple comparisons, or imply universal model superiority. A one-seed standard deviation of zero is an absence of seed replication, not evidence of deterministic accuracy.

## Preserved architecture excerpt

The separate [production-reference provenance manifest](../../research/production_reference/provenance.json) records full source hashes, selected symbol ranges, and transformations. It excludes original data loaders, database access, private configuration, weights, fitted private vocabularies, orchestration, and row-level data. Its generic training adapter is new code. Numerical architecture preservation does not reproduce an earlier training run, checkpoint, or deployed policy.

## Preserved three-stage cascade

The [cascade package](../../research/cascade/README.md) includes 122 preserved numerical definitions across eight archived modules, with source and syntax-tree hashes. The ordered routing policy tests the 504-hour tail first, then the 168-hour gate, then the 72-hour gate. Historical training objectives, gates and cohort eligibility are detailed in its [methods](../../research/cascade/METHODS.md); the gates are separate fitted decisions, not a conditional factorization of one calibrated duration distribution.

The portable adapters accept caller-supplied feature matrices, fit training-only preprocessing, perform neural optimizer steps, select ensembles and thresholds on development data, and save/load fitted bundles. The synthetic smoke executes all three stages. The release excludes private trained weights, fitted vocabularies, production connections and observation tables. Its numerical implementations are executable PyTorch code; the portable controllers replace environment-specific integration.

## Limitations and intended safeguards

The compact model's small width, binning, and training budget are configured for reproducibility. They are not evidence that a larger model would or would not help. Missing-modality robustness, calibration under drift, subgroup stability, live readiness coverage, and economic utility remain unestablished on real marketplace data. The synthetic generator contains relatively simple dependence and independent random censoring; real data observation failures can be informative.

Use requires an audited data contract, a declared event definition, positive valid follow-up under the public loader, disjoint episodes, versioned representations, and a frozen evaluation plan. Do not describe scores as verified transaction probabilities, causal price effects, guaranteed precision, or purchasing advice. Any deployment claim requires a separate prospective study capturing immutable inputs and predictions at actual decision time.
