# Empirical evidence and its interpretation

## Three kinds of evidence

The available results belong to three distinct layers. Historical reports and saved predictions describe models developed on the operational dataset. Saved replay and restoration checks assess whether the software reproduced its recorded behavior. New public synthetic experiments test a reproducible comparison pipeline under known data-generating mechanisms. The layers support different claims and should not be combined into a single performance number.

The historical project is substantial: it progressed from temporal feature stores and AFT tail screening to multimodal neural models, stage-specific decisions, calibration, policy selection, and deployed inference. A critical review should acknowledge that progression while examining which final populations were independent of development. The public comparison provides a cleaner experimental scaffold, but its synthetic population does not establish superiority on the historical market data. Aggregate evidence files accompanying this chapter make the numerical distinctions inspectable without releasing private rows.

## Neural ensembles improved on the earlier XGBoost result

The recorded 21-day tail-screening experiments show a clear historical improvement: **F1 increased from 0.8462 for the earlier XGBoost AFT model to 0.9209 for the later neural meta-ensemble**. This is an increase of **7.47 F1 percentage points**, or approximately **8.83% relative to the earlier score**. Precision increased from 0.8314 to 0.9802, while recall increased from 0.8614 to 0.8684. The later neural system therefore achieved substantially higher recorded precision while maintaining similar recall.

The earlier tree evaluation contains 941 sold records, including 166 slow outcomes. The neural holdout contains 964 sold records, including 114 slow outcomes. The table reports each model's recorded evaluation population; it does not silently treat these cohorts as identical.

| Historical model | Evaluation N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| XGBoost AFT | 941 | 0.8314 | 0.8614 | 0.8462 |
| Neural top-20 stack | 964 | 0.9307 | 0.8246 | 0.8744 |
| Neural optimized ensemble | 964 | 0.9091 | 0.8772 | 0.8929 |
| Neural meta-ensemble | 964 | **0.9802** | **0.8684** | **0.9209** |

The XGBoost result comes from the earlier empirical paper and its confusion matrix. The neural rows come from retained user-pasted execution output dated 11 February 2026. The output records 49 base candidates with selection and holdout predictions, a top-20 stack, a seed-specific optimized ensemble, and a mean-logit ensemble of ensembles. The accompanying code selects the original top-20 stack's regularization and threshold on SVAL, then applies that threshold to holdout. The later meta-ensemble output preserves its threshold and metrics; its complete selection history has not been reconstructed from that excerpt. The [aggregate comparison record](../../../research/thesis_evidence/historical_neural_comparison.json) identifies the retained sources by hash and provides the arithmetic behind the reported improvements.

![Historical tail-screening performance: the neural meta-ensemble records higher F1 and precision than the earlier XGBoost AFT system. The tree evaluation contains 941 records and the neural holdout 964; these are recorded historical cohorts.](../figures/r14_historical_neural_advantage.svg)

The recorded advantage concerns the selected neural-and-meta system on its historical task. It does not isolate the effect of replacing a tree with a neural network while holding every other input and decision fixed. Feature representations, fitting procedures, ensembles and cohorts evolved together. The older report uses a slow-tail boundary of at least 504 hours, while the recovered neural code uses greater than 504 hours. Those details must remain visible when reconstructing a row-matched comparison. They do not erase the higher neural metrics recorded during development.

### Scale of the completed experiments

The research already includes extensive model fitting and comparative optimization. A retained Stage 1 pool contains **233 neural trial prediction pairs plus three selected-model snapshots**, spanning three training phases. Its alignment checks examined 243 candidates, retained 236 and rejected seven for row mismatches. One subsequent ensemble-search run records **8,000 evaluated candidates across ten search seeds**, with 800 candidate rows per seed. These are ensemble configurations evaluated over retained neural predictions; they are distinct from the count of trained neural trials.

This distinction makes the experimental workload concrete. The project tested neural representations and training configurations, then tested how their outputs should be combined and converted into operational gates. The retained artifacts include competing combination methods and rejected candidates as well as selected results. The implemented XGBoost baseline is also available in the [original public training code](../../../modeling/slow21/train_slow21_gate_classifier.py), including its survival AFT objective, tuning, calibration, monotonic constraints and boundary-focused evaluation.

The historical neural advantage is therefore part of an experimental development program. The prospective protocol described later extends that work by measuring how the improvement persists across new periods and matched information sets.

## Earlier tail screening with structured anchors

The earlier AFT work focused on a slow tail, defined at 504 hours, or 21 days. The saved empirical chapter reports an evaluation cohort of 941 observations, including 166 slow outcomes, and a calibration cohort of 853 observations, including 171 slow outcomes. The confusion matrices are reproduced from the historical figure rather than reconstructed from hypothetical predictions.

Table. Historical slow-tail cohort sizes and confusion counts.

| Population | N | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|
| Evaluation | 941 | 143 | 29 | 23 | 746 |
| Calibration | 853 | 162 | 86 | 9 | 596 |

Table. Historical slow-tail prevalence and performance.

| Population | Prevalence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Evaluation | 17.64% | 0.8314 | 0.8614 | 0.8462 |
| Calibration | 20.05% | 0.6532 | 0.9474 | 0.7733 |

![Historical slow-tail confusion matrices](../../../research/thesis_evidence/figures/h04_slow21_confusion.png)

Historical figure H04. The source report's slow-tail confusion matrices use a 504-hour event threshold. The labels refer to the report's own cohorts and should not be confused with the later 72-hour neural-policy populations.

The evaluation precision interval reported in the earlier chapter is approximately 0.7683–0.8800 and the recall interval approximately 0.8007–0.9059. These finite-sample intervals are useful context. They do not account for model-search selection or any repeated consultation of the evaluation cohort. The report describes an optimizer objective using the evaluation F1 of 0.8462; consequently, the strongest safe interpretation is descriptive historical performance, not a claim of an untouched prospective test.

The earlier work also reports a sacrifice measure: how many items from faster duration bands are rejected by slow-tail screening. Evaluation sacrifice was 0.59% for durations below ten days and 26.88% for the middle ten-to-21-day band. Calibration sacrifice was 1.43% and 63.41%, respectively. The large difference in the middle band shows that similar high recall can coexist with very different selection behavior.

![Historical duration-band sacrifice](../../../research/thesis_evidence/figures/h05_slow21_sacrifice.png)

Historical figure H05. Duration-band rejection rates expose a tradeoff hidden by a single F1 score. These rates describe classification behavior; they are not realized financial losses or evidence of profitable interventions.

Anchor variables dominate the historical split-gain ranking: the strict anchor contributes approximately 28.15% and the flexible anchor 25.45%, with the next comparable statistic around 4.15%. This is consistent with the design emphasis on supported price comparisons. Split gain is an attribution within the fitted model and can distribute importance unevenly among correlated variables. It does not establish the causal effect of an anchor or its incremental out-of-sample value without an ablation.

![Historical anchor feature gain](../../../research/thesis_evidence/figures/h06_anchor_gain.png)

Historical figure H06. Anchor-derived variables carry substantial fitted-model gain in the earlier report. The plot is evidence about that fitted model's use of inputs, not a proof that removing an individual correlated feature would reduce performance by the plotted percentage.

The feature-store paper similarly explores live-stock pressure after stratifying by price. Its plots suggest conditional association in selected groups, with modest information-gain values in the accompanying analysis. These are useful hypotheses for subsequent validation. They should not be interpreted as a completed controlled experiment proving that inventory features improve every model.

![Historical stock-pressure association](../../../research/thesis_evidence/figures/h07_stock_conditional_lift.png)

Historical figure H07. The historical stock analysis compares slow-outcome rates across stock strata within price strata. It is an exploratory association plot; uncertainty and an independent incremental-performance comparison cannot be recovered from the image alone.

## Later 72-hour policy results

The later saved policy predictions allow direct recomputation of counts. The selection population contains 584 observations, the broader holdout 748, and a recent event-selected subset 283. The recent subset is nested within the 748-row holdout, with matching decisions. It is therefore useful for temporal characterization but is not an additional independent test that can be added to the holdout denominator.

Table. Saved 72-hour policy cohort sizes and confusion counts.

| Population | N | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|
| Selection | 584 | 126 | 14 | 203 | 241 |
| Holdout | 748 | 306 | 93 | 166 | 183 |
| Recent nested subset | 283 | 154 | 27 | 66 | 36 |

Table. Saved 72-hour policy prevalence and performance.

| Population | Prevalence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Selection | 56.34% | 90.00% | 38.30% | 0.5373 |
| Holdout | 63.10% | 76.69% | 64.83% | 0.7026 |
| Recent nested subset | 77.74% | 85.08% | 70.00% | 0.7681 |

These are dedicated-head policy decisions, including the selected policy combination; they are not automatically identical to a threshold on the neural survival curve's $1-S(72)$. The stored scalar head has a persistence-oriented interpretation in the inspected adapter, so conversion to a fast-event score requires its documented complement. Reversing that interpretation would change the meaning of every threshold.

The selection result shows a high-precision, low-recall operating point. The later holdout admits a broader fraction and has lower precision. This is compatible with changed prevalence, shifted feature distributions, imperfect calibration, and selection uncertainty; the aggregate table alone cannot identify the cause. The 56.34%, 63.10%, and 77.74% positive rates describe selected cohorts. They are not estimates of the entire market's turnover rate.

For perspective, an all-positive classifier has F1 approximately 0.7738 on the holdout and 0.8748 on the recent subset. These exceed the reported policy F1 because positive prevalence is high. That does not make the policy useless: an all-positive rule cannot meet a selective high-precision requirement. It does mean that F1 alone is an insufficient argument for the selected policy. Precision, recall, workload or selected fraction, and the user's utility constraints must be reported together.

## Duration-zero sensitivity and target provenance

The holdout contains 178 zero-duration records, all positive for the 72-hour target; the recent subset contains 70. A diagnostic that excludes these records changes holdout precision from 76.69% to 66.30% and recall from 64.83% to 62.24%. On the recent subset, precision changes from 85.08% to 78.91%. The sensitivity is material and should be visible in any empirical interpretation.

This exclusion is not a corrected benchmark. A zero duration can arise from the chosen origin, event timestamp resolution, same-time observations, or a processing rule. It is not automatically a corrupt label. The correct response is to trace how the decision or edited origin and event time were formed, whether negative intervals were clipped, and whether a zero-duration event could have been known at the claimed decision. The saved evidence establishes sensitivity; it does not by itself adjudicate every zero-duration case.

The research protocol should prespecify a treatment of same-time and already-resolved observations, report interval resolution, and distinguish the question of immediate event probability from the question of forecasting unresolved live cases. This avoids a model appearing strong because it predicts records whose outcomes were already effectively settled at the time origin. The detailed leakage chapter examines the available timestamp and overlap evidence without treating missing original exports as proof of either validity or invalidity.

## A reproducible synthetic comparison

The public smoke experiments each generate 1,200 synthetic observations and use a fixed chronological split: 593 training, 267 selection-validation, and 340 test rows. Both use the same prespecified model seed and one candidate configuration per family. One fixture contains nonlinear structure; the second follows a proportional-hazards regime. These experiments exercise all six estimators and the shared censoring, selection, scoring, and artifact-writing paths.

| Model | Nonlinear fixture test IBS | Proportional-hazards fixture test IBS |
|---|---:|---:|
| Kaplan–Meier | 0.242390 | 0.228664 |
| Cox proportional hazards | 0.180544 | 0.177519 |
| Random survival forest | 0.202620 | 0.194749 |
| Gradient-boosted survival analysis | 0.204181 | 0.189392 |
| Multilayer perceptron | 0.193811 | 0.190466 |
| Compact Perceiver mixture | 0.193862 | 0.183155 |

Lower integrated Brier score is better. Exact machine-readable values, selections, and protocols are retained in [the nonlinear aggregate](../../../research/examples/synthetic_smoke/summary.json) and [the proportional-hazards aggregate](../../../research/examples/synthetic_ph_smoke/summary.json). The comparison uses a common grid from 12 to 120 hours and training-derived censoring weights. It does not tune new configurations after observing the test table.

Cox has the lowest observed IBS in both fixtures. In the nonlinear fixture the two neural models outperform the two tree configurations, but Cox still performs better than either neural model. In the proportional-hazards fixture the compact Perceiver mixture outperforms the two tree configurations, while the dense neural model does not outperform the boosted model. These results directly contradict a blanket claim that either neural models or trees must win.

The paired difference between compact Perceiver and Cox IBS is approximately 0.01332 in the nonlinear fixture, with a 95% bootstrap interval of 0.00104–0.02441. In the proportional-hazards fixture it is approximately 0.00564, with interval −0.00305–0.01450. Positive differences favor Cox. The intervals use 500 paired resamples and are conditional on the fitted models; they are not a population-wide statement about model classes or uncertainty from repeated training.

The main result of these runs is methodological: the benchmark executes, freezes selection, produces comparable survival metrics, and reports an outcome that does not favor its most elaborate architecture. This is a useful property for subsequent real-data work. Synthetic smoke results remain a demonstration of that pipeline, not evidence that the historical operational model beats Cox or trees on genuine future listings.

## Extending the completed comparisons

Reconstructing a common-cohort comparison from the completed experiments requires a versioned decision-time cohort, explicit event provenance, resolved temporal overlap, train-fitted transforms, and the recorded model and policy choices. Extending it prospectively adds a final population that has not influenced those choices. Each duration region also needs enough observed events to assess its corresponding stage.

The most informative next study would combine the established operational evidence with a sealed future evaluation. Its first analysis would compare whole-curve performance and calibration across the six model families. Prespecified ablations would then isolate structured attributes, listing text, pooled visual vectors, and per-image reports. A temporal analysis would compare recent and older training windows while holding the test period fixed. A policy analysis would evaluate selected fraction, precision, recall, and explicitly stated utility, keeping prediction and intervention claims separate.

The current evidence supports a substantial experimental contribution: a large multimodal data system, hundreds of retained neural trial outputs, thousands of ensemble candidates, higher historical neural tail-screening metrics, operational policy artifacts and an executable neutral benchmark. A matched prospective comparison can establish how much of the recorded advantage persists under a common information set and later market conditions.
