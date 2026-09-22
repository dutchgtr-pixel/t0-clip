# Empirical evidence and its interpretation

## Three kinds of evidence

The available results belong to three distinct layers. Historical reports and saved predictions describe models developed on the operational dataset. Saved replay and restoration checks assess whether the software reproduced its recorded behavior. New public synthetic experiments test a reproducible comparison pipeline under known data-generating mechanisms. The layers support different claims and should not be combined into a single performance number.

The historical project is substantial: it progressed from temporal feature stores and AFT tail screening to multimodal neural models, stage-specific decisions, calibration, policy selection, and deployed inference. A critical review should acknowledge that progression while examining which final populations were independent of development. The public comparison provides a cleaner experimental scaffold, but its synthetic population does not establish superiority on the historical market data. Aggregate evidence files accompanying this chapter make the numerical distinctions inspectable without releasing private rows.

## Neural tail screening on a shared evaluation population

The retained exports establish a row-for-row comparison population: **all 964 neural evaluation records match the later XGBoost AFT export by entity, UTC time origin and duration label**. The AFT result recomputed on these 964 common observations is **F1 0.8560**, compared with the associated historical neural meta-ensemble's recorded **F1 0.9209**. This is an improvement of **about 6.5 F1 percentage points**. Precision increases from 0.8062 to 0.9802; recall changes from 0.9123 to 0.8684. The gain is a substantially more precise tail screen with a modest recall tradeoff.

| System on the 964-record evaluation population | Precision | Recall | F1 |
|---|---:|---:|---:|
| XGBoost AFT, recomputed common rows | 0.8062 | 0.9123 | 0.8560 |
| Neural top-20 stack, recorded holdout | 0.9307 | 0.8246 | 0.8744 |
| Neural optimized ensemble, recorded holdout | 0.9091 | 0.8772 | 0.8929 |
| Neural meta-ensemble, recorded holdout | **0.9802** | 0.8684 | **0.9209** |

The evidence has two explicit levels. The AFT value is recomputed from retained row-level predictions on the exact common keys. The neural ensemble values are retained execution results associated with the 964-row, 114-positive holdout stream. Retained neural prediction and evaluation exports also agree exactly in entity identity, UTC origin and duration in row-index order. The final meta-ensemble decision vector has not been recovered for a fresh paired bootstrap or disagreement test. This is a common-cohort aggregate comparison; it does not pretend that every final neural decision was re-executed during publication.

![Slow21 results on the shared 964-record evaluation population. The AFT value is recomputed from retained predictions; neural values are recorded ensemble results associated with that holdout. Source and final-decision recovery scopes are stated in the text.](../figures/r14_historical_neural_advantage.svg)

### Reconstructing the actual row relationship

The saved AFT export contains 971 unique entity-and-origin keys. All 964 neural keys are present, their duration labels match exactly, and no neural-only records remain. The seven AFT-only records comprise one correctly predicted slow outcome and six correctly predicted non-slow outcomes under the retained decision rule. Excluding them changes AFT F1 from $210/245=0.857143$ on 971 rows to $208/243=0.855967$ on the 964 common rows. Both exports cover the outcome week from 13 to 20 January 2026 and have no duration exactly at 504 hours.

| Evaluation record | Outcome window | N | Slow outcomes | AFT F1 |
|---|---|---:|---:|---:|
| Earlier AFT report | 6-13 January | 941 | 166 | 0.8462 |
| Later AFT run and retained export | 13-20 January | 971 | 115 | 0.8571 |
| Later AFT restricted to neural keys | 13-20 January | 964 | 114 | 0.8560 |

The contemporaneous later AFT run independently logs precision 0.8077, recall 0.9130 and F1 0.8571 on 971 observations, with confusion counts $(105,25,10,831)$. Its stored model identifier matches the retained prediction export. The shared-row recalculation therefore recovers a documented decision rule; it does not select a new threshold after inspecting the neural result.

This reconciliation explains the apparent 23-record difference in the earlier headline. The 941-row **evaluation** with F1 0.8462 belongs to the previous outcome week. In the later run, 941 also appears as the **calibration** population, with 166 positives. The actual recovered evaluation relationship is therefore **971 minus 7 equals 964**, rather than a verified 941-plus-23 subset. The earlier score remains part of the development history; it is not the tree score assigned to the later common rows. Differing sample counts did not establish unrelated model development, and the archive now supplies direct evidence of shared evaluation records.

The [cohort reconciliation](../../../research/thesis_evidence/cohort_reconciliation.json) records file and message hashes, exact time-window boundaries, cohort counts and reconstruction scopes. The [shared-cohort audit](../../../scripts/audit_shared_cohort.py) accepts private prediction exports with caller-specified neutral column mappings and emits aggregate checks without publishing identifiers. The [historical experiment receipt](../../../research/thesis_evidence/historical_neural_comparison.json) retains the original reports and ensemble-search evidence.

### What improved at the operating point?

On the common rows, AFT has $(TP,FP,FN,TN)=(104,25,10,825)$. Conditional on the meta result using the recorded 964-row, 114-positive holdout, its four-decimal precision, recall and F1 uniquely imply $(99,2,15,848)$. The latter matrix is inferred from rounded metrics and cohort counts, rather than copied from a separately printed meta confusion matrix.

The aggregate comparison has **23 fewer false-positive tail calls and five additional missed slow outcomes**. That is a meaningful change in operating behavior. It explains why precision and F1 improve while recall falls. Whether it is the best deployment policy depends on the relative costs of false-positive and missed-tail decisions; an F1 improvement alone is not a profit estimate. The F1 difference calculated from the two integer matrices is 6.4963 percentage points, consistent with the approximate 6.5-point statement above.

Both systems developed within SVAL and EVAL roles over the same source chronology. The later AFT calibration split has 941 records and 166 positives; the retained neural SVAL export has 933 records and the same 166 positive count, occupying the corresponding preceding outcome week. This supports common temporal roles while retaining model-specific row admissibility. It does not establish that every selection input or optimizer objective was identical. The preserved AFT optimizer scores EVAL labels, whereas the neural top-20 code selects its regularization and threshold on SVAL. The reported experiments are substantial retrospective development evidence, not a newly sealed future test.

Shared records control evaluation-population differences. They do not isolate the contribution of each representation, feature, survival objective, ensemble or selection decision. Those are components of the fitted systems being compared. Ablations can attribute the improvement more narrowly; they are not required to acknowledge the measured system-level F1 and precision advantage.

An earlier saved neural expected-duration output makes no tail calls when its mean is simply thresholded at 504 hours; the corresponding fixed-rule diagnostic is retained in the reconciliation receipt. It is a different output and checkpoint from the later dedicated tail-probability/meta system. Keeping this diagnostic prevents the final ensemble result from being attributed to every saved neural forecast or to a generic mean-duration threshold.

### A separate 23-record sensitivity calculation

The proposed 23-extra-record explanation can also be examined mathematically, without pretending it describes the dated exports. Under fixed labels and predictions, the inferred neural matrix $(99,2,15,848)$ has its smallest F1 after removing 23 observations when every removed observation is a true positive:

$$
F_{1,\min}=\frac{2(99-23)}{2(99-23)+2+15}
=\frac{152}{169}=0.899408.
$$

Removing all 17 errors and six true negatives yields an attainable maximum of 1.0000. Exhaustive enumeration of all 744 feasible confusion-cell deletion allocations confirms these sharp bounds. Thus adding 23 records alone could not explain the earlier reported F1 gap **under the specified unchanged-label subset assumption**. The audit identifies different weeks for the earlier 941-row report and later neural holdout, so this remains a counterfactual sensitivity result, not evidence that those particular 941 records were nested. The directly recovered 964-row intersection supplies the empirical cohort comparison.

The [aggregate sensitivity script](../../../scripts/analyze_cohort_sensitivity.py) reproduces the calculation and rejects the earlier 166-positive subset claim against a 114-positive parent. The bounds are not confidence intervals, do not include retraining or policy changes, and do not replace a paired uncertainty analysis of final decision vectors.

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

## Recovered fixed-input ensemble comparisons

The retained archive contains completed component comparisons beyond the February headline result. The new [experiment receipt](../../../research/thesis_evidence/experimental_contributions.json) records exact key matching, endpoint agreement, invariant-field agreement, source hashes, saved policies and selection scope. Reanalysis uses the existing binary decisions without refitting models or choosing new thresholds. These experiments must retain their separate cohorts and target definitions.

### Seven methods on the same Stage 0 population

Seven combination methods share 523 evaluation records, including 104 events lasting more than 504 hours. They operate on the retained ten-seed neural prediction surface. Thresholds and fitted stacks use the 369-row selection population; the stacks' scores on that same population are in-sample diagnostics. The table is a comparison of fitted combination methods and their policies, not a neural-versus-tree comparison on raw features.

\Needspace{22\baselineskip}

Table. Retained Stage 0 combination methods, with paired differences against mean-logit.

| Combination | Selection F1 | Evaluation F1 | Difference, 95% paired interval |
|---|---:|---:|---|
| Mean logit | 0.9595 | 0.8037 | Reference |
| Median logit | 0.9467 | 0.8073 | 0.0036 [-0.0140, 0.0237] |
| Mean probability | 0.9595 | 0.8057 | 0.0019 [-0.0174, 0.0205] |
| Vote | 0.9517 | 0.7938 | -0.0099 [-0.0541, 0.0329] |
| Weighted-logit stack | 0.9600 | 0.7909 | -0.0128 [-0.0411, 0.0162] |
| Gradient-boosted tree stack | 1.0000 | 0.7553 | -0.0484 [-0.1021, 0.0012] |
| SGD stack | 0.9536 | 0.8125 | 0.0088 [-0.0210, 0.0398] |

Every interval includes zero. The tree stack's perfect in-sample selection F1 is followed by lower evaluation F1 than the simple mean-logit combination. That observation supports retaining simple combinations as serious comparators. It does not establish that stacking is always harmful. The separate 472-configuration regularization sweep explicitly ranked holdout performance, so it belongs to development analysis; its winning result cannot be treated as a fresh independent confirmation.

### A paired Stage 1 hybrid comparison

The separate Stage 1 experiment contains 413 common records, including 296 positives for the retained event-within-168-hours target. The two methods use the same documented neural-prediction surface. The direct audit verifies matching record keys, labels, row ordinals, observed durations and event indicators; Stage 1 also preserves matching base probability, threshold and decision fields. The full underlying ten-seed score matrix was not independently reconstructed. The histogram-gradient-boosting meta-layer uses a different fitted score mapping and threshold from the mean-logit baseline.

Table. Saved Stage 1 policy results on 413 identical records.

| Policy | TP / FP / FN / TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Mean-logit baseline | 161 / 12 / 135 / 105 | 0.9306 | 0.5439 | 0.6866 |
| Neural-score/tree-stack hybrid | 256 / 33 / 40 / 84 | 0.8858 | 0.8649 | 0.8752 |
| All-positive reference | 296 / 117 / 0 / 0 | 0.7167 | 1.0000 | 0.8350 |

The hybrid gains 0.18865 F1 relative to mean-logit, with a paired 95% interval [0.14429, 0.23565]. It is correct on 97 records where mean-logit is wrong; mean-logit is uniquely correct on 23. The operating-point tradeoff is 95 fewer false negatives and 21 additional false positives. Against the all-positive reference, its F1 difference is 0.04024, with paired interval [0.00959, 0.07243] under the same resampling scope. The all-positive rule's precision, 0.7167, also falls below the saved policy precision floor of 0.8672; the hybrid's 0.8858 exceeds it on this cohort. The prevalence reference is informative but does not meet that operational constraint.

The intervals in both studies use 5,000 IID paired row resamples with seed 20260922. They condition on the fitted models and saved thresholds; they do not include training variability, tuning, experiment selection, temporal dependence or multiplicity adjustment. The complete history of final method selection and holdout inspection is not reconstructed. These uncertainty estimates quantify variation in fixed retained predictions under the stated resampling assumptions. They do not convert historical development into a preregistered test.

Together the studies demonstrate an executed component comparison and a stage/cohort-dependent result: a tree meta-layer is useful for the retained Stage 1 operating point, whereas it does not improve the retained Stage 0 mean-logit F1. The targets, populations and histories differ, so the between-study contrast is not a controlled causal estimate of stage identity. These results also rule out interpreting the system as a contest in which tree methods are inherently incapable of using neural representations.

## Fixed-model feature-source intervention and later replay

The retained forward-pass forensic report describes a frozen cohort of 443 observed records from 20-23 March 2026. There were 440 valid reconstructions, one degraded reconstruction and two invalid records; 441 received scores. After prior cohort/vector-contract fixes, a targeted change replaced the incorrect structured context and item-metadata relation with the decision-time feature source. The report records unchanged weights and labels and the same mean-logit ensemble threshold, approximately 0.720639.

Table. Documented fixed-policy source correction on the 441 scored records.

| Input-source state | Precision | Recall | F1 |
|---|---:|---:|---:|
| Before context-source correction | 0.5714 | 0.2105 | 0.3077 |
| After context-source correction | 0.8519 | 0.6053 | 0.7077 |

The corrected confusion counts are 46 true positives, eight false positives, 30 false negatives and 357 true negatives. This is a documented controlled input-source intervention, not retraining and not a new modality experiment. The complete paired before/after exports were not recovered at the report's recorded proof location, so this revision preserves its documentary evidence level. It supplies no invented paired confidence interval. The frozen cohort was reconstructed after outcomes and does not satisfy a sealed prospective protocol.

The report also records single phase-winner F1 values of 0.7647, 0.6929 and 0.7188 on the repaired benchmark. The deployed ensemble's 0.7077 is therefore not the highest recorded value on this diagnostic cohort. This is further evidence that ensembling and policy selection need direct evaluation rather than an assumed advantage.

A separate retained replay created on 20 April 2026 contains 587 matched records with 185 slow-tail positives. Fresh arithmetic gives base-policy F1 0.3701 and meta-policy F1 0.0909. The meta confusion matrix is 9/4/176/398; its precision is 0.6923 but recall only 0.0486. Saved serving checks report matching tabular metadata and no fallback path. Those checks do not establish complete decision-time feature fidelity, and the aggregate difference cannot isolate temporal decay. This unfavorable replay remains separate from the corrected 441-row experiment and from the February 964-row comparison.

## Which experimental gaps the recovered archive resolves

The archive establishes completed combiner comparisons, a documented frozen-policy source intervention, extensive development sweeps and reproducible reconstruction diagnostics. The present reanalysis adds paired uncertainty where row decisions are retained. It therefore supersedes a blanket characterization of all component experiments as merely proposed.

Other comparisons need more exact controls. The recovered context-present/context-absent training configurations also changed censoring minimums, dropout, loss weights and trial budgets. They cannot isolate the context block's incremental contribution. Matched raw-text, image, generated-report and fitted dual-band ablations were not recovered in the audited sources. Neither was a fixed-test recency ablation or a matched whole-cascade versus single-model experiment. The evidence ledger distinguishes these unresolved questions from the experiments actually completed; it does not infer their absence throughout every uninspected archive.

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

\Needspace{14\baselineskip}

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

Completing the score-to-snapshot reconciliation for the already shared-data experiments requires a versioned decision-time cohort, explicit event provenance, resolved temporal overlap, train-fitted transforms, and the recorded model and policy choices. Extending it prospectively adds a final population that has not influenced those choices. Each duration region also needs enough observed events to assess its corresponding stage.

The most informative next study would combine the established operational evidence with a sealed future evaluation. Its first analysis would compare whole-curve performance and calibration across the six model families. Prespecified ablations would then isolate structured attributes, listing text, pooled visual vectors, and per-image reports. A temporal analysis would compare recent and older training windows while holding the test period fixed. A policy analysis would evaluate selected fraction, precision, recall, and explicitly stated utility, keeping prediction and intervention claims separate.

The current evidence supports a substantial experimental contribution: a large multimodal data system, hundreds of retained neural trial outputs, thousands of ensemble candidates, higher historical neural tail-screening metrics, operational policy artifacts and an executable neutral benchmark. A matched prospective comparison can establish how much of the recorded advantage persists under a common information set and later market conditions.
