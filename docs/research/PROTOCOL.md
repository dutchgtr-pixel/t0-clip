# MarketNeural controlled comparison protocol

**Version 0.1 — 2026-09-21. Status: prospective specification; controlled real-data comparison NOT RUN.**

This document defines the next scientific experiment. It is not a claim that the experiment has happened. The compact synthetic benchmark implements a smaller execution lane and is described separately below. Any amendment after access to final-test labels must be dated and labeled exploratory.

## 1. Question and estimand

The primary question is whether a multimodal neural survival model improves event-time prediction over the strongest registered conventional comparator on future marketplace listing episodes, with comparable information and a fixed resource budget. The unit is one initial eligible decision per related-listing entity. The primary time origin is the actual decision instant when the required evidence is available. The event is a documented source-reported sold transition unless a stronger transaction endpoint is separately validated.

The estimand is the conditional survival distribution from that decision among eligible, live, score-ready listings. A metadata-origin historical score and a score generated for older live inventory are different estimands. Repeated landmarks, withdrawals as competing risks, causal price optimization, and profit prediction are outside the primary experiment.

## 2. Hypotheses

| ID | Hypothesis | Comparison | Status |
|---|---|---|---|
| H1 | Full neural model lowers integrated Brier score | Neural versus strongest conventional model chosen on development data under matched inputs | Not tested |
| H2 | Text and images add predictive information | Same model family with structured-only versus shared representations | Not tested |
| H3 | Token fusion improves beyond a fixed pooled representation | Attention versus pooled fusion with matched evidence and documented capacity | Not tested |
| H4 | Expert mixture improves on a single hazard head | Mixture versus no-expert ablation | Not tested |
| H5 | Any predictive advantage survives coverage, latency, and missing-modality constraints | Accuracy and operational measurements together | Not tested |

H1 is confirmatory once this protocol's remaining fields are frozen. H2–H5 are secondary, with multiplicity and exploratory status disclosed. A null or negative finding is a valid study result.

## 3. Items that must be frozen before the final test

The dataset does not yet supply public values for these real-data design fields. Leaving them explicit prevents a draft from being mistaken for a completed preregistration.

| Required field | Current state | Freeze criterion |
|---|---|---|
| Observed population and decision eligibility | Pending | Written inclusion/exclusion rules independent of future outcomes |
| Endpoint mapping and uncertain-status treatment | Pending | Raw transition audit and documented timestamp meaning |
| Four chronological date windows | Pending | Dates set before accessing final-test outcomes |
| Label observation lag and maturity policy | Pending | Fit-time availability justified for each label source |
| Related-item grouping and seller sensitivity | Pending | Hash/version of grouping implementation |
| Primary time grid and censoring estimator | Pending | Adequate development follow-up and fixed support rule |
| Minimum relevant effect and sample-size target | Pending | Development-only power/simulation analysis; no post-hoc margin |
| Compute budget and model search spaces | Pending | Per-family trials, time budget, hardware, seed plan recorded |
| Policy utility or precision constraint | Pending | Chosen from operational need before final-test review |
| Final-test access custodian or automated seal | Pending | No trial-level test metric exposure |

The controlled study does not begin until these fields have immutable values in a run manifest. The draft alone is not a registered experiment.

## 4. Data admission and temporal contract

Each raw fact has a source event time and an observation/availability time. An admissible feature depends only on facts available by the decision. Source timestamps alone do not prove availability. Historical text/images must have immutable version evidence; a deterministic representation recomputed from those bytes may be admissible for an explicitly retrospective representation study. Real-time readiness is evaluated separately.

The admission report must contain:

- Raw snapshot hashes, feature-definition revision, transform/encoder versions, and source licenses or permissions.
- Count of eligible episodes, events, censoring, uncertain terminal states, zero/negative durations, unresolved horizons, and unscored records.
- Availability audit for every feature family, including seller state, market aggregates, and enriched modalities.
- Missingness by lifecycle and label group, plus a missingness-only diagnostic baseline.
- Duplicate and related-entity overlap checks across every partition.
- Preprocessing fit-scope checks and explicit rejection of outcome-derived feature names and known unsafe sources.

A zero duration is quarantined for endpoint review rather than silently clipped. The final real-data decision to include, exclude, or interval-code such rows is frozen before final evaluation. The public loader's positive-follow-up requirement does not settle that scientific question.

## 5. Split and outcome construction

The real-data design uses four chronological blocks: **fit → selection → calibration/policy → final test**. Outcomes used to fit a model must be knowable at its simulated fit time. Training labels are censored at that cutoff; future eventual outcomes cannot leak into an earlier fold. Outcome-derived features are rebuilt as of each fold, not joined from a fully matured all-history export.

The selection block chooses architecture, regularization, binning, and ensemble composition. The next block fits calibration and the decision rule after architecture selection. The final test evaluates the frozen pipeline once. A walk-forward extension repeats this sequence using new future blocks while preserving the access rules. The earlier overlapping tail diagnostics remain regression tests and are excluded from claims of unseen-tail accuracy.

Related-listing groups cannot span partitions. One initial decision per entity is the primary design; repeated landmarks require an amended protocol and dependence-aware analysis. A seller-disjoint sensitivity assesses transfer to unseen sellers. A recent subset of the final test is a nested drift diagnostic, not extra independent sample size.

## 6. Comparators and information budgets

The complete comparison includes a cohort survival prior, penalized Cox, random survival forest, a gradient-boosting survival model, a boosted AFT model, a simple hazard MLP, and the multimodal attention/expert candidate. The compact implementation currently provides `km`, `coxph`, `rsf`, `gbsa`, `mlp`, and `perceiver_moe`; boosted AFT is an additional planned comparator.

Three information levels are evaluated separately:

1. Structured features only, identical admitted columns across model families.
2. Structured features plus identical fixed text/image representations, with the same underlying evidence and missingness indicators.
3. Token-level neural fusion versus an explicitly specified pooled representation, reported as a representation-and-architecture comparison.

The compact benchmark currently gives conventional and MLP models masked flattened slot values, while attention tokenizes those values. This is a common-information demonstration, not the entire three-level ablation study. Pretrained encoder choice, fine-tuning, feature selection, dimensionality reduction, and calibration must be counted in each model's information and computation budget.

Each family receives a reasonable predeclared search space and documented resource budget. Failed runs count toward reporting. No final-test-driven expansion of a favored model's search space is permitted. Seeds, parameter counts, wall-clock training, peak memory, feature extraction cost, and inference latency accompany accuracy.

## 7. Analysis plan

The proposed primary endpoint is integrated Brier score over a fixed grid, using a registered censoring estimator and support threshold. The grid cannot extend beyond credible follow-up. A marginal censoring model assumes independent censoring and transfer to the later evaluation cohort; feature-dependent censoring and changing collection practice require sensitivity analysis or a different estimator.

Secondary outputs are horizon Brier scores, calibration curves and summary calibration error, IPCW concordance, complete-outcome horizon precision/recall at the locked threshold, coverage, and operating workload. Binary diagnostics disclose censored unresolved records and do not call them negatives. The final report includes prevalence, confusion matrices, sample sizes, and the exact cohort definition next to every threshold metric.

Paired confidence intervals compare models on identical test cases. Resampling respects entity groups and time dependence. Compact smoke intervals condition on fitted models and censoring weights and do not include retraining uncertainty. The full study reports model-seed variation separately and uses a registered plan for repeated training or nested resampling if that uncertainty is to be included. Secondary ablations do not inherit the primary hypothesis's error guarantee.

Policy thresholds are fitted on the policy block. A validation precision target is not a future precision guarantee. If no feasible threshold exists, the result records that failure or an explicit reject-all policy; it does not lower the constraint after inspecting the test. Economic analysis uses measured costs where available and names hypothetical costs otherwise.

## 8. Required sensitivity analyses

Endpoint analyses compare the registered primary mapping with plausible interval bounds and a separately labeled zero-duration population sensitivity. Availability analyses exclude feature families with weaker historical version evidence. Population analyses compare all eligible scored records with stricter readiness windows and report excluded/unscored rates. Model analyses include no-text, no-image, no-market-anchor, no-expert, and simple-fusion ablations within a predeclared feasible budget.

Tail analysis uses a disjoint, mature, later cohort. A case already used in fitting cannot become an unseen tail test merely because a different target is scored. Group sensitivity examines related episodes and unseen sellers. Drift analysis compares time blocks and predeclared product groups, retaining uncertainty where group counts are small.

## 9. Public synthetic execution lane

The smoke lane generates artificial rows and vector arrays, uses three chronological blocks, administratively censors train labels before validation and validation labels before test, and selects configurations and thresholds on validation. It does not fit a separate calibration layer or implement the four-block real-data design. It is a software demonstration.

From the repository root, after installing the documented dependencies:

```shell
python -m marketneural synthetic --output data/synthetic --n 1200
python -m marketneural benchmark --config configs/synthetic_smoke.json --output results/synthetic_smoke
```

Output directories must be new where required by the CLI. Existing results are not overwritten to hide failed or changed runs. The exact commands, data/config/code hashes, versions, split audit, selection metrics, final metrics, and aggregate reports identify each run. Published example summaries live under `research/examples/synthetic_smoke/` and must retain a prominent synthetic designation.

Successful execution supports an implementation claim only. The default generator deliberately includes nonlinear and nonproportional-hazard structure; a separate PH generator is available. A ranking on either generator reflects those assumptions and the small tuning budget. It is not evidence that a model class is best for real listings.

## 10. Publication gate and amendments

A real-data comparison can be reported as confirmatory only after the freeze fields are completed, temporal/group audits pass, the endpoint is documented, and the final test is evaluated under the sealed configuration. Publish aggregate tables, uncertainty, coverage, failure counts, and enough configuration to reproduce the analysis where data access permits. Do not publish raw listing content, identifiers, contact details, private paths, or fitted vocabularies derived from private records.

If a leakage or label problem is discovered, preserve the original result, mark its limitation, issue a new version, and obtain a new clean confirmation cohort when the old test influenced repairs or selection. An amendment log records what changed, why, when, and which outcomes had already been viewed. A correction is stronger evidence of research integrity than silently replacing a result.
