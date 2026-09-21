# Operational engineering as a condition for reproducible research

## What was actually operated

The project is not merely a collection of model notebooks. Its retained artifacts show a coordinated platform for repeated data observation, structured interpretation, feature construction, vectorization, model inference, policy explanation, and recovery. The operator reports approximately 70,000 successful automation runs across the platform's lifetime. That is a platform-wide historical account. A separately inspected metadata snapshot preserves 15,360 workflow runs from 40 distinct workflows; this narrower retained population neither independently establishes nor contradicts the larger lifetime count.

The distinction between retained evidence and reported lifetime experience is essential. Systems migrate, histories are pruned, and backup dates can be later than the last retained execution. A retrospective reviewer should count the records that exist, state their scope, and avoid converting an incomplete retained database into a claim that earlier operation did not happen. The evidence is sufficient to establish sustained, multi-component operation while keeping the 70,000-run total explicitly attributed.

The retained scheduler history spans execution timestamps from September 16, 2025, at 23:30 UTC through March 31, 2026, at 07:03:18 UTC. Its snapshot was retained on April 29, 2026. It contains 15,254 successful, 103 failed, two running, and one queued workflow states, with no duplicate primary-key rows in the audited extraction. Among 15,357 terminal states, the stored success fraction is 99.3293%. This is a final-state workflow statistic, not an uptime measurement, an independent first-attempt success rate, or a guarantee that every successful workflow produced semantically correct data.

One retained survival-inference workflow contributes 262 successes and three failures. Those counts establish that inference was part of recurring operation. They are not the total number of predictions, the number of model training experiments, or the total number of stage-specific jobs. A single workflow may process a batch; another may perform maintenance. Workflow runs, task instances, scored rows, and listing observations should never be summed as though they were the same unit.

## The retained data scale and its denominators

The audited April 28 metadata bundle describes a database of 29,106,213,679 bytes, approximately 29.11 decimal gigabytes. Its principal relation counts are summarized below. The public [operational evidence record](../../../research/thesis_evidence/operational_counts.json) preserves the unit definitions and audit-receipt hashes.

| Retained object | Count | Interpretation |
|---|---:|---|
| Listing records | 55,260 | Stored rows, not an independently deduplicated lifetime entity population |
| Image assets | 240,622 | Multiple assets may belong to one listing |
| Image-feature records | 228,958 | Machine-derived feature records, potentially versioned |
| Structured text-enrichment records | 54,841 | Derived interpretations, not raw text tokens |
| Text-vector records | 62,244 | 768-dimensional records with their own coverage and versions |
| Image-vector records | 80,688 | 512-dimensional records, not one vector for every stored image |
| Post-event audit records | 34,307 | Later-stage audit evidence requiring a separate temporal boundary |

These denominators matter scientifically. The vector count can exceed the listing count because source coverage and versions differ. The image-feature count is not a number of independently labeled survival outcomes. The post-event audit count is evidence of a repair and reconciliation workload, not permission to use those records as initial-time predictors. Treating the database as one flat number of samples would overstate both training independence and temporal admissibility.

The system's scale also explains why operational design is part of the research contribution. Hundreds of thousands of visual records require stable identity, repeatable transforms, resource limits, failure isolation, and selective reprocessing. Without these controls, a model comparison would conflate architecture changes with changing or partially processed inputs. The platform makes it possible to ask a stable scientific question over heterogeneous evidence.

## Scheduling, readiness, and partial progress

The historical architecture uses scheduled workflows and specialized workers with explicit responsibilities. Text interpretation, image interpretation, vector construction, feature refresh, and survival inference can advance at different rates. A listing may be known to the platform before it has enough image or text evidence for the intended model. Readiness therefore needs its own state rather than being inferred from the listing's existence.

The inspected serving design reports eligible, blocked, and ready populations, including missing text, image, and image-slot evidence. Work is leased in bounded batches and scored outputs retain their execution time and version. A model result created after a substantial delay remains a result at that later decision time; its existence should not be backdated to the first observation. This makes latency an information-availability issue as well as a service-performance issue.

Failure isolation supports continuous progress. A malformed image response can leave a particular item incomplete while other items continue. A connection retry can recover a transient database failure without marking every item done. A content hash can prevent redundant vector recomputation. These mechanisms explain how a complex pipeline can operate repeatedly rather than requiring a human to restart a monolithic experiment after every local error.

There are limits to what a success state proves. A task may finish with partial coverage under an explicitly allowed policy. A cache may serve an older but valid artifact. A semantic mistake can pass a syntactic validator. Meaningful operational metrics therefore include readiness, stale-output age, failed-item counts, retry counts, and evidence completeness in addition to top-level workflow state. The retained run statistics show operational maturity, while these additional measures define the next level of observability.

## Replaying the model rather than only preserving its weights

Neural deployment requires a bundle, not just a checkpoint. The bundle includes the architecture, fitted preprocessing, feature names and order, category vocabulary, vector revisions, masks, output interpretation, calibration, policy thresholds, and reference inputs and outputs. Any one of these can change a decision while the checkpoint file remains identical. The historical serving work recognized this and preserved explicit replay fixtures and manifests.

A saved 523-row contract comparison from an older model generation checked expected and served identities, 12 seed or meta decision and bucket columns, and 91 numeric columns. The maximum meta-probability difference was approximately $1.01\times10^{-7}$, and the maximum point-estimate difference was approximately $9.16\times10^{-5}$ hours. Two separately captured 32-row live replays agreed exactly across 90 numeric columns. These are strong results for implementation parity within their stated artifacts and tolerances.

Parity has a precise scope. It establishes that two execution paths compute the same function on the checked inputs. It cannot establish that the function is well calibrated, that the labels are correct, or that its inputs were historically available. The older proof package also recorded a not-ready-for-live-forward status at creation. That status should not be silently promoted into certification of a later policy or current deployment. The research value lies in preserving and checking the evidence, including its release status.

One bounded archived inference run leased and scored eight rows with no failures. Its recorded stage duration was approximately 0.838 seconds, including roughly 0.773 seconds in forward computation. The vectors were already available. This supports the feasibility of a warm batch execution path, not a cold-start latency claim or a percentile service-level objective. Latency evaluation should separately measure evidence preparation, queue delay, model loading, forward computation, and publication of a usable decision.

## Explanations are a separately tested product

The platform also translates structured predictions and evidence into user-facing explanations. This layer has its own correctness risks: reversing a score's meaning, describing hidden damage as observed, treating an absent image as evidence of good condition, or presenting a stored prediction as fresh. Explanation tests must therefore validate evidence use and output semantics, not merely punctuation.

The saved narrative contract suite contains 70 passing tests. A separate representation-parity check compared two deterministic narrative implementations on 5,000 sampled rows from 6,049 eligible records and found zero mismatches. These results concern explanation logic and formatting. They must not be relabeled as 5,000 neural-model accuracy checks or 70 successful training experiments. Keeping the scopes separate makes both accomplishments more credible.

The implementation spans more than one language and service. Narrative calculation and wrappers should be attributed to their actual components; the presence of a compiled agent service does not make all explanatory logic part of that service. The later agentic-decisions chapter describes the real service boundary, controlled analytical access, cached context, streamed responses, and decision workflow. The operational point here is that a trustworthy explanation is an independent, testable interface between a probabilistic model and a human decision.

## Recovery and integrity of research artifacts

Recovery evidence is unusually concrete. Six saved restoration reports declare successful checks. The most recent, dated April 25, compares 10,230 files by hash and checks counts for 23 critical relations and eight data profiles, with exact matches on the declared checks. This is materially stronger evidence than the mere existence of backup files: it shows that retained copies were restored and compared against specified expectations.

The checks are still scoped. File equality does not prove application correctness, and matching relation counts do not detect every possible row-level difference. The hash comparisons and data profiles strengthen that evidence, but they do not imply every table and every operational dependency was tested. The publication reports the historical saved checks; it does not claim that a new restoration was performed during the review.

Recovery matters for research because feature and model artifacts can otherwise become irreproducible after an incident. A complete experiment may depend on a particular canonical text revision, encoder artifact, fitted scaler, slot manifest, model checkpoint, policy file, and prediction export. If only the database or only the weights are retained, the result may not be reconstructable. The architecture's backup and manifest work addresses that systems-level reproducibility problem.

The distinction between source code recovery and faithful historical reconstruction also deserves attention. Some retained components are reconstructed from a documented production contract because an original internal body was unavailable. Such a component may reproduce the documented interface without proving exact bitwise identity with the lost implementation. A publication should preserve that provenance rather than labeling every recovered file as the original execution source.

## Adaptation in a rapidly changing market

The operator observes that recent data, particularly roughly the latest two months, is especially important in this market. The project contains corresponding mechanisms: recent comparable windows, feature refresh, incremental representation updates, changed-input rescoring, model retraining, and stage-specific recency parameters in the tuning material. The historical anchor documentation includes thirty- and sixty-day windows. The stage-tuning documentation exposes recency half-life settings, including a search range of five to 35 days in one procedure. Inspected selected configurations use 30 days for Stage 0 members and 23 days for Stage 1.

These observations support the need to study temporal adaptation. They do not establish that a fixed 60-day window is universally optimal. A short window reduces staleness but also removes rare examples and tail events. A long window improves support but can average over a changed price regime, item mix, or observation process. Exponential weighting offers an intermediate option,

$$
w_i=2^{-\Delta_i/H},
$$

where $\Delta_i$ is age relative to the chosen reference date and $H$ is a half-life. The inspected trainer mean-normalizes these weights before using them. Thus a 30-day half-life downweights a 60-day-old observation to one quarter of an otherwise equivalent newest observation before common normalization; it does not discard every observation older than 60 days. The exact production weight remains tied to the selected training configuration and its reference date. The model chapter gives the implementation-level trace.

A controlled recency experiment fixes a future test period and compares training windows or weights selected on an earlier validation period. It reports not only mean error but event support in each stage, calibration drift, and readiness changes. The observed positive-prevalence shift from 56.34% to 63.10% to 77.74% across selected historical populations motivates such an analysis, but selection and nested cohorts prevent interpreting it as a direct estimate of market-wide acceleration.

Operational adaptation is therefore a sequence of versioned decisions. A refresh changes inputs; a recalibration changes probability interpretation; a new threshold changes action selection; a retraining changes the predictive function. Each has a different validation burden. The retained platform supplies much of the infrastructure needed to implement those changes safely. Evidence of autonomous online learning or measured improvement after every refresh is not required to recognize that substantial engineering, and is not claimed here.

## Engineering quality and scientific value

The most persuasive operational strengths are concrete: repeated workflow execution, large multimodal stores, owned field contracts, explicit readiness, numerical replay, explanation parity, and verified restoration. The main maintenance burden is also recognizable: many independently versioned components, recovered historical variants, cross-language interfaces, and complex lifecycle-dependent feature availability. These are the kinds of dependencies emphasized in the systems literature on machine-learning technical debt. [Sculley et al., 2015](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html).

The appropriate assessment is consequently neither a toy prototype nor a proof of flawless production behavior. It is a substantial operational research platform whose engineering made the modeling program possible. Its next scientific advance depends on using that capability to preserve a sealed sequence of decision-time predictions and outcomes. Its next engineering advance depends on making contracts, ownership, provenance, and release gates easier to inspect as the number of model and evidence versions grows.
