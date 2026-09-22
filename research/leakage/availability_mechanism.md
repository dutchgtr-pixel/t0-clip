# Coverage cutoffs and outcome-time cohort selection

## The demonstrated result

A feature block need not contain a future numerical value to become a misleading duration predictor. When an evaluation cohort is selected by its eventual event date, a coverage boundary indexed by prediction origin can encode duration through the geometry of that selection. This is distinct from a late correction that was unavailable at the intended prediction time. Both mechanisms can coexist, but different checks are needed to diagnose them.

The recovered historical record supports this more specific account of the population-context incident. A fresh audit of the two frozen evaluation files finds that the mixed zero-count/null-share pattern agrees exactly with a documented origin-date cutoff on **all 911 observations**. Their stored durations equal event time minus origin time to numerical precision. The previously reported missingness-only performance therefore has an explicit temporal explanation, rather than only a correlation with the endpoint.

This is a measured case, an elementary formal counterexample to a sufficient-safety claim, and a reproducible diagnostic. It does not establish that the general idea of selection bias or informative missingness is new. The contribution requiring comparison against prior work is the specific conjunction of asynchronous feature coverage, event-date survival cohorts, observed refresh behavior, and a practical audit that distinguishes clock violations from cohort-induced shortcuts.

## Exact mechanism and assumptions

For an observed event, write:

\[
O=\text{prediction origin},\quad S=\text{event time},\quad D=S-O\geq0,
\qquad Y_h=\mathbf 1[D\leq h].
\]

Let the evaluation select records satisfying \(a\leq S\leq b\). Suppose a coverage pattern follows a fixed origin cutoff:

\[
M=\mathbf 1[O\geq c].
\]

The indicator is defined from the feature block's zero/null pattern. It is not defined from the target. Within this cohort,

\[
M=\mathbf 1[D\leq S-c].
\]

Two sufficient conditions follow immediately:

1. If \(c\leq a-h\), then every fast event satisfies \(M=1\). Indeed, \(D\leq h\) implies \(O=S-D\geq a-h\geq c\).
2. If \(D>b-c\), then \(M=0\). Indeed, \(O=S-D\leq b-D<c\).

An additional end-of-window diagnostic follows without using a coverage feature: \(O\geq b-h\) implies \(D=S-O\leq h\). An origin-only predictor can therefore identify a mechanically fast region of an event-selected dataset.

These are deterministic statements under their stated assumptions, not significance tests. They do not require a causal effect of the coverage pattern on the event, access to a future value, or any predictive semantic content. Conversely, they do not imply that every calendar-related feature is illegitimate in deployment. They establish that performance in this selected cohort may answer a different question from performance on entities entering a system at prediction time.

The identities apply to observed events whose duration is defined by these two clocks. Censoring, clipped negative durations, updated origins, rounded event dates, and readiness delays need separate treatment. The audit rejects censored inputs, missing times, negative durations, and inconsistent duration identities rather than silently applying the proposition to them.

## Historical verification

The cutoff **2026-03-06 00:00 UTC** is taken from the retained reconstruction, then checked against the frozen data; it is not optimized against the endpoint in the new audit. Equivalent thresholds inside the gap between adjacent observed origins would yield the same classification, so the observations do not establish the exact instant of an underlying processing change.

| Quantity | Frozen validation | Frozen holdout |
|---|---:|---:|
| Observed-event rows | 408 | 503 |
| Fast events, at most 72 hours | 167 | 208 |
| Zero-count/null-share pattern | 219 | 335 |
| Pattern disagrees with origin cutoff | **0** | **0** |
| Maximum duration-identity error, hours | Less than 1e-12 | Less than 1e-12 |
| Fast events without the pattern | **0** | **0** |
| Pattern-only rule precision | 0.7626 | 0.6209 |
| Pattern-only rule recall | 1.0000 | 1.0000 |
| Pattern-only rule F1 | **0.8653** | **0.7661** |

Both cohorts satisfy \(c\leq\min(S)-72\text{ hours}\). Thus the perfect recall of this particular mask follows from the reconstructed cohort geometry. It is not evidence that the underlying population values identify all fast events in an origin-selected deployment population.

The machine-readable [receipt](availability_mechanism_results.json) contains file hashes, aggregate counts, timing residuals, sufficient-condition checks and confusion counts. No identifiers, source rows, private paths or numerical feature vectors are included. The audit reads only the timestamps, endpoint columns and two pattern-defining scalar columns. The earlier [aggregate record](aggregate_evidence.json) independently records the same 408/503 cross-tabs; this analysis adds direct verification of the date-cutoff relationship and duration identity.

The retained historical documentation also reports an exact forensic lookup replay of **47,704 rows across 13 context columns**: 46,793 training rows, 408 validation rows and 503 holdout rows. That proves faithful lookup restoration in the documented experiment, not correctness of the restored feature surface. The database was kept paused during the present review, so this count is explicitly documentary evidence rather than a newly rerun database comparison.

A later full-preparation audit identified a related failure after the original block had been removed: broad calendar-varying priors could stand in for origin time. Its documented origin-only diagnostic F1 values are 0.8261 and 0.8805. Inspection of the audit implementation establishes that it searches threshold and direction **separately using each split's own labels**. These numbers are descriptive shortcut diagnostics, not scores from a validation-selected classifier transferred unchanged to holdout. The original detailed output files were not recovered at the documented location in this review. The contemporaneous response removed broad temporal proxies and required an origin-time cohort with mature follow-up for final acceptance.

## A constructed counterexample and negative control

The public experiment creates the complete Cartesian product of 121 origin days and 30 possible durations: **3,630 artificial records**. Origin and duration are exactly independent. Every record has complete follow-up, and the fast-event rate is exactly 10%. A feature has a constant historical numerical value; its unavailable-state flag is \(M=\mathbf1[O\geq100]\), a calendar regime known at origin. No future event is used to construct that flag.

The same fixed, unfitted rule predicts a fast event whenever \(M=1\).

| Constructed evaluation | Rows | TP / FP / FN / TN | F1 | Balanced accuracy |
|---|---:|---|---:|---:|
| Select events on days 104 through 107 | 120 | 12 / 10 / 0 / 98 | 0.7059 | 0.9537 |
| Select origins on days 96 through 107; retain complete follow-up | 360 | 24 / 216 / 12 / 108 | 0.1739 | **0.5000** |
| Same first 120 rows, labels and fixed rule; fill the missing block | 120 | 0 / 0 / 12 / 108 | 0.0000 | 0.5000 |

The first comparison changes cohort selection and consequently its population; it is a controlled demonstration of selection geometry, not a paired remediation effect in the historical system. In the origin-selected control, both coverage groups have exactly 10% fast events. The apparent predictive relationship disappears. The final row is a paired artificial coverage intervention: only the mask changes. It illustrates why refreshing coverage can break a fixed shortcut-based decision rule without changing its input schema.

No model is trained, no threshold is selected, no random seed or bootstrap is involved, and no significance interval is appropriate for this enumerated finite construction. Its purpose is to disprove the claim that admissible feature values and timestamps alone guarantee an evaluation free of selection-induced shortcuts. It is not a replacement for the historical refresh experiments or an estimate of their neural-model effect.

## What the combined evidence supports

The historical data demonstrate an exact coverage/origin relationship and its algebraic connection to the endpoint in selected cohorts. The recorded reconstruction and refresh investigation connect that pattern to an operated feature pipeline. The constructed experiment establishes that a similar apparent advantage can arise with no semantic signal and with an origin-admissible mask. Together they justify checking the **cohort construction as well as the evidence clock**.

They do not establish that the historical mask itself was recorded and available at the intended origin. That requires the historical availability log, which is not supplied by a date-cutoff fit. They do not prove that every performance difference was caused by this block, that all repaired models are safe, or that removing every predictive missingness flag is beneficial.

An effective audit therefore has separate questions:

- Was each source version actually knowable at the decision time, and was the label available at fitting time?
- Was inclusion in evaluation determined using an outcome that a deployed decision would not know?
- Does a coverage mask or broad calendar proxy identify the endpoint through that inclusion rule?
- Does the relationship persist when eligibility is fixed at origin and outcome follow-up is subsequently observed?
- On fixed rows with frozen preprocessing, checkpoint and policy, does changing only the feature source change decisions?

The final question isolates model dependence; the preceding questions explain why a dependence may misrepresent deployment performance. A missingness association is a diagnostic flag, not a universal leakage verdict.

## Reproduction and diagram specification

Run the artificial experiment with:

```bash
python scripts/analyze_availability_mechanism.py
python -m pytest tests/test_availability_mechanism.py
```

For an authorized private dataset, use the configurable column mapping; Parquet additionally requires a supported pandas Parquet engine. The command emits only aggregate results and a content hash:

```bash
python scripts/analyze_availability_mechanism.py \
  --input /private/observed_events.csv \
  --origin-column origin_at --event-column event_at \
  --duration-column duration_hours --observed-column event_observed \
  --count-column context_count --share-column context_share \
  --cutoff 2026-03-06T00:00:00Z --horizon-hours 72
```

For a publication diagram, place origin \(O\) and duration \(D\) as independent parent nodes feeding event date \(S=O+D\). Draw \(O\to M\) for the coverage regime, \(D\to Y_h\) for the endpoint, and \(S\to Q\) for selection into an event-date window. Shade \(Q=1\) to indicate conditioning. Show the induced association between \(M\) and \(Y_h\) as a dashed path, not a causal arrow. An adjacent origin-selected panel conditions on a function of \(O\) and retains full follow-up; under the constructed independence assumption the mask no longer predicts duration. A third panel contrasts this mechanism with an actual late-created feature, whose availability time exceeds the origin. The latter is caught by an information-availability predicate; cohort-induced association requires a separate design check.
