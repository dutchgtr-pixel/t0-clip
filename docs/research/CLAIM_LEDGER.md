# Research claim ledger

Version 0.4 — 2026-09-22. Historical evidence, synthetic execution and proposed experiments retain their own populations and measurement scopes.

## Evidence categories

**Public historical source** means a historical project report that readers can inspect. **Retrospectively audited aggregate** means saved private predictions or code were inspected, but the row-level evidence is not released here. **Public implementation** means code is available; it does not imply a model was fitted on real data. **Synthetic execution** means the documented software run completed on artificial data. **Pending** means a proposed claim has not been established.

| ID | Bounded claim | Evidence and status | What it does not establish |
|---|---|---|---|
| C01 | Earlier work reported Slow21 precision 0.8314, recall 0.8614, F1 0.8462 on 941 sold listings | Public historical [empirical chapter](../../papers/part-3-modeling/ch05-empirical-results-sacrifice.pdf), pp. 2–4 | An untouched prospective all-listing test; neural superiority; performance on a different horizon |
| C02 | The same report gives fast-stratum gate false-positive fraction 0.0059 and middle-stratum fraction 0.2688 | Same source, duration strata below 240 h and 240–504 h | Lost profit, causal harm, or a universal false-positive rate |
| C03 | Calibration-cohort results were produced on the cohort used to fit the calibrator | Same historical report | Out-of-sample calibration quality |
| C04 | Historical feature engineering included temporal comparable windows, named feature contracts, and training guards | Historical [feature-store paper](../../papers/part-2-t0-certified-feature-stores.pdf), [governance chapter](../../papers/part-3-modeling/ch02-governance-layer.pdf), and public `feature-stores/`/`governance/` code | Universal historical observability or a permanent leakage-proof platform |
| C05 | A historical market-context missingness shortcut was reproduced and affected features were banned in the inspected successor lane | Retrospective code and saved-row audit, 2026-09-21; [leakage study](../../research/leakage/LEAKAGE_STUDY.md) | That every old score is valid, or that the current lane still uses the banned block |
| C06 | Earlier tail diagnostic construction reused the training-period source and did not exclude those cases from fitting | Retrospective split-code audit; complete overlap verified in an available rebuilt snapshot | Exact reconstruction of an unavailable original locked training matrix; unbiased unseen-tail accuracy |
| C07 | A later locked policy had approximately 76.69% precision and 64.83% recall on a 748-row holdout | Retrospectively audited saved-prediction aggregate | Public end-to-end reproduction, prospective marketplace effectiveness, comparison with C01 |
| C08 | A 283-row recent slice had approximately 85.08% precision and 70.00% recall | Retrospectively audited aggregate; all 283 occur within C07's 748 | An additional independent holdout or a combined validation size of 1,031 |
| C09 | Zero-duration composition materially changes C07's threshold metrics | 178/748 zero-duration rows; diagnostic exclusion leaves 570, precision approximately 66.30%, recall approximately 62.24% | That every zero is corrupt; a corrected ground truth; justification for post-hoc exclusion |
| C10 | Source metadata supplied both historical duration endpoints in the inspected lineage | Retrospective source-observation code and selected-row audit | Independently verified transaction time or an explanation of every identical timestamp |
| C11 | Some historical scores could be generated after the label's original time origin | Retrospective serving-code audit | That an unconditioned Fast72 score means sale within 72 h from the current score time |
| C12 | The project embodies operationally substantial engineering | Data observation/orchestration, feature governance, model pipelines, recovery and parity evidence described in historical reports | Zero failures over an undefined service set; any implication about predictive accuracy |
| C13 | The compact public benchmark implements six named survival estimators and a shared input/evaluation contract | [`marketneural/`](../../marketneural/) and [model card](MODEL_CARD.md) | A faithful reproduction of the entire earlier private production training system |
| C14 | The public nonlinear synthetic smoke run completed for all six estimators | [Aggregate summary](../../research/examples/synthetic_smoke/summary.json); seed 42; 1,200 synthetic records | Real marketplace performance, a general ranking of model classes, or a tuned large-scale comparison |
| C15 | Cox had the lowest observed test IBS in that smoke run | IBS 0.180544 versus MLP 0.193811 and compact Perceiver-MoE 0.193862 | Universal Cox superiority; a contradiction of the proposed neural research question |
| C16 | The compact benchmark checks temporal boundaries, entity disjointness, train-only preprocessing, and survival-output validity | Public [`data.py`](../../marketneural/data.py), [`models.py`](../../marketneural/models.py), [`metrics.py`](../../marketneural/metrics.py) | Verification of raw upstream timestamps, hidden image/text content, or delayed-label availability |
| C17 | A selected earlier numerical architecture is preserved with source and symbol hashes | [`research/production_reference/provenance.json`](../../research/production_reference/provenance.json) | Released private weights/data, a complete production controller, or a reproduction of historical metrics |
| C18 | Retained AFT/neural exports share 964 exact keys and duration labels; recomputed AFT F1 is 0.855967 versus associated neural meta aggregate 0.9209, with higher precision and lower recall | **SHARED-COHORT AGGREGATE RESULT**; [reconciliation](../../research/thesis_evidence/cohort_reconciliation.json) | A recovered final neural decision vector, paired significance, architecture-only causality, or universal superiority |
| C19 | The model improves real decision utility or profit | **PENDING / NOT ESTABLISHED** | Prediction scores do not establish causal or economic effects |
| C20 | The historical cascade contains three ordered survival-policy stages, preserved numerical definitions, portable fitting and inference, ensemble selection and persistence | [Cascade methods](../../research/cascade/METHODS.md), source hashes and executable tests | A single calibrated conditional-duration factorization, or reproduction of historical metrics without the private training artifacts |
| C21 | A retained database snapshot contains 55,260 listing rows and 240,622 image assets within 29.1 GB | [Operational evidence](../../research/thesis_evidence/operational_counts.json), snapshot dated 28 April 2026 | An equal number of independent survival outcomes or a complete lifetime census |
| C22 | The platform completed more than 70,000 successful Airflow workflow runs. A retained snapshot contains 15,360 runs over 40 workflow identifiers, including 15,254 successes | [Operations chapter](../../papers/marketneural/chapters/07_operations.md); lifetime operation and the retained scheduler subset have different time coverage | First-attempt success or system uptime |
| C23 | Retained configurations implement continuous age weighting with 30-day and 23-day half-lives | [Temporal platform](../../papers/marketneural/chapters/02_temporal_platform.md) and [cascade methods](../../research/cascade/METHODS.md) | An empirically proven optimal window or a hard two-month cutoff |
| C24 | Task-specific image/text generation, bounded schemas, validation and field ownership were implemented for enrichment | [Generative enrichment](../../papers/marketneural/chapters/03_generative_enrichment.md); [two approved image cases](../../papers/marketneural/chapters/04_image_case_studies.md) | A measured hallucination rate or independent perceptual accuracy from illustrative examples |
| C25 | The selected later K8 network contains 17,145,736 parameters and combines structured, text, image and report tokens | [Multimodal comparison](../../papers/marketneural/chapters/05_multimodal_comparison.md) and preserved source/configuration | Proof of superiority over trees or Cox, or confirmation that a raw-plus-whitened text branch was fitted in this selected network |
| C26 | A typed proposal service, deterministic action checks, Rust coordination and an operator-approved execution path were implemented | [Agentic decisions](../../papers/marketneural/chapters/10_agentic_decisions.md) | Audited profitable autonomy or a proven survival-powered decision policy in every deployed context |
| C27 | Existing quality controls support an architecture for future fraud/spam research | [Fraud and spam chapter](../../papers/marketneural/chapters/11_fraud_and_spam.md) | A supervised fraud classifier validated against adjudicated outcomes |
| C28 | Underlying research datasets are proprietary and excluded from the public release; researchers may request separately approved access | [Data access policy](../../DATA_ACCESS.md) | A public dataset download or automatic data rights under the code license |

## Interpretation rules

The earlier long-duration and later short-duration tables use different targets and populations. They are historical context, not points on a single progress chart. A nested recent slice must retain its relationship to its parent cohort. In-sample tail diagnostics remain valuable regression evidence when labeled correctly.

“Certified” means a named dataset or feature version passed declared checks. Hashes establish identity, not truth. Missingness checks identify suspicious associations, not automatic proof that every association is leakage. A repaired feature lane does not retroactively validate the earlier affected lane.

Historical infrastructure run totals are not repeated here as an unconditional uptime claim. A defensible total needs a stated service inventory, window, denominator, and retry/failure semantics. Successful restores or parity checks demonstrate those checks, not all possible operational failure modes.

No published synthetic ranking will be promoted to a real-world claim. Negative results will remain in the evidence ledger. If a claim changes, preserve the prior wording, date the amendment, identify newly available evidence, and state whether the old test influenced the change.

## Availability of historical evidence

The public PDFs are independently readable. The later audited aggregates rely on private artifacts inspected during preparation; raw records and their identities are deliberately not redistributed. Readers can inspect the limitations and arithmetic summaries, but cannot reproduce those historical scores from this repository alone. The public synthetic artifacts and future controlled-study manifests are separate evidence paths designed to improve that reproducibility boundary.

## Empirical evidence amendment: 22 September 2026

The previous edition marked the broad neural-over-Cox/trees claim pending and
described the real-data comparison as not run. Recovered user-pasted execution
logs, neural trial-pool manifests and ensemble-search histories establish
completed experimentation and higher recorded neural tail-screening metrics.
C18 now states that bounded historical result. The same-cohort and prospective
questions remain distinct. These historical evaluation results were already
visible during development; this amendment records existing results and does
not select a newly fitted model.

## Dated-run reconciliation and novelty revision

The earlier 941-row AFT result is now linked to its6-13January outcome window.
A later971-row AFT export and964-row neural export share the13-20January window;
all964 neural keys and duration labels match. The tree score on those common
rows is0.855967. The final neural meta score0.9209 remains a logged aggregate
associated with that cohort. The new contribution chapter states specific
implemented findings and their precedents. A sealed future cohort would add
prospective predictive validity; it is not used to erase completed retrospective
experiments or automatically settle novelty.
