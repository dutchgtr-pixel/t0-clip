# Research claim ledger

Version 0.1 — 2026-09-21. The distinction between historical evidence, synthetic execution, and a pending real-data experiment is part of the result, not a footnote.

## Evidence categories

**Public historical source** means a historical project report that readers can inspect. **Retrospectively audited aggregate** means saved private predictions or code were inspected, but the row-level evidence is not released here. **Public implementation** means code is available; it does not imply a model was fitted on real data. **Synthetic execution** means the documented software run completed on artificial data. **Pending** means a proposed claim has not been established.

| ID | Bounded claim | Evidence and status | What it does not establish |
|---|---|---|---|
| C01 | Earlier work reported Slow21 precision 0.8314, recall 0.8614, F1 0.8462 on 941 sold listings | Public historical [empirical chapter](../../papers/part-3-modeling/ch05-empirical-results-sacrifice.pdf), pp. 2–4 | An untouched prospective all-listing test; neural superiority; performance on a different horizon |
| C02 | The same report gives fast-stratum gate false-positive fraction 0.0059 and middle-stratum fraction 0.2688 | Same source, duration strata below 240 h and 240–504 h | Lost profit, causal harm, or a universal false-positive rate |
| C03 | Calibration-cohort results were produced on the cohort used to fit the calibrator | Same historical report | Out-of-sample calibration quality |
| C04 | Historical feature engineering included temporal comparable windows, named feature contracts, and training guards | Historical [feature-store paper](../../papers/part-2-t0-certified-feature-stores.pdf), [governance chapter](../../papers/part-3-modeling/ch02-governance-layer.pdf), and public `feature-stores/`/`governance/` code | Universal historical observability or a permanent leakage-proof platform |
| C05 | A historical market-context missingness shortcut was reproduced and affected features were banned in the inspected successor lane | Retrospective code and saved-row audit, 2026-09-21; sanitized account in [manuscript §6.1](../../papers/marketneural/manuscript.md#61-a-missingness-shortcut-that-was-detected-and-addressed) | That every old score is valid, or that the current lane still uses the banned block |
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
| C18 | Neural models improve over Cox/trees on controlled real marketplace data | **PENDING / NOT ESTABLISHED**; [protocol](PROTOCOL.md) | No superiority wording is permitted before a valid result |
| C19 | The model improves real decision utility or profit | **PENDING / NOT ESTABLISHED** | Prediction scores do not establish causal or economic effects |

## Interpretation rules

The earlier long-duration and later short-duration tables use different targets and populations. They are historical context, not points on a single progress chart. A nested recent slice must retain its relationship to its parent cohort. In-sample tail diagnostics remain valuable regression evidence when labeled correctly.

“Certified” means a named dataset or feature version passed declared checks. Hashes establish identity, not truth. Missingness checks identify suspicious associations, not automatic proof that every association is leakage. A repaired feature lane does not retroactively validate the earlier affected lane.

Historical infrastructure run totals are not repeated here as an unconditional uptime claim. A defensible total needs a stated service inventory, window, denominator, and retry/failure semantics. Successful restores or parity checks demonstrate those checks, not all possible operational failure modes.

No published synthetic ranking will be promoted to a real-world claim. Negative results will remain in the evidence ledger. If a claim changes, preserve the prior wording, date the amendment, identify newly available evidence, and state whether the old test influenced the change.

## Availability of historical evidence

The public PDFs are independently readable. The later audited aggregates rely on private artifacts inspected during preparation; raw records and their identities are deliberately not redistributed. Readers can inspect the limitations and arithmetic summaries, but cannot reproduce those historical scores from this repository alone. The public synthetic artifacts and future controlled-study manifests are separate evidence paths designed to improve that reproducibility boundary.
