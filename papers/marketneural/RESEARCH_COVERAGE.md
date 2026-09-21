# Research coverage and evidence traceability

This map records how the expanded research manuscript uses all nine earlier public papers and the inspected operational evidence families. Historical sources are evidence of the project's documented methods and results; their earlier wording is not automatically adopted as a universal guarantee. The new manuscript develops the methods in connected prose, uses eight tightly cropped historical figures, and separates descriptive historical results from the executable synthetic comparison and the planned controlled real-data experiment.

The final chapter numbering is assigned during typesetting. The stable chapter filenames and figure identifiers below provide source traceability independently of the rendered page numbers. Separately authored method sources are the [cascade chapter](../../research/cascade/METHODS.md), [leakage study](../../research/leakage/LEAKAGE_STUDY.md), and [portable feature-store chapter](../../research/feature_store_ddl/METHODS.md).

## Coverage of the nine historical papers

| Historical paper | Principal material incorporated | New manuscript destinations | Figure or formula traceability | Interpretation boundary |
|---|---|---|---|---|
| [Part 1: resilient data observation](../part-1-infra-resilient-data-observation.pdf) | Repeated observation, stable identity, staged processing, operational resilience and recovery | Chapters 02 and 07; agentic-decisions chapter | Operational counts and restoration tables use separately audited retained evidence | A historical design description does not prove every present deployment setting |
| [Part 2: decision-time certified feature stores](../part-2-t0-certified-feature-stores.pdf) | Feature-block composition, keys, decision-time contracts, stock pressure, historical comparisons | Chapters 01, 02, 05 and 06; leakage and schema chapter | H07 on page 13; conditional inventory discussion | Exploratory stock association is not a controlled performance increment; reconstructed stock needs an availability check |
| [Time leakage chapter](../part-3-modeling/ch01-time-leakage.pdf) | Temporal causality, label windows, event availability, forbidden future information | Chapters 01, 02 and 08; leakage and schema chapter | Decision-time measurability and censoring formulas | Tests prove named invariants under named timestamp semantics, not absence of every leakage mechanism |
| [Governance layer chapter](../part-3-modeling/ch02-governance-layer.pdf) | Registry, dependency closure, schema and dataset fingerprints, freshness and training preflight | Chapters 02 and 07; leakage and schema chapter | Feature-contract and experiment-manifest descriptions | A structural digest cannot establish the truth of an external event timestamp |
| [Tail gating and anchors](../part-3-modeling/ch03-tail-gating-and-anchors.pdf) | AFT distributions, tail objectives, conditional survival and anchor-based comparisons | Chapters 01, 05 and 06; neural cascade chapter | H01 on page 4; AFT and residual-time equations | Historical AFT implementation and compact public boosted baseline are different estimators |
| [Hierarchical Bayesian anchors](../part-3-modeling/ch04-hierarchical-bayesian-anchors.pdf) | Support-aware shrinkage, backoff hierarchy and robust price comparisons | Chapters 01 and 02 | H02 on page 3 and H03 on page 4; support-weight formula | Robust weighted anchors should not be called a conjugate posterior without the required distributional model |
| [Empirical results and sacrifice](../part-3-modeling/ch05-empirical-results-sacrifice.pdf) | Slow-tail confusion matrices, duration-band sacrifice and feature gain | Chapter 06 | H04 and H05 on page 3, H06 on page 5; exact cohort table | Historical selection independence is unresolved; sacrifice is not realized financial cost |
| [Slow-tail AFT stock supplement](../part-3-modeling/supplement-slow21-aft-stock-augmented.pdf) | Extended modeling configuration, feature families, tail emphasis, stock and attribution analyses | Chapters 01, 02, 05 and 06; neural cascade comparison | Feature-block inventory and distinction between attribution and ablation | Fitted gain or attribution is not a causal effect or proof of independent predictive contribution |
| [Generative enrichment and repair](../part-4-generative-ai-self-healing.pdf) | Text/vision decomposition, validation, reconciliation and repair feedback | Chapters 03 and 07; actual image case studies | H08 on page 8; enrichment ownership table | Later reconciliation must not be fed backward into an earlier decision-time input |

The [figure catalog](FIGURE_CATALOG.json) records every source PDF's page count and SHA256, a caption inventory, and exact one-based page and crop bounds for H01–H08. Each selected crop has its own SHA256, evidence class, and claim limits. Figure selection is a synthesis rather than a replacement of thesis chapters with complete source pages.

## Private evidence families represented publicly

Raw operational rows, personal information, source identifiers, local paths, and private endpoints are not redistributed. Neutral source receipts in [source_register.json](../../research/thesis_evidence/source_register.json) identify inspected code and document versions by hash. These hashes identify the bytes reviewed; they are not cryptographic proof that the source's assertions are correct.

| Evidence family | Content actually inspected | Public treatment | Main destination |
|---|---|---|---|
| Visual interpretation workers | Field ownership, bounded prompts, parsing retry, connection retry, completion and error behavior | Developed method description with exact behavioral limits | Chapter 03 |
| Text interpretation worker and recovered text encoder | Recurring execution, canonical tagged input, content hashes, masked mean pooling and normalization | Reconstruction provenance preserved; no invented exact original implementation | Chapters 02, 03 and 07 |
| Identity-correction incident | Conflict between visual proposal and textual consensus; stricter multi-view guard and reversal audit | Concrete incident and implemented guard, without claiming calibrated confidence | Chapter 03 |
| Eight-slot image/report design and builder | Role selection, padding, tensor dimensions, frozen encoders, hashes and status fields | [Documented build counts](../../research/thesis_evidence/image_slot_build.json) with undated-build warning | Chapters 03 and 05; neural cascade chapter |
| Actual image and paired interpretation snapshots | Retained visual evidence and corresponding generated judgments | Anonymized, visually reviewed case panels; generated judgments distinguished from truth | Image case-study chapter |
| Historical trainer, model and serving packages | Stage horizons, routing, tokenization, expert mixtures, losses, fitted transforms and deployment outputs | Exact selected-generation descriptions and distinctions from compact benchmark | Neural cascade chapter; Chapters 01, 05 and 07 |
| Feature exports, target audit and saved policy predictions | Missingness incident, availability rules, cohort nesting, zero-duration sensitivity and possible overlap | [Historical policy aggregates](../../research/thesis_evidence/historical_policy_aggregates.json); leakage limitations retained | Chapter 06; leakage and schema chapter |
| Retained database metadata and scheduler history | Storage bytes, relation counts, workflow states, execution interval and duplicates | [Operational counts](../../research/thesis_evidence/operational_counts.json) with denominators and receipt hashes | Chapter 07 |
| Saved restoration reports | Six reports, matching file hashes, critical relation counts and profiles | Historical checks, not a newly performed restore | Chapter 07 |
| Saved serving and narrative checks | Model replay, numerical tolerances, explanation tests and representation parity | [Historical checks](../../research/thesis_evidence/historical_checks.json), each with its distinct scope | Chapter 07 |
| Implemented decision and analytical services | Controlled analytical queries, context manifest, caching, serialization, streaming, quotas, timeouts and lifecycle | Actual architecture separately identified from decision theory | Agentic-decisions chapter |
| Fraud and spam pipeline | Filters, evidence measures and operational decision boundaries | Implemented detection distinguished from adjudicated fraud truth | Fraud and spam chapter |
| Platform operating history | More than 70,000 successful Airflow workflow runs; rapid market movement and emphasis on recent data | Lifetime scale is distinct from the retained scheduler subset; implemented recency weighting is documented separately from optimal-window experiments | Chapters 03, 07 and 08 |
| Representation design | Ordinary-plus-whitened text vectors | Fitted dual-band transform artifacts remain to be recovered; the selected inspected network uses one listing-text vector | Chapters 05 and 08 |

## New public experiment

The executable benchmark supplies Kaplan–Meier, Cox proportional hazards, random survival forest, gradient-boosted survival analysis, a dense neural baseline, and a compact Perceiver mixture. It uses common temporal partitioning, cutoff censoring, train-only preprocessing, validation configuration and threshold selection, and frozen selections before test prediction. The actual nonlinear and proportional-hazards smoke results are retained under [research/examples](../../research/examples/README.md).

The experiment uses three blocks. A separate calibration block and a matched real-data prospective comparison remain part of the research protocol. No table in the expanded manuscript converts synthetic results into a real-market superiority claim. Cox has the best observed integrated Brier score in both saved fixtures. The compact model is not asserted to be an exact reproduction of every archived cascade stage.

## Reviewer checklist

- Confirm that every empirical table names its cohort, outcome, unit, and selection status; the recent 283-row population is nested in the 748-row holdout.
- Confirm that dedicated-head historical policy scores are distinguished from survival-curve probabilities, and that persistence-oriented logits are complemented only where the inspected adapter requires it.
- Check that 72 hours is evaluated within its grid interval using the implementation's interpolation; it is not falsely called a bin boundary.
- Check that AFT, proportional-hazards boosting, discrete hazards, and survival-expert mixtures are described as distinct formulations.
- Check that missing tokens remain a possible learned signal, including where raw unavailable values are masked or replaced.
- Verify that source availability, transform completion, decision time, and event time remain distinct throughout all chapters and captions.
- Treat zero-duration exclusion as sensitivity analysis, not an automatically corrected target or proof that all zero durations are erroneous.
- Confirm that model replay, narrative tests, restoration checks, and predictive evaluation have separate scopes and denominators.
- Report more than 70,000 successful Airflow workflow runs as the platform lifetime total; label the retained snapshot as a subset rather than a lifetime census.
- Distinguish constrained generation from mathematical determinism and distinguish generated attribute confidence from calibrated correctness.
- Check the ordinary-plus-whitened representation account against any subsequently recovered fitted transform; do not infer a second inspected neural text branch from an operator description.
- Separate implemented recency parameters and refresh mechanisms from evidence that a particular 60-day window is optimal or that online learning was deployed.
- Confirm that historical image panels show admissible public content, that paired judgments are properly aligned, and that illustrative crops carry evidence limits.
- Verify source-neutral wording across prose, metadata, figure text, links, and archives; text extraction alone cannot detect every word embedded in a raster image.
- Run the public release audit, local-link check, reference validation, numerical table check, and rendered-page review after final integration.

## Review status

Independent mathematical cross-review of the foundations, enrichment, comparison, and results chapters found no major mismatch and corrected the image/report contract to retain separate availability masks. Operational cross-review verified the database counts, scheduler states, restoration counts, and receipt hashes against the retained audit evidence. A separate reader reviewed the orientation, image case studies, agentic decisions, and fraud extension for scope and source-neutrality. Final typesetting and integration include the separately authored cascade and schema/leakage methods. The parent publication workflow records the final release checks and visual inspection of the complete PDF; these are distinct from the content reviews recorded here.
