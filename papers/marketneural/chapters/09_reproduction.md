# Reproduction, release boundaries and further evaluation

## What the public package reproduces

The release contains three distinct model resources. The compact `marketneural` package implements a complete temporal comparison harness with six estimator families. The production-reference package preserves selected numerical definitions and architecture metadata for the later slot-based network. The cascade package preserves historical numerical definitions and adds portable controllers for three-stage fitting, probability combination, threshold selection, routing and local persistence. These resources are complementary; they are not interchangeable versions of one fitted model.

The repository also contains historical feature-store SQL, a dependency inventory, source-neutral derivatives of selected later SQL, and a portable empty-database contract with an artificial fixture. The portable contract was executed in an isolated in-memory PostgreSQL engine. Parsing historical SQL establishes syntactic validity within the recorded parser; it does not establish that every historical dependency exists in a fresh database or that rebuild statements are appropriate for a production installation.

No trained private checkpoint, category vocabulary, raw listing table, contact record, authentication secret or production connection is distributed. Selected anonymous photographs and their saved judgments are included as explicitly reviewed illustrative evidence. Their pairing and selection limitations are documented separately. The release provides reproducible algorithms, contracts and aggregate evidence while preserving the distinction between public demonstrations and the original operational environment.

The underlying datasets, including Parquet observations and training exports, remain proprietary and are not publicly downloadable. Researchers may request controlled access from the maintainer, subject to explicit approval and separately agreed access or licensing terms. The public software license does not grant access to those datasets. Approved delivery would use a separate controlled channel. The repository's [data-access policy](../../../DATA_ACCESS.md) records this boundary.

## Installation and execution

Use Python 3.11 or newer and an isolated environment. The development extra supplies the test runner, PDF inspection support and PostgreSQL parser. A CPU installation is sufficient for the compact checks; the original selected configurations are much larger than the smoke configurations.

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python -m research.cascade --smoke
python -m research.production_reference.training --smoke
```

Activate the environment using the command appropriate to the local shell before installation. Each smoke command performs actual numerical work on artificial inputs. Its short training budget and small architecture make it suitable for checking the path, not estimating final predictive performance. Shape-compatible arrays alone do not establish that a caller's embeddings satisfy the information boundary.

The six-model experiment is separate:

```sh
python -m marketneural synthetic --output data/synthetic --n 1200 --seed 2026
python -m marketneural benchmark --config configs/synthetic_smoke.json --output results/example
```

The configuration names its input paths. Use a new output directory for each experiment, and copy and update the configuration when using another dataset location. The runner refuses to overwrite existing outputs. This protects against accidental replacement; it cannot technically prevent an analyst from revisiting a test period.

For offline SQL inspection, run the inventory module and focused feature-store tests. The optional PGlite script executes only the five portable SQL files against an in-memory engine. The recorded result contains seven temporal fixture assertions and three challenged certificate failures: an empty uncertified state, changed content and an expired certificate. One artificial entity exercises these cases; it is not a scale benchmark or proof of production concurrency behavior.

## Reproducing the mechanism and retained-policy audits

The coverage-cutoff experiment is fully public and requires no proprietary data:

```sh
python scripts/analyze_availability_mechanism.py
python -m pytest tests/test_availability_mechanism.py
```

It enumerates an artificial population with independent origins and durations, applies two cohort-selection rules, and reports a fixed mask-only rule. Its historical-data mode accepts explicitly mapped timestamp and pattern columns and emits aggregate counts and hashes. It rejects censored inputs and inconsistent clock identities because the demonstrated proposition concerns observed event durations.

The retained-policy comparison tool accepts authorized private prediction exports and emits no row identities:

```sh
python scripts/analyze_retained_ensemble_comparison.py --help
python -m pytest tests/test_retained_ensemble_comparison.py
```

Every model must supply the same complete unique key set, endpoint labels and configured invariant fields. The tool rejects mismatches rather than silently taking a convenient intersection. Paired IID bootstrap intervals condition on the retained fitted decisions and their thresholds. They do not account for search, temporal dependence or final-method selection. Public aggregate receipts preserve the exact settings and evidence boundaries; proprietary input files remain available only through separately approved access.

## Building the monograph

The PDF is built from chapter files, methods chapters, reviewed historical crops, selected image cases, a bibliography and plots generated from public aggregate JSON. ReportLab produces the title page. Pandoc and Tectonic typeset the body, equations, contents, references and figures. The build script resolves inputs from the repository and records their hashes. It requires the `paper` extra and explicit executable paths when the typesetting tools are not on the system path.

```sh
python -m pip install -e ".[paper]"
python scripts/build_thesis.py --pandoc pandoc --tectonic tectonic
```

The first Tectonic run may obtain its public typesetting bundle. Intermediates stay under the ignored results directory. The publication artifact is `papers/marketneural/marketneural-thesis.pdf`; chapter files and the figure generator remain editable source. A development-only partial-build option exists for layout work. A release build must include every required chapter and pass visual and source-content review.

Rebuilding can change PDF byte hashes when dependencies or typesetting resources change. The release manifest binds review to the published bytes. A newly generated PDF requires inspection and a new reviewed hash; a matching filename is not evidence of unchanged content.

## What each validation establishes

Numerical tests check event and censor likelihoods, finite-support behavior, survival monotonicity, mixtures, horizon interpolation and stage-score direction. Training-adapter tests check optimizer steps, validation selection and bundle round trips. Policy tests check routing order, missing scores, constraints and fitted ensemble state. Temporal tests challenge unavailable features, repeated entities, split boundaries, label maturity and train-only preprocessing. SQL tests check parsing, dependency classification and the portable contract's declared cases.

Historical operational reports remain separate from newly executed checks. A seventy-test narrative report concerns explanation behavior. A five-thousand-row parity report concerns implementation agreement. A restoration report concerns its declared files and relations at its recorded date. None is relabeled as a new experiment. The release review records the final current test result, command scopes and resolved review findings.

Publication screening checks source terminology, private paths, secret-like assignments, data artifacts, archive members, PDF text and hash-bound reviewed images. Pixel content requires visual review because text extraction cannot read all embedded figures. This review corrected a source-specific word inside a historical raster diagram that an earlier text-only PDF audit could not see. That finding supports combining automated screening with visual inspection rather than treating either as complete.

The review scope is the current release tree. Earlier repository history is not rewritten. Source-neutral current files do not imply that every historical commit was anonymized. The release also does not restart or modify the paused private production stack.

## Reconstructing the comparisons and extending evaluation

The completed historical experiments include the XGBoost baseline, neural training trials, ensemble searches and the improved neural tail-screening results documented in the empirical chapter. Reconstruction should first connect each retained model, prediction file, target convention and selection decision. A further prospective experiment can then compare frozen models on a common future cohort with documented feature availability. Its protocol should predeclare the event definition, censoring treatment, initial decision rule, entity grouping, horizons, permitted features, training window and final observation cutoff. Original fit data, learned transforms and model selection must remain under content hashes so later repairs cannot silently replace the experiment.

Information ablations should compare structured-only inputs, structured plus text, structured plus images, and the full representation. Generated image reports require a separate ablation. If ordinary and whitened vectors are included, the whitening transform must be fitted inside training and preserved with its dimensions and population hash. Conventional estimators should receive meaningful reduced or regularized representations with equivalent information and a documented tuning budget.

Recency weighting should be evaluated against an unweighted baseline and alternative half-lives across multiple chronological origins. The historical half-lives demonstrate an implemented adaptation mechanism; only a controlled ablation establishes which setting is useful under a particular regime. Tests should also examine delayed enrichment, unavailable slots, changing category mix, stale prices and label-detection delays.

Selection and calibration must finish before final-test inspection. Report survival and policy results together: censoring-aware prediction error, discrimination, horizon calibration, precision and recall at a frozen policy, coverage, and eligible and excluded counts. Uncertainty intervals must state whether they condition on fitted models or include repeated training and selection. Small cohorts and overlapping slices must not be presented as independent replications.

Decision value needs its own study. A model can improve a statistical metric while making no useful difference to an operator or agent. Prospective shadow evaluation should record what was reviewed, what information was available, which proposals were accepted, what executed, and which outcomes became observable. Economic analysis must include costs and unresolved outcomes rather than assume that fast turnover implies profitable action.

These extensions test claims beyond the completed retrospective studies. The present release already contains measured failure analysis, fixed-policy feature reconstruction, comparisons of retained ensemble decisions, and a common-population neural/tree comparison. Each has a different experimental control. The new mechanism and experiment receipts make those controls explicit rather than describing all ablations as future work. Further evaluation should preserve the same distinction when a result is unfavorable.


## What a sealed future cohort would establish

A successful prospective replication would establish that the frozen system predicted later outcomes using information actually available before those outcomes. If it also met a prespecified comparison criterion against equally frozen baselines, it would support prospective comparative effectiveness for the registered population, time period and endpoint. That is a stronger transfer claim than retrospective model development. It is distinct from the originality of the method and does not make the completed experiments, leakage findings or engineering contributions disappear.

The eight operational requirements have concrete meanings in this platform:

1. **Freeze feature definitions.** Preserve the feature schema, source eligibility rules, decision-time SQL, image/report slot definitions and missing-value policy under hashes. The public feature-store contracts provide these kinds of boundaries; the study manifest must identify the exact versions used.
2. **Freeze preprocessing.** Preserve fitted encoders, category vocabularies, scalers and any projection or whitening transform. Fit them before the sealed period. The portable cascade explicitly requires the caller to retain its fitted feature encoder separately from the saved model bundle; a checkpoint alone is not a complete preprocessing freeze.
3. **Freeze the model.** Hash every stage checkpoint and ensemble member, its architecture/configuration and dependency versions. The cascade persistence path records stage membership and training audit fields. Successful save/load and numerical replay establish reproducibility of outputs; they are not evidence that a future cohort was sealed.
4. **Freeze calibration.** Record the fitted mapping, its training population and the horizon it calibrates. If a score has no independently fitted calibrator, state that fact rather than adding one after the sealed results are visible. Calibration and ranking answer different questions.
5. **Freeze policy and thresholds.** Record all stage thresholds, routing order, eligibility filters and abstention rules. The saved cascade policy and threshold-selection code supply the operational mechanism. Any policy change creates a new version evaluated on a subsequent cohort.
6. **Record predictions before outcomes.** Persist decision timestamp, feature-vintage hash, model/transform/policy versions, scores and decision before the event is known. Exclude already resolved cases at enrollment. Both neural and tree systems must receive the same admissible evidence for each registered comparison. Append-only or signed prediction receipts are the prospective evidence; a later replay of historic inputs is a different test.
7. **Allow outcomes to mature.** Prespecify follow-up, observation lag, event adjudication and censoring. A 21-day tail study must allow the horizon and reporting lag to pass. Disappearance, withdrawal and confirmed completion must follow the registered endpoint definition; they must not be silently interchanged when scoring.
8. **Evaluate without test-driven revision.** Close the cohort before examining the primary endpoint. Apply the locked policy and uncertainty method to all eligible records, including failures and abstentions. Report drift and calibration alongside discrimination and operating metrics. Retuning after inspection belongs to the next study, while the original result remains recorded.

The source material establishes components for feature contracts, fitted model persistence, policy routing and serving replay. It does not include one completed prospective receipt tying all eight conditions to the current comparison. That is the precise additional evidence being requested. Hundreds of training trials and tests answer substantive development and software questions; a sealed cohort answers a different question about predictions made into an unknown future.

Success must be defined before the outcomes are read. A primary horizon F1 criterion evaluates that policy; it does not by itself establish superiority of the whole survival distribution. A curve-level study instead needs an appropriate censored-data score, fixed evaluation grid and prespecified calibration analysis. The study should name the minimum useful effect, required event support, paired uncertainty method and treatment of multiple comparisons. Cohort size should be justified using the endpoint prevalence and plausible paired disagreements, rather than chosen after a favorable score appears.

A successful first window would support prospective validity in that window. Repeated windows and independent replication would strengthen evidence that the contribution transfers under market change. Neither result would automatically isolate architecture from all other system components or prove trading profit. Conversely, methodological and systems contributions can be supported by rigorous retrospective studies and reproducible failure analysis; prospective evaluation is an additional claim-specific experiment, not a universal definition of scientific originality.
