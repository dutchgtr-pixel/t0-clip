# Reproduction, release boundaries and the next decisive experiment

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

## The experiment needed for a stronger modeling claim

The next decisive experiment is a matched, forward-in-time comparison on a frozen real-data cohort with documented feature availability. Its protocol should predeclare the event definition, censoring treatment, initial decision rule, entity grouping, horizons, permitted features, training window and final observation cutoff. Original fit data, learned transforms and model selection must remain under content hashes so later repairs cannot silently replace the experiment.

Information ablations should compare structured-only inputs, structured plus text, structured plus images, and the full representation. Generated image reports require a separate ablation. If ordinary and whitened vectors are included, the whitening transform must be fitted inside training and preserved with its dimensions and population hash. Conventional estimators should receive meaningful reduced or regularized representations with equivalent information and a documented tuning budget.

Recency weighting should be evaluated against an unweighted baseline and alternative half-lives across multiple chronological origins. The historical half-lives demonstrate an implemented adaptation mechanism; only a controlled ablation establishes which setting is useful under a particular regime. Tests should also examine delayed enrichment, unavailable slots, changing category mix, stale prices and label-detection delays.

Selection and calibration must finish before final-test inspection. Report survival and policy results together: censoring-aware prediction error, discrimination, horizon calibration, precision and recall at a frozen policy, coverage, and eligible and excluded counts. Uncertainty intervals must state whether they condition on fitted models or include repeated training and selection. Small cohorts and overlapping slices must not be presented as independent replications.

Decision value needs its own study. A model can improve a statistical metric while making no useful difference to an operator or agent. Prospective shadow evaluation should record what was reviewed, what information was available, which proposals were accepted, what executed, and which outcomes became observable. Economic analysis must include costs and unresolved outcomes rather than assume that fast turnover implies profitable action.

This sequence would convert the current systems study into a stronger empirical contribution. The present release makes mechanisms, historical successes and discovered weaknesses inspectable. Its scientific credibility depends on preserving that openness when the next experiment produces an inconvenient result.
