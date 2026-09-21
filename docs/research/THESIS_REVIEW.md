# Release review: methods, execution and evidence boundaries

Review date: 21 September 2026. This technical review records the implementation
checks, evidence review and document inspection for the research release.

## Scope and division of evidence

The review examined the cascade implementation, its preserved numerical sources,
portable fitting and persistence adapters, tests, methods chapter, selected
operational evidence, feature-store contracts, leakage analysis, publication
builder and release audit. Private evidence was inspected read-only. Published
hashes identify the reviewed sources or receipts; they do not make private
datasets publicly reproducible.

The older three-stage cascade and the later 17,145,736-parameter slot-based model
are separate families. The release distinguishes original numerical definitions,
selected historical configurations, new portable controllers, artificial tests,
retained historical measurements and proposed experiments. Preserved source and
syntax-tree hashes must continue to match; source-neutral infrastructure examples
are explicitly adapted references and are not interchangeable with untouched
historical numerical bodies.

## Methodological review

The cascade accepts training outcomes and development outcomes during fitting.
Prediction accepts features, with no outcome argument. The artificial example
holds its final eight rows outside fitting and development. Train-fitted
preprocessing has tests for held-out extreme values and unseen categories.
Ensemble fitting uses declared development rows; prediction reuses fitted
calibrators, covariance-derived weights and stackers. Tests perturb excluded
rows and confirm that the fitted result does not change.

This interface separation does not by itself enforce chronological, entity or
campaign independence. The caller can still supply overlapping cohorts or
features assembled after the intended decision time. The primary cascade
controller also reuses its development population for neural early stopping,
ensemble fitting and threshold selection. The documentation correctly treats
that reuse as selection, not untouched evaluation. The separate inner/outer
ensemble-selection helper can support a stronger protocol, but it cannot turn
neural checkpoints already selected on those same rows into independent folds.

The historical 504-hour downstream training cap is explicit. It differs from
the predecessor-gated development population and uses outcome information to
define the retrospective fitting population. A future chronological experiment
must reconstruct labels at its fit cutoff rather than assuming the published
controller does this automatically. The default demonstration objective also
does not claim to reproduce every historical trial coefficient.

Numerical tests cover exact horizon boundaries, unresolved short censors,
fractional survival exposure, late-event recensoring, monotonic survival,
required-stage score failures and routing precedence. Each actual stage is
trained on artificial tensors and its saved/restored predictions are compared.
The complete cascade likewise round-trips its models, ensembles and policy.
Loading its learned ensemble bundle requires explicit trust because its
serialization format can execute code. Fitted feature-encoder persistence remains
the caller's responsibility and is documented.

## Findings corrected during review

Three substantive input-validation gaps were reproduced before correction:

- Outer ensemble selection accepted nonbinary event values and a nonfinite
  duration, yielding a finite selection score. Shared validation now rejects
  malformed inner and outer outcomes, probability alignment, horizon and class
  semantics before label construction.
- Some nonfinite training, architecture and ensemble controls were accepted at
  construction. Finite-value, integer-size, support and range checks now reject
  those configurations before fitting.
- Explicit empty stage or ensemble mappings silently activated defaults. Only
  an omitted mapping now selects defaults; empty supplied configurations fail.

The modeling reviewer added regression coverage for these cases without changing
the archived numerical function bodies. Narrow-support prediction also now
chooses default query times within the fitted horizon.

Two publication-builder issues were identified and corrected. Relative links
were previously copied from chapter locations into an assembled document at a
different directory, breaking six current local references. The builder now
resolves references relative to their source chapters, checks their existence,
emits portable manuscript paths, and uses repository links in the PDF.
Development-only partial builds now write under the ignored build directory
instead of overwriting the release PDF. These corrections were inspected in
source; final assembled-document verification belongs to the final build check.

Operational chapter counts were reconciled with retained aggregate receipts,
including receipt hashes: 15,360 scheduler rows and their state breakdown,
29,106,213,679 database bytes, seven principal relation counts, and six restoration
reports. The most recent recorded restoration compared 10,230 files, 23 relation
counts and eight profiles. These checks are historical evidence, not a new
restore performed during review. The platform completed more than 70,000 successful Airflow workflow runs; the
retained snapshot supplies a detailed breakdown for a subset of that history. Explanation parity,
neural serving parity, predictive accuracy and uptime remain distinct quantities.

One enrichment claim needed version qualification: the current private accessory
worker retries an invalid structured response once, while the earlier public
reference parses once and raises. The chapter now states this difference rather
than implying identical retry behavior in both versions.

## Feature-store validation and leakage limits

The release inventory covers 79 SQL files: 66 existing public files, eight
sanitized historical exports and five new portable contracts. Offline parsing
accepts 77 standalone files and two explicitly wrapped fragments, with no parser
rejections. Dependency extraction is static; it is not proof of complete column,
dynamic-SQL, function-body or deployment dependency resolution.

All five portable contract files were executed against artificial fixtures in
an isolated, in-process PostgreSQL-WASM instance: PGlite 0.5.8, PostgreSQL 18.3.
Seven temporal assertions passed. Empty uncertified output, content changed
after certification and expired certification were separately rejected. The
receipt is [validation_report.json](../../research/feature_store_ddl/validation_report.json).
Historical private SQL was not executed, and no production database was connected.

This execution validates the small portable contract, not the completeness of
the production feature store. In particular, certification hashes and as-of
value predicates do not prove historical availability. The leakage chapter
retains the documented counterexample in which lifecycle-dependent missingness
exposed a strong outcome shortcut. It separates the observed cross-tab counts,
an algebraic missingness-only diagnostic, documented remediation and unresolved
historical reconstruction limits. It does not invent a matched causal estimate
of the earlier metric increase. Tail-cohort overlap, a nested recent diagnostic
and zero-duration endpoint sensitivity also remain explicit limitations.

## Publication boundaries

The release gate has two complementary passes. Changed-file screening checks
credentials, private paths, blocked data formats, size and binary boundaries.
Full-tree screening checks source-neutral terminology, including legacy files.
Strict document mode requires exact path and content hashes for reviewed PDFs,
archives and PNGs. Archive members are independently screened. PNG approval
requires recorded visual review, a PNG header and bounded byte/dimension limits.
Tests reject missing reviews, modified bytes and mismatched dimensions.

PDF extraction checks visible text streams and metadata; it is not OCR and does
not inspect every embedded image's pixels. Consequently, visual and embedded
image review remains necessary even when text screening passes. The coordinating
reviewer is recording the final raster inspection, corrected document hashes and
visual checks separately. Earlier public Git history is outside this current-tree
screening scope.

A follow-up terminology check found residual product and payment names in eight
legacy text files. They were replaced with generic manufacturer, accessory and
payment descriptions. The repair-provider feature name was changed consistently
between its documented SQL definition and training query. Generic payment aliases
replace source-specific aliases in the adapted reference. A ninth initial match
was an ordinary browser engine identifier; it remains because it is a protocol
identifier rather than a source fingerprint. Audit regression tests distinguish
these cases. No preserved model module was changed by this cleanup.

## Execution record and remaining release check

An earlier repository-wide run recorded by the coordinating reviewer passed 89
tests. The cascade reviewer subsequently reported 29 passing cascade tests after
the validation fixes. The independent DDL, model-reference and release-boundary
run passed 26 tests before the additional terminology regression. These are
overlapping, differently timed runs and must not be added into a total. A later
combined run of the cascade, feature-store, image and public-reference test files
passed all 56 tests after the fixes and terminology regression, with two known
deprecation warnings from an unchanged archived regression estimator. The final
repository-wide count and final PDF/image audit must be appended after the last
edits.

No production service was started or modified during this review. No production
database connection, private inference run, private training run or restoration
was performed. Artificial numerical tests and the in-process SQL fixture are
the executed experiments described here.

The review does not establish prospective superiority, calibrated decision value,
fraud-detection accuracy or complete historical reproducibility. Those claims
require the separate, governed real-data studies described in the manuscript.

## Visual review receipt: PDF pages 67-83

The operations reviewer inspected all 17 pages in this physical PDF range using
the completed 100-dpi Poppler renders. The reviewed PDF had 99 pages and SHA-256
`30d257409e0f04eee6ebc0525d6722d8adeeff549b918d4cfb9f3bd03dc92958`.
The reviewed range covers the empirical chapter's final figures and tables,
the complete operational chapter, the discussion and the opening agentic-system
page. The later agentic diagram falls outside this assigned range.

No clipped content, overlapping table cells, unreadable chart labels, missing
glyphs or obstructed page furniture were observed. Operational row counts,
scheduler denominators, numerical parity scope and restoration counts agree with
the previously checked evidence. Figure-only pages 71 and 77 are sparse but
readable. One non-data-loss layout issue was reported for editorial consideration:
page 82 contains only the discussion's final two-line paragraph on an otherwise
empty page. Subsequent edits change the hash and require a new receipt for the
affected pages.

## Record-level privacy follow-up

A subsequent current-tree and existing-public-branch scan checked long numeric
record examples, row-bearing source URLs and seller/account-name literals in text,
extracted PDF content and archive members. Three legacy documentation or comment
locations used long numeric record examples of unestablished provenance. The
current release replaces them with explicitly synthetic small identifiers.
The trainer-derived package already described its five fixture identifiers as
synthetic; these were also shortened consistently, including the validation
lookup. Aggregate counts, dates, scientific references and architecture dimensions
were preserved. The full scope and historical-publication limitation are recorded
in [RECORD_PRIVACY_REVIEW.json](RECORD_PRIVACY_REVIEW.json).

The release gate now rejects contextual long numeric identifier assignments,
record-selection command examples and numeric source-record URLs, including
multiline assignments and lists that begin with a short fixture identifier. Tests
also verify that aggregate counts and model dimensions do not become false
record-identity findings. Current-tree corrections do not rewrite earlier public
Git history. Permission to retain anonymous example images does not authorize
original record identifiers, source URLs, seller details or raw datasets.

## Prior integrated release checks: edition 0.2

The final software suite passed **101 tests** with two known deprecation warnings
from a preserved historical estimator. The [validation receipt](../../research/thesis_evidence/current_validation.json)
records the commands, environment, source and test hashes, and installed-wheel
check. All five portable SQL files executed in the isolated in-memory PostgreSQL
engine; the historical SQL inventory parsed 77 standalone scripts and two wrapped
fragments without rejection. GitHub's research, Python and Go workflows passed
for the initial release commit.

The final monograph by **Ghaffar Masomi** contains **99 pages, 15 chapters,
32,704 chapter words, 22 figures and 16 bibliography entries**. Every page of the
completed typography edition was visually reviewed. The final author and
operating-total revision changed only physical pages 1, 2, 3, 9, 10, 72 and 81;
all seven were visually inspected again. The other 92 pages are pixel-identical
to the earlier reviewed edition in a page-by-page 72-DPI comparison. No clipping,
overlap, missing-glyph or pagination blockers were found.

The final PDF SHA-256 is
`eda6f34e88242a7a361d5ed79b59b628629714311354182fa9233192b5003401`.
The [build manifest](THESIS_BUILD_MANIFEST.json) records chapter hashes, tool
versions and the layout comparison; the [PDF manifest](PDF_RELEASE_MANIFEST.json)
binds review to this file. The author is credited on the cover, in PDF metadata,
the Markdown manuscript and the repository citation file. The Downloads copy
matches these exact PDF bytes.

The platform completed **more than 70,000 successful Airflow workflow runs** over
its operating lifetime. The retained scheduler snapshot supplies the detailed
breakdown for a subset of that history.

All 81 unique embedded raster images in the nine historical PDFs were visually
inspected and screened through cached OCR matched to their final image hashes.
Three source-specific raster issues were corrected and obsolete image objects
removed. The monograph uses reviewed figure derivatives and two expressly
approved anonymous photographs without original record identifiers or seller links.

A follow-up [Stage0 implementation map](../../research/cascade/STAGE0_IMPLEMENTATION.md)
compares the original recovery sources with the public package. It verifies
46 preserved Stage0-related numerical definitions within the 122-definition
cascade export and maps training, ensembles, scoring and integration boundaries.
Its complete artificial example trained for two epochs and produced finite
survival predictions. This follow-up changed documentation rather than runtime code.

The [data-access policy](../../DATA_ACCESS.md) keeps all working datasets
proprietary, including Parquet observations and training exports. Researchers may
request separately approved access. The public tree contains no Parquet, raw
tabular exports, private vocabulary or trained-weight files. Publication screening
covers the current files, reviewed PDFs, archive members and approved images.

Final publication audits passed for all 249 additions/changes since the original
public baseline and all 410 current public files, with zero findings.

## Historical neural comparison revision: edition 0.3

The results now state the recorded neural meta-ensemble improvement directly:
F1 0.9209 versus 0.8462 for the earlier XGBoost AFT system (+7.47 points),
precision 0.9802 versus 0.8314, and recall 0.8684 versus 0.8614. The four-row
comparison table and new vector chart retain the respective historical cohorts
of 964 and 941 records. The abstract, orientation, comparison and discussion
chapters, model card and claim ledger carry the same bounded result.

The [comparison evidence](../../research/thesis_evidence/historical_neural_comparison.json)
records sixteen hashed sources, 233 neural trial prediction pairs plus three
best snapshots, and 8,000 ensemble candidate evaluations. Mathematical review
verified confusion-matrix arithmetic, denominators and reported improvements.
Source review distinguishes actual execution output from surrounding generated
prose and preserves the recovered threshold-selection scope. The evidence does
not silently equate the two historical cohorts or every model-family comparison.

The rebuilt monograph by **Ghaffar Masomi** contains **102 pages, 15 chapters,
33,504 chapter words, 23 figures and 16 bibliography entries**. Two reviewers
visually inspected all physical pages using the final 90-DPI Poppler renders;
the integration review also enlarged the new table and chart on pages 66-67.
No blocking clipping, overlap, missing content or unreadable figures was found.
Physical pages 84 and 96 retain sparse continuation text, an optional pagination
refinement. The cover, metadata and direct 70,000+ successful workflow total
remain correct.

The final PDF SHA-256 is
`a938e2eca511e3c7ce7a8904df30478de2d33da3b15bd45a436e96cb917c398e`.
The build and PDF release manifests bind the review to those exact bytes; the
Downloads copy matches. Local Markdown links resolve. All 26 recorded runtime
source files and six test files remain identical to the previously tested
snapshot. This revision rebuilds documentation and records existing experiments;
it does not claim a new model-training or local software-suite run.

Publication audits passed for all 251 additions/changes since the original
public baseline and all 412 current public files, with zero findings. The
proprietary records, original identifiers and fitted weights remain excluded.

## Cohort reconciliation, novelty and prospective protocol: edition0.4

The archive now distinguishes the earlier941-row AFT evaluation week from the
later971-row run. An exact join finds964 common AFT/neural keys with964 matching
duration labels. The saved AFT rule reproduces its logged971-row confusion
matrix; restricting it to common keys gives F1 0.855967. The associated neural
meta run records F1 0.9209. Its final confusion matrix is inferred from rounded
metrics and cohort counts, and its final decision vector is not represented as
recovered. The manuscript states the precision/recall tradeoff and source scope.
The23-row deletion calculation is explicitly a separate counterfactual bound.

A dedicated original-contribution chapter develops five implemented research
claims and their precedents. The reproduction chapter answers all eight
requirements for a sealed future cohort, separating prospective validity from
novelty and from completed retrospective experimentation. One primary leakage
reference is added to the bibliography.

The release now contains **109 pages,16 chapters,36,145 chapter words,23 figures
and17 references**, credited to **Ghaffar Masomi**. All110 pages of its initial
build received independent visual review. After a compact table was kept
together and the new chapter's sparse continuation was removed,70 final page
bodies were pixel-identical to the reviewed edition; reviewers inspected all39
changed final pages, including shifted page furniture. No blocking defects were
found. Original sparse chapter endings and one table-caption continuation are
retained as minor typography observations.

Final PDF SHA-256:
`820999209d0134d0b7a92ab854866876a5e6a7a8ef5d29618f18f1a4d4b97b6e`.

Both aggregate audit scripts are public and accept neutral input mappings. Six
public synthetic regression tests were added; the full suite passed **107 tests**
with the same two historical-estimator deprecation warnings. Additional bounded
checks covered29 cohort-audit cases and1,536 small sensitivity cases. Source
hashes, commands and numerical scopes are recorded separately from predictive
performance. No new model training or prospective collection was performed.

The proprietary observation files, mappings, identifiers and model weights remain
private. The Downloads copy matches the released PDF bytes.

Final publication screening passed for257 additions/changes since the original
public baseline and all418 current public files, with zero findings.
