# Historical production model reference

This directory preserves a selected, sanitized implementation of the project's
historical multimodal survival network. It is separate from the new compact
public benchmark experiments. The public benchmark's measurements must not be
presented as measurements of this historical model or its private training data.

The export contains the actual K8 tokenizers, Perceiver fusion, dense mixture of
hazard experts, scalar gate, survival likelihood and masked binary losses.
These are executable PyTorch definitions, not pseudocode. It excludes production
data access, feature names, credentials, orchestration, fitted vocabularies,
checkpoints and row-level observations.

## What is preserved

`model_core.py` contains 24 definitions selected from two archived source files.
Each definition's body and decorators are unchanged. `provenance.json` records
the full source-file SHA-256, original symbol line ranges, source-span hashes,
AST hashes, canonical UTF-8/LF export hash and transformations. Normalize CRLF
before checking the export hash on a Windows checkout. The imports/module preamble were
replaced with minimal public dependencies. Private paths and an infrastructure-
specific source filename are omitted from the public provenance record.

`historical_architecture.json` contains only architecture metadata extracted from
an inspected historical checkpoint: numerical dimensions and categorical
cardinalities, without category values or learned parameters. Instantiating this
configuration creates **random weights**, not the trained model.

| Property | Inspected historical family |
|---|---:|
| Unique trainable parameters | 17,145,736 |
| Numerical / categorical inputs | 239 / 81 |
| Model width / latent tokens | 256 / 32 |
| Fusion blocks / attention heads | 6 / 8 |
| Dense hazard experts / time bins | 7 / 128 |
| Time support | 504 hours, linear bins |
| Compact numerical / categorical tokens | 8 / 8 |
| Listing-level text input | one 768-vector projected to 8 tokens |
| Image slots | eight 512-vectors |
| Image-report slots | eight 768-vectors |
| Total input tokens | 40 |

Summing all state-dict tensor elements gives 17,148,808, because shared LayerNorm
parameters appear under multiple names. Count `model.parameters()` to obtain the
unique trainable parameter count. Pretrained embedding encoders are upstream
feature extractors and are not included in either count.

The experts are dense: all expert MLPs execute. The network mixes their survival
distributions using softmax weights; it does not implement sparse expert routing.
K8 tokenization adds slot, modality and missing-slot embeddings. Missing slots
remain represented in attention. The text tokens are projections of one embedding,
not separately encoded passages.

## Run the reference

From the repository root, with NumPy and PyTorch installed:

```sh
python -m research.production_reference.training --count-historical
python -m research.production_reference.training --smoke
python -m unittest discover -s tests -p test_public_release.py -v
python scripts/audit_public_release.py
```

The count command instantiates the full architecture without loading weights or
running inference. The smoke command uses a **138,787-parameter reduced shape**
of the archived classes and four synthetic rows for one optimizer step. Its
reported loss is a software check, not model accuracy or a benchmark result.
No external services, downloads or data connections are needed by these commands.

`training.py` is a new, generic tensor adapter around the archived implementation.
It uses caller-supplied batches, AdamW in the smoke example, finite-input/loss/
gradient checks, clipping, and censoring-aware likelihood plus curve/head binary
supervision. It does not reproduce the historical training controller, search,
ensemble, calibration or exact trial objective coefficients. The archived
checkpoint did not contain all loss coefficients; the public adapter's defaults
are explicitly demonstration choices.

For integration, supply `TensorBatch` objects to `train_tensor_epoch`. Use an
optimizer of your choice and explicitly select `Objective` coefficients. Split
construction, fitted preprocessing, early stopping, model selection and final
held-out evaluation remain the caller's responsibility. Never fit preprocessing
or choose thresholds on the final test period.

## Input and outcome contract

- Features must contain only information available by the decision timestamp.
  Features computed later require an availability timestamp and an as-of join;
  matching column names alone does not establish time correctness.
- Numerical values must already be transformed using training-fitted statistics.
  Categorical integer IDs must use the frozen training vocabulary, with a reserved
  unknown value. The public metadata contains vocabulary sizes, not mappings.
- Text, image and report encoders must be version-pinned and identical across
  training and scoring. Shapes are respectively `(B,768)`, `(B,8,512)` and
  `(B,8,768)`. Slot masks are `(B,8)`, where one denotes missing.
- Outcomes contain nonnegative follow-up duration, a binary observed-event flag,
  and nonnegative sample weights. Right-censoring is a partially observed outcome,
  not a negative event label at every future horizon.
- For horizon `h`, an observed event by `h` is a known positive; observation through
  `h` without such an event is a known negative; shorter censored follow-up is
  unknown and excluded from that binary loss. Events beyond model support become
  right-censored at the support boundary for the survival likelihood.

`serving_contract.example.json` is a platform-neutral checklist/schema example
for packaging a trained successor. Its example values are not a declaration that
this archive is deployment-approved. The historic model code does not implement
all of that proposed runtime validation itself.

## Historical behavior retained deliberately

This reference does not silently repair the archived math or claim all model
choices were validated. The expected-time helper has `@torch.no_grad`; it is
appropriate for reporting but cannot supply a differentiable expected-time MAE
term. The public adapter does not use that optional term. Time summaries are
bounded by the finite support and approximate within-bin timing. The scalar
head is separately supervised and need not equal the survival curve at its
horizon. Provenance tests protect these historical semantics from accidental
changes; improvements should use separately named implementations and comparisons.

No weights or source data are distributed, so the original predictions cannot be
reproduced from this export alone. Architectural complexity and a successful
synthetic optimizer step do not establish predictive uplift. A matched temporal
benchmark and ablations are needed to compare the preserved architecture with
the compact public experiments.

## Release boundary

The release audit checks changed tracked files and nonignored untracked files.
Use `--base dc41f873b82ab5c1ac07862c7610bcf2d6a40e62` for this complete release diff
and `--all-public-text --include-reviewed-documents` to check source-neutral
terminology throughout the current public tree, including older files and the
reviewed PDF/ZIP documents. Local environments are excluded. It rejects model/data
binaries, row exports, raw logs, unreviewed PDFs, private paths, provider-specific
fingerprints and several credential forms. It reports paths and rule names,
never matching secret values. PDFs unchanged from the pinned original public
release (not moving HEAD), or explicitly reviewed path/hash manifests or SHA-256
values are permitted. Reviewed source archives also undergo bounded member-level
text, path and binary screening. The full-tree text mode lists non-text files for
separate review; the PDF and archive manifests document those reviews. Existing
Git history is outside this current-tree audit. This heuristic does not detect every
encoded credential, personal record, attribution requirement or fingerprint;
manual review remains necessary. PDF extraction does not perform OCR on raster images.
