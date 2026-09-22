# Reproducing MarketNeural

MarketNeural is the research extension of t0-clip. The new compact benchmark is
separate from the [preserved production network](production_reference/README.md).
The production reference includes real selected source definitions with hashes;
it does not include trained weights, private data, or the complete deployed stack.

The extended release also includes the [three-stage cascade](cascade/README.md),
[feature-store SQL](feature_store_ddl/README.md), [leakage study](leakage/LEAKAGE_STUDY.md),
and [full PDF monograph](../papers/marketneural/marketneural-thesis.pdf). The cascade
trains new models from caller-supplied matrices using preserved numerical routines
and explicitly new portable controllers. It does not reconstruct unavailable
historical fit datasets or private trained weights.

Private datasets remain proprietary. See the [research data-access policy](../DATA_ACCESS.md).
No public code license grants access to Parquet files or original record identifiers.

## Installation

Use Python 3.11 or newer in a clean virtual environment. From the repository root:

```sh
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# POSIX shells: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

PyTorch can be installed from its official CPU or accelerator distribution before
installing this package. The checked-in run records the actual Python and package
versions. Dependency ranges allow compatible installations; they are not a fully
locked operating-system or accelerator environment.

## Run the six-model demonstration

```sh
python -m marketneural synthetic --output data/synthetic --n 1200 --seed 2026 --regime nonlinear
python -m marketneural benchmark --config configs/synthetic_smoke.json --output results/synthetic_smoke
python -m pytest -q
python -m research.production_reference.training --count-historical
python -m research.production_reference.training --smoke
python scripts/audit_public_release.py
```

Use a new output directory for a new run. The CLI refuses to overwrite an existing
dataset or nonempty result directory. This prevents accidental replacement of a
scored experiment; it does not technically prevent a researcher from reusing the
test period in another run. External preregistration and an independent test-data
custodian are needed for stronger controls.

The fixture is small and deliberately uses a reduced architecture and a short
training budget. Its purpose is reproducibility and correctness, not architecture
ranking. No hyperparameters were adjusted after inspecting its test results.
The same fixed configurations can also run a proportional-hazards fixture:

```sh
python -m marketneural synthetic --output data/synthetic_ph --n 1200 --seed 2026 --regime ph
python -m marketneural benchmark --config configs/synthetic_ph_smoke.json --output results/synthetic_ph_smoke
```

The nonlinear fixture changes the Weibull shape across inputs and introduces
interactions. The PH fixture has a common shape and a linear log-risk predictor.
Both include text/image/report vectors, missing slots, independent observation
censoring, and additional administrative censoring at temporal cutoffs. Neither
fixture is a reconstruction of a private marketplace cohort.

## Model comparison

| Name | Estimator | Inputs |
|---|---|---|
| `km` | Unconditional Kaplan-Meier | Outcomes only |
| `coxph` | Penalized Cox proportional hazards | Tabular plus flattened vectors and availability masks |
| `rsf` | Random survival forest | Same information as Cox |
| `gbsa` | Gradient-boosted Cox-loss survival trees | Same information as Cox; not the historical boosted-AFT trainer |
| `mlp` | Discrete-time survival MLP | Same flattened information |
| `perceiver_moe` | Compact latent-attention mixture of survival distributions | Tabular/text tokens, image/report slots and masks |

All transformations of table features are fitted on TRAIN only. The new neural
models use their coherent survival curves for horizon probabilities; they do not
reproduce the historical independent FAST72 gate or K4 ensemble policy. The old
and new implementations also use different mask conventions, documented in their
respective model contracts. Do not pass one model's tensors to the other unchanged.

## Input contract

Configuration file paths are resolved relative to that configuration file. The
CSV requires `row_id`, `entity_id`, `decision_time`, `feature_observed_at`,
`observed_until`, `event`, and explicit numeric/categorical feature allowlists.
Times must include an unambiguous timezone; naive timestamps are interpreted as
UTC by the current loader and should be normalized upstream. Duration is measured
in hours from decision to observed event or censor instant. The event indicator is
one only when the event is observed, not merely inferred from disappearance.

The benchmark requires one landmark per entity and disjoint entities across all
partitions. Repeated-landmark questions need a separate protocol and cluster-aware
analysis. Training labels are censored at validation start, validation labels at
test start, and test labels at the prespecified final as-of time. Events exactly at
a label cutoff are conservatively censored. Zero/negative follow-up is rejected.

Optional vectors use an NPZ archive with non-object `row_ids`, plus `text` (N,D),
`image` and `report` (N,slots,D), and corresponding binary `image_mask` and
`report_mask` (N,slots), where **True means present**. IDs explicitly align vectors
with rows. Absent masks mean every supplied slot is present. Load uses
`allow_pickle=False`. No fitted encoders are downloaded or trained by the CLI.

`feature_observed_at` must attest to the latest source evidence used by all supplied
features, including vectors. The check cannot prove that this attestation is true,
that an edited description is its original version, or that an event was reported
promptly. Preserve immutable source versions, transformation/encoder versions,
event availability times and as-of join evidence upstream. Column denylisting is
an additional heuristic, never proof of leakage freedom.

## Selection and scoring

Each model family's candidate list is evaluated on SVAL. The configuration with
the lowest mean integrated Brier score across the declared seeds is selected;
candidate order resolves exact ties. Neural early stopping also uses SVAL.
Every family's selection is written to `selection.json` before any test scoring.
Models are not retrained on SVAL. The first declared seed is the primary seed for
paired bootstrap comparisons; all declared seeds are reported separately.

Operating thresholds maximize eligible SVAL recall subject to an empirical
precision target and minimum flagged support. If no threshold qualifies, the
policy rejects all rows and reports the reason. A target on SVAL is not a future
precision guarantee. Rows censored at/before the decision horizon without an
observed event are unresolved and excluded from the binary confusion matrix;
they remain in censor-aware survival evaluation and are counted separately.

Integrated Brier score uses a fixed declared time grid, with inverse-probability
censoring weights estimated from TRAIN. IPCW concordance uses horizon risk and
the same training censor estimator. This assumes independent censoring and a
transferable censoring distribution across periods. The harness fails if the
grid lacks training censoring support or sufficient evaluation follow-up. It
does not silently trim horizons to make a model look better.

Paired bootstrap confidence intervals compare per-entity integrated Brier losses
against Cox PH. Negative differences favor the named comparator. Intervals are
conditional on fitted models and the fixed TRAIN censoring estimator; they do
not include retraining/tuning uncertainty, temporal dependence, or multiplicity
adjustments. The [research protocol](../docs/research/PROTOCOL.md) describes the
stronger design required for publishable real-data claims, including independent
calibration/policy selection and temporal replication.

## Outputs and public release

Each run creates three aggregate JSON files:

- `protocol.json`: declared configuration, content hashes, split counts, train-only
  preprocessing metadata, source hash and dependency versions.
- `selection.json`: every candidate's SVAL results, chosen settings, fit times,
  seed-specific thresholds and neural early-stopping epoch.
- `summary.json`: frozen-selection hash, final test metrics, per-seed results and
  paired bootstrap intervals.

These contain no row-level predictions or private paths by default. Feature names,
category counts and aggregate statistics may still be confidential for a real
dataset: review them before copying any local output into the public repository.
`data/`, `results/`, virtual environments and caches are ignored by Git. Only
reviewed synthetic aggregates belong under `research/examples/`.

The [saved demonstrations and figure](examples/README.md) report the measured
results from two artificial-data regimes with unchanged model settings.

## Recovered mechanism and component experiments

The [coverage/cohort mechanism study](leakage/availability_mechanism.md) verifies
911 historical masks and provides an executable constructed negative control.
The [retained experimental receipt](thesis_evidence/experimental_contributions.json)
contains paired fixed-policy comparisons on 523 and 413 records, a documented
source intervention, and an unfavorable later replay. These are completed
retrospective studies with declared controls and selection limits.

The [closest-prior-work comparison](thesis_evidence/NOVELTY_PRIOR_ART.md) identifies
what these findings establish beyond a generic platform description. Raw input
records remain proprietary; public scripts expose the analysis and aggregate
receipts preserve source identity without distributing the observations.

The release scanner screens changed files for credentials, private paths, source
fingerprints and data/checkpoint artifacts. It is a heuristic and cannot replace
manual diff review. Historical papers are separately documented source-neutral
derivatives; the new manuscript distinguishes their results from this benchmark.
