# MarketNeural data card

Version 0.2 — 2026-09-21. The distributable benchmark data are **synthetic**.
The working datasets are **proprietary** and remain private, including Parquet
tables, original identifiers, source links, seller details, training exports and
fitted vocabularies. See [controlled research access](../../DATA_ACCESS.md).

The monograph includes two expressly approved anonymous photographs and their
limited saved judgments. They are illustrative media with reviewed filenames,
metadata and outputs, not a public image dataset. Their original record IDs and
source URLs are excluded. The [case provenance](../../research/thesis_evidence/image_case_studies.json)
describes pairing and interpretation limits without publishing the private keys.

## Synthetic dataset

The generator in [`marketneural/synthetic.py`](../../marketneural/synthetic.py) creates independent artificial entities. Its default is 1,200 rows with generator seed 2026. Every row has four Gaussian structured variables, one three-level categorical variable, a four-dimensional artificial text vector, two image slots with three dimensions each, and two report slots with three dimensions each. These names denote input channels; the values are random numerical fixtures, not embeddings of real photographs, descriptions, or reports.

Image slots are independently masked with probability 0.15 and report slots with probability 0.20. The event generator combines structured, text, image, report, and categorical signal. The `nonlinear` regime additionally includes a cross-modal interaction, a sinusoidal term, and a covariate-dependent Weibull shape; it deliberately departs from proportional hazards. The `ph` regime uses a constant shape. These are complementary software scenarios, not sampled market populations.

Independent random observation censoring is drawn uniformly between 12 and 420 hours. Chronological administrative censoring is added by the benchmark. Artificial decision times span 110 days from a fixed origin, and feature evidence is assigned one hour before the decision. The generator writes `cohort.csv`, `vectors.npz`, and a synthetic manifest. Existing outputs are not silently overwritten.

The default nonlinear smoke run uses 593 train, 267 validation, and 340 test rows, all distinct entities. Training and validation label cutoffs create 38 and 32 additional administratively censored rows respectively. Its 340 test rows contain 258 observed events. These counts describe one generator seed and one configuration; changing either changes the cohort.

## Public table contract

| Field | Meaning | Requirement |
|---|---|---|
| `row_id` | Unique record key used to align vector arrays | Nonempty, unique, no surrounding whitespace; use synthetic or privately managed anonymous keys |
| `entity_id` | Related-episode grouping key | One row per entity in the compact lane; cannot overlap partitions |
| `decision_time` | Time origin of the prediction | Parseable timestamp normalized to UTC |
| `feature_observed_at` | Attested latest availability time of contributing evidence | Must not exceed decision time; not an independent proof of provenance |
| `observed_until` | Event instant if `event=1`, otherwise last justified censoring instant | Strictly after decision time under the public loader |
| `event` | Indicator that the declared event was observed | Exactly 0 or 1 |
| Explicit numeric/categorical columns | Allowed model inputs | Declared unique feature allowlist; no contract fields or banned outcome-like tokens |

Duration is computed in hours from `observed_until - decision_time`, after applying the cohort's administrative cutoff. An event at the fit cutoff is conservatively not treated as already observed. Delayed reports require care: a backdated source event timestamp does not establish when the label became available. The compact contract assumes the upstream exporter has resolved that ambiguity; richer event-time and label-observation fields belong in the full study.

The loader rejects zero/negative follow-up so that an unresolved endpoint convention cannot pass silently. This requirement is not evidence that historical zero-duration records were necessarily errors. A real exporter must document and audit them before mapping data into this contract.

## Vector contract

The NPZ archive has `row_ids` plus optional `text`, `image`, and `report` arrays. `text` is shaped `[rows, dimensions]`; image/report arrays are `[rows, slots, dimensions]`. Optional `image_mask` and `report_mask` arrays are `[rows, slots]` with true meaning present. Masks default to present when omitted. Values must be finite, shapes must align, and every selected table row must have a vector key. Object pickles are not accepted.

Keys are used for explicit alignment rather than relying on file order. The archive does not itself establish where a vector came from. Real-data use must preserve encoder version, raw-content version/hash, evidence availability, and any learned transform's fit partition. A fixed vector computed later from immutable historical bytes and a vector computed from later edited content have different validity.

## Preprocessing and splits

Numeric imputation, missingness indicators, scaling, and categorical vocabulary are fitted on the training partition. Later unknown categories use the configured encoder's unknown handling. Published preprocessing metadata include counts and hashes rather than private vocabulary values. Representation dimensions and masks are checked across fitting and prediction.

The compact lane uses train/validation/test windows and rejects entity overlap. It does not discover real relists or sellers automatically; the upstream `entity_id` must represent the intended group. The full research protocol requires duplicate-content and related-item audits beyond exact key equality. It also adds a separate calibration/policy block and raw evidence versioning.

## Historical aggregate evidence

The manuscript's historical metrics come from historical project papers and a retrospective private-artifact audit. The public repository does not provide the historical row-level datasets needed to reproduce those metrics. Only cohort sizes, aggregate metrics, and methodological limitations are discussed. The recent 283-row short-horizon slice is a subset of the 748-row holdout. Their records must not be counted as two independent validation samples.

Source-reported status and metadata duration are not synonymous with a verified commercial transaction. Endpoint ambiguity, informative censoring, observation delay, relisting, missing evidence, and historical content mutation remain relevant real-data concerns. Synthetic generation does not resolve them.

## Privacy, licensing, and use boundary

Synthetic examples contain no real individuals, listing identities, contact details, private URLs, or raw marketplace content. The software license does not grant rights to collect or redistribute third-party data. A future dataset release needs its own permission, privacy, and licensing assessment. Hashing a real listing identifier alone does not guarantee anonymity when records can be linked to public content.

Use the synthetic dataset to verify execution, alignment, leakage guards, metric behavior, and result provenance. Do not use its accuracy as a product claim, as a sample-size substitute for real evaluation, or as evidence of generalization across marketplaces.
