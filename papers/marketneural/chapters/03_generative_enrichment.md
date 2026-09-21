# Generative enrichment as a measurement system

## Seeing condition at the scale of an operational dataset

A market listing is only partly represented by its structured fields. The same nominal item can have a damaged rear panel, an intact front, a protective layer, missing accessories, uncertain packaging, or an ambiguous battery display. These details affect how a person interprets the offer, yet many are expressed only in a photograph or a short, informal description. A survival model restricted to price and a categorical condition label cannot recover every distinction. The enrichment system was built to turn this otherwise difficult-to-use evidence into explicit, versioned measurements.

The retained snapshot contains 240,622 image-asset rows, 228,958 image-feature records, and 54,841 structured text-enrichment records associated with a listing store of 55,260 rows. These are different units and coverage populations. They demonstrate that visual and textual interpretation was a substantial production workload; they do not constitute 228,958 independently verified visual labels. A feature record is a machine-produced interpretation with a particular prompt, input, execution time, and revision. It becomes a useful research input only when those dependencies are preserved.

The practical description of the system as using artificial intelligence as its eyes is apt when interpreted operationally. Repeated visual judgments that would otherwise require individual inspection are converted into constrained fields and short evidence summaries. The system does not observe hidden components, infer a verified transaction, or establish the truth of a seller's claim. It interprets what is visible and what is written. The distinction is fundamental: generated attributes are measurements with error, not newly discovered ground truth.

Three separate computational layers participate. First, generative vision and language workers construct interpretable attributes. Second, frozen encoders map images, canonical listing text, and normalized image reports into vectors. Third, the survival network learns a relationship between the available attributes and vectors and a censored time-to-event target. The inspected implementation does not train a foundation model end to end on sale outcomes. Its research contribution lies in the contracts and learned fusion between these layers, together with the operational system that makes them consistently available.

## Decomposing visual interpretation into bounded tasks

The visual pipeline separates damage, accessory, and color or identity analysis. This is more than a convenient code organization. Each worker owns a limited field set, and a successful update must not overwrite a field owned by another worker. Damage analysis owns visible damage, photo quality, background clarity, stock-photo status, protective-layer evidence, and a short damage summary. Accessory analysis owns packaging, charger and cable evidence, other accessories, receipt evidence, battery-display interpretation, and related confidence fields. Color analysis owns body-color observations and tightly controlled identity-correction proposals.

This separation constrains the space in which an individual generative response can cause harm. A well-formed accessory response cannot silently clear an existing damage assessment. A color proposal cannot become a general authority to rewrite every product attribute. The database writer, not the language model's prose, determines which columns may change. Restricting ownership is an effective engineering control even when the underlying model remains probabilistic.

| Interpretation task | Representative outputs | Important boundary |
|---|---|---|
| Visible damage | Damage level, summary, protective-layer flags | Describe visible evidence; do not infer internal function |
| Image quality | Photo quality, background clarity, stock-image flag | Quality and authenticity cues are distinct from physical condition |
| Accessories and packaging | Box, cable, charger, case, receipt, seals | An object must be visible; absence of evidence may be uncertain |
| Battery display | Relevant screenshot flag and percentage | A generic settings screen does not establish battery health |
| Color and identity | Body color and confidence, correction proposal | Conflicts with independent evidence require a stronger guard |
| Text interpretation | Structured assertions and condition evidence | A seller statement remains an assertion rather than verified fact |

The accessory prompt handles a bounded set of images for one item and requests evidence at image level. It discourages false positives and constrains fields to a prescribed representation. A battery value is either an integer within the specified admissible range, 50–100 in the inspected contract, or null. The worker must establish that the relevant display is visible. This converts an unconstrained narrative task into a classification-and-extraction task with explicit abstention.

The same principle applies to damage. Distinguishing damage on a protective layer from damage on the underlying surface prevents a plausible sentence from becoming an unjustifiably strong physical-condition label. Background clutter is tracked separately from item condition. A stock illustration can communicate identity while providing weak evidence about the particular item's wear. These distinctions create features the downstream model can use and, equally importantly, distinctions an auditor can inspect.

## Prompt constraints, validation, and residual uncertainty

Prompt engineering in this system is a form of interface design. The model receives a bounded task, a field taxonomy, evidence instructions, and a structured response contract. Deterministic application code then parses, validates, and writes the response. The field contract reduces variation in spelling, label granularity, and narrative style, making repeated outputs easier to compare. It also supports selective retry: a malformed response need not become an accepted database record.

The inspected current private accessory worker requests a fresh response once after a JSON parse failure and raises an error if that repeated response is still unusable. The earlier public reference worker instead parses once and raises, so this retry behavior must remain tied to the inspected current source receipt. Its database helper can clear a failed connection and retry a connection operation once; other database exceptions roll back and propagate. Per-item errors are recorded while the larger loop continues. Completion is marked only after positive insertion progress. These are specific observed mechanisms, not an assertion that every worker has identical retry counts or complete transactional equivalence.

The damage worker has a bounded request timeout and similarly isolates item-level failures. Versioned image/report-vector records retain separate success flags, error text, and timestamps for the two representations. A report-vector failure can therefore be distinguished from an image-vector failure. Successful replacement can clear the relevant error rather than silently leaving a contradictory status. This is important for research because an apparently missing modality may reflect a pending job, a permanent input defect, or a failed execution, each with a different operational meaning.

Neither a low-variation generation configuration nor a JSON schema proves correctness. Structured wrong answers remain wrong. Repeated requests can differ because of backend changes, numerical behavior, or ambiguity in the input. The manuscript therefore describes constrained and repeatable generation, rather than mathematical determinism. It also avoids a zero-hallucination claim: no complete independently adjudicated error study was recovered for the full enriched population.

A useful decomposition of enrichment error is

$$
P(\widehat A\ne A)=P(\widehat A\ne A\mid Q=1)P(Q=1)+P(\widehat A\ne A\mid Q=0)P(Q=0),
$$

where $A$ is a target attribute and $Q$ indicates whether the supplied evidence is sufficient for a competent judgment. The expression emphasizes that model behavior and evidence sufficiency are separate. Prompt refinement may improve the first term without repairing an unreadable photograph. Explicit unknown values, quality measures, and abstention policies are therefore part of the measurement design rather than incidental missing data.

## Learning from an actual identity-correction failure

The retained development record documents a concrete failure in which an image interpretation overrode a title, description, and high-confidence text classification that agreed with one another. The proposed correction was a move to a higher tier within the same item family. Treating that case as a model improvement would have hidden a real evidence conflict. The implemented response was to make the override substantially harder and auditable.

The successor rule requires at least 0.95 model-reported confidence, at least two clear non-stock image-feature rows, a quality level of at least three, no stock-photo rows, and a rationale tied to relevant visible hardware evidence. The guard is applied in the specific situation where title, description, and text quality-control evidence form a consensus. Applied and skipped proposals are logged, and the mistaken change has a documented reversal record.

The confidence threshold is a control input, not a calibrated probability of correctness. Its value is that it participates in a conjunction of independent, inspectable conditions. Even a high self-reported confidence cannot bypass the requirements for multiple suitable views and conflict-aware evidence. The incident also illustrates why historical replay requires versioned rules: an older dataset may contain the original interpretation, the erroneous correction, or the repair, and these are not interchangeable at every decision time.

The scientific evaluation of such a guard should report the number of proposals, accepted overrides, rejected overrides, adjudicated correct changes, and unresolved cases, stratified by conflict type. The current evidence demonstrates a diagnosed failure and an implemented response. It does not provide enough adjudicated examples to estimate the guard's sensitivity or specificity precisely. That distinction preserves the engineering achievement without turning an incident fix into an unsupported accuracy statistic.

## Text interpretation and resource-aware decomposition

Text can supply direct condition statements, references to a defect, accessory descriptions, and uncertainty about operation. These statements can be normalized into structured fields and compared with visual evidence. The pipeline also constructs canonical encoder input from tagged title, description, and caption fields. Field tags retain some source structure, helping distinguish a short title from a longer condition account even when both eventually contribute to one vector.

The operational design uses recurring, incremental workers rather than repeating every expensive transformation at each survival prediction. The documented text worker has a 300-second loop in an active daily window. Text-vector construction skips unchanged canonical text using a content hash and refreshes when relevant caption evidence arrives. Reuse of stable intermediate representations is a credible resource-saving mechanism. The retained material does not, however, establish a complete per-item token bill or a controlled comparison of alternative providers and models. Descriptions of low-cost understanding should therefore be interpreted as a design objective unless accompanied by measured accounting.

A short textual statement can often resolve a condition ambiguity that would otherwise require several visual calls. Conversely, an image can contradict an optimistic description. The value of the multimodal design is that these sources need not collapse into a single unqualified truth field. A statement-derived condition, a visible-damage measurement, and their disagreement can remain distinct inputs. Their combination lets the survival model learn whether the disagreement is predictive while allowing human review of the underlying evidence.

This separation also supports a principled evaluation budget. Attribute-level annotation can concentrate on high-consequence or contradictory cases. Random sampling remains necessary to estimate ordinary error rates, while targeted review can find failure modes. Neither targeted examples nor a visually persuasive demonstration panel provides a population error estimate. The public examples accompanying the thesis illustrate actual evidence-to-judgment relationships under this constraint.

## From image judgments to eight structured evidence slots

The later multimodal design preserves image diversity through a fixed eight-slot representation. It selects at most one image for each primary role: front, rear or camera region, side or frame, damage, battery display, and accessories. Remaining slots are filled using quality and evidence preferences. The same image is not duplicated merely because it satisfies multiple roles; its metadata can preserve those multiple interpretations. Missing positions are padded and accompanied by explicit masks.

For each selected image, a canonical English report summarizes normalized structured facts. This report is not simply the raw seller description repeated beside each photo. Role mapping can use caption or text cues, but the report contract is based on the structured image analysis. The distinction matters because otherwise the supposed image-report branch could repeatedly amplify a seller's single assertion and appear to provide multiple independent observations.

The documented tensor interface is

$$
V^{\mathrm{img}}\in\mathbb R^{8\times512},\qquad V^{\mathrm{report}}\in\mathbb R^{8\times768},\qquad M^{\mathrm{img}},M^{\mathrm{report}}\in\{0,1\}^{8},
$$

with additional role and quality metadata. The two availability masks remain separate because an image vector can succeed while its report vector fails, or vice versa. The report and image vectors are aligned by item, slot, content version, and encoder revision. A shape match alone cannot prove alignment; the key and content hash are necessary parts of the contract.

An undated documented build recorded 44,939 listing manifests and 359,512 slot rows, exactly eight per manifest. It classified 42,487 manifests as ready and 2,452 as requiring weaker or additional evidence. Of the slot rows, 146,735 were padding; the other roles included 41,450 front, 40,747 rear, 15,332 side, 10,386 damage, 9,409 accessory, 3,036 battery, and 92,417 additional-image slots. These numbers describe a particular build population. They are not the denominator of the retained database snapshot or the final evaluation cohort.

A separately documented selected interval, March 21 through April 10 by event date, contained 2,458 listings, 2,450 manifests, and 2,313 ready manifests. The mean number of actual slots was 4.853, and 455 manifests had all eight real slots. These counts are useful evidence of real variable-length image sets. Selection by observed event date also means they should not be generalized to the initial live population without accounting for censoring and availability.

## Frozen semantic vectors and the meaning of whitening

The inspected image/report builder uses a frozen image encoder for 512-dimensional visual representations. Its language branch pools token-level hidden states using the attention mask, normalizes the pooled vector, and runs without gradient updates. The recovered listing-text job similarly constructs a 768-dimensional, normalized vector from canonical text. These are semantic representations produced by pretrained encoders; the supervised survival learner receives them as inputs.

Contrastive image-language pretraining offers a relevant precedent for reusable visual representations: Radford and colleagues (2021) show how paired language supervision can produce transferable visual features. That precedent motivates representation reuse, not a claim that the present system repeats the original pretraining experiment or inherits its published performance. [Radford et al., 2021](https://proceedings.mlr.press/v139/radford21a.html).

The operator also describes an ordinary-plus-whitened text-vector design. Whitening is a covariance transformation, not random white-noise augmentation. For a training-estimated mean $\mu$ and covariance eigendecomposition $\Sigma=U\Lambda U^\top$, a regularized transform may take the form

$$
z_{\mathrm{white}}=(z-\mu)U_r(\Lambda_r+\epsilon I)^{-1/2}.
$$

The ordinary vector can preserve the original embedding geometry while the transformed branch reduces dominant correlated directions. Whitening is studied as a sentence-representation transformation in its own right. [Huang et al., 2021](https://aclanthology.org/2021.findings-emnlp.23/).

The selected inspected neural contracts expose one listing-text vector of width 768. A fitted whitening artifact or two independently wired listing-text branches was not recovered in the bounded source audit. Accordingly, the ordinary-plus-whitened account remains an operator-described upstream design that requires artifact-level confirmation. It is not silently converted into an architectural claim about the selected network. A reproducible implementation would preserve the fitted mean, basis, eigenvalues, dimensionality, regularization, and training-population hash. Fitting these transforms using future cohorts would introduce distributional information even without outcome labels; the controlled benchmark requires train-only fitting.

## Availability, repair, and the boundary around outcome information

![Historical enrichment architecture](../../../research/thesis_evidence/figures/h08_enrichment_architecture.png)

Historical figure H08. A tightly cropped historical diagram shows separate text and vision interpretation, deterministic validation and merging, and later snapshot reconciliation. Its post-event reconciliation path is an audit and repair mechanism. It does not authorize using post-event facts in a feature vector for an earlier decision; every such input still requires a separate decision-time availability check.

The architecture's ability to repair data creates a temporal responsibility. A later audit can correct current records and improve future predictions. It cannot make an older score retrospectively informed by a fact that arrived after the outcome. Completion timestamps, raw-content hashes, transform revisions, and field ownership must therefore travel with the features. A successful worker execution is evidence that computation occurred, not evidence that its output was available at every earlier origin.

This is particularly important when enrichment scheduling depends on lifecycle state. If a context block is populated only after a terminal event, its missingness may reveal the outcome even when its numerical values look harmless. Historical audits identified and removed such a route in later exports. The appropriate lesson is to test both values and availability patterns, then confirm the revised pipeline on a new cohort. The leakage chapter develops that case in detail.

## What the enrichment system establishes

The retained evidence supports a substantial, functioning measurement pipeline: hundreds of thousands of image records, structured visual tasks, field ownership, retries, incremental vector construction, explicit slot contracts, and incident-driven correction rules. Its scientific role is to make heterogeneous evidence consistently available to a censored-outcome experiment. That role is valuable independently of which survival estimator ultimately wins.

The next controlled evaluation should measure two linked questions. First, how accurately do the generated attributes represent visible or stated evidence, including abstentions and disagreement? Second, what incremental predictive value do those attributes and representations provide after temporal validation, relative to structured features alone? Separating these questions prevents an attractive explanation from being mistaken for a correct measurement and prevents a correct measurement from being assumed useful for predicting duration. It also makes the central research hypothesis testable: that carefully governed visual and textual evidence improves decisions enough to justify its operational cost.
