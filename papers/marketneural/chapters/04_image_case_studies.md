# Real visual evidence and structured judgments

## Why include actual photographs

The image pipeline was intended to act as a scalable visual inspection layer. A product photograph contains information that is difficult to capture through a small fixed set of seller-supplied fields: cracks, reflections, protective covers, packaging, framing, accessories and whether a useful surface is visible at all. The pipeline converted this information into bounded judgments and short descriptions that could be stored, inspected and represented numerically. The examples in this chapter use retained photographs and saved model outputs, rather than generated illustrations.

The release selects two cases from a retained image-evidence snapshot generated on 18 April 2026. That snapshot contains 42,082 listing-level rows with nested per-image judgments; this is a different unit and date from the later platform-wide image-asset count. Local image files were matched through the private record key and stored image index. The private keys, source URLs, external-drive paths and original filenames are omitted from the public examples. Public files use anonymous case labels and carry image hashes in the evidence manifest.

This pairing has a material limit. The retained snapshot did not provide an immutable image-byte digest recorded at inference time. Agreement of record key and image index, supported by visual consistency, is weaker than a content-hash join. The examples therefore illustrate the saved output contract and its interpretability. They are not a blinded image benchmark, an estimate of accuracy or proof that every original input byte was recovered. Cases with obvious content/index inconsistency were not used as evidence of model correctness.

## Case A: visible damage converted to a bounded record

Case A shows a front surface with visible cracks. The saved model summary reads: **"front screen cracked with visible cracks"**. Its stored `visible_damage_level` is 2, `photo_quality_level` is 3, and `damage_on_protector_only` is false. A separate `image_damage_level` field is null. These values belong to different fields and scales; the null must not be filled by copying the visible-damage value merely because both names mention damage.

![Case A. Retained photograph paired with the saved summary "front screen cracked with visible cracks". The record contains visible damage level 2 and photo quality level 3; these are model-produced ordinal fields, not calibrated probabilities. Anonymous example, selected for illustration.](research/thesis_evidence/figures/case_a_damage.png){width=48%}

The associated local image caption states, in translation with the product name generalized, "Almost new [device]." The visible cracks show why this broad condition description cannot substitute for visual inspection. This is an illustrative evidence conflict, not a finding of intentional deception. Nor does an illuminated display establish complete functional condition: visible damage can coexist with an undisclosed internal fault or repair history that cannot be seen.

This example explains why feature construction should retain both the structured fields and the report. The bounded ordinal fields are easy to aggregate and validate. The report preserves a more specific location and visual pattern. A text encoder can represent those semantics for the survival model, while the original image encoder provides a separate visual representation. The generated report is an intermediate interpretation of the image, so its errors may be correlated with visual difficulty rather than independent noise.

## Case B: visibility limits must survive the pipeline

Case B shows an inactive display with strong reflections. The saved summary reads: **"Screen off with strong reflections and some smudges/dust; no clear scratches or cracks visible."** Its stored visible damage level is 2, photo quality level is 2, and protector-only flag is false. The separate image-damage field remains null.

![Case B. Retained reflective-screen photograph and its saved visibility-qualified judgment. "No clear scratches or cracks visible" describes this photograph; it does not establish absence of damage. The source record's ordinal fields are retained without reinterpretation.](research/thesis_evidence/figures/case_b_reflection.png){width=48%}

The photograph makes the distinction between absence and non-observation concrete. A bright reflection can obscure scratches. An inactive display cannot establish that the panel works. A clean-looking region does not certify surfaces outside the frame. An operational narrative that rewrites this output as "damage-free" would remove information that the image model correctly left uncertain.

The structured value also warrants scrutiny: a visible-damage score of 2 accompanies a summary that reports no clear crack or scratch. Without the original rubric, that number should not be translated into an intuitive label such as "moderate damage." A robust contract carries the rubric version, field meaning and raw interpretation together. The current case study preserves the recorded values and uses the mismatch as a reason to inspect semantics rather than an excuse to invent a more flattering interpretation.

## From per-image outputs to listing representations

A listing may contain several photographs that provide complementary or conflicting evidence. One may show the rear surface, another the front, another packaging, and another a settings screen. Aggregating these into a listing-level feature requires role-aware rules. A packaging image should not certify the condition of an unseen device. A screenshot of a generic settings page should not be treated as proof of a specific battery metric. A stock image should not be substituted for a photograph of the offered item.

The documented system addressed these distinctions through image-role labels, bounded attribute extraction, damage descriptions, quality fields, explicit missing values and downstream narrative rules. The later selected neural model used up to eight image vectors and eight image-report vectors. Retaining slots preserves more information than collapsing all photographs into a single unqualified maximum damage score. It also creates new responsibilities: slot ordering, vector dimensions, mask semantics and report-to-image alignment must remain stable between training and serving.

A model-produced score is not a probability merely because it is numeric. Ordinal damage and photo-quality fields should be encoded according to their documented scales. A confidence field requires calibration before it can support probability claims. Averaging incompatible scales across versions produces a precise-looking but uninterpretable feature. The feature store must therefore preserve producer versions and transform contracts alongside values.

## What would demonstrate performance at scale

These examples demonstrate inspectability, not measured hallucination suppression. A quantitative evaluation would select a representative image sample before examining model success, obtain independent annotations with adjudication, and report disagreement by image role, visibility, damage type and product subgroup. It would separately measure invalid schema outputs, unsupported specific claims, missed damage, incorrect localization and repeated-run variability. Each denominator must be explicit.

For text damage analysis, the parallel study should distinguish direct seller statements, negation, repaired historical damage, current functional faults and uncertain language. A short sentence about a replaced component must not be reclassified automatically as a current fault. Cross-modal agreement is useful, but correlated evidence must not be counted twice as independent confirmation.

The practical value of the implemented design is that it preserves material for these checks. Structured outputs, explanations, original content and downstream tests make it possible to investigate an error. The next research step is to quantify how often each error occurs and whether additional visual and semantic representations improve survival forecasts under an identical temporal protocol.
