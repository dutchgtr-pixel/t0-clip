# Contribution comparison against the closest prior work

Literature and implementation review: 22 September 2026. This comparison identifies the contribution that the available work can support, the established ideas it builds on, and the experiments that distinguish competing explanations. It is a targeted review of the closest mechanisms, not a claim that an exhaustive search has proved worldwide priority.

## What is being contributed

The strongest research object is a measured failure and its reconstruction in an operational multimodal survival system: feature availability and calendar selection can create an apparently powerful duration signal, and a feature-source intervention can materially change a frozen predictor's behavior. The associated source code, cohort rules, replay results and governed reconstruction make that finding inspectable. The general existence of leakage, informative missingness, temporal databases, neural survival, multimodal fusion and generated annotations is established prior work.

Consequently, two comparisons must be made separately. A comparison against earlier research establishes what differs conceptually or empirically. A controlled comparison inside this platform establishes what a particular intervention changes. A successful second comparison does not automatically establish priority, but an empirical systems contribution does not require inventing every component that it uses.

The current public evidence includes the [leakage study](../leakage/LEAKAGE_STUDY.md), [aggregate incident evidence](../leakage/aggregate_evidence.json), [historical checks](historical_checks.json), [cascade methods](../cascade/METHODS.md), [feature-store methods](../feature_store_ddl/METHODS.md) and [cohort reconciliation](COHORT_RECONCILIATION.md). Newly recovered experiments must retain their cohort, checkpoint, threshold and source-version identities when added to this argument.

## Information availability and missingness

Kaufman, Rosset and Perlich's 2011 leakage paper already describes downstream processing artifacts, prediction-time observability, and missing diagnosis-code combinations that reveal a target. It proposes a formal legitimacy condition and separation of learning from prediction. A claim that this work invented the idea that missingness or an ordinary-looking column can leak would therefore be incorrect. [Kaufman et al., KDD 2011, sections 2–4](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)

Informative missingness is also not synonymous with leakage. GRU-D explicitly models masks and elapsed time to exploit observation patterns for prediction. Its first preprint appeared on 6 June 2016 and its journal article on 17 April 2018. The question for this platform is whether a pattern existed at the intended decision time and whether its predictive relationship transports to the intended population. Correlation with an outcome alone does not answer either question. [Che et al., 2018](https://www.nature.com/articles/s41598-018-24271-9)

Informative presence provides another nearby explanation: the process that produces observations can change which outcomes are recorded. McGee and colleagues formalize this through misclassification mechanisms and simulations. Their article concerns estimation from clinical records; it does not establish that every availability pattern in another domain has the same causal structure. [McGee et al., Epidemiology 2022](https://pubmed.ncbi.nlm.nih.gov/34711733/)

The measured contribution here is more specific: the frozen matrices expose a mixed zero-count/null-share pattern covering all 167 fast cases in 408 validation rows and all 208 fast cases in 503 evaluation rows. The pattern-only rule's F1 values are 0.8653 and 0.7661. These counts quantify the incident; they are not the incremental causal effect of adding the feature block to a neural model. The post-refresh, fixed-model intervention and calendar-cutoff reconstruction address different parts of the mechanism and must be reported separately.

The [recomputed mechanism receipt](../leakage/availability_mechanism_results.json) establishes an additional, sharper result: across all 911 frozen validation and evaluation rows, the mixed missingness pattern exactly matches whether origin is on or after 6 March 2026 UTC. There are zero cutoff mismatches. The recorded duration equals event date minus origin to within 1e-12 hours, and the observed event-window bounds satisfy the condition that forces every event within 72 hours into the missingness group. This is direct reconstruction of the selected data's geometry. It does not establish when the historical mask was first available to a live predictor.

### A sharper distinction: future information versus cohort-induced shortcuts

Let O be a prediction origin, D an observed event duration, and S=O+D the event date. Suppose a feature-missingness indicator follows a calendar cutoff c:

`M = 1[O >= c]`.

If evaluation retains events with dates S in a narrow interval [a,b], then within that selected population:

`M = 1[D <= S-c]`.

This is an algebraic identity. It requires no future value to have been supplied to the model. Conditioning on event date couples origin with duration. A calendar-dependent mask that was actually known at O can consequently become a duration proxy in the selected evaluation population. Under the illustrative additional assumption that S conditional on D and selection is uniform on [a,b], the resulting relationship is:

`P(M=1 | D=d, S in [a,b]) = clip((b-c-d)/(b-a), 0, 1)`.

The uniformity assumption is a synthetic example, not an assertion about the historical population. The transition interval [a-c,b-c] gives a concrete diagnostic prediction that can be compared with real coverage and duration tables. If the real mask has the opposite coding, replace M with 1-M.

The accompanying constructed verification is an executed existence demonstration. It sets duration independent of origin and makes the mask known at origin, then applies alternative cohort rules. On the outcome-window cohort, the fixed mask-only rule has balanced accuracy 0.9537; on a fully matured origin-window cohort, both mask groups have event rate 0.10 and balanced accuracy is 0.50. This does not estimate the historical model's prospective performance. It demonstrates that the observed kind of association can arise without future feature access, and therefore prevents a mistaken causal diagnosis based on high missingness predictiveness alone. [Constructed experiment and aggregate historical audit](../leakage/availability_mechanism_results.json)

This mechanism is analytically distinct from a feature version arriving after O. For the latter, accurate availability timestamps and a known-time cutoff can exclude inadmissible evidence. For the former, a perfectly implemented known-time cutoff leaves the selection effect intact. The evaluation population itself needs to match the origin-defined population in which predictions will be used, with appropriate follow-up and censoring.

The closest identified recent comparator is Xu and colleagues' July 2026 preprint on administrative-cutoff leakage. They study inputs that reveal reference date and a fixed follow-up endpoint that makes later records less observable. Their simulations and clinical experiment concern censoring-induced date shortcuts. Here the specific identity concerns conditioning on an event-date window, with an operational coverage cutoff supplying the date proxy. Both involve observation design, but the selection predicates differ. This is a useful comparison to test rather than a basis for declaring either mechanism the first of its kind. [Xu et al., version 1, 11 July 2026](https://arxiv.org/html/2607.10466v1)

Classical length-biased prevalent sampling is related but also has a different inclusion rule: individuals with longer durations are more likely to be present at a sampling time. Selecting events in a calendar interval need not produce that same length-proportional sampling law. Likewise, immortal-time bias concerns follow-up during which an outcome cannot occur under the exposure or eligibility definition; that label should not be applied merely because a calendar cutoff is present. [Zelen, 2004](https://pubmed.ncbi.nlm.nih.gov/15690988/); [Suissa, 2008](https://pubmed.ncbi.nlm.nih.gov/18056625/)

The defensible claim is therefore a measured, explicitly reconstructed observation-process failure in this system, with separate tests for inadmissible evidence and admissible but nontransportable calendar proxies. The falsifier is a reconstruction that cannot reproduce the documented masks or their cutoff geometry. Genuine calendar drift remains an alternative explanation for residual date association; removing all calendar signal indiscriminately is not an appropriate definition of success.

## Event time, known time and fitted information

The distinction between valid time and transaction time predates modern feature stores. Snodgrass and Ahn's temporal database work explicitly separates these dimensions. The underlying two-clock idea is not new here. [Snodgrass and Ahn, Temporal Databases, 1986](https://www2.cs.arizona.edu/~rts/pubs/Computer.pdf)

The feature-store comparison is also more specific than saying that an existing product is or is not point-in-time correct. Feast issue 2980, opened on 28 July 2022, requested reproducible historical queries despite late-arriving or restated information. It proposed a global created-time boundary. Issue 6615, opened on 19 July 2026, proposed a per-entity-row boundary, and pull request 6617 merged on 31 July 2026. The documented opt-in rule constrains created time in addition to event time. A January or February 2026 implementation can precede that particular merge without establishing priority over the principle or the earlier discussion. [2022 issue](https://github.com/feast-dev/feast/issues/2980); [2026 proposal](https://github.com/feast-dev/feast/issues/6615); [merged implementation](https://github.com/feast-dev/feast/pull/6617)

MarketNeural's public temporal example composes several inspectable boundaries: source versions, image/report masks, external release availability, known outcomes used by priors, target-entity exclusion, fitted-anchor membership and label maturity, and guarded reads. That composition has a useful experimental unit: a decision and all the information that produced its input. Its contribution can be demonstrated by explicit counterexamples and a comparator that only constrains event time, followed by a comparator that also constrains known time. It should not be described as a theorem that arbitrary production data are leakage-free.

In particular, a timestamp on a learned anchor does not establish that its fitted population excluded evaluation labels. Conversely, a correct fitted-membership contract cannot make a revised source document available in the past. These are distinct requirements. The portable SQL is newly authored and tested; the surviving historical SQL is implementation evidence. Neither is evidence that every historic source retained a complete immutable revision history.

## Neural survival, fusion and staged decision horizons

Neural survival distributions and modality fusion have direct precedents. MCAT combines image and genomic representations with co-attention for survival prediction. MoME uses multimodal expert selection and progressive fusion. Perceiver supplies a precedent for latent attention over heterogeneous inputs. These works make claims about specific fusion methods, so describing MarketNeural only as attention plus experts plus survival does not sufficiently differentiate it. [MCAT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html); [MoME, MICCAI 2024](https://papers.miccai.org/miccai-2024/531-Paper2168.html); [Perceiver, ICML 2021](https://proceedings.mlr.press/v139/jaegle21a.html)

There is prior work on horizon-specific survival ensembles as well. WRSE combines binary classifiers at time resolutions chosen to emphasize short-term prediction. It is not the same as this platform's ordered rejection cascade, but it defeats a broad claim that assigning models to different horizons is itself unprecedented. [Heitz et al., 2021](https://proceedings.mlr.press/v146/heitz21a.html)

The specific system here has three distinct 504/168/72-hour policies, short-censor masking, fractional exposure in the discrete-time objective, recency weights, different fitting and routed populations, independent scalar and curve channels, and fitted ensemble policies. The source-derived implementation establishes these semantics. Whether the cascade improves utility over a simpler survival model is a separate empirical claim.

The sharp comparison uses the same admissible representations, origin-defined cohorts, censored labels and development budget for: one survival model queried at the three horizons; independent horizon heads with the same downstream policy; and the three-stage cascade. Report both stage-local and end-to-end outcomes, including fast cases rejected upstream. Different horizon values and many tuning trials establish neither incremental benefit nor a novel loss.

The recovered historical AFT/neural comparison remains a real system comparison. It does not isolate architecture from representation, ensembling, calibration and selection. Conventional models should receive the same frozen embeddings and train-fitted reductions before claiming that they cannot exploit high-dimensional representations. Ordinary and whitened text vectors are experimental factors only when the actual fitted transform, membership and wiring are recoverable.

## Generated reports as measurements

Concept bottleneck models establish an earlier route from inputs through interpretable predicted concepts to an outcome. MarketNeural's parallel embeddings and generated fields are not automatically a strict bottleneck, because other branches can bypass the generated concepts. [Koh et al., ICML 2020](https://proceedings.mlr.press/v119/koh20a.html)

A particularly close survival comparator is Song and colleagues' preprint, first submitted on 12 May 2025. Version 1 combines frozen image, expression and report representations with classical survival estimators, examines generated report summaries, and measures the effect of correcting hallucinations. Thus, generated text plus frozen multimodal embeddings for survival is established prior work. The version-specific citation matters because the later title changed. [Song et al., 2025, version 1](https://arxiv.org/abs/2505.07683v1)

The distinguishable implementation here is the field-ownership and conflict policy, bounded interpretation tasks, eight paired image/report slots, separate masks, and temporal provenance attached to generated measurements. Useful empirical questions are whether these controls reduce wrong-field updates and inconsistent outputs, whether corrections affect a frozen downstream model, and whether reports add benefit beyond their source images under matched availability.

Low-temperature generation and schema validity are not guarantees of correctness. A malformed-output test is a software result; a blind labeled audit measures extraction quality; repeated runs measure output stability; a matched downstream ablation measures predictive contribution. Keeping those endpoints separate gives the existing operational work a scientifically interpretable role.

## Evidence needed for each narrower claim

Existing retained artifacts should be recovered and reconciled before repeating expensive training. A saved result is eligible for a controlled comparison only when the relevant constants and changed factor can be identified.

| Narrow claim | Closest comparison and required evidence | What would refute or narrow it |
|---|---|---|
| A specific coverage cutoff creates a duration shortcut under event-date selection | Reconstruct coverage and cohort predicates; compare the derived transition with observed cross-tabs; evaluate a date-only or mask-only rule; contrast event-defined and origin-defined cohorts | The masks do not follow the cutoff, or the derived transition fails; residual origin-cohort association may instead reflect genuine drift or another process |
| A feature-source correction changes frozen-model behavior | Same rows, labels, checkpoint, preprocessing, score channel and threshold; change only the documented input source; retain paired scores or decisions | Cohort or threshold changes explain the apparent improvement, or unrecorded changes prevent isolation |
| The full reconstruction contract catches defects that a simpler join misses | Event-time-only, event-plus-known-time, and full contract against named failure cases; execute actual retrieval paths | A simpler comparator prevents the same failures, or the proposed contract permits one of its claimed exclusions |
| Staged routing improves an operational objective | Equal-input and equal-budget one-model, independent-head and cascade comparisons; upstream rejection accounting; temporal held-out results | Equivalent or better performance from the simpler model, or benefit disappears after matching inputs and budgets |
| Generated reports add useful information or improve usable representation | Image-only, report-only and combined inputs on the same rows; matched masks; field audit and frozen-model correction test | Gain is explained by availability, source revision, selection, or additional model budget rather than reports |
| Recency weighting improves adaptation | Frozen model family with no decay versus candidate half-lives selected on development data; evaluation across later origin cohorts | Benefit is confined to reused tuning rows, reverses across periods, or is explained by changed feature coverage |

The purpose of this matrix is to connect experiments to claims. Unit tests establish implementation properties. Source replay establishes reconstruction fidelity. Fixed-input interventions estimate changes under a defined intervention. Held-out temporal comparisons assess predictive transport. None should be discarded, and none should silently stand in for another.

## Contribution wording supported by this comparison

A defensible formulation is: this work documents and reconstructs a calendar- and availability-dependent shortcut in an operational multimodal survival system; separates prediction-time admissibility from selection-induced duration information; measures controlled feature-source interventions where retained artifacts permit them; and releases executable contracts and model semantics that expose the failure conditions to testing.

That formulation is stronger than an undifferentiated claim of a novel neural architecture because it identifies the observation, mechanism, implementation and falsifier. It remains open to refinement by recovered experiments. It does not claim that this work invented temporal correctness or that a new name for an established effect proves originality. A broader methodological claim requires showing which failure or guarantee remains different after the closest published approaches are given the same information and evaluated under the same contract.
