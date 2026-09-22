# Research contributions established by reconstruction and experiment

## The object of the contribution

The central research object is a multimodal survival decision whose inputs are produced asynchronously, whose outcome is observed later, and whose evaluation depends on how records enter a cohort. The platform made this object observable through source reconstruction, retained prediction files, feature-version audits, model experiments and serving investigations. The scientific contribution is expressed through specific findings about that object, with an explicit comparator and a test for each claim.

Three questions organize the completed work. First, can an apparently valid feature become a duration shortcut through the observation and cohort-selection process? Second, how much can restoring the feature contract change a fixed model's behavior? Third, what do retained common-population comparisons establish about the developed neural systems and their ensemble policies? The following sections connect these questions to existing experiments and new reanalysis of retained artifacts. Prospective prediction, modality attribution and optimal recency weighting remain separate questions.

The underlying ideas have precedents. Kaufman and colleagues already formalized prediction-time legitimacy and discussed missingness and downstream processing as leakage sources. Temporal databases distinguish valid time from transaction time. Neural survival distributions, multimodal attention and horizon-specific ensembles are established methods. Originality therefore concerns the measured failure, its explicit reconstruction, the tested intervention and the operational contract. The [prior-work comparison](../../../research/thesis_evidence/NOVELTY_PRIOR_ART.md) identifies the closest methods, differences and falsifiers. [Kaufman et al., 2011](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf); [Snodgrass and Ahn, 1986](https://www2.cs.arizona.edu/~rts/pubs/Computer.pdf).

## Finding 1: a cohort-induced shortcut can survive correct time filtering

The historical feature investigation found a mixed zero-count/null-share pattern that included every fast-event positive in two frozen matrices: 167 of 167 in 408 validation rows and 208 of 208 in 503 evaluation rows. The pattern alone gives F1 0.8653 and 0.7661. Those are calculated diagnostic results, not attributed neural-model gains. Their importance is the direct relationship between a production coverage pattern and the supervised endpoint.

A fresh audit of the retained matrices verifies that the mask follows the documented 6 March 2026 UTC origin cutoff on all 911 rows, with zero mismatches. Stored duration equals event time minus origin to numerical precision. Every row has an observed event. The association is consequently stronger evidence than a suspicious correlation alone: the calendar rule reproduces the pattern and the cohort rule explains its relation to duration. This verifies the stored pattern; it does not recover when that pattern first became available to a live predictor. The [mechanism receipt](../../../research/leakage/availability_mechanism_results.json) preserves aggregate counts and source hashes.

Let $O$ be the prediction origin, $D$ the observed duration and $S=O+D$ the event date. Let an operational coverage boundary produce a mask

$$
M=\mathbf{1}\{O\geq c\}.
$$

Within a cohort selected by $a\leq S\leq b$, the identity becomes

$$
M=\mathbf{1}\{D\leq S-c\}.
$$

A mask associated with origin time becomes a duration proxy when event date is constrained. This is an exact algebraic consequence of the selection rule. No future feature value needs to enter the model for the effect to occur. In particular, if $c\leq a-h$, every observed event with $D\leq h$ satisfies $O=S-D\geq a-h\geq c$ and therefore $M=1$. The historical cohorts satisfy this sufficient condition for $h=72$ hours.

If the selected event date is uniform on $[a,b]$ conditional on duration, an illustrative special case is

$$
P(M=1\mid D=d,\ a\leq S\leq b)
=\operatorname{clip}\!\left(\frac{b-c-d}{b-a},0,1\right).
$$

The uniformity condition belongs to this illustrative calculation; it is not imposed on the historical data. The transition interval $[a-c,b-c]$ supplies a diagnostic prediction to compare with observed coverage. Reversing the mask coding reverses the association.

The new public construction supplies a controlled counterexample with origin and duration independent before selection. The mask is known at origin and the feature value is nonpredictive. Nevertheless, selection by event date produces strong apparent discrimination. Selecting instead by origin with complete follow-up removes the induced balanced-accuracy advantage. Filling coverage on the same frozen event-selected rows changes the fixed mask rule again. These are constructed mechanism tests, separately labeled from historical performance and documented fully in the [executable mechanism study](../../../research/leakage/availability_mechanism.md).

Two distinct checks follow. An information-availability check asks whether the value and its presence state were known at $O$. A population check asks whether selected records represent the origin-defined decisions for which the model will be used. A system can pass the first and fail the second. A created-time cutoff addresses genuinely late information but cannot, by itself, change an event-selected evaluation population.

The closest identified comparison on calendar-dependent survival artifacts concerns administrative censoring: a fixed end of follow-up can let time-indexed inputs reveal how much observation is possible. The mechanism here conditions on event date and uses a coverage boundary as an origin proxy. These selection rules differ, even though both connect calendar information with apparent survival performance. This gives a specific comparison to test rather than a claim of first discovery of temporal bias. [Xu et al., 2026, version 1](https://arxiv.org/html/2607.10466v1).

Informative missingness can also be legitimate. GRU-D deliberately uses observed masks and elapsed time. Genuine changes in a market can make calendar features useful. The remediation objective is to remove inadmissible information and unintended cohort geometry while retaining legitimate information about current conditions. Removing every date or mask indiscriminately would not establish success. [Che et al., 2018](https://www.nature.com/articles/s41598-018-24271-9).

## Finding 2: feature reconstruction is an experimentally consequential part of the model

The retained forward-pass investigation records a fixed-policy intervention on a frozen cohort of 443 observed records, of which 441 were scored. Correcting the structured context and item-metadata source raised ensemble F1 from 0.3077 to 0.7077; precision moved from 0.5714 to 0.8519 and recall from 0.2105 to 0.6053. The documented model weights, labels and ensemble threshold were unchanged. The corrected confusion matrix is 46 true positives, eight false positives, 30 false negatives and 357 true negatives.

This experiment measures a different effect from retraining a neural model with an extra feature group. The intervention restores the input representation expected by an existing predictor. Its result shows that serving-source fidelity can dominate the observed score of that predictor. It supports treating reconstruction policy and source identity as part of the experimental model, rather than assuming checkpoint identity alone defines a repeatable comparison.

The result is documented in the retained forensic report. The complete paired before/after prediction exports have not been recovered in this revision, so the report is not relabeled as a newly executed paired experiment or given a reconstructed confidence interval. Other fixes were made during the broader investigation; the claimed contrast is the documented context-source correction within the frozen benchmark. The 443/441 eligibility boundary is retained. This reconstruction of observed outcomes does not constitute a sealed prospective cohort.

The methodological lesson has a direct counterfactual form: keep the evaluated cases, outcome labels, trained function and threshold fixed; change the defined source reconstruction; measure the resulting predictions and contract agreement. A report that changed any of those constants would support a weaker conclusion. Preserving those constants makes the incident an empirical systems result even though feature-serving skew is a known general problem.

## Finding 3: ensemble benefit depends on the stage and retained population

Retained row-level decisions support two additional completed comparisons. In the later Stage 0 experiment, seven combination methods share 523 records, matching labels and invariant fields; the positive class is duration greater than 504 hours, with 104 positives. The mean-logit combination has F1 0.803738; the histogram-gradient-boosting combiner has 0.755319. The latter minus the former gives a paired difference of -0.048419, with a 95% bootstrap interval from -0.102077 to 0.001227. The SGD combiner has F1 0.812500, but its difference from mean-logit also has an interval containing zero.

In the separate Stage 1 experiment, 413 matched records permit direct comparison of the mean-logit baseline and retained histogram-gradient-boosting hybrid. Its positive class is an event within 168 hours under the retained gate contract, with 296 positives. Their F1 values are 0.686567 and 0.875214. The hybrid minus the baseline gives a difference of 0.188647, with a paired 95% interval from 0.144285 to 0.235651. The hybrid is uniquely correct on 97 records, while the baseline is uniquely correct on 23.

Both analyses resample the same records jointly, using 5,000 paired IID bootstrap draws with seed 20260922. They condition on the fitted models, selected policies and available cohorts. They do not include model fitting, experiment selection, tuning uncertainty, temporal dependence or multiplicity adjustment. These are retrospective component comparisons; they are not fresh sealed tests.

The Stage 1 hybrid combines neural outputs using a tree-based meta-model. Its advantage is evidence for that hybrid combination on the recorded population, not evidence that neural methods universally beat tree methods. The less favorable Stage 0 result is retained because it limits the generalization of the finding. Different stages have different targets, cohorts and development histories; the contrast is not a controlled estimate of an intrinsic stage effect.

A separate retained replay on 587 records supplies a further negative result. At their saved operating points, the base output has F1 0.370079 and the meta output has F1 0.090909. Saved checks report exact tabular metadata, no serving fallback and no missing feature-contract fields; those checks do not establish semantic feature parity at the original decision time. This replay is distinct from the corrected 441-row benchmark. Its poor performance cannot be assigned uniquely to temporal decay, source reconstruction or the meta architecture, but it rules out presenting the archive as uniformly successful ensemble transfer. The [experimental receipt](../../../research/thesis_evidence/experimental_contributions.json) preserves these results alongside the favorable comparisons.

The research contribution is a reproducible comparison of existing policy-combination choices on fixed retained populations, with both favorable and unfavorable outcomes preserved. Matching verifies the retained decisions, labels and declared invariant fields; it does not reconstruct the entire underlying seed-score matrix. It establishes that an executed component study exists. It does not automatically establish the incremental contribution of a raw image, report, text embedding, whitening transform, loss term or recency setting.

## The common-population neural and AFT comparison

The earlier model-family comparison is reconciled separately. All 964 retained neural export keys and duration labels match the later AFT export. Applying the saved AFT rule to those common records gives F1 0.855967. The associated final neural meta-ensemble run records F1 0.9209, approximately 6.5 percentage points higher. Precision improves while recall decreases. This supports a developed-system comparison on that historical population.

The final neural meta decision vector for this February result has not been recovered. Its integer confusion matrix is uniquely compatible with the rounded metrics and recorded class counts, but separate confusion matrices do not identify the row-by-row disagreement structure. They therefore do not determine a unique paired bootstrap interval. No interval from the later 523- or 413-row experiments is assigned to the 964-row headline result.

A neural-system advantage does not isolate architecture from representation, ensembling, calibration and selection. Choosing a winner after examining evaluation scores does not produce an untouched final test. The archive distinguishes Stage 0's February prediction pool from Stage 1's later 8,000 ensemble candidates. Trial counts and search-seed counts are not additional independent evaluation populations.

## The executable method: information, population and reconstruction

The reusable method connects prediction-time evidence to cohort construction and serving fidelity. Its unit of analysis is a decision with a defined origin, admissible source versions, fitted transformations, available labels, model identity and downstream policy. The released implementation exposes these components separately so that a failed comparison can be traced to a specific boundary.

Event time and known time are necessary dimensions, but their distinction is established. Feast's 2022 historical-reconstruction discussion and its July 2026 per-row created-time implementation are concrete comparators. An implementation preceding the latter merge does not establish priority over the principle. The platform's additional research object includes image/report masks, label maturity, anchor fitting membership, cohort eligibility and fixed-policy reconstruction. Each additional condition needs a named failure case; a long checklist alone is not an evaluated method. [Feast discussion, 2022](https://github.com/feast-dev/feast/issues/2980); [created-time implementation, 2026](https://github.com/feast-dev/feast/pull/6617).

The public tests demonstrate declared contract behaviors. The documented historical replay reports agreement with frozen source values; the database comparison was not rerun in this revision. The calendar-selection construction demonstrates why a population error can survive a valid availability rule. The source intervention measures a documented downstream consequence of reconstruction. Together they address distinct failure mechanisms. They are not a guarantee that arbitrary production data are perfectly free from leakage.

## Model and enrichment contributions relative to established methods

The three-stage system implements separate 504-, 168- and 72-hour policies, stage-specific populations, short-censor masks, survival losses, recency weighting and ensemble selection. That is a concrete, released formulation. WRSE already combines horizon-specific classifiers; neural survival and multimodal co-attention have direct precedents. The distinguishing experiment for a broad cascade advantage compares this routing system with one survival model queried at all three horizons and independent horizon heads using the same inputs and policy. Current stage-local and ensemble results do not automatically answer that end-to-end question. [Heitz et al., 2021](https://proceedings.mlr.press/v146/heitz21a.html); [Chen et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html).

Generated interpretation is implemented through bounded tasks, field ownership, uncertainty, conflict checks and paired image/report slots. This makes generated attributes part of a governed measurement process. Concept bottleneck models and earlier multimodal survival work using generated reports provide relevant comparisons. A report remains a dependent transformation of its source, and schema compliance does not establish perceptual accuracy. The contribution concerns the implemented measurement contract and documented incidents; incremental report benefit requires a matched modality experiment. [Koh et al., 2020](https://proceedings.mlr.press/v119/koh20a.html); [Song et al., 2025, version 1](https://arxiv.org/abs/2505.07683v1).

The ordinary-plus-whitened text design remains distinct from the selected network actually inspected. A preserved covariance transform, fitted population and corresponding model wiring are required before treating it as an executed dual-band ablation. Whitening has established precedents. [Huang et al., 2021](https://aclanthology.org/2021.findings-emnlp.23/).

## What the research now demonstrates

The supported contribution is a reconstructed observation-process failure, an explicit distinction between information admissibility and outcome-conditioned cohort geometry, a documented fixed-policy reconstruction intervention, and retained common-population comparisons of survival systems and ensemble policies. Released code, aggregate receipts and named counterexamples connect these claims to inspectable experiments.

These claims identify what another researcher can reproduce, challenge or extend. A failed cutoff reconstruction would weaken the incident mechanism. A changed cohort or threshold would weaken the source-intervention interpretation. A simpler contract that catches the same defects would narrow the claim for additional controls. A matched future comparison that reverses the neural advantage would limit its predictive transfer.

A sealed prospective evaluation would add evidence about prediction under later conditions. It would not, by itself, establish that any general principle or neural component is new. The completed studies remain scientific evidence while answering narrower questions than universal model superiority. The originality argument rests on the measured mechanisms and comparisons, assessed against specific precedents rather than inferred from platform size or experiment count.
