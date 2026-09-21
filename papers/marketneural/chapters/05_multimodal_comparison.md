# Comparing neural, proportional-hazards, and tree models

## The comparison question

The main empirical question is whether a learned multimodal survival model improves the quality of predictions made from the same admissible evidence. Architecture size, the presence of attention, and the ability to process high-dimensional vectors do not answer that question. A fair comparison requires common populations, time origins, censoring rules, information cutoffs, evaluation horizons, and selection budgets. It must also distinguish the benefit of richer inputs from the benefit of a particular estimator.

The retained prediction exports establish a common 964-record evaluation population with identical entity keys, time origins and duration labels. AFT F1 recomputed on those rows is 0.8560; the associated historical neural meta-ensemble run records F1 0.9209 and precision 0.9802. The empirical chapter distinguishes recomputed AFT decisions from recorded neural aggregates and shows the accompanying recall tradeoff. The earlier AFT score of 0.8462 belongs to the preceding evaluation week. The controlled design below extends the shared-data evidence by separating the effects of estimator, representation and tuning.

This distinction is especially consequential here. The historical tree work used carefully engineered market and anchor features, while the later neural system consumes structured fields and semantic representations. Comparing a neural model with images against a tree trained only on price would confound representation and estimator. Conversely, forcing a tree to use thousands of raw coordinates without giving it a sensible regularized or reduced representation would test a particular implementation choice rather than the broad usefulness of trees.

The controlled research design therefore has two axes. One axis changes the estimator while holding the information set fixed. The other changes the information set within each estimator. Structured-only, structured-plus-text, structured-plus-images, and full evidence configurations identify where any improvement arises. Image reports should be added as a separate ablation because they are generated transformations of image evidence, not an independent sensor. Missingness indicators and preprocessing must be identical in meaning even when each estimator represents them differently.

## Why neural fusion is plausible and not guaranteed

Dense text and image vectors distribute information across coordinates. A small semantic change can move many coordinates together. Axis-aligned tree splits may require several partitions to represent such relationships, whereas a learned linear projection can combine them directly. Attention adds a mechanism for conditioning an item's representation on selected interactions between structured features and image or text tokens. Latent bottlenecks limit the number of internal positions and can make heterogeneous inputs computationally manageable.

These mechanisms offer plausible explanations for the observed historical improvement; their individual contributions require ablation. Dense vectors can be redundant, noisy, weakly related to duration, or dominated by irrelevant visual style. Neural optimization can fit those nuisances, especially when the effective number of independent observed events is much smaller than the number of image assets. Missing modalities can also create shortcuts. A high-capacity network can learn the pattern of which jobs finished rather than the item's condition.

Trees have complementary strengths: useful nonlinear interactions, robustness to monotone transformations, relatively direct handling of heterogeneous tabular scales, and strong performance in many medium-sized tabular problems. Grinsztajn, Oyallon, and Varoquaux (2022) provide a broad empirical reason to retain strong tree baselines rather than presume that deep learning wins. Their benchmark concerns typical tabular datasets; it does not settle this particular censored multimodal task. [Grinsztajn et al., 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).

The useful hypothesis is conditional: neural fusion may help when interactions among meaningful, available semantic representations contain information not captured by the structured baseline, and when the sample supports learning those interactions. Its falsifiable implication is improved held-out prediction under shared temporal controls. If a proportional-hazards or tree model performs better, the appropriate conclusion is that the added flexibility did not earn its cost under that experiment.

## A ladder of estimators

The public benchmark implements six families. A Kaplan–Meier model supplies an unconditional survival curve and establishes the value of conditioning on covariates. Cox proportional hazards adds a linear predictor with an estimated baseline survival function. Random survival forests provide nonlinear partitioning and aggregation. Gradient-boosted survival analysis provides a second tree family. A multilayer perceptron and a compact Perceiver-style mixture model learn discrete-time survival curves from the supplied features.

| Family | Main inductive assumption | Role in the experiment |
|---|---|---|
| Kaplan–Meier | Shared unconditional survival distribution | Population-only baseline |
| Cox proportional hazards | Multiplicative hazard effect with a fixed covariate score | Strong interpretable statistical baseline |
| Random survival forest | Nonlinear partitions and ensemble aggregation | Flexible tree baseline |
| Gradient-boosted survival analysis | Additive boosted predictor under the configured survival loss | A second strong tree baseline |
| Multilayer perceptron | Dense nonlinear feature interactions | Neural baseline without latent cross-attention |
| Compact Perceiver mixture | Latent attention and a mixture of survival experts | Structured multimodal fusion hypothesis |

The gradient-boosted implementation in the public demonstration uses its configured proportional-hazards loss. It is not an exact reproduction of the historical AFT model merely because both employ trees. Likewise, the compact neural benchmark is an executable comparison model, not a claim that the complete archived cascade has been re-created with all original training data and tuning.

The underlying literature motivates this ladder. Cox (1972) formalizes regression with a baseline hazard left unspecified. Ishwaran and colleagues (2008) develop random survival forests. DeepSurv supplies a nonlinear proportional-hazards predictor, while DeepHit and discrete-time neural survival methods demonstrate other ways to parameterize event distributions. Deep Survival Machines uses parametric survival mixtures. These methods differ in assumptions, losses, and flexibility, so an architecture label alone is an insufficient comparison unit. [Cox, 1972](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1972.tb00899.x); [Ishwaran et al., 2008](https://ishwaran.org/papers/IKBL.AOAS.pdf); [Katzman et al., 2018](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1); [Lee et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11842); [Gensheimer and Narasimhan, 2019](https://peerj.com/articles/6257/); [Nagpal et al., 2021](https://arxiv.org/abs/2003.01176).

## Information parity and representation parity

Information parity means each family is allowed the same evidence at the same decision time. Representation parity is a stricter condition and can be counterproductive if it prohibits an estimator's natural structure. The public benchmark flattens aligned masked vectors for conventional estimators and the dense neural baseline, while the attention model receives modality tokens. This makes evidence availability comparable while allowing a designed difference in architecture.

A fuller experiment should add dimension-controlled baselines. A train-fitted projection can reduce vector width before Cox or tree fitting. An embedding-only model can test whether most information resides in the pretrained representation rather than the fusion mechanism. A structured-only neural model tests whether the architecture helps without images. These experiments should be planned before opening the final outcomes so that the best-looking ablation is not retrospectively promoted to the primary claim.

Missingness needs special attention. The later slot-aware network and the compact benchmark use learned missing representations for unavailable branches; they do not universally remove those positions from attention. A slot mask can suppress or replace raw values while a learned missing token remains available to the model. The older legacy cascade accepts optional modality masks but does not use them to suppress attention, so its missing-vector behavior depends on the upstream representation. Both designs can expose availability to the model. This is legitimate when availability itself is admissible at the decision time and a possible leakage route when availability is downstream of the event. Consequently, masks and tokens must be audited as features, not treated as a purely technical implementation detail.

The Perceiver precedent supports learning through a fixed latent array rather than applying expensive interactions at every input position. The later mixture-of-multimodal-experts literature provides another relevant comparison point. Neither establishes novelty for every element used here. The defensible contribution is a particular survival formulation and evidence contract implemented within a temporally governed operational platform. [Jaegle et al., 2021](https://proceedings.mlr.press/v139/jaegle21a.html); [Xiong et al., 2024](https://papers.miccai.org/miccai-2024/531-Paper2168.html).

## Training, selection, calibration, and final evaluation

The ideal real-data protocol separates four chronological roles: fitting, architecture and hyperparameter selection, calibration and decision-policy fitting, and final evaluation. Entity identities cannot cross these partitions through revised listings or repeated snapshots. Observations that began before a cutoff but whose outcome was learned later are censored at that cutoff in the simulated training state. Preprocessing, category vocabularies, and learned projections are fitted using training evidence only.

The current executable demonstration uses three blocks: training, selection validation, and test. Training labels are administratively censored at validation start; validation labels are censored at test start. Configurations are selected using mean validation integrated Brier score across the configured seeds. The selected configuration and a validation-derived horizon threshold are frozen in a selection artifact before final-test predictions. Models are not refitted on combined training and validation after selection. An independent calibration block remains a requirement for the larger real-data comparison, rather than an implemented feature of the smoke experiment.

Selection validation is abbreviated SVAL in historical artifacts; EVAL denotes an evaluation population. Names alone do not establish independence. If a tuner receives EVAL outcomes, selects a policy on them, or repeatedly changes its features in response to them, EVAL has become part of development. The correct audit traces the actual optimizer inputs and saved predictions, then reserves a later cohort for confirmation. The historical evidence chapter keeps this distinction visible when discussing the older policy results.

Search budgets should be comparable and reported. A single default Cox model against hundreds of neural trials is not a balanced computational comparison, even if the final test remains untouched. Equal wall time, equal trial counts, and carefully justified family-specific search spaces answer different questions. The public smoke run uses one prespecified candidate per family to exercise the pipeline; it does not claim an exhaustive ranking of the model classes.

## Survival curves, scoring rules, and policy metrics

A good ranking can coexist with badly calibrated probabilities. The primary comparison therefore evaluates whole survival curves over a common, supported time grid. For horizon $t$, a censoring-adjusted Brier score can be written

$$
\operatorname{BS}(t)=\frac1n\sum_{i=1}^{n}\left[\frac{\mathbf 1(Y_i\le t,\delta_i=1)\widehat S_i(t)^2}{\widehat G(Y_i)}+\frac{\mathbf 1(Y_i>t)(1-\widehat S_i(t))^2}{\widehat G(t)}\right],
$$

where $\widehat G$ is the training censoring-survival estimate under the library's event/censor tie convention. The implementation uses the right-continuous prediction at $Y_i$ rather than silently substituting a different left-limit convention. Integrating this score over the fixed horizon grid yields the reported integrated Brier score. Graf and colleagues (1999) motivate censoring-adjusted prediction-error assessment. [Graf et al., 1999](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2819990915/30%2918%3A17/18%3C2529%3A%3AAID-SIM274%3E3.0.CO%3B2-5).

Inverse-probability weights require adequate censoring support and an appropriate censoring assumption. A tiny estimated $G(t)$ can make the score unstable. Common weights fitted on training data support a fair numerical comparison, but do not cure covariate-dependent informative censoring. A serious empirical analysis reports the support of the weighting distribution and sensitivity to horizon truncation.

IPCW concordance provides a complementary ranking statistic, while thresholded precision, recall, and selected fraction describe a decision policy. Calibration curves and horizon-specific calibration errors assess whether predicted probabilities correspond to observed frequencies. Calibration is not guaranteed by a strong ranking score. [Van Calster et al., 2019](https://link.springer.com/article/10.1186/s12916-019-1466-7).

The archived network also contains dedicated horizon heads and policy combinations. Their scores must not be relabeled as $1-\widehat S(72)$ when the stored decision actually uses a distinct head or conjunction. Historical reports and the compact benchmark can both discuss 72-hour events while computing materially different quantities. Exact output semantics are part of the estimator definition.

## Uncertainty and testing the historical advantage under shared controls

Paired bootstrap resampling of final-test observations compares models on the same cases and preserves their correlation. The public demonstration uses the first prespecified fitted seed as the primary paired comparison and holds training censoring weights fixed during resampling. Its intervals describe uncertainty conditional on those fitted models and that dataset-generation process. They do not include retraining variability, repeated feature selection, or the cost of trying many unreported comparisons.

The real-data study should repeat fitting across prespecified seeds, report temporal blocks separately, and use entity-level resampling where repeated observations are retained within a permitted partition. Subgroup analyses should emphasize effect size and sample support rather than a large collection of nominal significance claims. An incremental value claim for visual evidence requires a matched ablation; an economic value claim requires a separately specified decision and cost model.

The conclusion rule is deliberately symmetric. Persistence of the historical neural advantage under the sealed protocol would strengthen the result through improved primary survival prediction, adequate calibration, stable temporal behavior, and acceptable operational cost. A tie or a Cox/tree advantage would support a simpler estimator for that setting. Either outcome would clarify how broadly the documented historical gain transfers across populations and evaluation criteria.
