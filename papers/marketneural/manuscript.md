# MarketNeural: Temporal Data Contracts and Multimodal Survival Learning for Marketplace Decisions

**Research manuscript draft, empirical comparison pending**  
Version 0.1 — 21 September 2026  
Project: t0-clip / MarketNeural  
This is an independent research manuscript. It is not a submitted, examined, or awarded academic thesis.

## Abstract

Predicting how long a marketplace listing will remain available requires reasoning about incomplete outcomes, changing listing content, sparse comparable transactions, and the time at which information actually became usable. A complex neural architecture cannot repair a historical dataset that exposes tomorrow's information to yesterday's prediction. This manuscript develops a research framework in which temporal data contracts and operational provenance are part of the statistical method. It connects an existing marketplace data platform to a proposed controlled comparison of conventional and neural survival models.

The framework represents each prediction as a decision-time snapshot, distinguishes source event time from observation and feature-availability time, preserves censoring, and groups related listings across temporal splits. A multimodal survival architecture combines structured attributes, text representations, and image evidence through a latent attention bottleneck and an optional mixture of experts. Discrete hazards produce an internally consistent survival curve, from which short-horizon sale probabilities and long-horizon persistence probabilities can be derived. Penalized Cox regression, tree-based survival models, and simpler neural models provide necessary comparison points. The comparison is designed to separate additional information, architectural flexibility, and computation budget.

Existing project reports provide descriptive evidence of useful predictive signal and substantial operational engineering, but they do not establish neural superiority. An earlier Slow21 report describes 941 sold listings with precision 0.8314 and recall 0.8614; later short-horizon results concern different populations and cannot be placed on the same leaderboard. Historical leakage investigations also expose missingness shortcuts, dependent tail diagnostics, and uncertainty in endpoint provenance. These findings motivate a frozen, label-purged, decision-time evaluation protocol. The public synthetic benchmark exercises implementation and reporting; it is not evidence of marketplace effectiveness. The manuscript's principal present contribution is an explicit, falsifiable research design and an account of the engineering needed to make that design credible. The controlled real-data model comparison remains **not run**.

**Keywords:** survival analysis; multimodal learning; temporal leakage; point-in-time features; marketplace liquidity; reproducibility; censoring.

## 1. Introduction

### 1.1 The decision being modeled

A marketplace listing combines an asking price, a description, photographs, product attributes, seller-provided information, and a changing local supply of alternatives. A buyer deciding whether to investigate an offer, or an operator deciding where to direct limited attention, must act while the outcome is unknown. A prediction that a listing is likely to disappear quickly may help prioritize investigation. A prediction that it is likely to remain for weeks may inform a different workflow. Neither prediction, by itself, establishes a profitable purchase or a causal effect of changing the price.

The statistical target is therefore a distribution over a future event time under a specified observation process. It is not simply the eventual sale label available in a completed historical table. Some listings remain visible when collection ends. Others are withdrawn, expire, are relisted, or become temporarily inaccessible. The source may report a sold status without revealing an independently verified transaction time. A rigorous study must say which event is predicted and which cases remain unresolved.

The decision also has a clock. A prediction made when a listing first appears differs from one made after images have downloaded and enrichment has completed. A score generated for an older listing is not automatically a forecast from the present moment. If the training target is sale within 72 hours of a historical origin, the corresponding probability must retain that interpretation. This distinction becomes consequential when an asynchronous production pipeline scores records hours or days after their original timestamp.

### 1.2 Why the data system belongs in the method

The project underlying this manuscript contains data observation jobs, canonical tables, feature stores, temporal guards, model pipelines, operational checks, and recovery procedures. These components create the conditions under which a predictive experiment can be valid. They are not merely infrastructure around an otherwise self-contained learning algorithm. A database repair can change a training feature. An enrichment job can fill missing values selectively after sale. A current-state join can silently rewrite the past. An apparently harmless missingness indicator can then reveal an outcome more directly than any learned representation.

The central position of this manuscript is that a marketplace survival model must be evaluated together with its information boundary. This does not mean that a successful pipeline proves a successful model. It means that model performance is uninterpretable unless the experiment establishes what information the model was entitled to use. Reliability, temporal validity, predictive accuracy, and decision value are separate properties, each requiring its own evidence.

The existing work demonstrates considerable applied engineering: survival-aware labels, temporal aggregate rules, saved model configurations, train-only transformations, leakage postmortems, and operational controls. Its strongest lesson is not that complexity guarantees accuracy. It is that a serious system must be able to discover an attractive result is wrong, trace the failure, and preserve a revised account of the evidence. The public research extension makes this principle explicit.

### 1.3 Research questions and contributions

The primary research question is whether multimodal neural survival learning improves future decision-time predictions over well-tuned conventional survival models when the models receive comparable information. Secondary questions concern calibration, robustness to unavailable modalities, performance on sparse product segments, and the operational cost of generating a usable score. These questions admit negative results. If a tree model matches the neural model at lower cost, that is a useful outcome rather than a failure of the study.

The manuscript develops four contributions. First, it specifies a temporal data contract that separates event time, collection time, feature availability, and decision time. Second, it formulates a multimodal hazard model with explicit handling of missing modalities and survival-consistent outputs. Third, it specifies a controlled comparison that separates feature value from model-class value. Fourth, it documents how historical leakage and provenance findings change the interpretation of earlier results and the design of the next experiment.

These are contributions of synthesis, protocol, and implementation design at the present stage. Novel statistical superiority, a new state of the art, and commercial effectiveness are not established. Earlier project papers supply historical context. This manuscript narrows or qualifies some of their language: a contract can enforce declared checks without proving that every historical input was observable, and a successful operational run does not establish prospective prediction quality.

## 2. Related work and conceptual position

### 2.1 Survival models as a baseline family

Cox regression relates covariates to a baseline hazard through a multiplicative risk function. Its central proportional-hazards structure provides a useful, comparatively parsimonious starting point. A carefully regularized Cox model is particularly important when the number of independent outcomes is modest relative to feature dimension. Its inclusion makes the comparison more demanding than a contrast against an unregularized linear classifier. [Cox, 1972](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1972.tb00899.x).

Random survival forests offer a nonlinear alternative designed for censored outcomes. They can represent interactions and heterogeneous partitions without requiring the neural architecture proposed here. Consequently, they test whether the useful structure is mainly nonlinear tabular interaction rather than something requiring attention or representation learning. [Ishwaran et al., 2008](https://ishwaran.org/papers/IKBL.AOAS.pdf).

Accelerated failure-time models describe covariate effects on a time scale, often through a distribution for log duration. A boosted AFT model is especially relevant because earlier project experiments already used this family. Its historical success is a reason to include it as a strong comparator, not a reason to treat its earlier scores as directly comparable with new neural results. The loss distribution, censoring representation, feature set, and calibration procedure must all be specified in the new run.

### 2.2 Neural survival analysis

DeepSurv replaces a linear risk function with a neural function within a Cox-style framework. Nonlinearity in covariates does not, by itself, remove the proportional-hazards assumption. It therefore helps separate two questions: whether nonlinear representation is useful, and whether time-varying relative effects are necessary. [Katzman et al., 2018](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1).

Discrete-time neural survival models predict hazards over intervals and optimize a likelihood that accounts for events and censoring. This provides a natural route to minibatch training and horizon-specific probabilities. The present manuscript adopts that general modeling family, while treating bin selection and within-bin censoring conventions as explicit experimental choices. [Gensheimer and Narasimhan, 2019](https://peerj.com/articles/6257/).

DeepHit models event-time distributions and accommodates competing risks. Deep Survival Machines instead develops a fully parametric mixture approach with learned representations. Together these works show that neural survival analysis is not a single architecture or assumption set. They motivate comparisons among output parameterizations rather than an undifferentiated claim that a neural network is more expressive than survival analysis. [Lee et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11842); [Nagpal et al., 2021](https://arxiv.org/abs/2003.01176).

### 2.3 Multimodal representations

Perceiver uses cross-attention to compress high-dimensional inputs into a smaller latent representation, allowing subsequent computation in latent space. The relevant idea is a controlled information bottleneck for heterogeneous input tokens. It does not establish that attention improves marketplace survival predictions. That question remains empirical. [Jaegle et al., 2021](https://proceedings.mlr.press/v139/jaegle21a.html).

MoME studies mixtures of multimodal experts for cancer survival prediction, combining pathology and genomic information. Its relevance is the possibility that different cases require different degrees of reliance on each modality. The marketplace setting has different data, endpoint semantics, and missingness mechanisms. The proposed model is therefore inspired by a general fusion strategy, not presented as a reproduction of MoME or a transfer of its empirical conclusions. [Xiong et al., 2024](https://papers.miccai.org/miccai-2024/531-Paper2168.html).

### 2.4 Marketplace applications and the gap addressed here

Survival modeling of classified listings predates this project. Demiriz's used-car study explicitly treats delisting as a proxy sale event, illustrating both the usefulness of the framework and the importance of endpoint assumptions. Born and colleagues examine survival analysis for used-car price management. These are precedents for modeling market exposure duration; they do not validate any particular endpoint mapping in the present system. [Demiriz, 2018](https://acikerisim.gtu.edu.tr/items/5503898d-3684-44eb-b73c-f62a2e538ecd); [Born et al., 2018](https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers/discussion-papers-2017-1/irtg1792dp2018-065.pdf).

The gap addressed here is practical and methodological: how to compare multimodal models when labels mature asynchronously and historical data are assembled by a continuously changing pipeline. The manuscript does not claim to invent survival analysis, temporal databases, feature governance, or attention. It integrates these elements into a falsifiable research program with explicit boundaries between implemented software, historical measurements, and experiments still to be performed.

## 3. Data, time, and the T0 protocol

### 3.1 Observation unit and notation

Let an observation be a listing episode at a particular decision time. Write its anonymous episode key as \(i\), decision time as \(t_{0i}\), future event duration as \(T_i\), and censoring duration as \(C_i\). The observed outcome is

\[
Y_i=\min(T_i,C_i),\qquad \delta_i=\mathbf{1}(T_i\le C_i).
\]

The model estimates \(S_i(t)=P(T_i>t\mid X_i(t_{0i}))\), where \(X_i(t_{0i})\) contains only admissible information. The main proposed short-horizon target is \(F_i(72)=1-S_i(72)\), measured in hours from the declared decision time. A long-horizon persistence target is \(S_i(504)\). These are related summaries of the same event-time distribution, but separate binary models need not produce mutually consistent probabilities.

The primary real-data study will use one initial eligible decision per episode. A repeated-decision or landmark extension is a separate experiment because it changes the target population and dependence structure. Relisted items, repeated seller-item combinations, and near-duplicate content require grouping rules before splitting. Anonymous keys remain necessary inside a private evaluation environment for this purpose; they must not be included in public result tables.

### 3.2 Four clocks rather than one timestamp

Every source fact has at least a claimed event time and a time when the system observed it. Derived features also have a time when the required inputs were available and a time when computation completed. The decision time is a fourth boundary: the moment at which the proposed workflow could have acted. Confusing these clocks can produce a dataset that is temporally ordered but operationally impossible.

For a raw fact \(r\), record \(e(r)\), its source event time, and \(a(r)\), the earliest justified availability time. A feature is admissible only if the information needed to compute it was available by \(t_0\). A conservative replay therefore requires

\[
e(r)\le t_0,\qquad a(r)\le t_0
\]

for all contributing facts. An earlier event timestamp alone is insufficient. A sold comparable that occurred yesterday but was first learned tomorrow cannot enter today's historical price anchor. A backdated metadata field must not silently override the collection-time boundary.

Derived representations deserve a distinction. Recomputing a frozen image embedding later from an immutable photograph known to exist at the historical decision time can be a legitimate retrospective transformation. It does not require pretending that the GPU job ran in the past. However, a study must preserve the original bytes or an equivalent verifiable version, fix the encoder, and state that it evaluates the information content under a retrospective computation budget. Claims about real-time deployment additionally require measurements of feature readiness and latency. A current photograph substituted for an earlier unavailable one does not satisfy the same contract.

### 3.3 Endpoint provenance and censoring

The event definition must identify whether a record means a verified transaction, a source-reported sold status, disappearance, or another transition. These endpoints cannot be interchanged merely because each ends visibility. The primary protocol favors a documented sold-status transition, with the limits of its timestamp stated. Unknown disappearance remains a separate terminal category or censoring mechanism until a defensible mapping is available.

When the event is detected at periodic observations, its time may be interval-censored between the last live observation and the first sold observation. A source metadata timestamp may narrow that interval, but it is not automatically transaction ground truth. The first controlled experiment must select and freeze one handling rule: an interval-aware likelihood where supported, or a declared endpoint approximation accompanied by sensitivity analyses. Comparators must receive equivalent outcome information. Giving one model exact-looking timestamps and another coarse intervals would make the comparison difficult to interpret.

Zero durations require investigation rather than automatic deletion. They may arise from timestamp resolution, collection delay, shared metadata fields, or genuine rapid transitions. They may also identify a population that was effectively already sold when first available to the scoring system. The protocol retains an explicit zero-duration indicator for auditing outcomes, excludes it from predictors, and reports a predeclared sensitivity analysis. Such a sensitivity changes the population; it is not a corrected benchmark without additional provenance evidence.

Right censoring represents observation ending before the event is known. A listing followed for only 12 hours is not a negative example for sale within 72 hours. For a horizon \(h\), the binary label is observed for events by \(h\) and for listings known to remain event-free through \(h\). Other records require censoring-aware estimation or exclusion from a clearly labeled complete-outcome diagnostic. The full survival fit preserves them under its specified likelihood.

### 3.4 Feature blocks and historical aggregates

Structured inputs include product attributes, asking-price context, content counts, location at a permitted granularity, and historical market summaries. Text and image blocks contain representations of content available at the decision. Missingness is explicit because the absence of a photograph or attribute may itself matter. However, a model must not learn missingness created by future enrichment or post-event repair.

Historical anchors compare a listing with earlier comparable observations. For a group \(g\), a transparent shrinkage construction is

\[
\widetilde\mu_g = w_g\widehat\mu_g+(1-w_g)\mu_{\mathrm{parent}(g)},\qquad
w_g=\frac{n_g}{n_g+\lambda}.
\]

The support count and fallback level accompany the estimate. This is a practical shrinkage rule; describing it as a full Bayesian posterior requires an explicit likelihood and prior. Anchors must exclude the target episode, unavailable outcomes, and future observations. Any learned shrinkage constant, target encoding, or feature selection is fitted within the training partition. Cross-fitting is required where an outcome-derived encoding would otherwise include the same training observation's label.

Inventory reconstruction similarly needs information-time semantics. A current history of first and last visibility can describe past reality more completely than the system knew it at the time. For prospective simulation, the admissible inventory is the set inferable from observations available then. A reconstructed omniscient stock count may be useful for a separate scientific question, but it cannot be passed off as an operational T0 feature.

### 3.5 Chronological splits and label availability

The proposed design has four chronological partitions: fit, selection, calibration/policy, and final test. Their date boundaries and episode-group rules are registered before test labels are inspected. The training cutoff applies to both feature availability and outcome availability. A listing created before the cutoff but labeled by a sale after it cannot contribute that future event to a simulated model trained at the cutoff. It may contribute a censored training observation as of the cutoff if that is part of the frozen design.

A maturity gap alone is not a universal fix. It must match the label horizon, observation delay, and censoring policy. In a walk-forward study, each fold rebuilds outcomes and outcome-derived features as of its own fit time. A file exported once with all eventual outcomes cannot be reused uncritically across earlier folds.

Group disjointness is checked with episode keys, related-item groups, and content similarity diagnostics. The primary split keeps a related item in one partition. A seller-group sensitivity tests transfer to unseen sellers without making it the only deployment estimand. Final test data are accessed once for the preregistered comparison; later exploratory analyses are labeled as such and require a new temporal cohort for confirmation.

## 4. MarketNeural: proposed model family

### 4.1 Representation and fusion

MarketNeural denotes a research family, not an assertion that every architecture described here is already reproduced by the public demonstration. Its full configuration uses structured features, a text representation, and a variable number of image or image-region tokens. Each block is projected into a common width and receives modality and position metadata where appropriate. Validity masks distinguish unavailable tokens from padded storage.

For input token matrix \(U_i\in\mathbb{R}^{N_i\times d}\) and learned latent matrix \(Z\in\mathbb{R}^{L\times d}\), a cross-attention step has the familiar form

\[
Z_i' = Z + \operatorname{softmax}\left(\frac{(ZW_Q)(U_iW_K)^\top}{\sqrt{d_k}}+M_i\right)U_iW_V.
\]

The optional mask \(M_i\) can remove padded inputs. In the compact implementation and preserved architecture excerpt, an unavailable image/report slot instead retains a learned missing-token representation in attention; its unavailable content values are not exposed as evidence. Missing slots therefore need not be removed from the attention sequence. Subsequent latent processing can make computation less sensitive to the raw number of image tokens. This architectural choice is a hypothesis about useful information fusion. Its value must be measured against concatenation, pooled embeddings, and simpler tabular models using the same information.

The model must distinguish an absent modality from a zero vector that happens to be a valid representation. A modality-presence embedding and training-time modality dropout can make this distinction explicit. Dropout probabilities are selected on development data. Evaluation includes natural missingness and deliberately removed modalities, because robustness to random dropout does not establish robustness to the data observation failures seen in deployment.

### 4.2 Optional mixture of experts

An optional expert layer computes representations \(E_m(z_i)\), with nonnegative gate weights \(\pi_m(z_i)\) summing to one:

\[
r_i=\sum_{m=1}^{M}\pi_m(z_i)E_m(z_i).
\]

Experts might emphasize structured market context, text, image condition, or fused interactions. This is an implementation design choice; naming an expert does not prove that it learns that semantic role. Gate distributions, effective expert usage, and performance under modality removal must be inspected. A gate that consistently selects one expert may indicate that the extra complexity is unnecessary, while a gate dominated by collection artifacts may indicate shortcut learning.

The comparison includes a no-expert model with a matched representation width. Parameter counts, training time, inference cost, and tuning budget are reported. Otherwise a gain attributed to expert specialization could simply reflect greater capacity or more extensive search. Expert count and routing choices are locked before final testing.

The public compact `perceiver_moe` reference uses a more specific construction: experts emit separate hazard sequences and the gate mixes their survival distributions. Thus \(S_i(t)=\sum_m\pi_{im}S_{im}(t)\), and an observed outcome contributes \(\log\sum_m\pi_{im}L_{im}\). This is different from averaging hazards or mixing hidden representations before a single hazard head. The mixture weights are constant across a listing's prediction horizon in this reference. Both the compact implementation and the separately preserved numerical architecture excerpt must be identified by name when reporting results; they are not interchangeable model versions.

### 4.3 Discrete hazard head and likelihood

Let \(0=b_0<b_1<\dots<b_K\) be fixed time boundaries. The model returns interval hazards

\[
h_{ik}=P(b_{k-1}<T_i\le b_k\mid T_i>b_{k-1},X_i)
=\sigma(g_k(r_i)).
\]

At interval boundaries,

\[
S_i(b_k)=\prod_{j=1}^{k}(1-h_{ij}),\qquad F_i(b_k)=1-S_i(b_k).
\]

This construction guarantees a nonincreasing survival curve and nondecreasing cumulative event probability. It does not guarantee calibration. Bin boundaries are chosen using training information or a fixed grid. A requested output such as 72 hours need not coincide with a bin boundary; evaluation must use the documented within-bin interpolation. For example, the compact 120-hour/24-bin smoke model queries 72 hours inside (70, 75], while the preserved 504-hour/128-bin architecture queries it inside (70.875, 74.8125]. The handling of times beyond the last bin is stated rather than silently extrapolated.

For an event observed in bin \(k_i\), the contribution is the probability of surviving earlier bins and failing in that bin. For censoring after a fully observed bin \(m_i\), the contribution is survival through that bin. The negative log-likelihood is

\[
\mathcal L=-\sum_i\left[\delta_i\left(\sum_{j<k_i}\log(1-h_{ij})+\log h_{ik_i}\right)
 +(1-\delta_i)\sum_{j\le m_i}\log(1-h_{ij})\right].
\]

This expression intentionally discards partial-bin censoring information under a conservative discrete convention. A piecewise-constant hazard variant may use partial exposure instead, but that is a distinct implementation to document and test. Event-at-zero handling must also be specified. Numerical clipping stabilizes logarithms; it must not alter outcome times to manufacture valid labels.

The compact public implementation retains fractional-bin survival exposure for censoring and uses interval event probabilities for observed events. It interpolates log survival within a bin, giving \(S(b_{k-1}+u\Delta_k)=S(b_{k-1})(1-h_k)^u\) for \(0\le u\le1\). Events are therefore treated as bin membership, while censoring uses the available partial exposure. This convention is disclosed rather than described as an exact continuous-event likelihood. The table loader rejects zero or negative follow-up pending an upstream endpoint audit; that public input requirement is not a retrospective declaration that historical zero-duration outcomes were invalid.

### 4.4 Ensembles, calibration, and decision rules

An ensemble can average member survival probabilities with fixed nonnegative weights. A convex combination preserves survival monotonicity. Averaging logits or separately tuned horizon scores has different semantics and requires its own calibration assessment. Model seeds are not independent datasets; reporting many seeds cannot replace uncertainty over future listings.

Calibration is fitted on a dedicated partition after architecture selection. For horizon-specific probabilities, a monotone calibration map can improve reliability but does not automatically preserve coherence across independently calibrated horizons. The study therefore reports both raw survival curves and any calibrated decision probabilities, names the calibration method, and checks cross-horizon consistency. Calibration performance on the data used to fit the calibrator is not a generalization estimate.

A threshold is a policy parameter, not part of the meaning of a probability. It is chosen using a registered utility or an explicit precision/recall constraint on the policy partition. The final test reports the locked threshold even if another threshold looks better retrospectively. A useful threshold may change when workload capacity or event prevalence changes; such a policy update must be evaluated separately from a change to the underlying model.

### 4.5 Scoring after the original time origin

If a baseline model estimates survival from an earlier origin and a listing is still at risk at age \(a\), a residual-horizon calculation under that model is

\[
P(T\le a+h\mid T>a,X_0)=1-\frac{S(a+h\mid X_0)}{S(a\mid X_0)}.
\]

This is not the same as \(F(h\mid X_0)\). It also does not solve arbitrary covariate updates: replacing \(X_0\) with the latest content in a model trained on initial snapshots may introduce a new mismatch. A landmark model trained on live listings at decision time is the preferred extension when the actual workflow scores records at variable ages. The denominator must be well supported, and outcomes already realized before scoring are excluded from the live at-risk cohort.

## 5. The neural-versus-Cox-and-tree experiment

### 5.1 What would constitute a fair comparison?

A neural model given images, text, and a rich market history cannot establish architectural superiority over a Cox model given only price and age. The first comparison therefore uses the same structured feature contract for all eligible models. The second supplies the same fixed pooled text and image embeddings to each model. A third comparison tests token-level neural fusion against pooled representations, explicitly identifying this as a joint representation-and-architecture change.

The baseline ladder contains a training-cohort survival estimate, penalized Cox regression, a nonlinear tree survival model, a boosted AFT model, a simple discrete-hazard neural network, and the proposed multimodal architecture. Not every comparator must be in the minimal public demonstration. Any absent comparator is marked pending and cannot be used in a superiority claim. The complete real-data study records software versions, tie handling, feature transformations, hyperparameter ranges, and failed runs.

The public benchmark currently names six estimators: `km`, `coxph`, `rsf`, `gbsa`, `mlp`, and `perceiver_moe`. The `gbsa` comparator uses gradient-boosting survival analysis; it must not be relabeled as the historical boosted AFT model. Its conventional and MLP comparators receive the same available vector values, with masked slots flattened, while the attention model tokenizes those values. The controlled pooled-versus-tokenized ablations and the additional AFT comparator remain separate planned experiments.

Equal search counts are insufficient if one model requires vastly more computation. The study reports both trial counts and resource use, and uses a predefined feasible budget per family. Each family receives sensible regularization and early-stopping opportunities. All transformations that learn from data, including dimensionality reduction of shared embeddings, are fitted within the training partition and carried unchanged into later partitions.

The model assumptions provide useful explanatory contrasts. Cox writes \(\lambda(t\mid x)=\lambda_0(t)\exp(\beta^\top x)\), so its survival prediction is \(S_0(t)^{\exp(\beta^\top x)}\). Regularization can stabilize a high-dimensional shared representation, but proportional hazards remain a restriction. A boosted Cox-loss model can learn nonlinear risk while retaining that proportional structure. A survival forest instead aggregates tree-based survival estimates and can represent different time patterns across covariate regions. The neural discrete-hazard model directly parameterizes interval-specific conditional probabilities. These are distinct inductive biases, not an ordering from primitive to advanced. Limited data, noisy embeddings, and modest signal can favor the more constrained model even when the data-generating process is not exactly proportional-hazards.

### 5.2 Hypotheses and primary endpoint

The primary hypothesis is that the full neural model improves censoring-adjusted prediction error on a new temporal cohort compared with the strongest preregistered conventional comparator under a matched information contract. A lower integrated Brier score over a declared horizon range is the proposed primary statistical endpoint. The exact horizon grid, censoring estimator, and minimum follow-up requirements must be frozen before access to real final-test labels. A synthetic demo using an ordinary horizon Brier score does not fulfill this experiment.

Secondary hypotheses concern incremental value from text and images, whether latent attention improves on pooled fusion, whether experts improve on a single network, and whether any predictive gain survives latency and coverage constraints. A separate robustness hypothesis asks whether performance degrades acceptably when a modality is absent. The phrase “acceptably” is operationalized by a registered margin before the test, not chosen after inspecting the result.

The null outcome is meaningful. Confidence intervals that include no improvement, worse calibration, or a gain too small to justify extra cost argue against adopting the more complex model. A study designed only to find the best-looking neural score would not answer the stated question.

### 5.3 Metrics and uncertainty

For a horizon \(h\), a survival Brier score measures error between the predicted survival probability and event-free status. Under a declared independent-censoring model, an inverse-probability-weighted form is

\[
\widehat{BS}(h)=\frac{1}{n}\sum_i\left[
\frac{\mathbf 1(Y_i\le h,\delta_i=1)S_i(h)^2}{\widehat G(Y_i)}+
\frac{\mathbf 1(Y_i>h)(1-S_i(h))^2}{\widehat G(h)}\right],
\]

where \(\widehat G\) estimates the censoring survival distribution. This notation matches the compact implementation's reverse Kaplan–Meier estimator and the underlying library's right-continuous evaluation and tie convention at \(Y_i\). The estimator, fit population, and support rule are fixed in advance. If censoring depends on recorded covariates, a marginal estimator may be inadequate; conditional modeling and sensitivity analyses are then required. If support vanishes, the study shortens the evaluable horizon through a documented protocol amendment rather than reporting unstable tail estimates as precise.

The integrated score averages over a fixed time grid. Calibration plots, calibration-in-the-large, horizon discrimination, and locked-policy precision and recall provide complementary views. A concordance measure alone is insufficient because a useful ranking can still have poor absolute probabilities. Precision depends on prevalence and threshold; it cannot be transported unchanged to a different observed population.

Uncertainty uses paired resampling so each model is assessed on the same resampled cases. Resampling respects related-listing groups and temporal blocks where appropriate. The number of groups, not merely the row count, constrains effective sample size. Multiple seeds are summarized separately from sampling uncertainty. Secondary ablations are interpreted with multiplicity and exploratory status in view.

### 5.4 Decision and resource outcomes

Operational evaluation reports the fraction of eligible listings scored before the decision deadline, time from observation to feature readiness, model inference latency, memory use, and preprocessing cost. Accuracy among scored cases is accompanied by coverage because excluding difficult cases can improve apparent accuracy while reducing usefulness. A model that needs all photographs may lose the opportunity to score the fastest-moving listings.

A utility analysis must state the cost of investigation, false alarms, missed opportunities, and any capacity constraint. If actual transaction economics are unavailable, the study reports a transparent hypothetical utility curve rather than profit. A subsequent prospective workflow experiment would be needed to establish decision value, and a randomized intervention or other credible causal design would be needed to attribute changes in outcomes to the model.

## 6. Leakage investigations as methodological evidence

### 6.1 A missingness shortcut that was detected and addressed

A historical audit found a market-context feature block whose missingness pattern was strongly associated with the short-horizon label. In one saved validation cohort, every positive short-horizon example had a zero segment count, while corresponding ratio fields were absent. A similar pattern occurred in a second saved cohort. This was not a subtle gain from economic information: the availability pattern tracked the lifecycle of the enrichment process.

The inspected successor pipeline added explicit bans for the affected block and named post-event fields, guarded export boundaries, and restricted post-event enrichment writes. This is evidence of a detected-and-addressed failure in that lineage. It does not justify retaining earlier affected metrics as clean estimates, and it does not prove that every later feature is temporally valid. The public protocol includes missingness-only baselines and per-feature availability checks because an outcome column need not be present for leakage to occur.

### 6.2 Tail diagnostics and training overlap

Some historical long-tail diagnostics were constructed from the same earlier-period table used for fitting the main model. Independent inspection of an available rebuilt snapshot confirmed complete overlap for its two tail diagnostic subsets. The exact original locked fit matrix was not available for a complete row-for-row reconstruction; therefore the audit does not claim that a later alias is identical to the original artifact. The split construction itself nevertheless establishes the dependence problem.

Such a tail set can be useful as a regression or stress test. It can reveal that a model update suddenly gives high short-horizon scores to old persistent listings it previously handled. It cannot estimate performance on unseen tail cases when those cases also influence model fitting. Making its metric report-only during threshold tuning does not remove that training exposure. The new protocol reserves a genuinely disjoint mature-tail test and labels historical overlapping diagnostics as in-sample evidence.

### 6.3 Availability is stronger than an as-of filter

The historical system includes meaningful safeguards: sold-comparable windows exclude future and same-day observations under declared rules; transformations are fitted on training data; outcome-like columns are rejected at model boundaries; and model configuration accompanies checkpoints. These reduce specific failure modes. A temporal SQL predicate and a schema hash still cannot prove that mutable source content retained its historical version.

For example, an image enrichment table built today may legitimately describe an immutable old photograph, or it may describe an image replaced after the decision. The same join shape can support either case. The missing evidence is raw version identity and availability provenance, not necessarily a mathematical error in the embedding algorithm. Certification is therefore defined here as passing a named set of checks over a named dataset version. The manuscript does not use “leakage-proof” as an unconditional property of an entire platform.

### 6.4 Repeated exposure to evaluation results

The inspected model-selection code separates selection metrics from report-only holdout metrics and rejects holdout-named automated selection objectives. This is a real safeguard. However, a holdout reported during every trial can still influence later human choices. The audit did not establish that such adaptation occurred, and it would be inaccurate to present that risk as a proven violation. The next experiment avoids the ambiguity by hiding final-test outcomes until the configuration, calibrator, and policy are frozen.

The same principle applies to “forward” slices. A recent subset of a broader holdout can reveal temporal drift, but it is not another independent sample. The historical 283-row recent slice is contained in the 748-row broader short-horizon holdout. Their results must be presented as a parent cohort and a nested diagnostic, never added together to inflate validation size.

### 6.5 Label origin and score origin

An audit of historical endpoint construction found that duration was derived from source metadata timestamps observed at different lifecycle stages. In inspected rebuilt rows with zero duration, the stored endpoint and origin timestamps were exactly equal. This supports a provenance concern; it does not establish why the source emitted identical values or prove that all such rows are erroneous.

The serving path could also score an older, still-live record when its modalities became ready, while retaining a target defined from an earlier edited-time origin. Consequently, the historical short-horizon score should not automatically be described as the probability of sale within the next 72 hours from scoring. The revised research protocol makes actual decision time explicit and reports score-ready population exclusions. This changes the scientific question in a necessary way: it evaluates the prediction at the time someone could use it.

## 7. Operational architecture and scientific reproducibility

### 7.1 From data observation to a governed model input

The platform's operational design separates data observation, canonicalization, audit and repair, feature construction, training, and inference. Jobs operate under concurrency limits, bounded retries or timeouts, and explicit completion states. Database connection management, observable job outcomes, recovery procedures, and retained artifacts make sustained operation possible. These are substantive engineering achievements even where the research evaluation needs strengthening.

For reproducibility, the useful unit is an experiment manifest binding source snapshot identity, feature definitions, split rules, preprocessing state, model configuration, code revision, and predictions. A feature-schema hash detects one class of change; a dataset content digest detects another. Neither substitutes for a human-readable description of endpoint semantics. A manifest should also record exclusions and failures so that an apparently clean result does not conceal a large unscored population.

### 7.2 Repairs, change control, and fail-closed behavior

Repair processes need to preserve the distinction between improving current data and rewriting historical evidence. The proposed research lane writes a new immutable snapshot when a repair changes model inputs. It never overwrites a final-test export and then reports the old model's score against the new target as if the experiment were unchanged. A corrected analysis is a new result with an explanation of the difference.

A failed temporal contract prevents admission to the controlled experiment. Exploratory notebooks may inspect rejected rows, but those rows cannot silently re-enter the clean lane. The guards include outcome-column rejection, timestamp availability checks, group disjointness, preprocessing fit scope, schema and representation dimensions, and survival-output validity. Passing these checks is necessary, not sufficient: the raw evidence behind a timestamp can still be wrong.

### 7.3 What the public release reproduces

The public extension provides research documentation and an executable synthetic benchmark whose implementation scope is described in its own documentation. Synthetic records contain no real listing identities or contact details. A successful run demonstrates that the included models can train, produce outputs, and be compared under a defined software contract. It does not reproduce the historical private dataset, the complete earlier GPU training system, or the operational environment described by older papers.

Public reports should separate three statuses: implemented and exercised on synthetic data, implemented but not evaluated on the controlled real-data cohort, and proposed but not implemented. This distinction allows the repository to be useful without overstating completion. A negative or inconclusive comparison remains publishable if the protocol and artifacts are preserved.

The executable smoke lane uses train, validation, and test partitions, with validation used for configuration and threshold selection. The four-partition real-data protocol in Section 3 additionally separates calibration and policy fitting from architecture selection. The compact lane does not claim to implement that additional partition, raw image/text encoders, a fitted calibration layer, a prospective capture service, or a marketplace deployment. A separately attributed numerical excerpt under `research/production_reference/` preserves part of the earlier architecture and its source hashes, without releasing private data or a complete production controller.

## 8. Current evidence and results status

### 8.1 Historical Slow21 evidence

The historical [empirical-results chapter](../part-3-modeling/ch05-empirical-results-sacrifice.pdf) reports a Slow21 classifier on a last-seven-days sold cohort of 941 listings. The positive class is duration of at least 504 hours. Reported precision is 0.8314, recall 0.8614, and F1 0.8462, with positive prevalence 17.64%. These figures are evidence that the historical feature/model combination separated a substantial part of that defined cohort. They are not newly reproduced controlled-study results.

The same source reports a 0.0059 false-positive fraction among listings sold in under ten days and a 0.2688 fraction among those sold between ten and twenty-one days. These quantities describe the cost of a slow-listing gate within particular duration strata. They are not estimates of lost profit or general marketplace harm. The latter would require a decision model and outcomes beyond observed duration.

The report also distinguishes a calibration cohort with much higher recall and a much larger mid-duration false-positive fraction. Because calibration was fitted there, those values are in-sample. The text's use of an “evaluation objective” during optimization leaves uncertainty about how fully untouched the final evaluation was. This manuscript therefore treats the table as descriptive historical evidence and does not label it a pristine external test. A sold-only, sold-date-defined cohort additionally conditions on eventual sale and differs from the population of all listings available at a decision time.

### 8.2 Later short-horizon aggregate evidence

The later audit examined saved predictions from a locked short-horizon policy. Public reporting is limited to aggregates; underlying listing records and private artifacts are not included. The broader holdout contained 748 records and produced approximately 76.69% precision and 64.83% recall. A nested recent slice contained 283 records and produced approximately 85.08% precision and 70.00% recall. These are conditional historical observations, not a public reproducibility claim.

| Evidence | Population | Reported precision | Reported recall | Interpretation |
|---|---:|---:|---:|---|
| Legacy Slow21 | 941 sold records | 83.14% | 86.14% | Earlier long-duration target; descriptive report |
| Later Fast72 policy | 748 holdout records | 76.69% | 64.83% | Historical short-duration target; audited aggregate |
| Recent Fast72 slice | 283 of those 748 | 85.08% | 70.00% | Nested temporal diagnostic, not independent validation |
| New controlled comparison | Not run | — | — | No model-ranking conclusion available |

These rows intentionally do not rank models. Their labels, prevalences, information contracts, and cohort construction differ. Comparing 86.14% recall for slow outcomes with 70.00% recall for fast outcomes would not measure progress or regression.

The historical Fast72 policy also used dedicated classification-head scores and a locked combination rule, which must not be conflated with the new compact benchmark's curve-derived \(1-S(72)\). In the preserved numerical adapter, the dedicated scalar logit represents slow/persistence risk, so its complementary fast score is \(1-\sigma(z)\); the survival-curve fast probability is a distinct output. Reported historical policy metrics therefore do not validate calibration of the new compact survival curve.

### 8.3 Endpoint sensitivity

The 748-row short-horizon holdout contained 178 zero-duration records, all labeled positive under its endpoint definition. Removing those records solely as a diagnostic leaves 570 records and reduces precision to approximately 66.30%, with recall approximately 62.24%. The corresponding recent-slice precision changes from approximately 85.08% to 78.91%. These differences show that endpoint composition matters materially.

The analysis does not identify the zero-duration records as universally invalid and does not replace the original score with a “corrected” one. Exclusion changes prevalence and the evaluated population. Its value is diagnostic: the next experiment must explain why these timestamps coincide and whether a usable prediction preceded the event. The original historical predictions remain the original evidence, with this sensitivity attached.

### 8.4 Operational evidence and synthetic evidence

Earlier infrastructure documents report extensive sustained orchestration. Operational records, recovery checks, and input/output parity checks are relevant evidence of engineering maturity. This manuscript does not convert a reported run total into a claim of zero failures across all services or into a statistical confidence bound on future uptime. Counts require a defined service set, observation window, and treatment of retries, skipped tasks, and recovery. Nor does successful orchestration validate a model's labels.

The synthetic benchmark is a separate evidence class. Its known data-generating process makes it useful for testing censoring handling, chronological partitioning, feature contracts, and reproducibility. A neural model may perform well on a nonlinear synthetic generator because the generator favors its inductive bias. A Cox model may perform well when proportional hazards are built into the generator. Neither establishes superiority on a marketplace. Synthetic run outputs are reported with their generator and seed, and are never merged with historical real-data metrics.

### 8.5 Executed synthetic smoke comparisons

The first public smoke run generated 1,200 artificial records with generator seed 2026 and used model seed 42. Its temporal split contained 593 training, 267 validation, and 340 test records; 258 test events were observed. Integrated Brier score was evaluated at ten points from 12 to 120 hours, using the common training censoring estimate. The six estimators completed under the small configuration recorded in the [public aggregate summary](../../research/examples/synthetic_smoke/summary.json).

| Compact estimator | Nonlinear fixture test IBS ↓ | PH fixture test IBS ↓ |
|---|---:|---:|
| Cohort Kaplan–Meier | 0.242390 | 0.228664 |
| Penalized Cox | **0.180544** | **0.177519** |
| Random survival forest | 0.202620 | 0.194749 |
| Gradient-boosting survival | 0.204181 | 0.189392 |
| Hazard MLP | 0.193811 | 0.190466 |
| Compact Perceiver-MoE | 0.193862 | 0.183155 |

Cox had the best observed score in this fixture, including against the nonlinear neural models. The Perceiver-MoE minus Cox IBS difference was 0.013318, with a conditional paired-bootstrap 95% interval of approximately [0.001042, 0.024410] using 500 resamples. Positive values mean worse error than Cox. This interval conditions on fitted models and training censor weights; it does not include model-training uncertainty or multiplicity correction. It is reported to demonstrate the comparison machinery, not to generalize from an artificial generator to a marketplace.

The fixture contains deliberately nonlinear and nonproportional-hazard signal, yet a constrained model can still win at this sample size and budget. There was one configured candidate per family and one model seed; validation still governed neural stopping and operating thresholds. This is not a comprehensive hyperparameter search. The result is retained without changing the hypothesis to favor its winner. It supports execution and provenance claims, while the controlled real-data comparison remains pending.

A second [PH-generator smoke run](../../research/examples/synthetic_ph_smoke/summary.json) used the same fixed model configurations and split sizes, with 263 observed test events. Cox again had the lowest observed IBS. The Perceiver-MoE minus Cox difference was approximately 0.005636, with a conditional 95% paired interval of [-0.003050, 0.014498], which includes zero. Neither generator was used to tune a new model configuration after observing the first test. The [comparison figure](../../research/examples/synthetic_comparison.svg) visualizes these two fixtures; it is a software demonstration of differing inductive biases, not an empirical marketplace leaderboard.

## 9. Discussion

### 9.1 Where complexity may earn its cost

Multimodal learning has a plausible role when visible condition, linguistic descriptions, and structured market context contain complementary information. An image may reveal wear absent from a short description; text may clarify accessories invisible in photographs; historical context may distinguish a fair price from an attractive-looking but ordinary offer. A fusion model may represent interactions among these signals. The plausibility of that mechanism motivates an experiment but does not supply its result.

There are equally plausible reasons a simpler model could win. Structured anchors may already summarize most useful variation. Embeddings may introduce noise, missingness, or unavailable information. Limited independent event counts may not support the neural model's capacity. Marketplace drift can reduce the value of finely fitted interactions. Latency may erase a predictive gain by delaying scores until attractive listings have already disappeared. The proposed experiment measures these possibilities rather than treating them as objections to be dismissed.

The relevant adoption decision is therefore conditional. A neural system is attractive only if a credible improvement survives temporal validation, calibration checks, coverage constraints, and resource accounting. If it improves a secondary metric while weakening the primary endpoint or withholding scores from many listings, the result needs a narrower interpretation. Operational convenience or maintainability may legitimately favor a simpler model even when statistical performance is close.

### 9.2 Governance as an engineering contribution

The feature-governance layer is valuable because it turns temporal assumptions into inspectable contracts. Registry entries, schema identities, named invariant checks, and hard failures make it harder to consume a broken feature silently. Their contribution is strongest when paired with immutable evidence and a clear account of what each check cannot establish.

Overstating certification weakens this contribution. A statement that a view passed specified temporal and schema checks is testable. A statement that an entire evolving platform is permanently leakage-proof is not supported by the same evidence. The historical discovery of an enrichment-related shortcut demonstrates why the narrower statement is both more accurate and more useful. Correcting the data contract is a research result when it changes what can validly be claimed from a model.

### 9.3 From prediction to decisions

Time-to-event prediction describes an observational distribution. It does not establish what would happen if a seller changed the price, if an operator bought an item, or if a platform altered exposure. Price and seller behavior are selected within the market, and unmeasured quality may affect both price and duration. A model can rank observational risk without identifying a causal pricing policy.

Similarly, a high-probability fast-sale listing is not necessarily a good purchase. It may have low margin, hidden defects, high logistics cost, or an event label unrelated to a completed transaction. A decision system needs a separate utility model and prospective outcome collection. The survival model can supply one input while preserving its limited claim: an estimated event-time distribution under a specified observation process and population.

## 10. Limitations and research integrity

The controlled real-data comparison has not been run. No present result establishes that the proposed multimodal architecture beats Cox regression, random survival forests, boosted AFT, or a simpler neural network. The manuscript's architectural detail is a specification and research rationale, not evidence of completed empirical validation. The minimal public benchmark may implement only a subset of this design; its model card is authoritative about that scope.

Historical records were created by a system that evolved. Some original locked artifacts were unavailable during the retrospective audit, and later aliases cannot substitute for them without qualification. Current enriched tables do not universally establish the exact version of every input available at a historical decision time. Source-reported sold timestamps are not independently verified transaction timestamps. These limitations constrain interpretation even where saved prediction metrics are arithmetically reproducible.

The source population may differ from other marketplaces, product categories, seasons, and operating policies. Generalization across those conditions requires new cohorts. Censoring may be informative, relisting may be incompletely linked, and sparse subgroups may have imprecise estimates. Aggregate calibration can conceal subgroup errors. A public synthetic dataset avoids disclosure of private records but cannot reproduce all these dependencies or provide external validity.

The proposed temporal protocol reduces opportunities for leakage but is not immune to incorrect source timestamps, undocumented upstream changes, or human adaptation after seeing results. Research integrity therefore requires preserving unsuccessful experiments, registering amendments, distinguishing exploratory from confirmatory analyses, and reporting coverage and exclusions. A scientifically useful outcome may be that the available data do not support the proposed comparison until endpoint and availability provenance improve.

## 11. Conclusion

Marketplace survival modeling is a joint problem of data semantics, temporal validity, representation learning, and operational delivery. The project provides a substantial engineering foundation and historical evidence of predictive signal. Its leakage investigations also show why operational success and attractive offline scores must be assessed separately.

MarketNeural specifies how to build on that foundation: define the actual decision time, preserve censoring and endpoint provenance, isolate related observations across temporal splits, fit transformations only on training data, and compare model families under matched information and resource budgets. A latent multimodal survival model is a plausible candidate within that design. Its superiority remains an empirical question.

The public release is intended to make that question answerable. It offers a research protocol, explicit claim boundaries, model and data documentation, and a synthetic execution path. The next decisive result is a frozen, disjoint, label-purged temporal comparison followed by prospective prediction capture. Until those experiments are complete, the appropriate claim is a reproducible research program with audited historical context, not a proven neural advantage.

## References

1. Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–202. [Publisher and DOI](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1972.tb00899.x).
2. Ishwaran, H., Kogalur, U. B., Blackstone, E. H., and Lauer, M. S. (2008). Random survival forests. *Annals of Applied Statistics*, 2(3), 841–860. [Author-hosted paper](https://ishwaran.org/papers/IKBL.AOAS.pdf).
3. Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., and Kluger, Y. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18, 24. [Publisher](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1).
4. Lee, C., Zame, W. R., Yoon, J., and van der Schaar, M. (2018). DeepHit: A Deep Learning Approach to Survival Analysis With Competing Risks. *Proceedings of AAAI*, 32(1). [Publisher](https://ojs.aaai.org/index.php/AAAI/article/view/11842).
5. Gensheimer, M. F., and Narasimhan, B. (2019). A scalable discrete-time survival model for neural networks. *PeerJ*, 7, e6257. [Publisher](https://peerj.com/articles/6257/).
6. Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., and Carreira, J. (2021). Perceiver: General Perception with Iterative Attention. *Proceedings of ICML*, PMLR 139, 4651–4664. [Proceedings](https://proceedings.mlr.press/v139/jaegle21a.html).
7. Nagpal, C., Li, X., and Dubrawski, A. (2021). Deep Survival Machines: Fully Parametric Survival Regression and Representation Learning for Censored Data With Competing Risks. *IEEE Journal of Biomedical and Health Informatics*. [Author preprint](https://arxiv.org/abs/2003.01176); [institutional record](https://publications.ri.cmu.edu/deep-survival-machines-fully-parametric-survival-regression-and-representation-learning-for-censored-data-with-competing-risks-2).
8. Xiong, C., Chen, H., Zheng, H., Wei, D., Zheng, Y., Sung, J. J. Y., and King, I. (2024). MoME: Mixture of Multimodal Experts for Cancer Survival Prediction. *MICCAI 2024*, LNCS 15004, 318–328. [Proceedings](https://papers.miccai.org/miccai-2024/531-Paper2168.html).
9. Demiriz, A. (2018). Used Car Pricing and Beyond: A Survival Analysis Framework. *First IEEE International Conference on Artificial Intelligence for Industries*. DOI: 10.1109/AI4I.2018.00023. [Institutional record](https://acikerisim.gtu.edu.tr/items/5503898d-3684-44eb-b73c-f62a2e538ecd).
10. Born, A., Kovachka, N., Lessmann, S., and Seow, H.-V. (2018). Price Management in the Used-Car Market: An Evaluation of Survival Analysis. *IRTG 1792 Discussion Paper 2018-065*. [Institutional paper](https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers/discussion-papers-2017-1/irtg1792dp2018-065.pdf).

Project evidence: the historical reports in [the papers directory](../README.md) are historical project reports, not external peer review. The [claim ledger](../../docs/research/CLAIM_LEDGER.md) separates their statements from inspected aggregates and from experiments pending under the [research protocol](../../docs/research/PROTOCOL.md).
