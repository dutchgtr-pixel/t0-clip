# Statistical foundations for decisions in a changing market

## The observation is a decision, not merely a listing

A time-to-event dataset is often presented as a matrix of attributes, a duration, and an event indicator. In a changing marketplace, that representation is the end of a construction process rather than its starting point. A listing may be edited, enriched, observed repeatedly, temporarily unavailable, or offered again. The question asked of the model depends on when a decision becomes possible and on which version of those attributes is visible. Two records describing the same physical object at different times need not represent the same prediction problem.

The primary observation in this research is an eligible listing episode at a declared decision time $t_0$. Eligibility means that the item remains at risk of the event and that the specified information is available. Let $X(t_0)$ denote that information, $T$ the future duration to the event, and $C$ the remaining observation duration. The observed pair is

$$
Y=\min(T,C), \qquad \delta=\mathbf{1}\{T\le C\}.
$$

The survival function $S(t\mid X)=P(T>t\mid X)$ is the probability of remaining event-free beyond duration $t$. It supports several decisions without requiring separate definitions of elapsed time. A short-horizon event score is $1-S(72\mid X)$; a twenty-one-day persistence score is $S(504\mid X)$. These quantities describe a declared event, such as a reported sold transition. They are not automatically probabilities of an independently verified transaction, profitable purchase, or satisfactory item condition.

The same discipline applies to denominators. The retained operational snapshot contains 55,260 listing rows and 240,622 image assets. These establish a substantial data-processing scale, but they are not 295,882 independent survival observations. Several assets can belong to one listing, representations can be versioned, and many listings may lack mature outcomes at a particular cutoff. Statistical sample size is determined by the evaluated entities and their observed outcomes, not by the sum of database row counts.

## Censoring and the meaning of an unresolved outcome

Right censoring means that an event has not been observed before follow-up ends. A record followed for twelve hours cannot be labeled a negative for an event within seventy-two hours. It contributes evidence that the event time exceeds twelve hours. The survival likelihood preserves this information instead of either inventing a negative or discarding the record entirely.

For an exact event time with density $f$ and a right-censored observation, the usual contribution is

$$
L_i=f(Y_i\mid X_i)^{\delta_i}S(Y_i\mid X_i)^{1-\delta_i}.
$$

This factorization relies on the observation mechanism being ignorable under the assumed censoring model. If unavailable records are systematically those most likely to experience the event, a purely administrative interpretation is inadequate. Collection gaps, withdrawals, and observation policies can create censoring associated with attributes or outcomes. The research protocol therefore records the reason follow-up ended where possible and treats independent censoring as an assumption to investigate.

Periodic observations also create interval uncertainty. If a listing is live at one observation and marked sold at the next, an event interval $(L_i,U_i]$ may be more defensible than an exact timestamp. Its contribution is $S(L_i\mid X_i)-S(U_i\mid X_i)$. Right censoring is recovered by taking $U_i=\infty$. A source-provided timestamp can narrow the interval only to the extent that its semantics are understood. A modification timestamp is not transaction ground truth merely because it is precise to the second.

Historical zero-duration outcomes illustrate this issue. Equal origin and endpoint metadata can arise from coarse resolution, observation delay, or a source convention. Some such cases may already have completed before the system could score them. The appropriate response is to inspect provenance and define an admissible decision cohort. Removing every zero after seeing its effect on accuracy changes the population and cannot by itself establish a corrected benchmark.

## Proportional hazards and accelerated failure time

Cox regression provides a useful reference because its structure is explicit:

$$
\lambda(t\mid X)=\lambda_0(t)\exp(\beta^\top X), \qquad
S(t\mid X)=S_0(t)^{\exp(\beta^\top X)}.
$$

Covariates multiply a shared baseline hazard. Penalization can stabilize estimates when structured variables and fixed embeddings are numerous relative to events. The limitation is not that the model is “simple” in an informal sense, but that relative hazards remain constant over time under the model. Nonlinear transformations or a neural risk function can make covariate effects more flexible without removing that proportional-hazards structure. [Cox, 1972](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1972.tb00899.x); [Katzman et al., 2018](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1).

The earlier project instead emphasized accelerated failure-time modeling. In its location-scale form,

$$
\log T=f_\theta(X)+\sigma\varepsilon,
\qquad
S(t\mid X)=1-F_\varepsilon\!\left(\frac{\log t-f_\theta(X)}{\sigma}\right).
$$

The learned function shifts log duration, while the error distribution and scale determine the shape of the conditional distribution. Tree boosting makes $f_\theta$ nonlinear and allows interactions among price context, item attributes, and generated condition features. Its censoring-aware objective remains a survival objective; thresholding a point prediction at twenty-one days is a subsequent decision rule.

The error-family figure below is methodological, not a fit to an observed duration histogram. Different tail shapes can materially change $S(504)$ even where central predictions are similar. Choosing a distribution therefore deserves development-set comparison and tail calibration checks, rather than selection by visual plausibility alone.

![Standardized AFT error families](../../../research/thesis_evidence/figures/h01_aft_error_families.png)

Historical figure H01. Standardized distributions from the earlier tail-model chapter, PDF page 4. The curves illustrate assumptions; they are not empirical evidence for one distribution.

## Flexible survival functions and discrete hazards

Random survival forests model nonlinear structure through ensembles of survival trees and provide a strong alternative to neural methods. A discrete-hazard neural network parameterizes conditional event probabilities over time intervals. These are different ways of controlling flexibility; neither is intrinsically superior across datasets. [Ishwaran et al., 2008](https://ishwaran.org/papers/IKBL.AOAS.pdf); [Gensheimer and Narasimhan, 2019](https://peerj.com/articles/6257/).

For boundaries $0=b_0<\cdots<b_K$, define

$$
h_k(X)=P(b_{k-1}<T\le b_k\mid T>b_{k-1},X),
\qquad S(b_k\mid X)=\prod_{j=1}^{k}(1-h_j(X)).
$$

The product enforces survival monotonicity. It does not enforce calibration, correct endpoint construction, or temporal admissibility of $X$. The compact public implementation interpolates log survival within each bin. A seventy-two-hour query can therefore lie inside a bin: the 120-hour, 24-bin demonstration evaluates it between 70 and 75 hours. The preserved 504-hour, 128-bin architecture places it between 70.875 and 74.8125 hours. Reporting the horizon precisely requires the interpolation convention as well as the number of bins.

A survival curve and a dedicated binary head may disagree because they optimize different objectives. The historical cascade contains auxiliary classification scores as well as survival outputs. A historical policy combining classification heads cannot be evaluated as if it were simply $1-S(72)$. This distinction becomes especially important when discussing calibration or comparing the new compact implementation with earlier saved policy results.

## Sparse segments and hierarchical price anchors

A market can be large overall and sparse within the segment relevant to an individual decision. Conditioning jointly on product family, capacity, condition, locality, and recent date can leave few comparable observations. A raw segment mean may then be unstable, while a fully pooled market average may ignore economically important distinctions. The historical anchor system addresses this by expressing asking price relative to a supported reference and recording how much pooling was required.

For a normal-normal explanatory model with within-group variance $\sigma^2$ and prior variance $\tau^2$, the posterior mean has weight

$$
\widetilde\mu_g=w_g\bar y_g+(1-w_g)\mu_{p(g)},
\qquad w_g=\frac{n_g}{n_g+\sigma^2/\tau^2}.
$$

This equation explains why low-support groups borrow information. The implemented historical method uses robust medians or quantiles and a discrete support-based backoff hierarchy. It is inspired by hierarchical shrinkage but is not equivalent to fitting that conjugate model. Likewise, choosing the first sufficiently supported level is an algorithmic rule, not a posterior probability calculation unless an explicit probabilistic model supplies that interpretation.

![Support-dependent shrinkage](../../../research/thesis_evidence/figures/h02_support_shrinkage.png)

Historical figure H02. Illustrative shrinkage weight from the anchor chapter, PDF page 3. The support threshold shown is an explanatory setting, not an independently optimized universal constant.

A price-relative feature can be written $r_i=\log(p_i/a_i)$, where $p_i$ is the ask and $a_i$ an admissible historical anchor. Support counts, age of contributing observations, and the chosen fallback level accompany $r_i$. Without these reliability features, two identical ratios could conceal very different evidence bases. Outcome-derived anchors must also exclude the target record and any comparison whose outcome was unavailable at the decision.

## Time from the present and adaptation to change

A listing still live at age $a$ has supplied additional survival information. Under a model fitted at the initial origin, its residual event probability is

$$
P(T\le a+h\mid T>a,X_0)=1-\frac{S(a+h\mid X_0)}{S(a\mid X_0)}.
$$

The expression requires $S(a\mid X_0)>0$ and generally differs from $1-S(h\mid X_0)$. It also does not justify replacing initial covariates with later values in an initial-snapshot model. A landmark model trained at the actual later decision, or an appropriately specified time-varying model, can incorporate substantial new information. Fast-moving markets make this a practical requirement: a prediction delivered too late may describe an outcome window already partly or wholly elapsed.

Observed short-horizon prevalence changed from 56.34% in one historical selection cohort to 63.10% in the broader holdout and 77.74% in its recent nested slice. Those are selected-cohort rates, not estimates of market-wide turnover. They nevertheless demonstrate why calibration and policy evaluation must state their population. Refreshing features, retraining a model, recalibrating probabilities, and moving a threshold are different adaptations with different validation requirements.

Finally, observational duration models do not identify the effect of changing the asking price. Price, quality, urgency, and seller behavior may share unobserved causes. The research can establish prediction quality for a defined population before it can establish a beneficial intervention. This separation keeps statistical evidence useful without inflating it into a causal or economic claim.
