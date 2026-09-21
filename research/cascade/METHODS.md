# The three-stage survival cascade: implementation, training and interpretation

## Scope and evidence

The historical cascade is an implemented sequence of three multimodal survival models and their decision policies. Stage0 identifies long-persistence cases around a 504-hour boundary. Stage1 separates faster and slower cases around 168 hours among records passed downstream. Stage2 makes the final 72-hour selection. Each stage contains a neural event-time model; the cascade is not simply three comparisons applied to one shared prediction vector. Separate heads, checkpoint combinations, calibration choices and thresholds create separate score channels with different training and selection populations.

The public package preserves numerical implementations and provides a complete portable path through feature preparation, stage fitting, validation, probability combination, policy selection, sequential prediction and local model-bundle persistence. It does not distribute fitted private weights, category vocabularies, row-level observations or source-specific ingestion and infrastructure adapters. A new training controller is explicitly distinguished from the preserved numerical code. Reproducing an earlier run also requires its exact input snapshot, fitted transformations, trial settings and random state; executable architecture alone does not reconstruct those artifacts.

The evidence has three levels. First, `provenance.json` records full source-file hashes, original definition ranges, source-span hashes, abstract-syntax-tree hashes and canonical export hashes. Its selected definitions include the actual legacy network, survival functions, three stages' numerical policies, calibration functions and routing. Second, `historical_evidence.json` contains safely selected numerical configuration from inspected historical manifests, with the manifest hashes retained. Third, artificial execution tests verify the new portable adapters. These levels establish different things: source identity, historical configuration and current software behavior respectively. None alone establishes prospective predictive accuracy.

The recovered Stage1 trainer has exactly the source hash recorded in its historical PhaseA manifest. The full base-module hash referenced by that manifest differs from the recovered base available for export. This difference remains visible rather than being hidden behind a claim of complete original-run reconstruction. The nineteen exported Stage2 training functions were also compared against an earlier recovered Stage2 trainer: their syntax-tree hashes match, despite differences elsewhere in the later file. This corroborates continuity of those numerical routines across the legacy and later slot-aware implementation.

## Architecture of the legacy stages

The legacy backbone accepts numerical features, categorical integer identifiers, one 768-dimensional listing-text vector and one 512-dimensional image vector. Upstream embedding encoders are outside this network. Their pretrained parameters are not counted as trainable survival-model weights, and their execution is not joint end-to-end training with the survival loss. The model assumes that the supplied representations and tabular values obey a fixed feature contract.

Numerical feature tokenization learns a separate affine representation for each scalar. Categorical values index a distinct embedding table for each categorical column. The feature-token variant retains the individual tokens. The compact variant first constructs those representations and then uses learned queries to pool numerical and categorical groups into a smaller fixed number of tokens. This is learned attention pooling, not a simple average of every input column. It reduces the token count entering the main fusion network while allowing different feature groups to contribute differently.

Text and image vectors are projected into multiple tokens. Eight text tokens in a selected configuration are eight learned projections of one listing-level embedding; they are not eight independently encoded paragraphs. Similarly, the legacy image path projects one supplied image representation. The later slot-based architecture instead accepts multiple individual image and report vectors, which changes both its information representation and missingness contract.

The fusion network initializes a learned latent array. Each block cross-attends from the latent queries to the input tokens, then performs self-attention and feed-forward processing over the latent sequence. Layer normalization and residual connections stabilize the repeated transformations. Pooling the final latents produces the shared representation supplied to the hazard experts and an independent scalar classification head. This pattern concentrates multimodal interaction in a fixed latent space; it does not demonstrate by itself that the interaction improves accuracy.

An inspected older Stage0 serving-bundle member and an inspected Stage1 PhaseA manifest both specify width 256, 32 latent tokens, nine fusion blocks, eight attention heads, seven dense hazard experts and 128 linear time bins over 504 hours. They use eight numerical-summary tokens, eight categorical-summary tokens, eight text tokens and eight image tokens: 32 input tokens in total. These are recovered configurations, not source defaults and not a claim that every trial shared those values. The metadata of the six selected Stage0 bundle members is published separately so that differences can be inspected rather than averaged away.

The expert layer is dense: every expert network runs for every scored row. Its softmax gate allocates nonnegative mixture weights that sum to one. It is not sparse expert dispatch and does not provide a corresponding sparse-computation saving. The scalar head shares the fused representation but has its own final linear map. Training this scalar head requires a loss connected to that output; a survival-curve loss alone does not train its separate final parameters.

The legacy forward method accepts optional text and image masks but does not use them to suppress attention. Missing vectors therefore require a fixed upstream representation. This behavior is preserved. It must not be described as the later slot model's explicit missing-token mechanism. The later model zeros unavailable slot values and adds learned missingness, position and modality information while retaining the resulting tokens in attention.

## Event-time distribution and censor-aware likelihood

Let the time boundaries be $0=b_0<b_1<\cdots<b_K$. For expert $m$, the network emits an interval hazard $h_{imk}$ for row $i$. Its meaning is the conditional probability of an event in the interval $(b_{k-1},b_k]$, given survival to its left boundary. Expert survival at a boundary is

$$
S_{im}(b_k)=\prod_{j=1}^{k}(1-h_{imj}).
$$

The gate weights $\pi_{im}$ mix complete expert distributions:

$$
S_i(t)=\sum_m\pi_{im}S_{im}(t),\qquad \sum_m\pi_{im}=1.
$$

The implementation does not average expert hazard logits and then treat that result as the mixture distribution. For an observed event in interval $k$, an expert contributes $S_{im}(b_{k-1})h_{imk}$. The sample likelihood mixes these event probabilities across experts. Its logarithm is evaluated with log-sum-exp after clipping probabilities for numerical stability. A nonincreasing survival curve follows from this construction. Calibration does not follow automatically.

For a censored observation at $t=b_{k-1}+u(b_k-b_{k-1})$, the code retains partial-bin exposure:

$$
S_{im}(t)=S_{im}(b_{k-1})(1-h_{imk})^u,\qquad 0\leq u\leq1.
$$

This is linear interpolation in log survival, corresponding to a constant event rate inside an interval. Observed events still contribute interval probability rather than continuous-time event density. The combined convention should therefore be described as a discrete-event objective with fractional censor exposure, not as an exact likelihood for precisely observed continuous event times. Analytical tests cover interval events, partial exposure, expert mixtures and censoring at finite support.

Events beyond the modeled support become censored at the support boundary before the loss is called. The raw archived likelihood helper assumes that its caller has already performed this outcome transformation; it does not automatically reinterpret every late event. The portable adapter performs that transformation explicitly. Its prediction interface rejects extrapolation beyond the fitted horizon. Under 504 hours divided into 128 linear bins, the interval width is 3.9375 hours, so 72 hours is evaluated by interpolation rather than being an exact bin boundary.

Point summaries have their own limitations. The archived expected-time helper allocates remaining probability mass to the finite horizon and uses within-bin approximations. It is decorated with `no_grad`, so it cannot supply a differentiable expected-time training penalty. The portable objective deliberately does not advertise that optional historical MAE term as an active learning component. Preserving a helper for reporting does not establish that its corresponding optional training argument had the intended gradient effect.

## Stage targets and objective terms

All stages share survival machinery, but their binary targets and policy directions differ. The scalar head is slow-positive at the stage's operational horizon. Stage0 uses 504 hours, Stage1 uses 168 hours and Stage2 uses 72 hours. An observed event exactly at the horizon belongs to the fast class under the archived code. An observation censored exactly at the horizon is treated as a known slow example under its stated boundary convention. Earlier censored observations have no binary label at that horizon and are masked out of that binary loss.

For a generic horizon $H$, the known-label mask is an observed event or follow-up through $H$. The slow-positive target is an event after $H$, or censoring with follow-up through $H$. The fast-positive target is an event by $H$. Unknown short-censored rows can still contribute to survival NLL even though they contribute no horizon-classification loss. This avoids the error of treating every unresolved listing as a negative example at all future horizons.

Stage0's archived training loop combines weighted survival NLL, optional multi-horizon curve BCE and an optional dedicated tail-head BCE. The global curve term can supervise probabilities at 24, 72, 168 and 504 hours. Stage1 explicitly supervises the survival probability at 168 hours and its separate scalar head; optional curve terms at 24 and 72 hours reinforce shorter-horizon ordering. Stage2 moves the principal scalar and curve boundary to 72 hours and includes optional auxiliary survival supervision at 168 and 240 hours as well as short-horizon curve targets.

The Stage1 and Stage2 binary functions implement sample weighting, positive-class weighting and optional focal factors. A focal factor multiplies a row's loss by a power of one minus the probability assigned to its correct class. This changes emphasis toward difficult examples; it is not evidence that their labels are correct or that probability calibration improves. Optional consistency regularization penalizes disagreement between the slow curve and scalar head. Optional pairwise ranking draws known fast/slow pairs and applies a softplus loss to their score difference, with configurable margin and pair weights.

The inspected historical Stage1 manifest records NLL, slow-curve BCE and slow-head BCE weights of one, with MAE, consistency and ranking weights zero. Those values are evidence for that saved run, not a universal recipe. The later direct Stage2 checkpoint does not preserve every main-trainer objective coefficient, so base-configuration defaults cannot recover its exact trial objective. Duration-based contamination penalties used during threshold or model selection are also distinct from differentiable training losses; a large selection penalty does not mean the same penalty was backpropagated through the network.

A concrete retained Stage0 behavior deserves attention. Inspected members of the older serving bundle record zero dedicated tail-head BCE weight while their selected score channel is the average of the curve and scalar head. In the inspected training implementation, that scalar's separate final parameters receive their training signal only when its dedicated term is enabled. One must therefore not assume that every combined-score component in every historical member was independently supervised. The export preserves the behavior, the metadata reports it, and the new adapter requires its objective coefficients to remain explicit. Its demonstration defaults train both the head and curve; they are not presented as recovered historical settings.

## Training populations and cascade routing

Historical input preparation differs between training and later evaluation. Stage1 training is drawn from the earlier training table with a duration cap of 504 hours; an optional setting additionally requires an observed event. Its validation and holdout tables instead contain records passed by the Stage0 ensemble threshold. The Stage1-to-Stage2 preparer copies the Stage1 training population, again with an optional 504-hour safety cap. It does not generally restrict Stage2 training to true durations of at most 168 hours. Stage2 validation and holdout are filtered by the predecessor's predicted 168-hour fast gate.

This design deliberately exposes downstream models to difficult longer-duration negatives, including some 168–504-hour outcomes. It also creates a distinction between their fitting population and their routed evaluation population. Filtering a training table by an eventually observed duration is a modeling-population choice, not proof that those examples were knowable at an earlier simulated training date. A new chronological experiment must censor labels at each fit cutoff and independently justify its population definition. The portable controller makes the historical label cap explicit and leaves temporal cohort construction to the caller's governed data protocol.

The inference policy is a strict ordered cascade:

1. If the Stage0 tail score reaches its threshold, assign `TAIL_21PLUS`.
2. Otherwise, if the Stage1 fast-168 score is below its threshold, assign `SLOW_168PLUS`.
3. Otherwise, if the Stage2 fast-72 score reaches its threshold, assign `FAST_72H`.
4. Otherwise, assign `MID_72_168H`.

Score-threshold comparisons are inclusive on the positive side. The code preserves this exact order, so a Stage0 rejection takes precedence over apparently favorable later scores. The portable implementation only evaluates later models for rows that reach them. An unavailable required score causes an error rather than being replaced by another horizon's output. The old operational code offered fallbacks such as deriving a tail score from the 168-hour fast probability; that substitution does not have the same event-time meaning and is not enabled silently in the portable path.

The bucket names describe policy actions, not observed ground-truth intervals. A record sent to `SLOW_168PLUS` can in reality sell quickly; that error is one of the quantities a rejection policy must measure. Nor can the three independent score channels be multiplied to obtain calibrated bucket probabilities without an additional conditional model. The downstream training populations, hard predecessor gates, independently calibrated outputs and potentially inconsistent horizons prevent that shortcut. The package returns gate scores and route decisions separately.

Stage1 and Stage2 expose curve-derived slow probability, scalar-head slow probability and their arithmetic average. Their fast scores are the complement of the selected slow channel. Stage0 exposes the corresponding slow or tail score directly. Consequently, a historical Fast72 policy based on the scalar head or a calibrated ensemble is not automatically identical to $1-S(72)$ from one model's survival curve. Comparing these channels requires naming the selected channel and its calibration, not merely printing the same horizon beside both numbers.

## Recency, tuning and ensemble policies

The original training implementation explicitly weights recent observations. Its age-only factor is

$$
w_i^{\mathrm{age}}=2^{-a_i/\tau},
$$

where $a_i$ is age in days relative to the declared reference date and $\tau$ is a half-life. Weights are normalized to mean one. Inspected Stage0 serving members use a 30-day half-life; the inspected Stage1 configuration uses 23 days. Before other weighting, a row aged 60 days consequently receives one quarter of a fresh row's weight at a 30-day half-life, or approximately 0.164 at a 23-day half-life. This is a continuous decay mechanism, not an exact two-month exclusion boundary.

Additional weights can emphasize observations near a decision boundary using a Gaussian function of duration, persistent cases beyond selected cutoffs and very fast cases protected against false rejection. These operations make the optimized empirical distribution differ from unweighted population frequency. They may help an operational objective, but calibration and unweighted performance still require independent measurement. A reference date or decay hyperparameter chosen after examining final-test outcomes would compromise that measurement.

The preserved Stage0 three-phase search functions expose a real staged design. PhaseA searches optimization settings, recency half-life, curve-loss balance and boundary focus, with an optional censoring floor. PhaseB searches tail emphasis, fast-case protection and the dedicated tail-head coefficient. PhaseC searches architecture and regularization, including latent count, fusion depth, expert count, token counts and attention/dropout settings. This is more specific than describing every phase as an undifferentiated parameter sweep. The portable fitting adapter is a new controller; historical crash recovery, storage paths and infrastructure launchers are not copied into it.

The archived ensemble implementations support arithmetic probability averaging, average logits, trimmed logit aggregation with a dispersion penalty, covariance-based minimum-variance weights, discriminant weights and learned logistic weights. A second level can combine outputs from several ensemble searches. An inspected cascade forward path uses mean-logit combination for the Stage0 and Stage2 seed-level scores and a histogram-gradient-boosting meta classifier over Stage1 logits. The portable ensemble adapter supports these mechanisms with fitted state retained for prediction. Covariance estimation, temperature scaling, isotonic fitting and learned stackers use only the development rows explicitly passed to `fit`.

Temperature scaling changes logits by a fitted scalar temperature. Isotonic calibration learns a monotone mapping. Neither creates new outcome information. A fitted curve evaluated on the same records used to fit it is a calibration diagnostic, not an out-of-sample calibration result. Likewise, a meta model trained on validation predictions is still using validation labels, even when every underlying neural checkpoint was fitted only on training rows.

Historical threshold objectives go beyond raw accuracy. Exported routines calculate precision, recall, contamination among accepted cases, sacrifice of true fast cases, minimum bucket support and optional duration-dependent false-positive costs. Other routines use Wilson lower confidence bounds and threshold validation splits. Some archived searches return a closest infeasible threshold when no candidate meets all constraints; their negative objective or explicit feasibility checks must not be discarded. The new policy adapter instead raises when no feasible candidate exists, making that portability decision explicit.

An ensemble search seed and a neural training seed are different experimental units. Several search seeds can repeatedly combine one common pool of already trained checkpoints. This may explore policy instability, but it does not establish that all pool members were independently retrained for every search seed. Threshold folds operating on saved SVAL predictions also are not independently refitted neural folds. Extensive checkpoint, calibration, mixture and policy exploration over a small SVAL cohort increases selection reuse; more search does not increase its independent sample size.

## The later direct slot-based lane

The later direct Stage2 lane must be distinguished from the legacy cascade. It can score a 72-hour policy directly without mandatory Stage0 and Stage1 routing. Its preserved architecture is available under `research/production_reference`. An inspected checkpoint configuration has 17,145,736 unique trainable parameters, 239 numerical and 81 categorical inputs, width 256, 32 latent tokens, six fusion blocks, seven experts and 128 linear bins. Summing state-dictionary tensor entries instead gives 17,148,808 because tied normalization parameters appear under multiple names.

Its 40 input tokens comprise 16 pooled tabular tokens, eight projections of one listing-text vector, eight image-slot tokens and eight report-slot tokens. Frozen upstream representations feed the model; explicit slot positions, modality indicators and missingness representations distinguish the supplied evidence. The historical raw and transformed embedding ecosystem may provide multiple upstream representations, but the inspected selected network interface still has one listing-level text-vector input. Two available upstream vector versions do not by themselves prove two explicit neural text branches.

The documented later split contains 50,227 training rows, 45,087 rows eligible for the short-horizon training target, 584 SVAL rows and 748 holdout rows. Saved evaluation predictions directly establish 72 censored SVAL rows and 98 censored holdout rows. The latest complete training matrix was not recovered at its stated artifact location, so its exact censoring fraction remains unknown. Capacity relative to these counts motivates matched regularization and ablation studies; it does not prove either overfitting or architectural superiority.

## Leakage investigations and scientific limits

The system contains substantive safeguards: training-fitted preprocessing, named feature exclusions, temporal comparable windows, frozen model configuration, explicit outcome masks and saved numerical serving checks. These controls address identifiable mechanisms. They do not make arbitrary late-computed features historically observable. A vector computed from a changed image, a description edited after the target event or an aggregate assembled from future labels can violate the decision-time contract while matching every expected column name and dimension.

A historical market-context feature family illustrates the limitation. Its missingness pattern was strongly associated with short-horizon labels because enrichment availability tracked later lifecycle processing. Such a shortcut can improve validation metrics even though the model never receives an obvious target column. The inspected successor pipeline banned the affected family and added export/write guards. That is evidence of diagnosis and remediation in the inspected lineage, not a reason to relabel affected earlier scores as clean. Missingness-only probes, predecessor-state checks and availability-version audits are therefore part of the proposed next experiment.

An available later rebuilt training table contains all rows in the historical tail diagnostic subsets. The original fit exports were not recovered, so this establishes reconstruction overlap rather than proving that those exact rows trained the original checkpoint. Such exploratory diagnostics cannot establish unseen-tail generalization without independent lineage. A recent 283-row diagnostic was contained in a broader 748-row holdout; these populations cannot be added together as independent validation evidence. Source-derived duration endpoints and zero-duration observations require provenance analysis rather than automatic assumptions about verified transaction times.

Operational parity provides a different kind of evidence. A saved older Stage0 comparison covers 523 rows, reports matching decision/bucket columns and a maximum meta-probability discrepancy of about $1.01\times10^{-7}$. Captured-row replays also establish repeatability for that saved bundle. These are substantial model-serving checks. They do not measure predictive accuracy, and a proof tied to an older contract bundle does not certify every later model version. Similarly, one saved eight-row live batch with zero failures establishes that recorded execution, not a universal latency guarantee.

The available code supports adapting models and policies to a changing market through recent cohorts, decay weighting, development-set tuning and retraining. It does not establish automatic online learning or that a policy remains calibrated after population drift. Nor is there a completed matched real-data study proving that conventional models fail because of noisy vectors while the neural model succeeds. Attention, nonlinear fusion, token pooling and regularization provide plausible mechanisms; a fair comparison must give conventional and neural estimators the same admissible information and report their actual outcomes.

## Reusable release and verification

The exact-source modules are identified in `provenance.json`. `portable.py` supplies the new feature encoder and per-stage estimator; `ensemble.py` retains fitted calibration/combination state; `pipeline.py` fits and routes all three stages; the package command runs a small artificial example. Individual estimators expose `fit`, `evaluate`, `predict`, `predict_survival`, `save` and `load`. The cascade exposes fitting, sequential prediction and bundle persistence. Caller-created bundles contain fitted parameters and must remain separate from the source-only public release.

Verification uses artificial data and analytical arithmetic. Tests check source hashes, syntax-tree identity, exact horizon boundaries, short-censor masking, fractional survival exposure, late-event recensoring, training-only feature vocabularies, frozen ensemble fitting, gate precedence, required-score failures, model round trips and actual training of all three stages. The artificial example separately reports its fitting and development counts and identifies its prespecified thresholds. Passing these checks establishes runnable modeling software and preserved semantics. The decisive research result still requires a frozen, independently governed temporal dataset and final-test evaluation that has not participated in architecture, calibration or policy selection.
