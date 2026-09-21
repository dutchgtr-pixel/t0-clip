# The temporal platform as part of the research method

## Why the feature pipeline determines the experiment

The temporal platform exists because the model's information set is produced by many asynchronous processes. Listing attributes, seller text, photographs, structured interpretations, historical comparison statistics, and learned vectors do not arrive simultaneously. An export assembled from their latest versions may be internally consistent and still describe a world that was unavailable at the claimed prediction time. The feature pipeline therefore participates directly in the experiment's validity.

The historical system divides these responsibilities into observation, canonicalization, audit and repair, feature construction, certification, and model consumption. This decomposition is technically substantive. It gives each stage a contract and allows failures to be localized. A damaged image can block one modality rather than silently changing a listing's outcome. A schema change can invalidate a feature contract rather than reaching a model as a different column order. A repair can create a new version instead of replacing the evidence behind an old result.

The feature-store report described roughly 320–350 assembled variables in a representative historical AFT run. Its inventory included geographic, socioeconomic, device, vision, text-image fusion, market, anchor, missingness, and trainer-derived blocks. Those approximate dimensions belong to that reported configuration; they are not the input size of every later neural run. Their significance is architectural: a model consumes a composition of independently maintained information products, and the composition must preserve their temporal and identity contracts.

## Event time, evidence time, computation time, and decision time

Four clocks need explicit names. Source event time records when a fact claims to apply. Evidence time records when the system first had access to the supporting observation. Computation time records when a transform or enrichment finished. Decision time records when a score could have been used. A temporal filter on the first clock does not establish validity with respect to the other three.

Let $\mathcal F_t$ denote the information available by time $t$. The intended feature condition is $X(t_0)\in\mathcal F_{t_0}$, understood as measurability rather than set membership of a raw record. Operationally, every dependency must have an admissible version, and every outcome-derived statistic must use labels available by the relevant cutoff. An old event discovered later remains unavailable to an earlier decision unless the research explicitly changes its question to retrospective reconstruction.

There is a legitimate distinction between later computation and later information. A frozen image encoder applied today to immutable bytes known to have existed at the decision may study what that historical information could predict. It does not show that the production system could have completed the encoding on time. Conversely, a current photo substituted for an earlier one is a change in information, even if the same encoder is used. The experiment records both raw-content identity and computation readiness so that these claims cannot be conflated.

Outcome availability follows the same rule. A listing originating before a training cutoff cannot contribute an eventual sale learned after that cutoff to a simulated model trained then. Administrative censoring as of fit time preserves what was known. A fixed maturity gap can help for horizon labels, but it is not a substitute for an availability model when reports are delayed or observation policies vary.

## Composable stores and identity-preserving joins

A feature block must specify its observation key, granularity, version, and missingness semantics. The historical design commonly uses an item identity, a product-family identifier, and a time origin. Joining an image-level table directly to a listing-level table can multiply rows. Joining an unversioned current seller profile can change history. Both errors can occur while every SQL statement executes successfully.

The composition contract therefore checks uniqueness at the intended key and preserves counts before and after joins. For a listing-level matrix, image-level evidence is aggregated or assigned to a fixed slot contract before the final join. The absence of a block remains explicit. A missing image vector, a padded image slot, and a valid all-zero numerical feature are different states and cannot share an undocumented representation.

Historical feature blocks also attach support metadata to derived quantities. A market median based on many recent observations differs from one reached through a broad fallback. A condition estimate supported by a clear rear image differs from a guess based on packaging. Recording these distinctions allows the downstream model and audit to reason about evidence quality rather than treating every scalar as equally grounded.

![Historical anchor backoff hierarchy](../../../research/thesis_evidence/figures/h03_anchor_backoff.png)

Historical figure H03. The earlier anchor design pools selected strata while retaining core item-family constraints, from PDF page 4 of the anchor chapter. The diagram specifies a construction policy; temporal availability must still be verified separately.

## Certification as a declared, executable guarantee

The governance papers propose a registry of feature entry points, structural identities, freshness checks, temporal invariants, and training preflight guards. Static analysis inspects dependency closures and forbidden expressions. Dynamic checks assess keys, null patterns, cutoffs, and data fingerprints. Consumption can fail when a block is stale or violates the declared contract. These mechanisms are valuable because they convert assumptions into repeatable checks at a boundary.

The strongest defensible guarantee is conditional: a named feature version passed named checks against named source versions at a recorded time. A schema digest establishes interface identity; it cannot prove that a source timestamp is truthful. A sampled dataset digest detects certain changes; it cannot prove the absence of changes outside the sample. A dependency scan can reject direct outcome columns while missing an enrichment process whose availability pattern indirectly reveals the outcome.

This qualification does not make certification unhelpful. It identifies the proof obligations that remain outside the automated checks. Raw content provenance, timestamp semantics, and external process behavior require evidence beyond a SQL view definition. The platform's historical missingness incident shows why the distinction matters: an apparently ordinary context block became predictive through its lifecycle-dependent availability, and successor exports explicitly removed it.

The detailed schema and leakage chapter develops these invariants and failure cases. Here the platform-level point is that contracts must compose. A valid numeric table and a valid vector archive do not automatically form a valid experiment if their keys are misaligned or their timestamps refer to different decisions. The compact public loader consequently aligns vectors by explicit row keys and rejects cross-partition entity overlap rather than relying on file order.

## Historical anchors and reconstructed inventory

Price anchors use completed comparable observations before the target decision, often in recent windows and with an embargo. The historical report illustrates thirty- and sixty-day windows and a five-day embargo. These are documented design settings, not evidence that five days covers every possible reporting delay. An embargo is effective only under a justified bound or probabilistic account of that delay. Availability timestamps remain preferable when they exist.

Inventory features count competing listings believed to be live at the decision. An offline history can retrospectively reveal that a listing stayed live between its first and last observation. That reconstructed interval may contain information that was not available at the earlier time. The correct prospective inventory is the one inferable from observations then, with an explicit staleness policy. A retrospectively reconstructed count can support exploratory analysis, but its result must not be presented as an operationally observable feature without that additional check.

These distinctions are especially important in a fast-changing market. Historical comparisons lose relevance; new stock changes the competitive set; missing evidence can delay a score beyond its useful window. Temporal correctness and freshness are therefore complementary. A perfectly historical but badly stale feature may be operationally weak, while a fresh current-state feature may be invalid for an earlier evaluation.

## Reproducibility across research and serving

An experiment manifest binds the data snapshot, raw-content versions, feature definitions, split rules, fitted preprocessing, model configuration, code revision, and output predictions. Reproducing a neural score requires more than the checkpoint tensors. Category vocabulary, numeric scaling, feature order, missing-value rules, slot selection, and output interpretation all influence the result. The inspected serving work preserves these contracts and tests numerical agreement across execution paths.

The platform also distinguishes current operational repair from historical scientific evidence. A corrected attribute should be available to future decisions under a new version. It should not silently rewrite an old final-test matrix and make the old prediction appear to have been based on corrected information. Reanalysis is legitimate when the correction and changed population are recorded. Confirmation then requires a cohort that has not already influenced those choices.

The public compact benchmark operationalizes a subset of this discipline: train-only preprocessing, chronological administrative censoring, identity checks, configuration selection on validation, and frozen selection before final-test prediction. Its three-block demonstration is smaller than the proposed four-block real-data design, which separates architecture selection from calibration and policy fitting. The distinction is explicit so that software completeness is not confused with a completed scientific comparison.

## Market movement and controlled adaptation

The retained platform has documented routes for feature refresh, new-record scoring, changed-input rescoring, model retraining, and threshold or calibration updates. These routes are forms of adaptation infrastructure. They do not establish autonomous online learning, a continuously optimized policy, or measured improvement after every update.

A controlled adaptation cycle first identifies what changed: the evidence distribution, event prevalence, observation process, or relationship between evidence and outcome. It then chooses the smallest justified intervention. Refreshing an inventory statistic differs from refitting a calibrator; a new encoder changes the representation contract; a new survival model changes the predictive function. Each update needs a version, a bounded evaluation, and a rollback path.

The research consequence is a prequential perspective: freeze a model, capture predictions before outcomes mature, evaluate the next cohort, and only then decide how to update. A rolling stream of such cohorts provides stronger evidence about adaptation than repeatedly optimizing one historical holdout. The platform supplies much of the machinery needed for that process. The remaining task is to use it under a sealed evaluation protocol rather than infer future reliability from successful feature refresh alone.
