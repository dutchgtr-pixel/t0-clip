# The implemented agentic decision system

## From a forecast to an operational workflow

The platform included an implemented operator system for candidate review, message drafting, approval, conversation mirroring and outcome follow-up. It was more than a proposal to connect a language model to a database. The inspected source contains a typed action-proposal schema, a context-packet builder, a model-call runtime, a deterministic proposal validator, persistent action and execution records, a stateful execution service, a Rust cache and realtime layer, and operator interfaces. The historical documentation records a known operational baseline and subsequent read-path verification.

These facts establish the existence and structure of a working decision-support and execution architecture. They do not establish an independently profitable autonomous trading strategy. The implementation deliberately distinguishes a proposed action, a staged draft, an approved send, an uncertain send, an observed reply and an applied outcome. This distinction is a strength of the design: the system does not have to pretend that model-generated prose is execution evidence.

The role of the survival model is to add information about time and liquidity to a broader decision packet. A listing that may disappear quickly can justify prompt review; a long-tail estimate may justify waiting, lowering priority or demanding a greater margin of safety. The forecast does not establish the item's condition, the seller's reliability, the realized resale price or the causal effect of an offer. Those require separate evidence and models.

## The implemented division of responsibilities

The inspected conversation-agent runtime has four principal layers. A web interface displays candidate and conversation state. A cockpit API proxies the operator-facing endpoints. A Python execution service manages authenticated interaction, thread reconciliation, composers, drafts and send flows. A Rust service provides caching, serialized refresh work, server-sent events and latency instrumentation. The authenticated session is stateful infrastructure; it cannot be treated as a disposable stateless request worker without changing execution behavior.

The Rust layer's contribution is operational coordination. Its source includes per-run locks, conversation caches, queued background refreshes, queue status, event streams, bounded latency samples and correlation identifiers. These mechanisms reduce duplicate concurrent work and let the interface update while retaining durable state elsewhere. Rust is not the language of every model or every narrative rule in the system. The project combines Python, SQL, TypeScript and Rust according to responsibility; describing all reasoning as a Rust model would misrepresent the inspected architecture.

A separate Rust language-model service supports analytical questions over curated documentation and approved database views. It loads a document manifest, selects relevant documentation, authenticates users, stores conversations, reserves request quota, requests a structured response and validates generated SQL before execution. This analytical service and the conversation execution service are different control paths. Access to analytical data does not itself grant permission to send a message or commit funds.

## Context packets as an explicit information contract

The implemented context builder packages the listing, deal-engine fields, evidence, thread state, session readiness and policy. It includes the current ask, recommended opening offer, maximum buy ceiling, fast and standard resale floors, route and lane explanations, condition and damage fields, a bounded description excerpt, recent messages and operational readiness. Compact, standard and expanded modes limit description length to 500, 1,200 and 2,500 characters respectively. The message count is bounded between zero and fifty, with twelve as the default.

Those limits are a concrete form of context engineering. The language model receives selected structured evidence rather than the entire database or an unbounded conversation history. The package also declares the distinction between proposal, executor and browser lanes. The browser is not exposed as an actuator to the proposal model. The model is instructed to use the supplied economic fields and return exactly one action proposal.

The inspected context does not by itself prove that a survival curve is present in every agent prompt. Some downstream fields can reflect model-derived ranking or candidate selection, but the visible packet must be audited separately from upstream data availability. A stronger survival-aware extension would include the model version, forecast origin, scoring time, horizon probabilities, uncertainty and freshness explicitly. That is an extension of an existing packet architecture, not evidence that every component was already connected in the retained version.

The context builder also identifies conditions that block or restrict actions, including incompatible seller scope, bidding-style listings, unavailable listings without a known thread, and session readiness. Buy-offer drafting is included among allowed actions only when the opening offer and ceiling are present. Survey campaigns use a different message-only policy. A downstream executor must still enforce these restrictions at action time because readiness and listing state can change after the packet was built.

## Structured proposals and deterministic checks

The action schema admits five proposal types: wait, stage a draft, refresh the thread, mark the case for review, and apply a candidate reported outcome price. Supported objectives cover next-action advice for an outcome survey, buy-offer advice and thread-reply advice. A proposal contains an action, campaign, optional message and offer, a short rationale, evidence references, confidence and an optional candidate outcome price.

The inspected model runtime requests JSON output with a low temperature of 0.15, parses the returned JSON, and validates it against the typed schema. This reduces formatting variability and narrows the action space. It does not make language-model reasoning deterministic, certify the truth of a rationale, or calibrate the supplied confidence. The model's own confidence is a bounded numeric field, not a validated probability of a successful trade.

The policy validator checks that the objective and action are supported, that the action is allowed by the packet, that the campaign matches, and that evidence references are present. It checks message-language constraints and blocks direct-send instructions in generated message content. Draft actions require message text. A message-only campaign cannot contain an offer field or detected offer language. A buy-offer draft requires an offer value, and a supplied offer is compared with the supplied ceiling. Confidence must fall between zero and one.

These are implemented checks, not a proof of complete safety. In the inspected validator, evidence references are required to be nonempty but are not resolved there into verified source claims. The ceiling comparison is conditional on both numeric values being present; the context builder's buy-offer eligibility rule is therefore part of the end-to-end contract. The validator's action allowlist and the packet's separate blocked-action descriptions are also distinct fields. A complete assurance case must test the composed path, including the executor, rather than assume a check in one layer enforces every statement in another.

The distinction between schema validity and semantic validity is the same distinction encountered in image enrichment. A well-formed proposal can still misunderstand a photograph, misread a seller's statement, rely on stale prices or cite an irrelevant field. Deterministic checks bound the proposal; they do not eliminate the need for evidence evaluation and outcome measurement.

## Execution, persistence and operational truth

The system stores candidate staging, runs, run events, mirrored thread state, context packets, action intents and execution batches. The inspected database definitions include batch items, item-level events, candidate sets, session state and session events. The data model can therefore distinguish a cohort of proposed actions from individual execution outcomes. Persisted state is materially more informative than a single transient model response.

The context contract states that live sending requires an existing run awaiting approval, a ready session and authentication state, a fresh heartbeat, operator approval outside the proposal controls, and duplicate or uncertain-send checks. Execution proof events distinguish a staged draft, a clicked send, a completed run, an uncertain send, a reply and an applied outcome. A narrative may claim that an action occurred only when a result packet or event supports it.

This contract matters for retries. A timeout after a send attempt does not prove that the external action failed. Retrying without reconciling the conversation can duplicate a message. Conversely, recording a successful local click does not prove the remote recipient received it. The implementation's uncertain state is a useful representation of this ambiguity. A future unattended workflow would need the same discipline for offers, purchases, cancellations and funds, with idempotency and reconciliation applied to each irreversible step.

The historical read-path documentation records a test in which five unread conversations were opened through the passive read path and the externally reported unread counts remained unchanged. This supports the behavior for those tested conversations and that interface version. It is not an immutable guarantee about future external behavior. The broader known-good baseline records exact manual-draft preservation, conversation refresh, inbox mirroring and operator handoff from a live candidate into a real draft run.

## Evidence-aware listing narratives

The narrative layer turns structured facts into explanations suitable for review. Retained regression results dated 21 April 2026 report seventy passing tests. Their cases include receipt claims versus visual proof, repair history versus current damage, functional fault wording, screen-protector-only damage, stock photographs, packaging claims, missing battery evidence, unsupported damage location and local-market conflicts. These tests show that the implementation encoded distinctions that a generic fluent summary could easily erase.

A separate retained report dated 19 April 2026 compared five thousand sampled live listing narratives from 6,049 eligible rows and reported zero mismatches between the compared implementations. The sample included variation in condition, image damage, text severity, photographic quality and image source. That is strong evidence of agreement within the tested path. Two implementations can agree on the same wrong rule, so parity must not be reported as a five-thousand-example accuracy study against independently labeled truth.

The research significance lies in the separation of evidence roles. A seller's text claim, an image-derived observation, a model forecast and a deterministic business rule should remain distinguishable in the explanation. When evidence conflicts, the narrative should expose the conflict or request review. It should not silently choose the most favorable story. This approach gives a human or a frontier reasoning model a better basis for action than an unqualified score alone.

## Analytical access and its limits

The Rust analytical service parses proposed SQL into a syntax tree, requires exactly one query statement, collects referenced table names through supported query shapes and compares them with an allowlist. Execution applies a statement timeout, wraps the query with a result limit and rolls back the transaction. These are concrete resource and access controls in the inspected code.

The implementation should not be described as a formally proven SQL sandbox. The inspected transaction creation does not itself set a database read-only transaction mode. Its table traversal focuses on supported FROM, JOIN, derived-table and set-operation structures; a comprehensive security claim would require adversarial tests of expression subqueries, functions, database privileges and all accepted syntax. The defensible present statement is that the system implements a restricted analytical query path with several guards. Database-level read-only permissions and complete syntax validation would strengthen that boundary.

This is relevant to the research rather than an unrelated security checklist. An agent that can reason over industry data must receive controlled, correctly interpreted evidence. Analytical access should remain separate from action authority. A plausible SQL answer is not permission to mutate data, message a seller or buy an item.

## How survival information can strengthen the existing agent

Survival estimates add a time dimension that a static bargain score lacks. For a newly eligible listing, a calibrated short-horizon event probability can rank review urgency. A long-tail estimate can identify candidates for patient negotiation or further inspection. A full survival curve can express differences between short-term availability and longer-term persistence that a single binary score conceals.

For an older listing that remains active, the relevant future probability is conditional on survival to its current age. Under a compatible fixed-origin model, the probability of an event over the next interval of length $h$ is

$$
P(a<T\leq a+h\mid T>a,X)=1-\frac{S(a+h\mid X)}{S(a\mid X)},
$$

when the denominator is positive and the conditioning information remains appropriate. If listing content, price or market state has changed, a landmark model using current admissible features may be preferable. Reusing an unconditional historical FAST72 score as a fresh next-72-hour forecast would give the agent a misleading clock.

The frontier-model extension can therefore be described concretely: use the existing context packet, add versioned and time-qualified survival evidence, request a bounded proposal, validate it against deterministic policy, and reconcile execution through the existing event ledger. The reasoning model can explain tradeoffs and ask for missing evidence; the policy layer defines action limits. A model-generated explanation cannot override a stale-feature rejection or invent a buy ceiling.

Economic value remains a separate target. A rapidly disappearing listing may be desirable, mispriced, withdrawn or otherwise unobservable; it is not automatically profitable inventory. An evaluation must measure realized acquisition cost, fees, repair and handling costs, resale outcomes, capital duration and unsuccessful attempts. It must compare policies prospectively or through a defensible off-policy design, not infer profits from survival discrimination alone.

## What scaling would mean scientifically

The platform already supplied data volume, enrichment, candidate generation, typed proposals and tracked execution. Scaling this into a more autonomous system is therefore an engineering extension of implemented components. The missing evidence is not whether an agent interface can be imagined; it is whether the composed policy remains reliable, calibrated and economically useful as volume and market conditions change.

A rigorous next experiment would start with shadow decisions on a frozen policy, record the proposals and the information available at each moment, and compare them with later observable outcomes. Any staged expansion of action authority should have explicit exposure limits, reconciliation tests and stopping criteria. The experimental report should distinguish analytical throughput, accepted proposals, actually executed actions, completed transactions and realized net outcomes. Combining those counts into a single success rate would hide where the system succeeds or fails.

The implemented architecture is powerful because it connects evidence, prediction, explanation and action through inspectable contracts. Its research value is strongest when those contracts retain uncertainty and failure states. The appropriate claim is an operational foundation for data-informed agentic decisions, with identifiable components and testable extensions, rather than an unmeasured assertion of autonomous trading superiority.
