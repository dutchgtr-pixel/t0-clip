# Status Sanity Audit (Public Release)

## Summary

`status_sanity_audit` is a job-mode auditor that validates and reconciles **status correctness** in a marketplace listings database. It is intentionally designed as a “Swiss Army knife” for status auditing:

- Works across **any status taxonomy** (status is stored as free-form text).
- Supports precision targeting (one listing, one segment, one scope) or broad sweeps.
- Records full provenance: what was checked, what was detected, what differed, and what (if anything) was changed.
- Applies corrections only under explicit, guardrailed modes.

This public release preserves the architecture, SQL audit telemetry, and operational UX, while abstracting target-specific connectors behind an adapter layer.

## What problems it solves

Common failure modes in real marketplaces:

- Listings that were sold externally but still stored as `live`.
- Listings removed externally but still stored as `live`.
- Listings that are still live but are flagged `inactive` upstream (or the inverse), causing inconsistent downstream behavior.
- Incorrect or missing sale timestamps and sale prices due to partial ingestion, transient failures, or schema drift.

`status_sanity_audit` provides a repeatable, measurable process to detect and correct these issues while maintaining a forensic audit trail.

## Core concepts

### Stored vs detected state

- **Stored**: the canonical listing record in your database.
- **Detected**: the state inferred from the marketplace at audit time, via the adapter.

The auditor compares these, computes mismatches, and can suggest or apply corrections.

### Primary status vs secondary flags

Most teams benefit from separating:
- primary status (e.g., `live|sold|removed|stale_bucket|...`)
- secondary flags (e.g., `is_inactive`, `is_bidding`, `is_shadow_hidden`)

This auditor supports both:
- “status sanity” (primary status reconciliation)
- “flag sanity” (boolean/metadata synchronization)

### Guardrailed mutation philosophy

The default posture is: **observe, log, and propose**. When enabled, mutations are constrained:

- `--apply none`: detect + log only
- `--apply safe`: apply monotonic, low-risk transitions and safe flag sync
- `--apply all`: enable broader transitions only with explicit opt-ins

A recommended governance model:
1) run daily with `--apply none` to build mismatch baselines,
2) enable `--apply safe` once you trust the detector + thresholds,
3) reserve `--apply all` for supervised runs or incident response.

## Marketplace Adapter Layer

Public repositories should not embed target-specific scraping or parsing logic. The auditor expects a connector interface that provides:

- `fetch_listing(listing_id)` → raw payload
- `parse_payload(raw)` → normalized detection fields:
  - `detected_status` (e.g., `sold|live|removed|inactive|unknown_*`)
  - `sold_at`, `sold_price` (if sold)
  - `live_price` (if live)
  - `is_inactive` (optional)
  - `evidence` (structured debug context)

A default “mock adapter” is recommended for synthetic runs and CI.

## Execution model

This tool runs in **job mode**:

1) start run
2) select candidates (cadence-aware)
3) audit candidates concurrently under bounded pressure controls
4) write per-listing events
5) update run summary
6) exit

This makes it suitable for:
- Airflow `DockerOperator`
- Kubernetes CronJobs
- CI smoke tests (with mock adapter)
- manual operator runs for incident response

## Modes

### `--mode init`
Ensures SQL objects exist (tables/indexes/views).

### `--mode run`
Executes the audit. All power is exposed via CLI flags: selection, ordering, cadence, rate control, logging, exports, and apply guardrails.

### `--mode export`
Exports the latest mismatch per listing (a ready triage queue).

## What audits can it run?

Think of “audit type” as a *candidate selection + detector configuration*.

### Built-in audit profiles (recommended patterns)

1) **Sold integrity audit**
- Candidate set: stored status is `sold`
- Goal: verify that sold listings remain sold and have coherent sold fields.

2) **Live integrity audit**
- Candidate set: stored status is `live`
- Goal: catch “silent” sells/removals and flag mismatches (e.g., upstream inactive).

3) **Removed integrity audit**
- Candidate set: stored status is `removed`
- Goal: verify removals and detect unexpected reactivations (usually manual review).

4) **Stale bucket audit**
- Candidate set: stored status is your internal “aging bucket” (e.g., `stale_bucket`)
- Goal: confirm whether stale listings are actually removed/sold/live.

5) **Inactive flag synchronization**
- Candidate set: `--inactive-only` or `--scope live`
- Goal: synchronize `is_inactive` flag without changing primary status.

Because statuses are free-form, you can define any audit profile you want by using `--statuses`.

## Candidate selection (power and control)

Selection is database-driven and highly controllable:

- `--scope`: convenience presets (e.g., `sold`, `live`, `removed`, `stale_bucket`, `all`)
- `--statuses`: explicit CSV list overriding `--scope`
- `--gens`: optional segmentation (CSV ints; e.g., product generation/category version)
- `--only-ids`: targeted IDs for incident response (CSV)
- `--inactive-only`: only rows currently flagged inactive
- `--limit`: cap candidates (smoke test or throttled ops)

### Cadence controls

- `--cadence-days N`: skip listings audited within the last N days
- `--force`: ignore cadence (use with `--only-ids` or during incidents)

Cadence is essential for keeping audits cheap and predictable at scale.

### Ordering controls

Ordering ensures deterministic behavior:

- `--order first_seen_asc|first_seen_desc`
- `--order sold_date_asc|sold_date_desc`
- `--order last_seen_asc|last_seen_desc`
- `--order listing_id_asc|listing_id_desc`

This is operationally useful for:
- prioritizing oldest unverified records,
- triaging earliest sold records first,
- spreading load deterministically.

## Concurrency and bounded pressure

The job is designed for high throughput without destabilizing upstream systems:

- `--workers`: concurrent workers
- adaptive limiter:
  - `--rps` (start), `--rps-max`, `--rps-min`
  - `--rps-step`, `--rps-down`, `--ok-every`, `--burst`
- retry/backoff:
  - `--retry`, `--retry-backoff`, `--throttle-sleep`, `--jitter`
- safety caps:
  - `--timeout`, `--max-body-bytes`

### Why an adaptive limiter?

Fixed RPS is brittle across:
- upstream traffic spikes,
- transient throttling,
- uneven listing page weight.

An adaptive limiter increases throughput when healthy and backs off on throttle signals.

## Apply mode (guardrailed corrections)

### `--apply none`
- No mutations.
- Writes events and mismatch suggestions only.

### `--apply safe`
Only applies “monotonic, low-risk” transitions and safe flag sync, typically:

- `live|stale_bucket -> sold` (when sold is detected)
- `live|stale_bucket -> removed` (when removal is detected)
- `is_inactive` sync (when the upstream flag exists)

### `--apply all` (advanced)
Allows broader transitions only when explicitly enabled:

- “unsell”: `sold -> live` (requires `--allow-unsell`)
- “revive”: `removed|stale_bucket -> live` (requires `--allow-revive`)

`--apply all` is intended for supervised runs and post-incident cleanups.

### Audit-trail enforcement

If mutations are enabled, the job requires that DB event logging remains enabled so every change is paired with an audit event.

## Outputs

### Database telemetry
- `marketplace.status_sanity_runs`
- `marketplace.status_sanity_events`

### Optional files
- `--out-jsonl`: JSONL export of each audited row
- `--out-csv`: CSV export

### Optional stdout telemetry
- `--log-json`: one-line JSON per audited listing (ops-friendly)
- `--mismatch-summary`: end-of-run summary and a CSV of mismatched listing IDs

## CLI reference (full operational surface)

The public repo may map flags to env vars for clean Docker orchestration.

### Required
- `--dsn` (recommended via `PG_DSN`)
- `--schema` (default: `marketplace`)

### Mode
- `--mode init|run|export`

### Selection
- `--scope`
- `--statuses`
- `--gens`
- `--only-ids`
- `--inactive-only`
- `--limit`

### Ordering and cadence
- `--order`
- `--cadence-days`
- `--force`

### Request behavior
- `--workers`
- `--timeout`
- `--head-first`
- `--allow-ui-fallback`

### Rate limiting and backoff
- `--rps`, `--rps-max`, `--rps-min`
- `--rps-step`, `--rps-down`, `--ok-every`, `--burst`
- `--retry`, `--retry-backoff`, `--throttle-sleep`, `--jitter`
- `--max-body-bytes`
- `--print-limiter` (optional periodic limiter state)

### Logging / exports
- `--write-events`
- `--log-json`
- `--log-url`
- `--out-jsonl`
- `--out-csv`
- `--mismatch-summary`

### Apply modes and safety toggles
- `--apply none|safe|all`
- `--apply-changes` (alias for `--apply safe`)
- `--allow-unsell` (only honored when `--apply all`)
- `--allow-revive` (only honored when `--apply all`)

## Runbook recipes (public)

### Create tables
```bash
status_sanity_audit --mode init
```

### Nightly read-only sweep
```bash
status_sanity_audit --mode run --scope all --apply none --cadence-days 30 --workers 16
```

### Nightly safe sweep
```bash
status_sanity_audit --mode run --scope all --apply safe --cadence-days 30 --workers 16
```

### Tight, low-pressure mode (for sensitive upstreams)
```bash
status_sanity_audit \
  --mode run \
  --scope live \
  --apply none \
  --workers 4 \
  --rps 0.5 --rps-max 1.0 --rps-min 0.25 \
  --retry 2 --timeout 20s
```

### Incident response (single listing)
```bash
status_sanity_audit --mode run --only-ids 123456789 --force --apply none --log-json
```

### Export mismatch triage queue
```bash
status_sanity_audit --mode export
```

## SQL: schema and logic (public)

This public release includes two SQL companions:

- `status_sanity_audit_schema_public.sql`  
  Tables, indexes, and a helper view for latest mismatches.

- `status_sanity_audit_logic_public.sql`  
  Transparent candidate selection, triage queries, rollups, review workflow, and example “safe apply” patterns.

## Review workflow

The events table contains review fields to support human-in-the-loop governance:

- `review_action` (e.g., `accept|reject|ignore|manual_fix`)
- `review_note`, `reviewed_by`, `reviewed_at`

This is critical for:
- documenting why mismatches were ignored,
- capturing operational knowledge,
- training future automation policies.

## Extending the auditor

### Adding new status concepts
- Implement platform semantics in the adapter.
- Normalize to your internal taxonomy.
- Keep the canonical `status` small and stable; model specialized states as flags or derived views.

### Adding new “audit dimensions”
If you want to expand beyond status:
- add additional normalized detection fields (e.g., shipping availability, region tags, category drift),
- include them in `evidence_json`,
- compute new mismatch booleans and suggestions,
- keep mutations guardrailed and auditable.

## Public-release safety guarantees

- No embedded credentials (use env vars or orchestration secrets).
- No platform identifiers or target fingerprints.
- No real listing IDs or real logs in docs; all examples are synthetic.

