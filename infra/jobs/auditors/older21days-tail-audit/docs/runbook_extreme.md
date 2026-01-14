# Stale Listing Audit Job Runbook (Public Template)

This document describes the **stale listing audit job** implemented in `audit_older21days.go`.

The public release is intentionally **platform-agnostic**. Marketplace-specific connectivity and parsing are isolated behind the adapter interface in `adapters/marketplace_adapter.go`.

---

## 1) Purpose

The job audits listings already persisted in Postgres that have been marked as *stale* (default: `status='older21days'`).

For each stale listing, it:

1. Fetches a normalized listing snapshot via the **marketplace adapter**.
2. Reconciles terminal outcomes in the main table:
   - `older21days → sold` (writes `sold_date` when available)
   - `older21days → removed` (for HTTP 404/410 or adapter-reported terminal state)
   - It does **not** flip terminal rows back to `live`.
3. Writes sparse history:
   - Terminal events are always written to `marketplace.price_history`.
   - For `live` rows, it writes at most one daily baseline (UTC day bucket) plus optional change events.
4. Records inactive/bidding state signals when present, in:
   - `marketplace.listings` (current flag state)
   - `marketplace.listing_inactive_state_events` (append-only state-change log)

---

## 2) Data model (minimal public schema)

See `sql/schema.sql`.

Tables used by the job:

- `marketplace.listings`
  - primary key: `(generation, listing_id)`
  - includes `status`, `price`, `sold_date`, timestamps, and `marketplace_*` state flags

- `marketplace.price_history`
  - sparse history stream; `source='audit-stale'` identifies job writes

- `marketplace.listing_inactive_state_events`
  - append-only event log for inactive state changes

---

## 3) Adapter layer

The job never embeds target-site logic. All external calls go through:

- `FetchListing(listing_id)` → `adapters.Listing`
- `SearchListings(params)` → optional helper for discovery use cases
- `ParsePayload(raw)` → optional helper for decoding upstream payloads

This repo ships with:
- `ADAPTER_KIND=mock` — safe synthetic responses for demos/CI
- `ADAPTER_KIND=http` — a generic HTTP JSON adapter targeting a placeholder API (`MARKETPLACE_BASE_URL`)

---

## 4) Configuration (env vars)

### Required
- `PG_DSN` — injected at runtime (do not commit)
- `PG_SCHEMA` — default: `marketplace`

### Common
- `GENERATION` — integer cohort key (default `0`)
- `STALE_STATUS` — default: `older21days`
- `AUDIT_DAYS` — optional time-window filter on `last_seen` (0 disables)
- `AUDIT_LIMIT` — optional row limit per run (0 disables)
- `AUDIT_ONLY_IDS` — optional comma-separated `listing_id` allowlist

### Adapter
- `ADAPTER_KIND` — `mock` or `http`
- `MARKETPLACE_BASE_URL` — used by the HTTP adapter
- `MARKETPLACE_AUTH_HEADER` — optional auth header (inject at runtime)

### Bounded-pressure controls
- `WORKERS`
- `REQUEST_MIN_RPS`, `REQUEST_MAX_RPS`, `REQUEST_STEP_UP_RPS`, `REQUEST_DOWN_MULT`
- `REQUEST_BURST_FACTOR`, `REQUEST_JITTER_MS`
- `REQUEST_RETRY_MAX`, `REQUEST_FALLBACK_THROTTLE_MS`

---

## 5) Operations

### Run locally (Go)

```bash
export PG_DSN=...
export PG_SCHEMA=marketplace
export ADAPTER_KIND=mock
go run . --mode audit-stale
```

### Run in Docker

See `docs/runbook_docker_image.md`.

---

## 6) Verification SQL (no real data)

These queries are safe to publish and are designed to be run against the template schema.

### A) Daily baseline snapshots (UTC bucket)

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') + interval '5 minutes' AS bucket
)
SELECT
  generation,
  COUNT(*) AS baselines_today
FROM marketplace.price_history, p
WHERE source = 'audit-stale'
  AND status = 'live'
  AND observed_at = p.bucket
GROUP BY 1
ORDER BY 1;
```

### B) Anti-bloat check: rows per listing per day

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') AS d0,
         date_trunc('day', now() AT TIME ZONE 'UTC') + interval '1 day' AS d1
)
SELECT
  generation,
  max(n) AS max_rows_per_listing_today,
  percentile_disc(0.99) WITHIN GROUP (ORDER BY n) AS p99_rows_per_listing_today
FROM (
  SELECT generation, listing_id, count(*) AS n
  FROM marketplace.price_history, p
  WHERE source='audit-stale'
    AND observed_at >= p.d0 AND observed_at < p.d1
  GROUP BY 1,2
) t
GROUP BY 1
ORDER BY 1;
```

### C) Terminal flips attributed to the audit job today

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') AS d0,
         date_trunc('day', now() AT TIME ZONE 'UTC') + interval '1 day' AS d1
)
SELECT
  l.generation,
  COUNT(*) FILTER (WHERE l.status='sold')   AS sold_flipped_today,
  COUNT(*) FILTER (WHERE l.status='removed') AS removed_flipped_today
FROM marketplace.listings l, p
WHERE l.status IN ('sold','removed')
  AND l.last_seen >= p.d0 AND l.last_seen < p.d1
  AND EXISTS (
    SELECT 1
    FROM marketplace.price_history ph
    WHERE ph.generation=l.generation
      AND ph.listing_id=l.listing_id
      AND ph.source='audit-stale'
      AND ph.status=l.status
      AND ph.observed_at >= p.d0 AND ph.observed_at < p.d1
  )
GROUP BY 1
ORDER BY 1;
```

### D) Inactive event log is event-based (no spam)

```sql
WITH p AS (
  SELECT now() - interval '7 days' AS t0
)
SELECT
  generation,
  listing_id,
  COUNT(*) AS events_last_7d
FROM marketplace.listing_inactive_state_events, p
WHERE observed_at >= p.t0
GROUP BY 1,2
ORDER BY events_last_7d DESC
LIMIT 50;
```

