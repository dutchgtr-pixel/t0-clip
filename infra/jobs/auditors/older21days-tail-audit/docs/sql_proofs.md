# Audit Job Proof/Verification SQL (Public Template)

This document contains **copy-paste SQL** to validate the behavior of the stale listing audit job.

All examples are platform-agnostic and contain no real listing identifiers or outputs.

---

## 1) Tail population “right now”

```sql
SELECT
  generation,
  COUNT(*)                                            AS stale_total_now,
  COUNT(*) FILTER (WHERE marketplace_is_inactive)      AS stale_inactive_now,
  COUNT(*) FILTER (WHERE marketplace_is_bidding)       AS stale_bidding_now
FROM marketplace.listings
WHERE status = 'older21days'
GROUP BY 1
ORDER BY 1;
```

---

## 2) Daily baselines at the UTC bucket

The audit job can write one daily baseline snapshot per listing at the UTC day bucket (`00:05` in this template).

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') + interval '5 minutes' AS bucket
)
SELECT
  generation,
  COUNT(*) AS baselines_today
FROM marketplace.price_history, p
WHERE source='audit-stale'
  AND status='live'
  AND observed_at = p.bucket
GROUP BY 1
ORDER BY 1;
```

---

## 3) Missing baseline detection (for live tail rows)

This query shows stale rows that are currently **not** flagged inactive in the main table, but are missing today’s baseline.

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') + interval '5 minutes' AS bucket
),
live_tail AS (
  SELECT generation, listing_id
  FROM marketplace.listings
  WHERE status='older21days'
    AND marketplace_is_inactive=false
),
have_baseline AS (
  SELECT generation, listing_id
  FROM marketplace.price_history, p
  WHERE source='audit-stale'
    AND status='live'
    AND observed_at = p.bucket
)
SELECT lt.generation, lt.listing_id
FROM live_tail lt
LEFT JOIN have_baseline hb
  ON hb.generation=lt.generation AND hb.listing_id=lt.listing_id
WHERE hb.listing_id IS NULL
ORDER BY 1,2
LIMIT 200;
```

---

## 4) Terminal flips attributed to the audit job

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') AS d0,
         date_trunc('day', now() AT TIME ZONE 'UTC') + interval '1 day' AS d1
)
SELECT
  l.generation,
  l.listing_id,
  l.status,
  l.last_seen,
  l.external_url
FROM marketplace.listings l, p
WHERE l.status IN ('sold','removed')
  AND l.last_seen >= p.d0 AND l.last_seen < p.d1
  AND EXISTS (
    SELECT 1
    FROM marketplace.price_history ph
    WHERE ph.generation=l.generation AND ph.listing_id=l.listing_id
      AND ph.source='audit-stale'
      AND ph.status=l.status
      AND ph.observed_at >= p.d0 AND ph.observed_at < p.d1
  )
ORDER BY l.last_seen DESC
LIMIT 200;
```

---

## 5) Inactive event log: last known inactive state for terminal rows

```sql
WITH p AS (
  SELECT date_trunc('day', now() AT TIME ZONE 'UTC') AS d0,
         date_trunc('day', now() AT TIME ZONE 'UTC') + interval '1 day' AS d1
),
term_today AS (
  SELECT l.generation, l.listing_id, l.status, l.last_seen, l.external_url
  FROM marketplace.listings l, p
  WHERE l.status IN ('sold','removed')
    AND l.last_seen >= p.d0 AND l.last_seen < p.d1
),
last_inact AS (
  SELECT
    t.*,
    ie.is_inactive AS last_inactive_state,
    ie.observed_at AS last_inactive_ts,
    ie.evidence    AS last_inactive_evidence
  FROM term_today t
  LEFT JOIN LATERAL (
    SELECT is_inactive, observed_at, evidence
    FROM marketplace.listing_inactive_state_events
    WHERE generation=t.generation AND listing_id=t.listing_id
      AND observed_by='audit-stale'
      AND observed_at <= t.last_seen
    ORDER BY observed_at DESC, event_id DESC
    LIMIT 1
  ) ie ON true
)
SELECT *
FROM last_inact
ORDER BY last_seen DESC
LIMIT 200;
```

