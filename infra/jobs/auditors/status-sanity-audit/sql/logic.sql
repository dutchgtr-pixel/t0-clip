-- status_sanity_audit_logic_public.sql
-- Public release SQL logic for the Status Sanity Audit job.
--
-- This file provides:
-- - Candidate selection patterns (cadence-aware)
-- - Triage and rollups
-- - Review workflow helpers
-- - Example “safe apply” patterns (manual SQL)

-- -------------------------------------------------------------------
-- 1) Candidate selection: cadence-aware
-- -------------------------------------------------------------------
-- Use this pattern to avoid re-auditing the same listing too frequently.
-- Parameters (example):
--   :statuses       text[]
--   :generations    int[] (nullable)
--   :inactive_only  boolean
--   :cadence_days   int (0 disables cadence)
--   :limit          int (0 disables)
--
-- NOTE: In the public Go job, this selection is executed by the program;
-- this query is provided for transparency and customization.

WITH last_audit AS (
  SELECT
    generation,
    listing_id,
    max(observed_at) AS last_audited_at
  FROM marketplace.status_sanity_events
  GROUP BY 1,2
)
SELECT
  l.generation,
  l.listing_id,
  l.url,
  l.status,
  l.first_seen,
  l.edited_at,
  l.last_seen,
  l.sold_at,
  l.sold_price,
  l.live_price,
  l.is_inactive,
  l.is_bidding,
  la.last_audited_at
FROM marketplace.listings l
LEFT JOIN last_audit la
  ON la.generation = l.generation AND la.listing_id = l.listing_id
WHERE l.status = ANY(:statuses)
  AND (:generations IS NULL OR l.generation = ANY(:generations))
  AND (:inactive_only IS FALSE OR l.is_inactive IS TRUE)
  AND (
    :cadence_days = 0
    OR la.last_audited_at IS NULL
    OR la.last_audited_at < now() - make_interval(days => :cadence_days)
  )
ORDER BY l.first_seen ASC NULLS LAST, l.generation ASC, l.listing_id ASC
LIMIT NULLIF(:limit, 0);

-- -------------------------------------------------------------------
-- 2) Latest mismatch triage queue
-- -------------------------------------------------------------------
SELECT
  observed_at,
  generation,
  listing_id,
  main_status,
  detected_status,
  mismatch_status,
  mismatch_inactive,
  suggested_status,
  suggested_is_inactive,
  suggested_reason,
  http_status,
  page_ok,
  run_id,
  url
FROM marketplace.status_sanity_latest_mismatches_v
ORDER BY observed_at DESC
LIMIT 200;

-- -------------------------------------------------------------------
-- 3) Operational rollups
-- -------------------------------------------------------------------
-- 3.1 Mismatch counts by detected_status and main_status (last 7 days)
SELECT
  detected_status,
  main_status,
  count(*) AS n
FROM marketplace.status_sanity_events
WHERE observed_at >= now() - interval '7 days'
  AND (mismatch_status OR mismatch_inactive)
GROUP BY 1,2
ORDER BY n DESC;

-- 3.2 Error hot list (non-empty apply_error or evidence->error)
SELECT
  observed_at,
  generation,
  listing_id,
  detected_status,
  http_status,
  page_ok,
  apply_error,
  evidence_json->>'error' AS detect_error,
  run_id
FROM marketplace.status_sanity_events
WHERE observed_at >= now() - interval '7 days'
  AND (
    apply_error IS NOT NULL
    OR (evidence_json ? 'error')
  )
ORDER BY observed_at DESC
LIMIT 200;

-- 3.3 Run-level KPIs (last 30 runs)
SELECT
  started_at,
  finished_at,
  observed_by,
  scope,
  candidate_count,
  processed_count,
  mismatch_count,
  error_count,
  applied_count,
  apply_mode,
  run_id
FROM marketplace.status_sanity_runs
ORDER BY started_at DESC
LIMIT 30;

-- -------------------------------------------------------------------
-- 4) Review workflow
-- -------------------------------------------------------------------
-- Mark a mismatch row as reviewed (example).
UPDATE marketplace.status_sanity_events
SET
  review_action = 'accept',
  review_note   = 'Validated; safe correction applied downstream',
  reviewed_at   = now(),
  reviewed_by   = 'analyst@example'
WHERE event_id = :event_id;

-- Bulk-mark the latest mismatch per listing as “ignore” for 30 days by adding a note.
-- (If you want enforcement, create a suppressions table and join it in candidate selection.)
UPDATE marketplace.status_sanity_events e
SET
  review_action = 'ignore',
  review_note   = 'Known discrepancy; suppress until next vendor reconciliation',
  reviewed_at   = now(),
  reviewed_by   = 'analyst@example'
FROM marketplace.status_sanity_latest_mismatches_v m
WHERE e.generation = m.generation
  AND e.listing_id = m.listing_id
  AND e.observed_at = m.observed_at
  AND (m.mismatch_status OR m.mismatch_inactive);

-- -------------------------------------------------------------------
-- 5) Manual “safe apply” patterns (SQL)
-- -------------------------------------------------------------------
-- In most deployments, mutations should be executed by the auditor in apply-mode
-- so every change is coupled with an audit event. These statements are provided
-- only as transparent examples.

-- 5.1 Safe: set sold when detected sold
UPDATE marketplace.listings
SET
  status     = 'sold',
  sold_at    = :detected_sold_at,
  sold_price = :detected_sold_price,
  last_seen  = :observed_at
WHERE generation = :generation
  AND listing_id = :listing_id
  AND status IN ('live', 'stale_bucket');

-- 5.2 Safe: set removed when detected removed
UPDATE marketplace.listings
SET
  status    = 'removed',
  last_seen = :observed_at
WHERE generation = :generation
  AND listing_id = :listing_id
  AND status IN ('live', 'stale_bucket');

-- 5.3 Safe: synchronize inactive flag
UPDATE marketplace.listings
SET
  is_inactive = :detected_is_inactive,
  inactive_observed_at = CASE WHEN :detected_is_inactive THEN :observed_at ELSE NULL END,
  inactive_meta_edited_at = CASE WHEN :detected_is_inactive THEN COALESCE(:inactive_meta_edited_at, inactive_meta_edited_at) ELSE NULL END,
  inactive_evidence = CASE WHEN :detected_is_inactive THEN :evidence_json ELSE NULL END
WHERE generation = :generation
  AND listing_id = :listing_id;

-- -------------------------------------------------------------------
-- 6) Suggested dashboard queries
-- -------------------------------------------------------------------
-- 6.1 Mismatch rate over time (hourly)
SELECT
  date_trunc('hour', observed_at) AS hour,
  count(*) FILTER (WHERE mismatch_status OR mismatch_inactive) AS mismatches,
  count(*) AS audited,
  round(100.0 * count(*) FILTER (WHERE mismatch_status OR mismatch_inactive) / NULLIF(count(*), 0), 2) AS mismatch_pct
FROM marketplace.status_sanity_events
WHERE observed_at >= now() - interval '7 days'
GROUP BY 1
ORDER BY 1;

-- 6.2 Top listings by repeated mismatches (last 30 days)
SELECT
  generation,
  listing_id,
  count(*) AS mismatch_events
FROM marketplace.status_sanity_events
WHERE observed_at >= now() - interval '30 days'
  AND (mismatch_status OR mismatch_inactive)
GROUP BY 1,2
ORDER BY mismatch_events DESC
LIMIT 100;
