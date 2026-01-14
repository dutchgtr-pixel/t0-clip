-- status_sanity_audit_schema_public.sql
-- Public release schema for the Status Sanity Audit job.
-- This is platform-agnostic: no site names, no platform identifiers.
--
-- Assumptions
-- ----------
-- 1) You have a canonical listings table, e.g. marketplace.listings, that stores the status used downstream.
-- 2) This auditor writes audit telemetry into marketplace.status_sanity_runs and marketplace.status_sanity_events.
--
-- Notes
-- -----
-- - This schema is intentionally flexible: status values are free-form TEXT.
-- - If you want hard enums, add CHECK constraints in your private deployment.

BEGIN;

CREATE SCHEMA IF NOT EXISTS marketplace;

-- -------------------------------------------------------------------
-- Canonical listings table (minimal public contract)
-- -------------------------------------------------------------------
-- If you already have a production listings table, keep yours and adapt
-- candidate selection queries to it. This table is provided as a runnable
-- public template only.
CREATE TABLE IF NOT EXISTS marketplace.listings (
  generation          int          NOT NULL,
  listing_id          bigint       NOT NULL,

  url                 text,
  status              text         NOT NULL,  -- e.g. live|sold|removed|stale_bucket|...
  first_seen          timestamptz,
  edited_at           timestamptz,
  last_seen           timestamptz,

  sold_at             timestamptz,
  sold_price          int,
  live_price          int,

  is_inactive         boolean,
  is_bidding          boolean,

  inactive_observed_at      timestamptz,
  inactive_meta_edited_at   timestamptz,
  inactive_evidence         jsonb,

  PRIMARY KEY (generation, listing_id)
);

CREATE INDEX IF NOT EXISTS listings_status_idx
  ON marketplace.listings (status);

CREATE INDEX IF NOT EXISTS listings_last_seen_idx
  ON marketplace.listings (last_seen DESC);

-- Optional provenance table (used for “first marked sold” and previous event context)
CREATE TABLE IF NOT EXISTS marketplace.listing_price_history (
  generation   int        NOT NULL,
  listing_id   bigint     NOT NULL,
  observed_at  timestamptz NOT NULL,
  status       text       NOT NULL,
  source       text,
  price        int,
  PRIMARY KEY (generation, listing_id, observed_at)
);

CREATE INDEX IF NOT EXISTS listing_price_history_lookup_idx
  ON marketplace.listing_price_history (generation, listing_id, observed_at DESC);

-- -------------------------------------------------------------------
-- Status sanity audit telemetry
-- -------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS marketplace.status_sanity_runs (
  run_id uuid PRIMARY KEY,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,

  observed_by text NOT NULL,
  note text,

  scope text NOT NULL,
  statuses text[] NOT NULL,
  generations int[] NULL,
  inactive_only boolean NOT NULL DEFAULT false,

  cadence_days int NOT NULL,
  force_run boolean NOT NULL DEFAULT false,

  apply_mode text NOT NULL DEFAULT 'none',
  allow_unsell boolean NOT NULL DEFAULT false,
  allow_revive boolean NOT NULL DEFAULT false,

  workers int NOT NULL,
  head_first boolean NOT NULL,
  allow_ui_fallback boolean NOT NULL,

  rps_start double precision NOT NULL,
  rps_max double precision NOT NULL,

  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,

  candidate_count int NOT NULL DEFAULT 0,
  processed_count int NOT NULL DEFAULT 0,
  mismatch_count int NOT NULL DEFAULT 0,
  error_count int NOT NULL DEFAULT 0,
  applied_count int NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS marketplace.status_sanity_events (
  event_id bigserial PRIMARY KEY,

  run_id uuid NOT NULL REFERENCES marketplace.status_sanity_runs(run_id) ON DELETE CASCADE,
  observed_at timestamptz NOT NULL DEFAULT now(),

  generation int NOT NULL,
  listing_id bigint NOT NULL,
  url text,

  http_status int,
  page_ok boolean,

  detected_status text NOT NULL,
  detected_price int,
  detected_price_src text,
  detected_sold_at timestamptz,

  detected_is_inactive boolean,
  inactive_key_seen boolean,
  inactive_meta_edited_at timestamptz,

  hydration_src text,
  evidence_json jsonb,

  main_status text,
  main_sold_at timestamptz,
  main_sold_price int,
  main_is_inactive boolean,
  main_last_seen timestamptz,

  first_marked_sold_at timestamptz,
  marked_sold_by text,
  sold_price_at_mark int,
  prev_event_at timestamptz,
  prev_status text,
  prev_source text,
  prev_price int,

  mismatch_status boolean NOT NULL DEFAULT false,
  mismatch_inactive boolean NOT NULL DEFAULT false,

  suggested_status text,
  suggested_is_inactive boolean,
  suggested_reason text,

  applied boolean NOT NULL DEFAULT false,
  applied_at timestamptz,
  apply_action text,
  apply_error text,

  review_action text,
  review_note text,
  reviewed_at timestamptz,
  reviewed_by text,

  UNIQUE(run_id, generation, listing_id)
);

CREATE INDEX IF NOT EXISTS status_sanity_events_listing_idx
  ON marketplace.status_sanity_events (generation, listing_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS status_sanity_events_run_idx
  ON marketplace.status_sanity_events (run_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS status_sanity_events_mismatch_idx
  ON marketplace.status_sanity_events (observed_at DESC)
  WHERE mismatch_status OR mismatch_inactive;

-- -------------------------------------------------------------------
-- Helper view: latest mismatches per listing
-- -------------------------------------------------------------------
CREATE OR REPLACE VIEW marketplace.status_sanity_latest_mismatches_v AS
WITH latest AS (
  SELECT DISTINCT ON (generation, listing_id)
    generation, listing_id, url,
    observed_at,
    detected_status, main_status,
    mismatch_status, mismatch_inactive,
    suggested_status, suggested_is_inactive, suggested_reason,
    http_status, page_ok,
    run_id
  FROM marketplace.status_sanity_events
  WHERE mismatch_status OR mismatch_inactive
  ORDER BY generation, listing_id, observed_at DESC
)
SELECT *
FROM latest;

COMMIT;
