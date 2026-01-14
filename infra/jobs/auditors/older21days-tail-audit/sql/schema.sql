-- Public template schema for the marketplace audit job.
--
-- This schema is intentionally minimal and platform-agnostic.
-- It supports:
--   - a main listings table (job input + terminal updates)
--   - a sparse price/history table (daily baselines + change events)
--   - an append-only inactive state events table
--
-- NOTE: Review and adjust types/constraints for your environment.

CREATE SCHEMA IF NOT EXISTS marketplace;

-- Main table
CREATE TABLE IF NOT EXISTS marketplace.listings (
    generation                      integer      NOT NULL,
    listing_id                       bigint       NOT NULL,
    external_url                     text,
    status                           text         NOT NULL DEFAULT 'live',

    price                            integer,
    sold_price                       integer,
    sold_date                        timestamptz,

    first_seen                       timestamptz,
    last_seen                        timestamptz,
    edited_date                      timestamptz,

    marketplace_is_inactive          boolean      NOT NULL DEFAULT false,
    marketplace_inactive_observed_at timestamptz,
    marketplace_inactive_evidence    text,
    marketplace_inactive_source_timestamp timestamptz,

    marketplace_is_bidding           boolean      NOT NULL DEFAULT false,
    marketplace_bidding_evidence     text,

    created_at                       timestamptz  NOT NULL DEFAULT now(),
    updated_at                       timestamptz  NOT NULL DEFAULT now(),

    PRIMARY KEY (generation, listing_id)
);

CREATE INDEX IF NOT EXISTS listings_status_idx
    ON marketplace.listings (generation, status);

CREATE INDEX IF NOT EXISTS listings_last_seen_idx
    ON marketplace.listings (generation, last_seen DESC);

-- History table (sparse)
CREATE TABLE IF NOT EXISTS marketplace.price_history (
    generation   integer     NOT NULL,
    listing_id   bigint      NOT NULL,
    observed_at  timestamptz NOT NULL,
    observed_hour timestamptz GENERATED ALWAYS AS (date_trunc('hour', observed_at)) STORED,

    price        integer     NOT NULL,
    status       text        NOT NULL,
    source       text        NOT NULL,

    PRIMARY KEY (generation, listing_id, observed_at),

    -- Deduplicate noisy probes: at most one row per hour per price.
    CONSTRAINT price_history_hour_dedupe UNIQUE (generation, listing_id, observed_hour, price)
);

CREATE INDEX IF NOT EXISTS price_history_listing_time_idx
    ON marketplace.price_history (generation, listing_id, observed_at DESC);

-- Inactive state events (append-only)
CREATE TABLE IF NOT EXISTS marketplace.listing_inactive_state_events (
    event_id        bigserial   PRIMARY KEY,
    generation      integer     NOT NULL,
    listing_id      bigint      NOT NULL,
    observed_at     timestamptz NOT NULL,
    is_inactive     boolean     NOT NULL,
    source_timestamp timestamptz,

    observed_by     text        NOT NULL,
    main_status     text        NOT NULL,
    evidence        text,
    http_status     integer
);

CREATE INDEX IF NOT EXISTS inactive_events_listing_time_idx
    ON marketplace.listing_inactive_state_events (generation, listing_id, observed_at DESC);
