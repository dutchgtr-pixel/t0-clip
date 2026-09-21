-- Public-release schema for marketplace-ingest-template
-- -----------------------------------------------------
-- This schema is intentionally generic and contains no target-specific keys.

-- NOTE:
-- The ingest/audit binaries qualify tables using PG_SCHEMA (default: "public").
-- If you set PG_SCHEMA to something else, create that schema and create the
-- tables in that schema (or adjust these statements accordingly).

CREATE TABLE IF NOT EXISTS marketplace_listings (
    listing_id       TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    price            INTEGER,
    url              TEXT,
    description      TEXT,
    updated_at       TIMESTAMPTZ,
    location_city    TEXT,
    postal_code      TEXT,
    attribute_num_1  INTEGER,
    attribute_text_1 TEXT,
    attribute_text_2 TEXT,
    category         TEXT,
    score            DOUBLE PRECISION,
    price_per_unit   DOUBLE PRECISION,
    status           TEXT NOT NULL DEFAULT 'live',
    first_seen       TIMESTAMPTZ,
    last_seen        TIMESTAMPTZ,
    last_fetched     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS marketplace_listings_last_seen_idx
    ON marketplace_listings (last_seen DESC);

CREATE INDEX IF NOT EXISTS marketplace_listings_price_idx
    ON marketplace_listings (price);

-- Audit run metadata (for job traceability)
CREATE TABLE IF NOT EXISTS marketplace_audit_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    job_name      TEXT NOT NULL,
    adapter       TEXT,
    dry_run       BOOLEAN NOT NULL DEFAULT FALSE,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    checked_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS marketplace_audit_runs_started_at_idx
    ON marketplace_audit_runs (started_at DESC);
