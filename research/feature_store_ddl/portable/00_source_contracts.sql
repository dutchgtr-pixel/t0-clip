-- Newly authored reference contracts, not a production schema dump.
-- Apply only to a disposable empty PostgreSQL database you explicitly control.
CREATE SCHEMA ref_input;
CREATE SCHEMA ref_feature;
CREATE SCHEMA ref_audit;

CREATE TABLE ref_input.decision (
    entity_id text PRIMARY KEY,
    t0 timestamptz NOT NULL,
    label_cutoff timestamptz NOT NULL CHECK (label_cutoff >= t0),
    region_key text NOT NULL,
    cohort_key text NOT NULL
);
CREATE TABLE ref_input.listing_version (
    entity_id text NOT NULL REFERENCES ref_input.decision,
    revision integer NOT NULL,
    observed_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL CHECK (available_at >= observed_at),
    content_digest text NOT NULL,
    ask_price numeric CHECK (ask_price > 0),
    condition_score real,
    PRIMARY KEY (entity_id, revision)
);
CREATE TABLE ref_input.image_report_version (
    entity_id text NOT NULL REFERENCES ref_input.decision,
    slot integer NOT NULL CHECK (slot BETWEEN 0 AND 7),
    revision integer NOT NULL,
    evidence_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL CHECK (available_at >= evidence_at),
    source_digest text NOT NULL,
    encoder_digest text NOT NULL,
    damage_score real CHECK (damage_score BETWEEN 0 AND 1),
    image_vector real[] CHECK (array_length(image_vector, 1) = 4),
    report_vector real[] CHECK (array_length(report_vector, 1) = 4),
    PRIMARY KEY (entity_id, slot, revision)
);
-- Four-dimensional fixture vectors exercise contracts only; real adapters use
-- the separately declared historical encoder dimensions, without pad/truncate.
CREATE TABLE ref_input.geo_release (
    region_key text NOT NULL,
    release_id text NOT NULL,
    valid_from timestamptz NOT NULL,
    published_at timestamptz NOT NULL,
    density numeric,
    PRIMARY KEY (region_key, release_id)
);
CREATE TABLE ref_input.market_observation (
    observation_id text PRIMARY KEY,
    source_entity_id text NOT NULL,
    cohort_key text NOT NULL,
    event_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL CHECK (available_at >= event_at),
    ask_price numeric NOT NULL CHECK (ask_price > 0),
    duration_hours numeric NOT NULL CHECK (duration_hours >= 0)
);
CREATE TABLE ref_input.outcome_version (
    entity_id text NOT NULL REFERENCES ref_input.decision,
    revision integer NOT NULL,
    observed_until timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    event_at timestamptz,
    PRIMARY KEY (entity_id, revision),
    CHECK (event_at IS NULL OR event_at <= observed_until)
);
CREATE TABLE ref_input.fitted_anchor (
    artifact_id text NOT NULL,
    cohort_key text NOT NULL,
    fitted_through timestamptz NOT NULL,
    label_available_through timestamptz NOT NULL,
    published_at timestamptz NOT NULL,
    fold_policy text NOT NULL CHECK (fold_policy IN ('forward_disjoint', 'out_of_fold')),
    log_odds numeric NOT NULL,
    PRIMARY KEY (artifact_id, cohort_key)
);
