-- 00_prereqs.sql
-- Minimal prerequisites for the socio_economic_store package.
-- NOTE: Public release: a minimal audit/cert helper layer is provided in
--   sql/00a_minimal_audit_framework.sql
-- If you already have an audit framework in your DB, you can skip that file.

CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS audit;

-- Used for SHA256 viewdef baselines
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Registry tables (created IF NOT EXISTS; safe to re-run)
CREATE TABLE IF NOT EXISTS audit.t0_viewdef_baseline (
  entrypoint     text NOT NULL,
  object_fqn     text NOT NULL,
  relkind        char NOT NULL,
  viewdef_sha256 text NOT NULL,
  captured_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(entrypoint, object_fqn)
);

CREATE TABLE IF NOT EXISTS audit.t0_dataset_hash_baseline (
  entrypoint     text NOT NULL,
  t0_day         date NOT NULL,
  sample_limit   int  NOT NULL,
  dataset_sha256 text NOT NULL,
  computed_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(entrypoint, t0_day, sample_limit)
);

CREATE TABLE IF NOT EXISTS audit.t0_cert_registry (
  entrypoint      text PRIMARY KEY,
  status          text NOT NULL,
  certified_at    timestamptz NOT NULL DEFAULT now(),
  viewdef_objects int NOT NULL,
  dataset_days    int NOT NULL,
  notes           text
);
