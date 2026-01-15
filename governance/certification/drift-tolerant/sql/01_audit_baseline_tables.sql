-- 01_audit_baseline_tables.sql
-- Creates baseline + registry tables (safe if already exist).

CREATE SCHEMA IF NOT EXISTS audit;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
  viewdef_objects int NOT NULL DEFAULT 0,
  dataset_days    int NOT NULL DEFAULT 0,
  notes           text
);
