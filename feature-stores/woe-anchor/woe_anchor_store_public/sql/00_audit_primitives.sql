-- 00_audit_primitives.sql
-- Minimal, public-friendly certification primitives used by the feature store.
-- If you already have an audit/certification framework, you can omit this file and
-- map the store's calls to your equivalent objects.

CREATE SCHEMA IF NOT EXISTS audit;

-- pgcrypto provides digest() for SHA256 hashing.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Stores the transitive closure of view definitions (views + matviews) required by an entrypoint.
CREATE TABLE IF NOT EXISTS audit.t0_viewdef_baseline (
  entrypoint     text        NOT NULL,
  object_fqn     text        NOT NULL,
  relkind        char(1)     NOT NULL,  -- 'v' (view) or 'm' (matview)
  viewdef_sha256 text        NOT NULL,
  captured_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entrypoint, object_fqn)
);

-- Stores per-day dataset hashes for a given certified entrypoint (T0 safe).
CREATE TABLE IF NOT EXISTS audit.t0_dataset_hash_baseline (
  entrypoint     text        NOT NULL,
  t0_day         date        NOT NULL,
  sample_limit   int         NOT NULL,
  dataset_sha256 text        NOT NULL,
  computed_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entrypoint, t0_day, sample_limit)
);

-- Certification registry: one row per entrypoint (plus optional alias rows).
CREATE TABLE IF NOT EXISTS audit.t0_cert_registry (
  entrypoint      text        PRIMARY KEY,
  status          text        NOT NULL, -- e.g., 'CERTIFIED', 'UNCERTIFIED', 'STALE'
  certified_at    timestamptz NOT NULL DEFAULT now(),
  viewdef_objects jsonb,
  dataset_days    int,
  notes           text
);

-- Compute a stable-ish SHA256 hash of a bounded sample for a given (entrypoint, t0_day).
-- Contract: the entrypoint view must expose an `edited_date` column (timestamptz/date-like).
CREATE OR REPLACE FUNCTION audit.dataset_sha256(p_entrypoint regclass, p_t0_day date, p_sample_limit int)
RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
  v_sql text;
  v_hash text;
BEGIN
  -- Use a stable ordering where possible. If the entrypoint has `generation` and `listing_id`,
  -- ordering by those is ideal. Otherwise we fall back to ordering by the row JSON text.
  v_sql := format($fmt$
    WITH s AS (
      SELECT to_jsonb(t) AS j
      FROM %s t
      WHERE (t.edited_date)::date = $1
      ORDER BY
        (t.edited_date),
        (t.generation) NULLS LAST,
        (t.listing_id) NULLS LAST,
        (to_jsonb(t))::text
      LIMIT $2
    )
    SELECT encode(digest(COALESCE(string_agg(j::text, E'\n' ORDER BY j::text), ''), 'sha256'), 'hex')
    FROM s
  $fmt$, p_entrypoint::text);

  EXECUTE v_sql INTO v_hash USING p_t0_day, p_sample_limit;
  RETURN v_hash;
END $$;

-- Fail-closed guard for production consumers.
CREATE OR REPLACE FUNCTION audit.require_certified_strict(p_entrypoint text, p_max_age interval)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  v_status text;
  v_certified_at timestamptz;
BEGIN
  SELECT status, certified_at
  INTO v_status, v_certified_at
  FROM audit.t0_cert_registry
  WHERE entrypoint = p_entrypoint;

  IF v_status IS NULL THEN
    RAISE EXCEPTION 'Entry point % is not present in audit.t0_cert_registry', p_entrypoint;
  END IF;

  IF v_status <> 'CERTIFIED' THEN
    RAISE EXCEPTION 'Entry point % is not certified (status=%)', p_entrypoint, v_status;
  END IF;

  IF v_certified_at < now() - p_max_age THEN
    RAISE EXCEPTION 'Entry point % certification is stale (certified_at=%; max_age=%)', p_entrypoint, v_certified_at, p_max_age;
  END IF;
END $$;
