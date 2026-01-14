-- 00_audit_primitives_public.sql
-- Public-release, minimal certification primitives for T0 feature stores in Postgres.
--
-- This module provides:
--   - audit.t0_viewdef_baseline           (hashes of view/MV definitions in the entrypoint closure)
--   - audit.t0_dataset_hash_baseline      (per-day dataset hashes for the entrypoint)
--   - audit.t0_cert_registry              (latest certification status)
--   - audit.capture_viewdef_baseline(...) (captures/updates viewdef baselines)
--   - audit.dataset_sha256(...)           (computes a deterministic per-day dataset hash)
--   - audit.rebaseline_last_n_days(...)   (updates dataset hash baselines for last N days)
--   - audit.run_t0_cert_trainer_derived_store_v1(p_check_days)
--   - audit.require_certified_strict(...) (fail-closed guard)
--
-- Notes:
-- - This is intentionally generic and platform-agnostic.
-- - It assumes entrypoints expose an `edited_date` column for T0 hashing.
-- - The example certification procedure name is kept for continuity; rename freely.
--
-- Dependencies:
--   - pgcrypto extension (for SHA256)
--
-- Security:
--   - No secrets/credentials are used or required by this module.

BEGIN;

CREATE SCHEMA IF NOT EXISTS audit;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------
-- Baseline tables
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audit.t0_viewdef_baseline (
  entrypoint      text        NOT NULL,
  object_fqn      text        NOT NULL,
  relkind         char(1)     NOT NULL,
  viewdef_sha256  text        NOT NULL,
  captured_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entrypoint, object_fqn)
);

CREATE TABLE IF NOT EXISTS audit.t0_dataset_hash_baseline (
  entrypoint      text        NOT NULL,
  t0_day          date        NOT NULL,
  sample_limit    int         NOT NULL,
  dataset_sha256  text        NOT NULL,
  computed_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entrypoint, t0_day, sample_limit)
);

CREATE TABLE IF NOT EXISTS audit.t0_cert_registry (
  entrypoint     text        PRIMARY KEY,
  status         text        NOT NULL,          -- 'certified' | 'failed'
  certified_at   timestamptz NOT NULL DEFAULT now(),
  checked_days   int         NOT NULL DEFAULT 0,
  dataset_days   int         NOT NULL DEFAULT 0,
  notes          text        NULL,
  details        jsonb       NULL
);

-- ---------------------------------------------------------------------
-- Helper: capture view/MV definition baselines in the entrypoint closure
-- ---------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE audit.capture_viewdef_baseline(p_entrypoint regclass)
LANGUAGE plpgsql
AS $$
BEGIN
  DELETE FROM audit.t0_viewdef_baseline
  WHERE entrypoint = p_entrypoint::text;

  WITH RECURSIVE rels(oid) AS (
    SELECT p_entrypoint::oid
    UNION
    SELECT c2.oid
    FROM rels r
    JOIN pg_rewrite w ON w.ev_class = r.oid
    JOIN pg_depend  d ON d.objid = w.oid
    JOIN pg_class   c2 ON c2.oid = d.refobjid
    WHERE c2.relkind IN ('v','m')
  ),
  objs AS (
    SELECT DISTINCT r.oid, n.nspname, c.relname, c.relkind
    FROM rels r
    JOIN pg_class c ON c.oid = r.oid
    JOIN pg_namespace n ON n.oid = c.relnamespace
  )
  INSERT INTO audit.t0_viewdef_baseline(entrypoint, object_fqn, relkind, viewdef_sha256)
  SELECT
    p_entrypoint::text AS entrypoint,
    o.nspname||'.'||o.relname AS object_fqn,
    o.relkind,
    encode(digest(convert_to(pg_get_viewdef(o.oid,true),'utf8'),'sha256'),'hex') AS viewdef_sha256
  FROM objs o
  ON CONFLICT (entrypoint, object_fqn)
  DO UPDATE SET
    viewdef_sha256 = EXCLUDED.viewdef_sha256,
    captured_at = now();
END;
$$;

-- ---------------------------------------------------------------------
-- Helper: compute a deterministic per-day dataset hash for an entrypoint
-- ---------------------------------------------------------------------

-- Requirements on the entrypoint:
--   - exposes edited_date (timestamptz or timestamp)
--   - exposes generation (int), listing_id (bigint), t0 (timestamptz) for deterministic ordering
--
-- If your store uses different key column names, either:
--   (a) create a wrapper view exposing these columns, or
--   (b) adapt this function accordingly.
CREATE OR REPLACE FUNCTION audit.dataset_sha256(
  p_entrypoint   regclass,
  p_t0_day       date,
  p_sample_limit int DEFAULT 2000
) RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
  sql text;
  out_sha text;
BEGIN
  sql := format($fmt$
    SELECT encode(
      digest(
        convert_to(
          COALESCE(
            string_agg(to_jsonb(t)::text, E'\n' ORDER BY t.generation, t.listing_id, t.t0),
            ''
          ),
          'utf8'
        ),
        'sha256'
      ),
      'hex'
    )
    FROM (
      SELECT *
      FROM %s
      WHERE edited_date IS NOT NULL
        AND edited_date::date = $1
      ORDER BY generation, listing_id, t0
      LIMIT $2
    ) t
  $fmt$, p_entrypoint);

  EXECUTE sql INTO out_sha USING p_t0_day, p_sample_limit;
  RETURN out_sha;
END;
$$;

-- ---------------------------------------------------------------------
-- Helper: UPSERT dataset hash baselines for the last N distinct days
-- ---------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE audit.rebaseline_last_n_days(
  p_entrypoint    regclass,
  p_n             int,
  p_sample_limit  int DEFAULT 2000
)
LANGUAGE plpgsql
AS $$
BEGIN
  EXECUTE format($fmt$
    WITH picked AS (
      SELECT edited_date::date AS t0_day
      FROM (
        SELECT DISTINCT edited_date
        FROM %s
        WHERE edited_date IS NOT NULL
      ) d
      ORDER BY t0_day DESC
      LIMIT %s
    )
    INSERT INTO audit.t0_dataset_hash_baseline(entrypoint, t0_day, sample_limit, dataset_sha256)
    SELECT
      %L AS entrypoint,
      p.t0_day,
      %s AS sample_limit,
      audit.dataset_sha256(%s, p.t0_day, %s) AS dataset_sha256
    FROM picked p
    ON CONFLICT (entrypoint, t0_day, sample_limit)
    DO UPDATE SET
      dataset_sha256 = EXCLUDED.dataset_sha256,
      computed_at = now();
  $fmt$,
    p_entrypoint,
    p_n,
    p_entrypoint::text,
    p_sample_limit,
    p_entrypoint,
    p_sample_limit
  );
END;
$$;

-- ---------------------------------------------------------------------
-- Fail-closed guard: require a recent successful certification
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION audit.require_certified_strict(
  p_entrypoint text,
  p_max_age    interval DEFAULT interval '24 hours'
) RETURNS boolean
LANGUAGE plpgsql
AS $$
DECLARE
  st text;
  ts timestamptz;
BEGIN
  SELECT status, certified_at
  INTO st, ts
  FROM audit.t0_cert_registry
  WHERE entrypoint = p_entrypoint;

  IF st IS NULL THEN
    RAISE EXCEPTION 'CERT GUARD FAIL: no certification record for %', p_entrypoint;
  END IF;

  IF st <> 'certified' THEN
    RAISE EXCEPTION 'CERT GUARD FAIL: % status=%', p_entrypoint, st;
  END IF;

  IF ts < now() - p_max_age THEN
    RAISE EXCEPTION 'CERT GUARD FAIL: % certification too old (% < now() - %)', p_entrypoint, ts, p_max_age;
  END IF;

  RETURN true;
END;
$$;

-- ---------------------------------------------------------------------
-- Example certification procedure for this store family
-- ---------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE audit.run_t0_cert_trainer_derived_store_v1(p_check_days int DEFAULT 10)
LANGUAGE plpgsql
AS $$
DECLARE
  entrypoint regclass := 'ml.trainer_derived_feature_store_t0_v1_v'::regclass;
  mismatches int;
  checked int;
BEGIN
  PERFORM 1 FROM audit.t0_viewdef_baseline WHERE entrypoint = entrypoint::text LIMIT 1;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'CERT FAIL: missing viewdef baselines for % (run audit.capture_viewdef_baseline first)', entrypoint::text;
  END IF;

  -- Viewdef drift check
  WITH cur AS (
    WITH RECURSIVE rels(oid) AS (
      SELECT entrypoint::oid
      UNION
      SELECT c2.oid
      FROM rels r
      JOIN pg_rewrite w ON w.ev_class = r.oid
      JOIN pg_depend  d ON d.objid = w.oid
      JOIN pg_class   c2 ON c2.oid = d.refobjid
      WHERE c2.relkind IN ('v','m')
    ),
    objs AS (
      SELECT DISTINCT r.oid, n.nspname, c.relname, c.relkind
      FROM rels r
      JOIN pg_class c ON c.oid = r.oid
      JOIN pg_namespace n ON n.oid = c.relnamespace
    )
    SELECT
      entrypoint::text AS entrypoint,
      o.nspname||'.'||o.relname AS object_fqn,
      o.relkind,
      encode(digest(convert_to(pg_get_viewdef(o.oid,true),'utf8'),'sha256'),'hex') AS viewdef_sha256
    FROM objs o
  ),
  drift AS (
    SELECT b.object_fqn
    FROM audit.t0_viewdef_baseline b
    JOIN cur c
      ON c.entrypoint = b.entrypoint
     AND c.object_fqn = b.object_fqn
    WHERE c.viewdef_sha256 <> b.viewdef_sha256
  )
  SELECT COUNT(*) INTO mismatches FROM drift;

  IF mismatches > 0 THEN
    INSERT INTO audit.t0_cert_registry(entrypoint, status, certified_at, checked_days, dataset_days, notes, details)
    VALUES (entrypoint::text, 'failed', now(), p_check_days, 0, 'viewdef drift detected', jsonb_build_object('viewdef_mismatches', mismatches))
    ON CONFLICT (entrypoint)
    DO UPDATE SET status='failed', certified_at=now(), checked_days=EXCLUDED.checked_days, dataset_days=EXCLUDED.dataset_days,
                  notes=EXCLUDED.notes, details=EXCLUDED.details;

    RAISE EXCEPTION 'CERT FAIL: viewdef drift detected for % (mismatched objects=%)', entrypoint::text, mismatches;
  END IF;

  -- Dataset hash drift check (strict; rebaseline last-N first if you allow backfills)
  WITH picked AS (
    SELECT edited_date::date AS t0_day
    FROM (
      SELECT DISTINCT edited_date
      FROM ml.trainer_derived_feature_store_t0_v1_v
      WHERE edited_date IS NOT NULL
    ) d
    ORDER BY t0_day DESC
    LIMIT p_check_days
  ),
  cur_hash AS (
    SELECT
      p.t0_day,
      audit.dataset_sha256(entrypoint, p.t0_day, 2000) AS dataset_sha256
    FROM picked p
  ),
  base AS (
    SELECT t0_day, dataset_sha256
    FROM audit.t0_dataset_hash_baseline
    WHERE entrypoint = entrypoint::text
      AND sample_limit = 2000
      AND t0_day IN (SELECT t0_day FROM picked)
  ),
  drift AS (
    SELECT c.t0_day
    FROM cur_hash c
    LEFT JOIN base b USING (t0_day)
    WHERE b.dataset_sha256 IS NULL OR b.dataset_sha256 <> c.dataset_sha256
  )
  SELECT COUNT(*) INTO mismatches FROM drift;

  SELECT COUNT(*) INTO checked FROM picked;

  IF mismatches > 0 THEN
    INSERT INTO audit.t0_cert_registry(entrypoint, status, certified_at, checked_days, dataset_days, notes, details)
    VALUES (entrypoint::text, 'failed', now(), p_check_days, checked, 'dataset hash drift detected', jsonb_build_object('dataset_mismatch_days', mismatches))
    ON CONFLICT (entrypoint)
    DO UPDATE SET status='failed', certified_at=now(), checked_days=EXCLUDED.checked_days, dataset_days=EXCLUDED.dataset_days,
                  notes=EXCLUDED.notes, details=EXCLUDED.details;

    RAISE EXCEPTION 'CERT FAIL: dataset hash drift detected for % (mismatched days=%)', entrypoint::text, mismatches;
  END IF;

  -- Success
  INSERT INTO audit.t0_cert_registry(entrypoint, status, certified_at, checked_days, dataset_days, notes, details)
  VALUES (entrypoint::text, 'certified', now(), p_check_days, checked, 'ok', jsonb_build_object('checked_days_present', checked))
  ON CONFLICT (entrypoint)
  DO UPDATE SET status='certified', certified_at=now(), checked_days=EXCLUDED.checked_days, dataset_days=EXCLUDED.dataset_days,
                notes=EXCLUDED.notes, details=EXCLUDED.details;
END;
$$;

COMMIT;
