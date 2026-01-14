-- 00a_minimal_audit_framework.sql
-- Minimal, self-contained audit/certification helpers required by this package.
--
-- If you already have an audit framework in your database, you can skip this file.
--
-- Provides:
--   - audit.dataset_sha256(regclass, date, int [, text])
--   - audit.assert_viewdef_baseline(text)
--   - audit.require_certified_strict(text, interval)

CREATE SCHEMA IF NOT EXISTS audit;

-- Required for digest(..., 'sha256')
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -----------------------------------------------------------------------------
-- 1) Deterministic per-day dataset fingerprinting (sample-based)
-- -----------------------------------------------------------------------------
-- Computes a SHA256 fingerprint over a deterministic sample of rows from a relation.
-- This is designed for T0 feature-store entrypoints that expose:
--   - generation (int)
--   - listing_id (int/bigint/text)
--   - edited_date (timestamp/timestamptz) OR a compatible T0 timestamp column
--
-- Parameters:
--   p_rel         : the relation (view/matview/table) to hash
--   p_t0_day      : the day bucket to hash (based on p_t0_col::date)
--   p_sample_limit: number of rows included (ORDER BY generation, listing_id, p_t0_col)
--   p_t0_col      : name of the T0 timestamp column (default 'edited_date')
--
-- Notes:
--   - This is intentionally sample-based; it is a drift sentry, not a full checksum.
--   - Any schema/column-order change will change the hash. That is desirable: schema drift
--     should force re-certification.
CREATE OR REPLACE FUNCTION audit.dataset_sha256(
  p_rel          regclass,
  p_t0_day       date,
  p_sample_limit int,
  p_t0_col       text DEFAULT 'edited_date'
) RETURNS text
LANGUAGE plpgsql
AS $$
DECLARE
  v_sql  text;
  v_hash text;
BEGIN
  v_sql := format($fmt$
    WITH sample AS (
      SELECT *
      FROM %s
      WHERE %I IS NOT NULL
        AND (%I)::date = $1
      ORDER BY generation, listing_id, %I
      LIMIT $2
    )
    SELECT encode(
             digest(
               COALESCE(string_agg(row_to_json(sample)::text, E'\n' ORDER BY generation, listing_id, %I),''),
               'sha256'
             ),
             'hex'
           )
    FROM sample;
  $fmt$, p_rel, p_t0_col, p_t0_col, p_t0_col, p_t0_col);

  EXECUTE v_sql USING p_t0_day, p_sample_limit INTO v_hash;
  RETURN v_hash;
END;
$$;

-- -----------------------------------------------------------------------------
-- 2) View-definition drift assertion (certified closure integrity)
-- -----------------------------------------------------------------------------
-- Asserts that the current view definitions of all objects captured in
-- audit.t0_viewdef_baseline match their recorded SHA256 digests.
CREATE OR REPLACE FUNCTION audit.assert_viewdef_baseline(p_entrypoint text)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  n_baseline int;
  n_mismatch int;
  mismatch_list text;
BEGIN
  SELECT COUNT(*) INTO n_baseline
  FROM audit.t0_viewdef_baseline
  WHERE entrypoint = p_entrypoint;

  IF n_baseline = 0 THEN
    RAISE EXCEPTION 'No viewdef baselines found for entrypoint=% (run sql/08_baseline_viewdefs.sql)', p_entrypoint;
  END IF;

  WITH cur AS (
    SELECT
      b.object_fqn,
      b.viewdef_sha256 AS baseline_sha,
      encode(digest(convert_to(pg_get_viewdef(b.object_fqn::regclass, true),'utf8'),'sha256'),'hex') AS current_sha
    FROM audit.t0_viewdef_baseline b
    WHERE b.entrypoint = p_entrypoint
  ),
  mism AS (
    SELECT object_fqn
    FROM cur
    WHERE baseline_sha <> current_sha
    ORDER BY object_fqn
  )
  SELECT COUNT(*) INTO n_mismatch FROM mism;

  IF n_mismatch > 0 THEN
    SELECT string_agg(object_fqn, ', ' ORDER BY object_fqn)
    INTO mismatch_list
    FROM (SELECT object_fqn FROM mism LIMIT 10) x;

    RAISE EXCEPTION 'Viewdef drift detected for entrypoint=% (mismatched_objects=%). Re-run baseline + certification.',
      p_entrypoint, mismatch_list;
  END IF;
END;
$$;

-- -----------------------------------------------------------------------------
-- 3) Fail-closed consumption guard
-- -----------------------------------------------------------------------------
-- The guard is intended to be invoked inside a view via:
--   SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');
--
-- It fails closed if:
--   - the entrypoint has no registry record
--   - status is not CERTIFIED
--   - certified_at is older than max_age
--   - viewdef drift is detected
CREATE OR REPLACE FUNCTION audit.require_certified_strict(p_entrypoint text, p_max_age interval)
RETURNS boolean
LANGUAGE plpgsql
AS $$
DECLARE
  v_status text;
  v_certified_at timestamptz;
BEGIN
  -- Ensure the certified closure has not changed.
  PERFORM audit.assert_viewdef_baseline(p_entrypoint);

  SELECT status, certified_at
  INTO v_status, v_certified_at
  FROM audit.t0_cert_registry
  WHERE entrypoint = p_entrypoint;

  IF v_status IS NULL THEN
    RAISE EXCEPTION 'Entrypoint % is not registered in audit.t0_cert_registry (not certified)', p_entrypoint;
  END IF;

  IF v_status <> 'CERTIFIED' THEN
    RAISE EXCEPTION 'Entrypoint % is not CERTIFIED (status=%)', p_entrypoint, v_status;
  END IF;

  IF v_certified_at < (now() - p_max_age) THEN
    RAISE EXCEPTION 'Entrypoint % certification is stale (certified_at=%, max_age=%)', p_entrypoint, v_certified_at, p_max_age;
  END IF;

  RETURN true;
END;
$$;
