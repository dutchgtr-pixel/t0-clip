-- sql/stale_sweeper.sql
--
-- Public template: DB-side “mark stale” sweeper + trigger wiring.
--
-- This pattern can enforce an invariant like:
--   No row may remain status='live' once it is older than a cutoff duration,
--   where “older” is defined by edited_date.
--
-- IMPORTANT:
-- - Review SECURITY DEFINER usage for your environment.
-- - Tune batch sizes and throttle windows for your workload.
-- - Consider running the sweeper via a scheduled job instead of a trigger in high-write systems.

CREATE SCHEMA IF NOT EXISTS marketplace;

-- Watermark table: stores last run timestamp for throttling/observability
CREATE TABLE IF NOT EXISTS marketplace.maintenance_watermark (
  job_name     text        PRIMARY KEY,
  last_run_at  timestamptz,
  last_rows    integer,
  last_note    text,
  last_run_by  text,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Run log table: append-only audit log of sweeper calls
CREATE TABLE IF NOT EXISTS marketplace.stale_sweep_runs (
  run_id         bigserial   PRIMARY KEY,
  ran_at         timestamptz  NOT NULL,
  reason         text         NOT NULL,
  eligible_total bigint       NOT NULL,
  updated_rows   bigint       NOT NULL,
  dry_run        boolean      NOT NULL,
  min_gap        interval     NOT NULL,
  batch          integer      NOT NULL
);

-- Sweeper function: marks eligible LIVE rows as stale (default: older21days).
CREATE OR REPLACE FUNCTION marketplace.sweep_mark_stale(
  p_min_gap     interval DEFAULT interval '15 minutes',
  p_batch       integer  DEFAULT 5000,
  p_dry_run     boolean  DEFAULT true,
  p_cutoff      interval DEFAULT interval '21 days',
  p_stale_status text    DEFAULT 'older21days'
)
RETURNS TABLE(ran boolean, reason text, eligible_total bigint, updated_rows bigint, last_run_at timestamptz)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO marketplace, public
AS $function$
DECLARE
  v_lock_key bigint := 88442121; -- arbitrary constant; change if you have a lock-key registry
  v_prev_run timestamptz;
  v_now      timestamptz := clock_timestamp();
  v_bad      boolean := false;
BEGIN
  -- Transaction-scoped lock (auto-released on commit/rollback)
  IF NOT pg_try_advisory_xact_lock(v_lock_key) THEN
    INSERT INTO marketplace.stale_sweep_runs
      (ran_at, reason, eligible_total, updated_rows, dry_run, min_gap, batch)
    VALUES
      (v_now, 'locked', 0, 0, p_dry_run, p_min_gap, p_batch);

    ran := false; reason := 'locked'; eligible_total := 0; updated_rows := 0; last_run_at := NULL;
    RETURN NEXT; RETURN;
  END IF;

  SELECT mw.last_run_at
    INTO v_prev_run
  FROM marketplace.maintenance_watermark mw
  WHERE mw.job_name = 'mark_stale';

  IF v_prev_run IS NOT NULL AND v_prev_run > v_now - p_min_gap THEN
    INSERT INTO marketplace.stale_sweep_runs
      (ran_at, reason, eligible_total, updated_rows, dry_run, min_gap, batch)
    VALUES
      (v_now, 'throttled', 0, 0, p_dry_run, p_min_gap, p_batch);

    ran := false; reason := 'throttled'; eligible_total := 0; updated_rows := 0; last_run_at := v_prev_run;
    RETURN NEXT; RETURN;
  END IF;

  SELECT COUNT(*)
    INTO eligible_total
  FROM marketplace.listings l
  WHERE l.status = 'live'
    AND l.edited_date IS NOT NULL
    AND l.edited_date <= v_now - p_cutoff;

  IF eligible_total = 0 THEN
    INSERT INTO marketplace.maintenance_watermark(job_name, last_run_at, last_rows, last_note, last_run_by)
    VALUES ('mark_stale', v_now, 0, 'no_eligible', current_user)
    ON CONFLICT (job_name) DO UPDATE
      SET last_run_at = EXCLUDED.last_run_at,
          last_rows   = EXCLUDED.last_rows,
          last_note   = EXCLUDED.last_note,
          last_run_by = EXCLUDED.last_run_by,
          updated_at  = now();

    INSERT INTO marketplace.stale_sweep_runs
      (ran_at, reason, eligible_total, updated_rows, dry_run, min_gap, batch)
    VALUES
      (v_now, 'no_eligible', 0, 0, p_dry_run, p_min_gap, p_batch);

    ran := false; reason := 'no_eligible'; updated_rows := 0; last_run_at := v_now;
    RETURN NEXT; RETURN;
  END IF;

  IF p_dry_run THEN
    INSERT INTO marketplace.maintenance_watermark(job_name, last_run_at, last_rows, last_note, last_run_by)
    VALUES ('mark_stale', v_now, 0, 'dry_run', current_user)
    ON CONFLICT (job_name) DO UPDATE
      SET last_run_at = EXCLUDED.last_run_at,
          last_rows   = EXCLUDED.last_rows,
          last_note   = EXCLUDED.last_note,
          last_run_by = EXCLUDED.last_run_by,
          updated_at  = now();

    INSERT INTO marketplace.stale_sweep_runs
      (ran_at, reason, eligible_total, updated_rows, dry_run, min_gap, batch)
    VALUES
      (v_now, 'dry_run', eligible_total, 0, p_dry_run, p_min_gap, p_batch);

    ran := false; reason := 'dry_run'; updated_rows := 0; last_run_at := v_now;
    RETURN NEXT; RETURN;
  END IF;

  WITH cand AS (
    SELECT l.tableoid, l.ctid
    FROM marketplace.listings l
    WHERE l.status = 'live'
      AND l.edited_date IS NOT NULL
      AND l.edited_date <= v_now - p_cutoff
    ORDER BY l.edited_date ASC
    LIMIT GREATEST(p_batch, 1)
  ),
  bad AS (
    SELECT 1
    FROM cand c
    JOIN marketplace.listings l
      ON l.tableoid = c.tableoid AND l.ctid = c.ctid
    WHERE NOT (
      l.status = 'live'
      AND l.edited_date IS NOT NULL
      AND l.edited_date <= v_now - p_cutoff
    )
    LIMIT 1
  ),
  upd AS (
    UPDATE marketplace.listings l
    SET status     = p_stale_status,
        last_seen  = v_now,
        updated_at = v_now
    FROM cand c
    WHERE l.tableoid = c.tableoid AND l.ctid = c.ctid
    RETURNING 1
  )
  SELECT (SELECT COUNT(*) FROM upd), EXISTS(SELECT 1 FROM bad)
    INTO updated_rows, v_bad;

  -- Hard assertions / guardrails
  IF updated_rows > GREATEST(p_batch, 1) THEN
    RAISE EXCEPTION 'sweep_mark_stale: updated_rows % exceeds batch %', updated_rows, p_batch;
  END IF;

  IF updated_rows > eligible_total THEN
    RAISE EXCEPTION 'sweep_mark_stale: updated_rows % exceeds eligible_total %', updated_rows, eligible_total;
  END IF;

  IF v_bad THEN
    RAISE EXCEPTION 'sweep_mark_stale safety violation: candidate set drifted before update';
  END IF;

  INSERT INTO marketplace.maintenance_watermark(job_name, last_run_at, last_rows, last_note, last_run_by)
  VALUES ('mark_stale', v_now, updated_rows::integer, 'ok', current_user)
  ON CONFLICT (job_name) DO UPDATE
    SET last_run_at = EXCLUDED.last_run_at,
        last_rows   = EXCLUDED.last_rows,
        last_note   = EXCLUDED.last_note,
        last_run_by = EXCLUDED.last_run_by,
        updated_at  = now();

  INSERT INTO marketplace.stale_sweep_runs
    (ran_at, reason, eligible_total, updated_rows, dry_run, min_gap, batch)
  VALUES
    (v_now, 'updated', eligible_total, updated_rows, p_dry_run, p_min_gap, p_batch);

  ran := true; reason := 'updated'; last_run_at := v_now;
  RETURN NEXT; RETURN;
END;
$function$;

-- Trigger function: optional "heartbeat" that invokes the sweeper.
CREATE OR REPLACE FUNCTION marketplace.trg_heartbeat_mark_stale()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO marketplace, public
AS $function$
BEGIN
  -- Prevent recursion due to our own UPDATEs
  IF pg_trigger_depth() > 1 THEN
    RETURN NULL;
  END IF;

  -- Run a throttled sweep; batch size tuned for safety
  PERFORM marketplace.sweep_mark_stale(
    p_min_gap := interval '15 minutes',
    p_batch   := 5000,
    p_dry_run := false
  );

  RETURN NULL;
END;
$function$;

-- Statement-level trigger on marketplace.listings
DROP TRIGGER IF EXISTS zzz_heartbeat_mark_stale ON marketplace.listings;
CREATE TRIGGER zzz_heartbeat_mark_stale
AFTER INSERT OR UPDATE ON marketplace.listings
FOR EACH STATEMENT
EXECUTE FUNCTION marketplace.trg_heartbeat_mark_stale();
