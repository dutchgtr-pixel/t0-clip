-- 07_create_cert_functions.sql
-- Certification assertion + runner for the SOCIO+MARKET certified entrypoint.
--
-- This file creates:
--   - audit.assert_t0_certified_socio_market_v1(p_check_days int)
--   - audit.run_t0_cert_socio_market_v1(p_check_days int)
--
-- Dependencies:
--   This package includes minimal versions in sql/00a_minimal_audit_framework.sql.
--   If you use your own audit framework, ensure equivalent functions exist:
--     - audit.dataset_sha256(regclass, date, int [, text]) -> text
--     - audit.require_certified_strict(text, interval)
--     - audit.assert_viewdef_baseline(text)
--
-- The runner writes to audit.t0_cert_registry.

CREATE OR REPLACE FUNCTION audit.assert_t0_certified_socio_market_v1(p_check_days int DEFAULT 30)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  ep regclass := 'ml.socio_market_feature_store_t0_v1_v'::regclass;
  token_ct int;
  socio_bad bigint;
  market_bad bigint;
  base_days int;
  drift_ct int;
BEGIN
  -- Gate A2: time-of-query tokens in closure are forbidden
  WITH RECURSIVE rels(oid) AS (
    SELECT ep
    UNION
    SELECT c2.oid
    FROM rels r
    JOIN pg_rewrite w ON w.ev_class = r.oid
    JOIN pg_depend  d ON d.objid = w.oid
    JOIN pg_class   c2 ON c2.oid = d.refobjid
    WHERE c2.relkind IN ('v','m')
  )
  SELECT COUNT(*) INTO token_ct
  FROM rels
  WHERE lower(pg_get_viewdef(oid,true)) ~
    '\m(current_date|current_timestamp|now\(|clock_timestamp\(|transaction_timestamp\(|statement_timestamp\(|localtimestamp)\M';

  IF token_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (socio_market): time-of-query tokens detected in closure: %', token_ct;
  END IF;

  -- SOCIO invariant (must be 0)
  SELECT COUNT(*) INTO socio_bad
  FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
  WHERE (postal_kommune_snapshot_date IS NOT NULL AND postal_kommune_snapshot_date > t0::date)
     OR (socio_snapshot_date IS NOT NULL AND socio_snapshot_date > t0::date);

  IF socio_bad > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (socio_market): n_bad_socio=%', socio_bad;
  END IF;

  -- MARKET invariant (must be 0)
  SELECT COUNT(*) INTO market_bad
  FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
  WHERE (comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0)
     OR (comp_max_t0_kommune     IS NOT NULL AND comp_max_t0_kommune     >= t0);

  IF market_bad > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (socio_market): n_bad_market=%', market_bad;
  END IF;

  -- Drift check vs dataset hash baselines (bounded to last p_check_days)
  SELECT COUNT(*) INTO base_days
  FROM audit.t0_dataset_hash_baseline
  WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v'
    AND sample_limit=2000;

  IF base_days <= 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (socio_market): no dataset hash baselines exist for entrypoint';
  END IF;

  WITH base AS (
    SELECT t0_day, sample_limit, dataset_sha256
    FROM audit.t0_dataset_hash_baseline
    WHERE entrypoint='ml.socio_market_feature_store_t0_v1_v'
      AND sample_limit=2000
    ORDER BY t0_day DESC
    LIMIT GREATEST(1, LEAST(p_check_days, base_days))
  ),
  cur AS (
    SELECT
      b.t0_day,
      audit.dataset_sha256(ep, b.t0_day, b.sample_limit) AS current_sha
    FROM base b
  )
  SELECT COUNT(*) INTO drift_ct
  FROM base b
  JOIN cur c USING (t0_day)
  WHERE b.dataset_sha256 IS DISTINCT FROM c.current_sha;

  IF drift_ct > 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (socio_market): dataset hash drift vs baseline on % day(s)', drift_ct;
  END IF;
END $$;

CREATE OR REPLACE PROCEDURE audit.run_t0_cert_socio_market_v1(p_check_days int DEFAULT 30)
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.socio_market_feature_store_t0_v1_v';
  v_viewdef_objects int := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days    int := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=2000);
  msg text;
BEGIN
  BEGIN
    PERFORM audit.assert_t0_certified_socio_market_v1(p_check_days);
    msg := format('T0 CERT PASS (socio_market): invariants + drift checked on last %s day(s); baselines=%s',
                  LEAST(p_check_days, v_dataset_days), v_dataset_days);

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='CERTIFIED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;

    -- Register inherited aliases as CERTIFIED (optional but recommended)
    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    SELECT
      x.entrypoint,
      'CERTIFIED',
      v_viewdef_objects,
      v_dataset_days,
      'Alias points to ml.socio_market_feature_store_t0_v1_v; certification inherited.'
    FROM (VALUES
      ('ml.socio_market_feature_store_train_v'),
      ('ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv'),
      ('ml.market_relative_socio_t0_v1_mv'),
      ('ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv')
    ) AS x(entrypoint)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='CERTIFIED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;

  EXCEPTION WHEN others THEN
    msg := SQLERRM;

    INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
    VALUES (ep,'FAILED',v_viewdef_objects,v_dataset_days,msg)
    ON CONFLICT (entrypoint) DO UPDATE SET
      status='FAILED',
      certified_at=now(),
      viewdef_objects=EXCLUDED.viewdef_objects,
      dataset_days=EXCLUDED.dataset_days,
      notes=EXCLUDED.notes;
  END;

  RAISE NOTICE '%', msg;
END $$;
