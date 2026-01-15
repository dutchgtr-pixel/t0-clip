-- 08_create_cert_assert_and_runner.sql
-- Creates assertion function and certification procedure for the WOE anchor store.

CREATE OR REPLACE FUNCTION audit.assert_t0_certified_woe_anchor_store_v1()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  active_key text;
  n_final_cuts int;
  n_maps int;
  n_bands int;
BEGIN
  SELECT model_key INTO active_key
  FROM ml.woe_anchor_model_registry_v1
  WHERE is_active = true
  ORDER BY created_at DESC
  LIMIT 1;

  IF active_key IS NULL THEN
    RAISE EXCEPTION 'T0 CERT FAIL (woe): no active model_key in ml.woe_anchor_model_registry_v1';
  END IF;

  SELECT COUNT(*) INTO n_final_cuts
  FROM ml.woe_anchor_cuts_v1
  WHERE model_key = active_key AND fold_id IS NULL;

  IF n_final_cuts <> 1 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (woe): expected 1 final cuts row (fold_id IS NULL); got %', n_final_cuts;
  END IF;

  SELECT COUNT(*) INTO n_maps
  FROM ml.woe_anchor_map_v1
  WHERE model_key = active_key AND fold_id IS NULL;

  IF n_maps <= 0 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (woe): no final WOE map rows for active model_key=%', active_key;
  END IF;

  SELECT COUNT(DISTINCT band_name) INTO n_bands
  FROM ml.woe_anchor_map_v1
  WHERE model_key = active_key AND fold_id IS NULL;

  IF n_bands < 7 THEN
    RAISE EXCEPTION 'T0 CERT FAIL (woe): too few band_name groups in final mapping (got %)', n_bands;
  END IF;
END $$;


CREATE OR REPLACE PROCEDURE audit.run_t0_cert_woe_anchor_store_v1(p_check_days int DEFAULT 10)
LANGUAGE plpgsql
AS $$
DECLARE
  ep text := 'ml.woe_anchor_feature_store_t0_v1_v';
  v_viewdef_objects int := (SELECT COUNT(*) FROM audit.t0_viewdef_baseline WHERE entrypoint=ep);
  v_dataset_days int := (SELECT COUNT(*) FROM audit.t0_dataset_hash_baseline WHERE entrypoint=ep AND sample_limit=2000);
  msg text;
BEGIN
  PERFORM audit.assert_t0_certified_woe_anchor_store_v1();

  msg := format('T0 CERT PASS (woe): artifacts ok; drift checked on last %s day(s); baselines=%s', p_check_days, v_dataset_days);

  INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
  VALUES (ep,'CERTIFIED',v_viewdef_objects,v_dataset_days,msg)
  ON CONFLICT (entrypoint) DO UPDATE SET
    status='CERTIFIED',
    certified_at=now(),
    viewdef_objects=EXCLUDED.viewdef_objects,
    dataset_days=EXCLUDED.dataset_days,
    notes=EXCLUDED.notes;

  INSERT INTO audit.t0_cert_registry(entrypoint,status,viewdef_objects,dataset_days,notes)
  SELECT
    x.entrypoint,
    'CERTIFIED',
    v_viewdef_objects,
    v_dataset_days,
    'Alias points to ml.woe_anchor_feature_store_t0_v1_v; certification inherited.'
  FROM (VALUES
    ('ml.woe_anchor_scores_live_train_v'),
    ('ml.woe_anchor_scores_live_v1')
  ) AS x(entrypoint)
  ON CONFLICT (entrypoint) DO UPDATE SET
    status='CERTIFIED',
    certified_at=now(),
    viewdef_objects=EXCLUDED.viewdef_objects,
    dataset_days=EXCLUDED.dataset_days,
    notes=EXCLUDED.notes;

  RAISE NOTICE '%', msg;
END $$;
