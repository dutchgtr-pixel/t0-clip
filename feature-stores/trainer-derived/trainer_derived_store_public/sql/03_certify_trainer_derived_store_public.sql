-- 03_certify_trainer_derived_store.sql
-- Captures viewdef baselines, sets/rebuilds dataset hash baselines, and certifies.

-- 0) Viewdef baseline capture
DELETE FROM audit.t0_viewdef_baseline
WHERE entrypoint = 'ml.trainer_derived_feature_store_t0_v1_v';

WITH RECURSIVE rels(oid) AS (
  SELECT 'ml.trainer_derived_feature_store_t0_v1_v'::regclass
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
),
defs AS (
  SELECT
    'ml.trainer_derived_feature_store_t0_v1_v'::text AS entrypoint,
    nspname||'.'||relname AS object_fqn,
    relkind,
    encode(digest(convert_to(pg_get_viewdef(oid,true),'utf8'),'sha256'),'hex') AS viewdef_sha256
  FROM objs
)
INSERT INTO audit.t0_viewdef_baseline(entrypoint, object_fqn, relkind, viewdef_sha256)
SELECT entrypoint, object_fqn, relkind, viewdef_sha256
FROM defs
ON CONFLICT (entrypoint, object_fqn)
DO UPDATE SET
  viewdef_sha256 = EXCLUDED.viewdef_sha256,
  captured_at = now();

-- 1) Dataset baselines (choose ONE strategy)
-- 1A) Development strategy: rebaseline last 10 days (fast; accepts backfill drift)
CALL audit.rebaseline_last_n_days('ml.trainer_derived_feature_store_t0_v1_v'::regclass, 10, 2000);

-- 1B) Deep strategy: rebuild 365 day baselines (slower)
-- DELETE FROM audit.t0_dataset_hash_baseline
-- WHERE entrypoint='ml.trainer_derived_feature_store_t0_v1_v'
--   AND sample_limit=2000;
--
-- WITH picked AS (
--   SELECT t0_day
--   FROM (
--     SELECT DISTINCT edited_date::date AS t0_day
--     FROM ml.trainer_derived_feature_store_t0_v1_v
--     WHERE edited_date IS NOT NULL
--   ) d
--   ORDER BY t0_day DESC
--   LIMIT 365
-- )
-- INSERT INTO audit.t0_dataset_hash_baseline(entrypoint, t0_day, sample_limit, dataset_sha256)
-- SELECT
--   'ml.trainer_derived_feature_store_t0_v1_v',
--   p.t0_day,
--   2000,
--   audit.dataset_sha256('ml.trainer_derived_feature_store_t0_v1_v'::regclass, p.t0_day, 2000)
-- FROM picked p
-- ON CONFLICT (entrypoint, t0_day, sample_limit)
-- DO UPDATE SET dataset_sha256 = EXCLUDED.dataset_sha256,
--               computed_at    = now();

-- 2) Certification (p_check_days should match your drift policy window)
CALL audit.run_t0_cert_trainer_derived_store_v1(10);

-- 3) Enforce fail-closed
SELECT audit.require_certified_strict('ml.trainer_derived_feature_store_t0_v1_v', interval '24 hours');

-- 4) Quick sanity read
SELECT * FROM ml.trainer_derived_features_train_v LIMIT 5;
