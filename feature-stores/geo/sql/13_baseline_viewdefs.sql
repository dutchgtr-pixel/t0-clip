-- 13_baseline_viewdefs.sql
-- Viewdef closure baseline for geo entrypoint.

DELETE FROM audit.t0_viewdef_baseline
WHERE entrypoint = 'ml.geo_feature_store_t0_v1_v';

WITH RECURSIVE rels(oid) AS (
  SELECT 'ml.geo_feature_store_t0_v1_v'::regclass
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
    'ml.geo_feature_store_t0_v1_v'::text AS entrypoint,
    nspname||'.'||relname AS object_fqn,
    relkind,
    encode(digest(convert_to(pg_get_viewdef(oid,true),'utf8'),'sha256'),'hex') AS viewdef_sha256
  FROM objs
)
INSERT INTO audit.t0_viewdef_baseline(entrypoint, object_fqn, relkind, viewdef_sha256)
SELECT entrypoint, object_fqn, relkind, viewdef_sha256
FROM defs
ON CONFLICT (entrypoint, object_fqn)
DO UPDATE SET viewdef_sha256 = EXCLUDED.viewdef_sha256,
              captured_at = now();
