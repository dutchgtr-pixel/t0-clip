# Phases, Proof Artifacts, and How to Reproduce

This file documents the certification program as a sequence of falsifiable gates (Phase 1–6), plus the concrete scripts used to generate proof artifacts.

---

## Phase 1 — Gate A: Code integrity proof (extraction + hash verification)

### Goal
Prove that what is running in the DB is exactly what you inspected.

### Inputs
- Feature store extraction ZIP (inventory, dependency edges, and viewdef bytes):
  - `inventory/inventory_views_matviews.csv`
  - `deps/deps_relations.csv`
  - `viewdef_bytes/*.sql`

### Pass criteria
- All extracted viewdefs and function definitions match expected SHA256 manifests:
  - `verify_viewdef_sha256.csv` all `match=True`
  - `verify_functions_sha256.csv` all `match=True`

### Operational notes
- Hashing must be byte-stable (UTF-8 no BOM).
- If you regenerate the extraction, you should store the outputs in version control or immutable artifact storage.

---

## Phase 2 — Gate A2: Time-of-query dependence ban

### Goal
Ban time-of-query tokens that make feature values depend on query execution time rather than event time.

### Tokens banned (minimum)
`CURRENT_DATE`, `CURRENT_TIMESTAMP`, `now()`, `clock_timestamp()`, `transaction_timestamp()`, `statement_timestamp()`, `localtimestamp`

### Proof method
Search the view/matview dependency closure for those tokens.

### Remediation (what happened here)
The legacy object `ml.tom_speed_anchor_v1_mv` contained `CURRENT_DATE` and was replaced (Phase 5) by a T0-safe as-of anchored MV:
- `ml.tom_speed_anchor_asof_v1_mv`

---

## Phase 3 — Gate B: Structural anchoring (as-of contracts)

### Goal
Prove every feature in the training feature store is an **as-of T0** function of time-varying sources.

### How it works (contract-based)
1. Compute the dependency closure for selected entrypoints.
2. For each node, compute transitive source tables (facts).
3. Classify sources as `TIME_VARYING` vs `STATIC`.
4. For nodes with time-varying sources, enforce an anchor contract:
   - anchor exists in outputs
   - all event-time columns are constrained by `< anchor`

### Critical tooling fix (Postgres matviews)
Do not use `information_schema.columns` for materialized view columns; it is not reliable.
Use `pg_attribute` / `pg_class` / `pg_namespace` instead.

#### Fixed output-column query (works for views + matviews)
```sql
SELECT
  n.nspname AS table_schema,
  c.relname AS table_name,
  a.attnum  AS ordinal_position,
  a.attname AS column_name,
  pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type
FROM pg_namespace n
JOIN pg_class c     ON c.relnamespace = n.oid
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = :schema
  AND c.relname = :name
  AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY 1,2,3;
```

### Firewall rule (labels + current dims)
After role-mapping nodes, run a firewall check:
- Feature-store entrypoints must not depend on:
  - `LABEL_ONLY`
  - `DIM_CURRENT`
  - `TRAINING_VIEW_MIXED`

This was used to catch:
- label leakage via `ml.tom_labels_v1_mv`
- “current selector” instability via `ref.geo_mapping_current`

### Outcome in this program
Phase 3 concluded with remediation (Phase 5) that removed:
- `ml.tom_speed_anchor_v1_mv`
- `ref.geo_mapping_current` and `ml.geo_dim_super_metro_v4_current_v` from certified closure

and replaced them with anchored/pinned T0-safe objects.

---

## Phase 4 — Gate C: Dynamic “future perturbation” proof

### Goal
Static scans are not enough. You need a behavioral proof that running the same historical query later gives the same result.

### Definition
For a certified entrypoint:
- choose a historical `t0_day`
- sample deterministically (ORDER BY key, LIMIT N)
- hash rows deterministically:
  1. serialize `to_jsonb(row)::text`
  2. sha256 hash each row
  3. sha256 hash concatenation of row hashes sorted by key

### Pass criteria
Re-running later yields the same hash.
If it changes:
- either you have leakage, or
- upstream facts changed (backfill)

Both cases must trigger alert + review.

---

## Phase 5 — Remediation rule (what gets rebuilt)
Only rebuild objects that fail gates (and the minimal upstream needed to fix them).
No DB rebuilds. No table rebuilds.

This program rebuilt:
- speed anchor MV (as-of)
- speed-enriched base MV
- AI clean T0 MV
- pinned geo dimension
- T0-gated image MV + derived damage MV
- T0 device meta MV
- final survival feature-store entrypoint view

Legacy objects remain but are marked as NOT T0 SAFE.

---

## Phase 6 — Certification registry + training enforcement

### Goal
Prevent reintroducing leakage by:
- storing certification state in DB
- enforcing it at query time for training

### Mechanism
- `audit.t0_cert_registry` stores status/timestamps/notes
- `audit.require_certified_strict(...)` checks:
  - status + freshness
  - viewdef drift vs baseline
- `ml.survival_feature_store_train_v` embeds `audit.cert_guard(...)` so reads fail closed.

---

## PowerShell reproducibility notes (non-truncated, no secrets)

### 1) General conventions
- Never commit passwords. Use `$env:PGPASSWORD` (set via secret manager or interactive prompt).
- Always force UTF-8 **without BOM** when writing hashable artifacts.

### 2) Common PowerShell parser pitfall
PowerShell forbids a pipeline directly after a `foreach (...) { ... }` expression in some contexts.

**Bad (can trigger “An empty pipe element is not allowed”):**
```powershell
$nodeTargets = foreach ($n in ($scope | Sort-Object)) {
  ...
} | Sort-Object schema,name -Unique
```

**Good:**
```powershell
$nodeTargets = foreach ($n in $scope) {
  ...
}
$nodeTargets = $nodeTargets | Sort-Object schema,name -Unique
```

---

## Required outputs for Phase 3 (Gate B)
Your Gate B workflow must produce:
- `meta/gateB_scope_nodes__survival.txt`
- `meta/gateB_scope_source_tables.csv`
- `meta/gateB_scope_source_tables_timecols.csv`
- `meta/gateB_time_varying_source_classification.csv`
- `meta/gateB_scope_nodes_output_columns.csv`
- `meta/gateB_structural_anchor_contract.csv`
- `meta/gateB_structural_anchor_findings.csv`

These are the falsifiable artifacts you keep with the certification record.

