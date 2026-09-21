# Feature-store SQL research release

Start with [the methods chapter](METHODS.md), [inventory](inventory.json),
[dependency graph](dependency_graph.json) and [rebuild plan](rebuild_plan.json).

- `archived/`: eight substantive historical Stage2 SQL derivatives. Identifiers
  are generic and private seller seed data was removed completely. See
  [provenance](provenance.json). These assume external source tables and include
  destructive rebuild statements; they are not an empty-database installer.
- `portable/`: five newly authored SQL files defining complete artificial source
  contracts, point-in-time views, historical and learned anchors, guards and a
  synthetic fixture. They demonstrate mechanisms, not the entire deployed schema.
- Existing public definitions remain under `feature-stores/` and are included in
  the inventory. Alternative historical revisions must be selected deliberately.

Offline syntax/dependency inspection, without a database:

```sh
python -m research.feature_store_ddl.inventory --output research/feature_store_ddl/inventory.json
python -m pytest tests/test_feature_store_ddl.py -q
```

Optional isolated execution uses the pinned package `@electric-sql/pglite@0.5.8`.
Install it in a disposable directory, set `NODE_PATH` to that directory's
`node_modules`, then run:

```sh
node research/feature_store_ddl/smoke_pglite.cjs
```

The script creates only an in-memory engine, applies the five portable SQL files,
checks the fixture, challenges three certification failures and closes the engine.
It uses no database URL or production connection. The retained
[validation report](validation_report.json) documents the successful run.

For a native PostgreSQL experiment, use only an explicitly controlled empty test
database and apply `portable/*.sql` in lexical order. No native deployment,
extension, concurrency or large-data performance claim follows from the in-memory
smoke. The [leakage study](../leakage/LEAKAGE_STUDY.md) explains why correct SQL
syntax and current certificates cannot establish every historical information
boundary.
