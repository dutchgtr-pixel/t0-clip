# Proof Queries (Copy/Paste)

## A) Artifact presence (after training)

```sql
SELECT model_key, created_at, is_active
FROM ml.woe_anchor_model_registry_v1
ORDER BY created_at DESC
LIMIT 5;

SELECT COUNT(*) AS n_cuts FROM ml.woe_anchor_cuts_v1;
SELECT COUNT(*) AS n_maps FROM ml.woe_anchor_map_v1;
SELECT COUNT(*) AS n_scores FROM ml.woe_anchor_scores_v1;

SELECT * FROM ml.woe_anchor_scores_live_v1 LIMIT 20;
```

Expected:
- 1 active model_key row
- n_cuts > 0, n_maps > 0
- n_scores equals train_sold count (when persisted)

## B) Certification status

```sql
CALL audit.run_t0_cert_woe_anchor_store_v1(10);

SELECT *
FROM audit.t0_cert_registry
WHERE entrypoint = 'ml.woe_anchor_feature_store_t0_v1_v';

SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours');

SELECT * FROM ml.woe_anchor_scores_live_train_v LIMIT 10;
```

Expected:
- CERTIFIED registry row + inherited alias rows
- strict guard passes
- guarded scoring view returns rows

## C) “Why is it empty?” troubleshooting

If `ml.woe_anchor_scores_live_v1` returns 0 rows, check:
- active model exists:
  `SELECT * FROM ml.woe_anchor_model_registry_v1 WHERE is_active=true;`
- final cuts exist:
  `SELECT * FROM ml.woe_anchor_cuts_v1 WHERE fold_id IS NULL;`
- final map exists:
  `SELECT COUNT(*) FROM ml.woe_anchor_map_v1 WHERE fold_id IS NULL;`

If any of those are 0, training persistence did not run or failed before commit.
