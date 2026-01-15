# 5. Inference Usage

## 5.1 SQL-native scoring (recommended)

Use the fail-closed, certified scoring view:

- `ml.woe_anchor_scores_live_train_v`

This guarantees:
- an active model exists
- artifacts exist and pass certification
- dataset baselines and viewdef baselines are valid (per your policy)

Example: attach WOE prior to a live scoring surface:

```sql
SELECT
  f.generation,
  f.listing_id,
  f.t0,
  w.woe_anchor_p_slow21
FROM ml.socio_market_feature_store_train_v f
LEFT JOIN ml.woe_anchor_scores_live_train_v w
  ON w.generation = f.generation
 AND w.listing_id    = f.listing_id
 AND w.t0         = f.t0;
```

## 5.2 Notes on keys

The WOE scoring view exposes:
- `(generation, listing_id, t0, model_key)`

This is deliberate: WOE is conceptually time-anchored and model-versioned.

## 5.3 What to do when a new model becomes active

Every training run that persists WOE artifacts sets a new active model_key.
After that you should re-run WOE certification:

```sql
CALL audit.run_t0_cert_woe_anchor_store_v1(10);
SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours');
```

This is required because:
- the entrypoint’s output changes when the active model_key changes
- your certification baseline and registry entry should reflect that.

