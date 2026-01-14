# Feature specification

This package defines a derived store keyed by:

## Keys
- `generation` (int)
- `listing_id` (bigint)
- `t0` (timestamptz)

## Calendar features
- `dow` (int): Monday=0 .. Sunday=6 (derived from ISO day-of-week)
- `is_weekend` (int): 1 if Saturday/Sunday, else 0

## Trailing activity counts (strict T0 windows)
- `gen_30d_post_count` (int): count of rows in the same generation with `t0` in `(t0-30d, t0)` **strictly before** the current row’s `t0`
- `allgen_30d_post_count` (int): same as above across all generations

## Cross-modal fusion and consistency flags (example: battery)
Inputs are expected from upstream training-safe surfaces:
- `battery_pct_img` from `ml.image_features_unified_v1_train_v`
- `battery_pct_effective` from `ml.market_feature_store_train_v`

Outputs:
- `battery_pct_effective_fused` (float8): `COALESCE(battery_pct_img, NULLIF(battery_pct_effective,0))`
- `battery_img_minus_text` (float8): `battery_pct_img - battery_pct_text` when both exist
- `battery_img_conflict` (int): 1 when both exist and `abs(diff) >= 7` (threshold can be changed under change control)

## Rule / pattern features (examples)
These are deterministic rules based on T0-available fields:
- `rocket_clean` (int)
- `rocket_heavy` (int)
- `fast_pattern_v2` (int)
- `is_zombie_pattern` (int)

The specific semantics should be treated as policy and may evolve; the store’s certification process ensures any change is explicit and auditable.

