# 3. Banding Specification (Schema v1)

Banding compresses continuous/raw features into categorical keys.
Each categorical key indexes a WOE table entry.

## 3.1 Input features used for banding

From `ml.socio_market_feature_store_train_v`:
- `delta_vs_sold_median_30d`
- `condition_score`
- `damage_severity_ai`
- `battery_pct_effective`
- `seller_rating`, `review_count`, `member_since_year`
- `ai_ship_can`, `ai_ship_pickup`, `ai_pickup_only_bin`
- `ai_sale_mode_firm`, `ai_sale_mode_bids`, `ai_sale_mode_obo`
- `generation`, `listing_id`, `t0`

From `ml.iphone_image_features_unified_v1_train_v`:
- `image_count`, `caption_share`, `stock_photo_share`
- `photo_quality_avg`, `bg_clean_avg`
- `dmg_band_hq_struct` (vision damage band)
- `charger_bundle_level`, `box_present`, `receipt_present`
- join key: `(generation, listing_id)`

## 3.2 Band definitions

### 3.2.1 dSold band (`dsold_band`)

Input: `delta_vs_sold_median_30d`

Thresholds are stored in `ml.woe_anchor_model_registry_v1`:
- `dsold_t1..dsold_t4`

Mapping (example names):
- NULL → `dSold_missing`
- <= t1 → `dSold_cheap`
- <= t2 → `dSold_under`
- <= t3 → `dSold_fair`
- <= t4 → `dSold_over`
- else → `dSold_overpriced`

### 3.2.2 Trust tier (`trust_tier`)

Uses:
- seller_rating
- review_count
- member_since_year (tenure)

Example logic:
- HIGH: rating >= 9.7 and reviews >= 50 and tenure >= 3 years
- MED:  rating >= 9.0 and reviews >= 10
- LOW:  otherwise

### 3.2.3 Ship band (`ship_band`)

- pickup_only or ship_pickup → `pickup_heavy`
- can_ship → `shipping_ok`
- else → `ship_unknown`

### 3.2.4 Sale band (`sale_band`)

- firm → `firm`
- bids → `bids`
- obo → `obo`
- else → `sale_unspec`

### 3.2.5 Condition band (`cond_band`)

- NULL → `cond_missing`
- >= 0.90 → `cond_hi`
- >= 0.70 → `cond_mid`
- else → `cond_lo`

### 3.2.6 AI damage band (`dmg_ai_band`)

- NULL → `dmg_missing`
- <= 0 → `dmg_0`
- == 1 → `dmg_1`
- == 2 → `dmg_2`
- else → `dmg_3p`

### 3.2.7 Battery band (`bat_band`)

- NULL → `bat_missing`
- >= 95 → `bat_hi`
- >= 88 → `bat_mid`
- else → `bat_lo`

## 3.3 Vision-dependent bands

These are only produced when the listing has vision assets (`has_assets=1`), otherwise NULL.

### 3.3.1 Presentation score (`presentation_score`)

A weighted score:
- more images, better photo quality and background
- fewer stock photos
- higher caption share

This is then binned into:
- `present_lo`, `present_mid`, `present_hi`
using `c1/c2` cutpoints stored per model_key in `ml.woe_anchor_cuts_v1`.

### 3.3.2 Vision damage band (`vdmg_band`)

Based on `dmg_band_hq_struct`:
- missing → `vdmg_missing`
- 0 → `vdmg_0`
- 1 → `vdmg_1`
- 2 → `vdmg_2`
- else → `vdmg_3p`

### 3.3.3 Accessories band (`accessories_band`)

Accessory score derived from:
- charger_bundle_level
- box_present
- receipt_present

Binned to:
- `acc_lo`, `acc_mid`, `acc_hi`

