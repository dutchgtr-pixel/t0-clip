# 1. Anchor Priors Store — Purpose and Scope

## Purpose
The anchor priors store provides **strong, stable priors** derived from historical SOLD outcomes and computed
**as-of each row’s reference time (T0)**.

These priors are typically used as:
- a **regularizer** against noisy/high-dimensional features,
- a **market thickness indicator** (support-aware fallbacks when comparable data is thin),
- a **stability mechanism** under distribution shift.

## What this repo means by “T0 / as-of”
For an active listing row with reference date `t0 = edited_date::date`, all historical SOLD comparables used to
compute anchor priors must satisfy an **embargo** such as:

- `sold_date::date < t0 - interval '5 days'` (strict/advanced anchors), or
- `sold_day < anchor_day` (speed anchors keyed by anchor_day)

This ensures the anchor values are **not contaminated** by outcomes that occur on or after the reference time.

## Store boundaries (feature inventory)
This package defines three anchor families:

### A) Speed anchors (database MV)
- `speed_fast24_anchor`
- `speed_fast7_anchor`
- `speed_slow21_anchor`
- `speed_median_hours_ptv`
- `speed_n_eff_ptv`

### B) Strict price anchor (SQL fragment)
- `anchor_30d_t0`, `n30_t0`
- `anchor_60d_t0`, `n60_t0`
- `anchor_blend_t0`
- `ptv_anchor_strict_t0` = price / anchor_blend_t0

### C) Advanced anchor v1 (SQL fragment)
- `anchor_price_smart`
- `anchor_tts_median_h`
- `anchor_n_support`
- `anchor_level_k` (fallback level indicator)
- `ptv_anchor_smart` = price / anchor_price_smart
