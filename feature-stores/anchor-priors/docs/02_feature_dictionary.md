# 2. Feature Dictionary (anchor_priors_store)

## 2.1 Speed anchors (database MV)
These are bucketed market priors (e.g., generation × storage_bucket × relative-price bucket).

- `speed_fast24_anchor`: recency-weighted P(time-to-sell ≤ 24h)
- `speed_fast7_anchor`: recency-weighted P(time-to-sell ≤ 7d)
- `speed_slow21_anchor`: recency-weighted P(time-to-sell > 21d)
- `speed_median_hours_ptv`: median `duration_hours` within the bucket
- `speed_n_eff_ptv`: “effective sample size” implied by the recency weights

All are anchored to explicit `anchor_day` (T0-safe) rather than `CURRENT_DATE`.

## 2.2 Strict anchor (30/60-day blend; SQL fragment)
Computed per listing using sold medians from historical SOLD rows:

- 30-day window: [t0 − 35d, t0 − 5d)
- 60-day window: [t0 − 65d, t0 − 5d)

Cohort keys (strict):
- generation
- model_norm
- storage bucket
- condition bucket (discretized)
- severity (damage severity)

Outputs:
- `anchor_30d_t0`: median sold_price in 30-day window
- `n30_t0`: sold count in 30-day window
- `anchor_60d_t0`: median sold_price in 60-day window
- `n60_t0`: sold count in 60-day window
- `anchor_blend_t0`: weighted blend with small pseudocount priors when n is small
- `ptv_anchor_strict_t0`: price / anchor_blend_t0

## 2.3 Advanced anchor v1 (SQL fragment; “smart anchor”)
Window:
- [t0 − 365d, t0 − 5d)

Hard locks (never relaxed):
- generation
- model_norm
- storage bucket

Rich cohort features (relaxed via fallback cascade):
- condition bucket
- severity
- battery bucket (optional domain-specific signal)
- trust tier (optional domain-specific signal)

Outputs:
- `anchor_price_smart`: median sold_price under best available cohort level
- `anchor_tts_median_h`: median duration_h (time-to-sell proxy) under best cohort level
- `anchor_n_support`: support count (n) under the chosen cohort level
- `anchor_level_k`: fallback level chosen (0=best, 4=most pooled)
- `ptv_anchor_smart`: price / anchor_price_smart
