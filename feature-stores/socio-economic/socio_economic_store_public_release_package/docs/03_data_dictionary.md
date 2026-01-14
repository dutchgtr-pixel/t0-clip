# 3. Data dictionary (socio_economic_store)

This section documents the socio_economic_store feature columns, grouped by sub-layer.

## 3.1 Socio context (as-of T0)

| Column | Type | Description | T0 contract |
|---|---:|---|---|
| kommune_code4 | text | Kommune code derived from postal code mapping snapshot | mapping snapshot_date <= t0::date |
| centrality_class | int | Kommune centrality class | socio snapshot_date <= t0::date |
| income_median_after_tax_nok | float8 | Kommune median after-tax income (NOK) | socio snapshot_date <= t0::date |
| postal_kommune_snapshot_date | date | snapshot date used for postal→kommune mapping | must be <= t0::date |
| socio_snapshot_date | date | snapshot date used for kommune socio | must be <= t0::date |
| miss_kommune | int | 1 if kommune mapping missing, else 0 | derived |
| miss_socio | int | 1 if socio record missing, else 0 | derived |
| model_tier_socio | text | model tier bucket used for within-segment normalization | derived from model string (T0) |

## 3.2 Affordability features (as-of T0)

### price_months_income
- Definition:
  `price_months_income = price / (income_median_after_tax_nok / 12) = 12*price/income`
- NULL if price <= 0 or income <= 0.

### log_price_to_income_cap
- Definition:
  `ln((price+1)/income_median_after_tax_nok)`
- Winsorized to pinned constants:
  - p01 = -5.964859
  - p99 = -3.610855

### aff_resid_in_seg
- Definition:
  `log_price_to_income_cap − mean(log_price_to_income_cap | generation × model_tier_socio)`
- Mean computed within the MV at refresh time.

### aff_decile_in_seg
- Definition:
  `NTILE(10)` within `(generation × model_tier_socio)` ordered by `log_price_to_income_cap`
- NULL if `log_price_to_income_cap` is NULL.

## 3.3 Market-relative pricing (strict trailing 30d leave-one-out)

All market stats are computed using:

- `log_price_mkt = ln(price)` in the market base table (your current implementation)
- strict trailing window:
  `RANGE BETWEEN '30 days' PRECEDING AND '1 microsecond' PRECEDING`
  which ensures:
  - no same-row contamination
  - no future-row contamination

Market stats fields:

| Column | Type | Description |
|---|---:|---|
| comp_n_30d_super_metro | int | # comps in same super_metro_v4_geo over trailing 30d |
| comp_log_price_mean_30d_super_metro | float8 | mean(log_price_mkt) over comps |
| comp_max_t0_super_metro | timestamptz | max comp t0 used (must be < row t0) |
| comp_n_30d_kommune | int | # comps in same kommune_code4 over trailing 30d |
| comp_log_price_mean_30d_kommune | float8 | mean(log_price_mkt) over comps |
| comp_max_t0_kommune | timestamptz | max comp t0 used (must be < row t0) |

Derived relative pricing fields (computed in the wide MV):

| Column | Type | Definition |
|---|---:|---|
| rel_log_price_to_comp_mean30_super | float8 | ln(price) − comp_log_price_mean_30d_super_metro (only if comp_n>=20) |
| rel_log_price_to_comp_mean30_kommune | float8 | ln(price) − comp_log_price_mean_30d_kommune (only if comp_n>=20) |
| rel_price_best | float8 | coverage-aware fallback (super first in current implementation) |
| rel_price_source | int | 1=super_metro, 2=kommune, NULL missing |
| miss_rel_price | int | 1 if rel_price_best is NULL else 0 |

Note: your implementation uses comp thresholds of 20 before computing rel scores.

