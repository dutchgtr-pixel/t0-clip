# 1. Overview

## 1.1 What is the socio_economic_store?

In the Slow21 gate classifier taxonomy, **socio_economic_store** is the feature-store block that provides:

1) **Socio-economic context** (as-of the listing’s T0):
- kommune mapping (`kommune_code4`) derived from postal code
- `centrality_class`
- `income_median_after_tax_nok`
- missingness flags: `miss_kommune`, `miss_socio`
- model segmentation key: `model_tier_socio`

2) **Affordability features** (listing-level, as-of T0):
- `price_months_income`
- `log_price_to_income_cap` (winsorized to pinned p01/p99 constants)
- `aff_resid_in_seg`
- `aff_decile_in_seg`

3) **Market-relative pricing features** (strict trailing 30d leave-one-out windows, as-of T0):
- `rel_log_price_to_comp_mean30_super`
- `rel_log_price_to_comp_mean30_kommune`
- `rel_price_best`
- `rel_price_source`
- `miss_rel_price`

This block is designed to provide high-signal, leak-safe “pricing context” for slow-tail risk.

## 1.2 Why does this layer exist?

Slow21 performance is dominated by *overpricing relative to the relevant reference set*.
This layer provides two complementary perspectives:

- **Affordability (income-normalized pricing)**: the listing price relative to local income, and residualization within model segments.
- **Market comps (recent supply pricing)**: the listing price relative to comparable listings in the same local market over a strict trailing window.

Together these supply a stable, interpretable “liquidity prior” that generalizes across geography.

## 1.3 Where this store lives in your pipeline

This store is implemented inside the **certified features_view** used for training:

- `ml.socio_market_feature_store_train_v` (guarded; fail closed)
  → `ml.socio_market_feature_store_t0_v1_v` (certified entrypoint)
  → `ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv` (wide MV)
     - built from:
       - `ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv` (SOCIO+AFF)
       - `ml.market_relative_socio_t0_v1_mv` (MARKET)

In your SHAP reporting taxonomy, the socio_economic_store block is populated by the 12 modeling columns:

`centrality_class, miss_kommune, miss_socio, price_months_income, log_price_to_income_cap,
 aff_resid_in_seg, aff_decile_in_seg, rel_log_price_to_comp_mean30_super,
 rel_log_price_to_comp_mean30_kommune, rel_price_best, rel_price_source, miss_rel_price`

(Other socio context columns may exist in the wide MV but are not necessarily selected into the trainer’s “attached cols” log.)
