# 1. Overview: WOE Anchor Store

## 1.1 Problem the store solves

Your Slow21 gate classifier targets a specific operational decision:
- Determine whether a listing is likely to **survive >= 21 days (>= 504 hours)**.

You already have strong priors (anchors), socio-market context, vision, fusion, geo, and device meta.
However, the long tail is dominated by a small set of categorical “market + trust + presentation” regimes.

A classic way to compress such regimes into a high-signal scalar prior is **WOE (Weight of Evidence)**:
- Learn per-category log-odds contributions for the binary label **slow21**.
- Score each listing with an additive logit:
  `woe_logit = base_logit + Σ woe(band_i)`
- Convert to probability:
  `woe_anchor_p_slow21 = sigmoid(woe_logit)`

## 1.2 Why WOE is not a normal feature store

WOE is **supervised target encoding**. The mapping uses **labels** during training.
Therefore it must be managed like a model artifact:

- The mapping must be **versioned** and tied to the trained model run (`model_key`).
- Training must use **OOF scores** for TRAIN rows (to avoid self-influence leakage).
- Inference must use the **frozen mapping** for the active model (not re-learned from current labels).

This package implements exactly that.

## 1.3 Guarantees provided by this implementation

- **No self-influence leakage:** TRAIN SOLD rows are scored OUT-OF-FOLD.
- **Run-level reproducibility:** mapping + cuts + thresholds stored under `model_key`.
- **SQL-native inference:** WOE probability computed in DB using stored artifacts.
- **Fail-closed governance:** scoring view is guarded by `audit.require_certified_strict`.
- **Fast certification:** dataset hash baselines are kept for the last **10** `t0_day` values (policy choice).

