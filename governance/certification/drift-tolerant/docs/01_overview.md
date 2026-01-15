# 01 — Drift‑Tolerant Certification Overview

## Core concepts

### 1) Entry points vs “feature blocks”
Your SHAP rollups use feature_block labels (e.g., main_feature_store, vision_store).
Those do not necessarily correspond 1:1 to registry entrypoints.

Registry certification applies to a DB entrypoint surface (a view/MV that includes edited_date or an equivalent T0 column),
and the guarded consumer view reads from that surface with audit.require_certified_strict.

Example:

- main_feature_store, ai_enrichment_store, socio_economic_store, orderbook_store, anchor_priors_store
  are columns within the SOCIO/MARKET entrypoint:
  ml.socio_market_feature_store_t0_v1_v (guarded consumer: ml.socio_market_feature_store_train_v)

### 2) Baselines
Two baseline types are used:

- Viewdef baselines: sha256 of every view/matview definition in the dependency closure of an entrypoint.
  Purpose: detect silent code drift.

- Dataset hash baselines: sha256 of a stable serialization of entrypoint output for a given t0_day (and sample_limit).
  Purpose: detect data drift / backfill drift.

### 3) Staleness vs drift
- Staleness: certification is older than max_age (e.g., >24h). Training fails closed.
  Fix: recertify.

- Drift: dataset hashes differ from baseline. Could be expected (backfills) or unexpected (leakage, bad joins).
  Fix: depends on policy.

## Drift‑tolerant policy (last N days)

We accept drift in the most recent N days (commonly 10) and automatically rebaseline those days.
We keep longer baseline history for audit, but we do not recompute it every dev run.

## Why device_meta needs special handling

device_meta depends on vision-derived signals; delayed/retroactive image labeling creates expected drift.
A drift‑allowed cert runner checks invariants, computes drift in last N days, optionally updates baselines, and certifies.

