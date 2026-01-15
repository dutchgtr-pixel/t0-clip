# WOE Anchor Store (Slow21) — Fully Certified Package

Package name: **woe_anchor_store_certified_package_20260111**  
Generated: 2026-01-11 15:30:37Z  

This package documents and recreates the **WOE Anchor Feature Store** used by your Slow21 gate training/inference stack, including:

- The **OOF (out-of-fold) WOE training** logic inside Python (no self-influence leakage).
- The **Postgres artifact tables** that store a versioned WOE mapping keyed by `model_key`.
- The **SQL-native scoring views** (`bands → logit → probability`) for inference.
- The **T0 + leak-proof certification** and registry enforcement (fail-closed) using your `audit.*` guardrails.
- Full SQL scripts for creating/migrating objects and reproducing certification baselines.

## What this store is (one sentence)

A **versioned, supervised target-encoding prior** for `P(survive >= 21 days)` that is:
- trained on **TRAIN SOLD**,
- applied as **OOF** on TRAIN SOLD for leak-free training,
- persisted to Postgres as immutable artifacts keyed by `model_key`,
- scored in SQL for inference via an **active model pointer**,
- protected by `audit.require_certified_strict(...)` gating.

## Folder layout

- `docs/` — long-form documentation (extremely detailed).
- `sql/` — all SQL required to recreate objects (tables, views, cert + runbook).
- `python/` — Python-side details (where in the trainer WOE is computed and persisted).
- `proofs/` — copy/paste verification queries and expected outcomes.

## Recommended execution order (fresh DB)

0. `sql/00_audit_primitives.sql` (only if you do not already have compatible `audit.*` certification primitives)

1. `sql/00_create_tables.sql`
2. `sql/01_nullable_fold_id_migration.sql` (only if you created PKs that forced fold_id NOT NULL)
3. `sql/02_add_registry_fields.sql`
4. `sql/03_create_banding_view.sql`
5. `sql/04_create_scoring_view.sql`
6. Run training once (the trainer will populate the tables and set an active `model_key`).
7. `sql/05_create_entrypoint_and_guard.sql`
8. `sql/06_baseline_viewdefs.sql`
9. `sql/07_baseline_dataset_hashes_last10.sql`
10. `sql/08_create_cert_assert_and_runner.sql`
11. `sql/09_run_certification.sql`

## Minimal “is it working?” checks

After training has run and certification executed:

- `SELECT * FROM audit.t0_cert_registry WHERE entrypoint='ml.woe_anchor_feature_store_t0_v1_v';`
- `SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours');`
- `SELECT * FROM ml.woe_anchor_scores_live_train_v LIMIT 10;`
