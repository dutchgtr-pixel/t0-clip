# System Spine (Closed-Loop Execution + Governance + Certified Surfaces)

This document describes the operational spine of the system: what runs, what is governed, and what models consume.
DAGs are only the scheduling surface. The actual spine includes certified feature-store entrypoints, fail-closed
governance primitives, and the controller scripts that assemble multi-stage pipelines.

## 1) Orchestration Surface (what runs)

| DAG ID | Schedule (expr) | Schedule (resolved default) | File |
|---|---|---|---|
| audit_stale_listings_daily | os.getenv("AUDIT_SCHEDULE", "45 23 * * *") | 45 23 * * * | etl\dags\older21days_tail_audit_job_public.py |
| marketplace_db_healing_pipeline_30m | */30 * * * * | */30 * * * * | etl\dags\db_healing_pipeline_30m_public.py |
| marketplace_image_pipeline_30m | */30 * * * * | */30 * * * * | etl\dags\image_labeling_pipeline_30m_public.py |
| marketplace_ingest_and_audit | */30 * * * * | */30 * * * * | etl\dags\discovery_job_public.py |
| status_sanity_audit_midnight | os.getenv("DAG_SCHEDULE_CRON", "0 0 * * *") | 0 0 * * * | etl\dags\status_sanity_audit_job_public.py |
| survival_marketplace_daily | os.getenv("DAG_SCHEDULE_CRON", "15,45 8-23 * * *") | 15,45 8-23 * * * | etl\dags\reverse_monitor_job_public.py |

## 2) Certified Feature-Store Entrypoints (what models consume)

Models and training code are designed to consume governed entrypoints (T0-correct views) rather than raw tables.

- ml.fusion_feature_store_t0_v1_v
- ml.geo_feature_store_t0_v1_v
- ml.socio_market_feature_store_t0_v1_v
- ml.survival_feature_store_t0_v1_v
- ml.trainer_derived_feature_store_t0_v1_v
- ml.woe_anchor_feature_store_t0_v1_v

## 3) Governance / Certification Primitives (fail-closed controls)

Certification is enforced via audit tables and guard functions. Training/scoring can be blocked when stores are stale,
drifting, or out of compliance.

### Guard calls / assertions found in SQL
- audit.assert_t0_certified_fusion_store_v1
- audit.assert_t0_certified_geo_store_v1
- audit.assert_t0_certified_socio_market_v1
- audit.assert_t0_certified_survival_v1
- audit.assert_t0_certified_woe_anchor_store_v1
- audit.require_certified
- audit.require_certified_strict

### Baseline / registry tables
- audit.t0_cert_registry
- audit.t0_dataset_hash_baseline
- audit.t0_viewdef_baseline

## 4) Key Controllers (where pipeline and model assembly happens)

These controller scripts coordinate multi-stage pipelines (training, tuning, QC/healing, labeling, embeddings).
Complexity concentrates here by design.

- modeling\slow21\train_slow21_gate_classifier.py
- modeling\meta\fast24_flagger3.py
- infra\jobs\quality-control\db-healing\src\dedupe_ai_sql.py
- infra\jobs\quality-control\db-healing\src\battery_refit_llm_v2.py
- infra\jobs\labeling\image-pipeline\src\scrape_images_playwright.py
- infra\jobs\labeling\image-pipeline\src\label_damage.py
- infra\jobs\labeling\image-pipeline\src\label_color.py
- modeling\embeddings\sbert\sbert_vec_upsert_title_desc_caption.py
- modeling\embeddings\image\build_img_vec512_openclip_pg.py

