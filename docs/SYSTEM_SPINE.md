# System Spine (Closed-Loop Orchestration)

This repository runs as a closed loop: ingestion and monitoring feed an operational database; scheduled auditors and
self-healing jobs reconcile and repair state; feature-store surfaces are refreshed/certified; models consume only
certified entrypoints; results and audits feed back into governance.

## Airflow DAG Inventory

| DAG ID | Schedule (expr) | Schedule (resolved default) | File |
|---|---|---|---|
| audit_stale_listings_daily | os.getenv("AUDIT_SCHEDULE", "45 23 * * *"), | 45 23 * * * | etl\dags\older21days_tail_audit_job_public.py |
| marketplace_db_healing_pipeline_30m | */30 * * * * | */30 * * * * | etl\dags\db_healing_pipeline_30m_public.py |
| marketplace_image_pipeline_30m | */30 * * * * | */30 * * * * | etl\dags\image_labeling_pipeline_30m_public.py |
| marketplace_ingest_and_audit | */30 * * * * | */30 * * * * | etl\dags\discovery_job_public.py |
| status_sanity_audit_midnight | os.getenv("DAG_SCHEDULE_CRON", "0 0 * * *") | 0 0 * * * | etl\dags\status_sanity_audit_job_public.py |
| survival_marketplace_daily | os.getenv("DAG_SCHEDULE_CRON", "15,45 8-23 * * *") | 15,45 8-23 * * * | etl\dags\reverse_monitor_job_public.py |

## Operational Meaning (edit as needed)

- marketplace_ingest_and_audit: ingestion/discovery pipeline (writes canonical operational rows + audit events).
- marketplace_db_healing_pipeline_30m: periodic self-healing / reconciliation loop (repairs state and enforces invariants).
- marketplace_image_pipeline_30m: multimodal enrichment pipeline (image-derived features/labels).
- status_sanity_audit_midnight: daily sanity audit (detects inconsistency and drift signals).
- udit_stale_listings_daily: daily stale/tail audit (enforces long-tail state transitions).
- survival_marketplace_daily: monitoring loop feeding survival/tail modeling state updates.

