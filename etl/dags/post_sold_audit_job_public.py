# etl/dags/post_sold_audit_midnight.public.py
"""Public-release Airflow DAG (sanitized): post-sold snapshot audit (job-mode).

This DAG runs once per day at midnight and (optionally) waits for the status sanity audit DAG
to succeed for the same logical date before proceeding.

Public-release constraints:
- No embedded credentials.
- No target-platform identifiers or scraping fingerprints.
- Marketplace-specific logic must be behind the adapter layer inside the container image.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# Optional: enforce ordering across DAGs (status audit -> post-sold audit)
# Set WAIT_FOR_STATUS_DAG=true to enable the ExternalTaskSensor.
from airflow.sensors.external_task import ExternalTaskSensor


DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")
WAIT_FOR_STATUS_DAG = os.getenv("WAIT_FOR_STATUS_DAG", "true").lower() in ("1", "true", "t", "yes", "y", "on")

DEFAULT_ARGS = dict(
    owner="airflow",
    depends_on_past=False,
    email_on_failure=False,
    email_on_retry=False,
    retries=0,
    execution_timeout=timedelta(minutes=int(os.getenv("JOB_TIMEOUT_MINUTES", "90"))),
)

with DAG(
    dag_id="post_sold_audit_midnight",
    description="Daily post-sold snapshot audit (job mode) for marketplace listings",
    schedule=os.getenv("DAG_SCHEDULE_CRON", "0 0 * * *"),
    start_date=datetime(2025, 5, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["marketplace", "audit", "post_sold"],
) as dag:

    wait_for_status_sanity = ExternalTaskSensor(
        task_id="wait_for_status_sanity_audit",
        external_dag_id=os.getenv("STATUS_DAG_ID", "status_sanity_audit_midnight"),
        external_task_id=os.getenv("STATUS_TASK_ID", "run_status_sanity_audit"),
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule",
        timeout=int(os.getenv("STATUS_WAIT_TIMEOUT_SECONDS", "21600")),  # default 6h
        poke_interval=int(os.getenv("STATUS_WAIT_POKE_SECONDS", "60")),
    )

    run_post_sold_audit = DockerOperator(
        task_id="run_post_sold_audit",
        image=os.getenv("POST_SOLD_IMAGE", "post-sold-audit:latest"),
        api_version="auto",
        auto_remove=True,
        docker_url=os.getenv("DOCKER_HOST", "unix://var/run/docker.sock"),
        network_mode=DOCKER_NETWORK,
        entrypoint=None,
        command=None,
        mount_tmp_dir=False,
        environment={
            "PG_DSN": os.getenv("PG_DSN", ""),
            "PG_SCHEMA": os.getenv("PG_SCHEMA", "marketplace"),

            "MARKETPLACE_ADAPTER": os.getenv("MARKETPLACE_ADAPTER", "mock"),
            "MARKETPLACE_BASE_URL": os.getenv("MARKETPLACE_BASE_URL", "https://example-marketplace.invalid"),
            "MARKETPLACE_AUTH_HEADER": os.getenv("MARKETPLACE_AUTH_HEADER", ""),

            # Job behavior (optional)
            "MODE": os.getenv("MODE", "run"),
            "WORKERS": os.getenv("WORKERS", "32"),
            "REQUEST_RPS": os.getenv("REQUEST_RPS", "3.0"),
            "REQUEST_RPS_MAX": os.getenv("REQUEST_RPS_MAX", "12.0"),
            "REQUEST_RPS_MIN": os.getenv("REQUEST_RPS_MIN", "0.7"),
            "MIN_AGE_DAYS": os.getenv("MIN_AGE_DAYS", "7"),
            "DAY_OFFSET": os.getenv("DAY_OFFSET", "7"),
            "BATCH": os.getenv("BATCH", "0"),
            "PER_SEGMENT_CAP": os.getenv("PER_SEGMENT_CAP", "0"),
            "SEGMENTS": os.getenv("SEGMENTS", ""),
            "ONLY_IDS": os.getenv("ONLY_IDS", ""),
            "DRY_RUN": os.getenv("DRY_RUN", "false"),
        },
    )

    if WAIT_FOR_STATUS_DAG:
        wait_for_status_sanity >> run_post_sold_audit
