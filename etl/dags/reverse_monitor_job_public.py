# etl/dags/Go_survival_marketplace_daily.py
"""Public-release Airflow DAG (sanitized).

This DAG demonstrates the "job-mode" orchestration pattern: an Airflow-scheduled Docker container
runs a bounded reverse-survival sweep and exits.

Redactions / public-release constraints:
- No embedded credentials (PG_DSN must be supplied via environment or Airflow Connections).
- No target-platform identifiers or scraping fingerprints.
- Platform-specific logic must live behind an adapter inside the container image.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


# Compose/Docker network so the container can reach dependencies by service name.
# Public default is a generic network mode; override in deployment.
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")

DEFAULT_ARGS = dict(
    owner="airflow",
    depends_on_past=False,
    email_on_failure=False,
    email_on_retry=False,
    retries=0,
    # Keep a hard upper bound on runtime to prevent runaway runs.
    execution_timeout=timedelta(minutes=int(os.getenv("JOB_TIMEOUT_MINUTES", "90"))),
)

with DAG(
    dag_id="survival_marketplace_daily",
    description="Reverse-survival sweep (job mode) for marketplace listings; DB-only",
    # Example cadence: every 30 minutes between 08:15..23:45 UTC (adjust as needed)
    schedule=os.getenv("DAG_SCHEDULE_CRON", "15,45 8-23 * * *"),
    start_date=datetime(2025, 5, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["survival", "marketplace"],
) as dag:

    run_survival = DockerOperator(
        task_id="run_survival_sweep",
        # Container image that encapsulates the job; publish as a template image name.
        image=os.getenv("SURVIVAL_IMAGE", "survival-marketplace:latest"),
        api_version="auto",
        auto_remove=True,
        docker_url=os.getenv("DOCKER_HOST", "unix://var/run/docker.sock"),
        network_mode=DOCKER_NETWORK,
        # Use the image ENTRYPOINT (sequential sweep and exit)
        entrypoint=None,
        command=None,
        mount_tmp_dir=False,
        environment={
            # Database connection MUST be provided externally (env/Airflow Connection/Docker secret).
            # Do not embed DSNs with credentials in public repos.
            "PG_DSN": os.getenv("PG_DSN", ""),
            "PG_SCHEMA": os.getenv("PG_SCHEMA", "public"),

            # Sweep behavior (the container ENTRYPOINT reads these)
            "SCAN_SINCE_DAYS": os.getenv("SCAN_SINCE_DAYS", "30"),
            "MAX_PROBE": os.getenv("MAX_PROBE", "5000"),
            "WORKERS": os.getenv("WORKERS", "32"),
            "REQUEST_RPS": os.getenv("REQUEST_RPS", "10"),

            # Optional: pause between segments (if the container runs multiple segments sequentially)
            "SLEEP_BETWEEN": os.getenv("SLEEP_BETWEEN", "5"),

            # Optional: run a single segment (string key), otherwise run container default sweep list.
            # "SEGMENT_ONLY": os.getenv("SEGMENT_ONLY", ""),
        },
    )
