from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# marketplace_db_healing_pipeline_30m.py
#
# Public-release DAG to run the 8-stage database-healing pipeline every 30 minutes.
#
# Security posture:
# - No secrets are embedded.
# - PG_DSN and API keys must be supplied via Airflow Variables/Connections/Secrets backend.

DOCKER_URL = os.getenv("DOCKER_URL", "unix://var/run/docker.sock")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")

IMAGE = os.getenv("HEALING_JOBS_IMAGE", "marketplace-db-healing-jobs:public")

# Templates for secrets (Airflow Variables are shown here for simplicity)
PG_DSN_TPL = os.getenv("AIRFLOW_PG_DSN_TEMPLATE", "{{ var.value.PG_DSN }}")
OPENAI_API_KEY_TPL = os.getenv("AIRFLOW_OPENAI_API_KEY_TEMPLATE", "{{ var.value.OPENAI_API_KEY }}")
OPENAI_BASE_URL_TPL = os.getenv("AIRFLOW_OPENAI_BASE_URL_TEMPLATE", "{{ var.value.OPENAI_BASE_URL }}")

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "execution_timeout": timedelta(minutes=int(os.getenv("TASK_TIMEOUT_MIN", "55"))),
}

with DAG(
    dag_id=os.getenv("DAG_ID", "marketplace_db_healing_pipeline_30m"),
    description="Run 8-stage DB healing pipeline (public release) every 30 minutes",
    schedule="*/30 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["public", "marketplace", "db-healing", "llm"],
) as dag:

    run_pipeline = DockerOperator(
        task_id="run_db_healing_pipeline",
        image=IMAGE,
        api_version="auto",
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        entrypoint=None,  # use container ENTRYPOINT
        command=os.getenv("PIPELINE_COMMAND", ""),  # empty: just run entrypoint
        environment={
            "PG_DSN": PG_DSN_TPL,
            "OPENAI_API_KEY": OPENAI_API_KEY_TPL,
            "OPENAI_BASE_URL": OPENAI_BASE_URL_TPL,
            "MODEL_NAME": os.getenv("MODEL_NAME", "llm-model"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "WORKERS": os.getenv("WORKERS", "16"),
            "QPS": os.getenv("QPS", "0"),
            "PIPELINE_FAIL_FAST": os.getenv("PIPELINE_FAIL_FAST", "true"),
            "PIPELINE_SLEEP_BETWEEN_SEC": os.getenv("PIPELINE_SLEEP_BETWEEN_SEC", "0"),
            # Optional: Go auditor adapter
            "MARKETPLACE_ADAPTER": os.getenv("MARKETPLACE_ADAPTER", "mock"),
            "MARKETPLACE_BASE_URL": os.getenv("MARKETPLACE_BASE_URL", "https://example-marketplace.invalid"),
            # If used, supply via secrets backend:
            # "MARKETPLACE_AUTH_HEADER": "{{ var.value.MARKETPLACE_AUTH_HEADER }}",
        },
    )
