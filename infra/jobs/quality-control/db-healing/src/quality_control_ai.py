from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

"""
marketplace_quality_control_ai.public.py

Public-release Airflow DAG that runs the LLM-assisted quality-control job
(quality_control_ai.public.py) in a container, then optionally runs the enrichment
upserter.

Security posture:
- No DSNs, passwords, API keys, cookies, or platform identifiers are embedded here.
- Secrets must be supplied via Airflow Variables/Connections/Secrets backend.
- All knobs are env-driven; safe defaults are provided.

Trigger pattern:
- Recommended: make your ingest DAG run this DAG immediately after ingest completes
  (either by upstream task dependency in a single DAG, or TriggerDagRunOperator).
- This DAG is schedule-less (schedule=None) by default so it is purely event-driven.
"""

# Docker daemon + network
DOCKER_URL = os.getenv("DOCKER_URL", "unix://var/run/docker.sock")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")

# Images (build from Dockerfile.ai_jobs.public)
AI_JOBS_IMAGE = os.getenv("AI_JOBS_IMAGE", "marketplace-ai-jobs:public")

# Airflow Variable templates (override via env if you use a different secret backend)
PG_DSN_TEMPLATE = os.getenv("AIRFLOW_PG_DSN_TEMPLATE", "{{ var.value.PG_DSN }}")
OPENAI_API_KEY_TEMPLATE = os.getenv("AIRFLOW_OPENAI_API_KEY_TEMPLATE", "{{ var.value.OPENAI_API_KEY }}")
OPENAI_BASE_URL_TEMPLATE = os.getenv("AIRFLOW_OPENAI_BASE_URL_TEMPLATE", "{{ var.value.OPENAI_BASE_URL }}")

# Generic DB object names (public template)
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")
LISTINGS_TABLE = os.getenv("LISTINGS_TABLE", f'"{PG_SCHEMA}".listings').strip()

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "execution_timeout": timedelta(minutes=int(os.getenv("TASK_TIMEOUT_MIN", "45"))),
}

with DAG(
    dag_id=os.getenv("DAG_ID", "marketplace_quality_control_ai"),
    description="Run listing quality control (LLM-assisted) then enrichment (public template)",
    schedule=None,  # event-driven; trigger from ingest DAG or run manually
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["public", "marketplace", "quality", "enrich"],
) as dag:

    # Task 1: quality control
    run_quality = DockerOperator(
        task_id="run_quality_control",
        image=AI_JOBS_IMAGE,
        api_version="auto",
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        entrypoint=None,  # use image ENTRYPOINT (entrypoint_ai_jobs.public.sh)
        command=os.getenv("QUALITY_COMMAND", "quality"),  # quality_control_ai.public.py
        environment={
            # DB
            "PG_DSN": PG_DSN_TEMPLATE,
            "PG_SCHEMA": PG_SCHEMA,
            "LISTINGS_TABLE": LISTINGS_TABLE,

            # LLM (supply via Airflow Variables/Secrets)
            "OPENAI_API_KEY": OPENAI_API_KEY_TEMPLATE,
            "OPENAI_BASE_URL": OPENAI_BASE_URL_TEMPLATE,
            "MODEL_NAME": os.getenv("MODEL_NAME", "llm-model"),

            # Runtime knobs
            "WORKERS": os.getenv("QUALITY_WORKERS", "16"),
            "CAP_PER_GEN": os.getenv("QUALITY_CAP_PER_GEN", "1000"),
            "ONLY_LIVE_HOURS": os.getenv("QUALITY_ONLY_LIVE_HOURS", "0"),
            "VERSION_TAG": os.getenv("QUALITY_VERSION_TAG", "llm-quality-template-v1"),
            "DRY_RUN": os.getenv("DRY_RUN", "false"),

            # For public template runs without keys, set MOCK_LLM=true
            "MOCK_LLM": os.getenv("MOCK_LLM", "false"),
            "MOCK_DB": os.getenv("MOCK_DB", "false"),
        },
    )

    # Task 2: enrichment upserter (optional)
    run_enrich = DockerOperator(
        task_id="run_enrich_upserter",
        image=AI_JOBS_IMAGE,
        api_version="auto",
        auto_remove=True,
        docker_url=DOCKER_URL,
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        entrypoint=None,
        command=os.getenv("ENRICH_COMMAND", "enrich"),  # enrich_upserter.public.py
        environment={
            "PG_DSN": PG_DSN_TEMPLATE,
            "PG_SCHEMA": PG_SCHEMA,
            "LISTINGS_TABLE": LISTINGS_TABLE,

            "OPENAI_API_KEY": OPENAI_API_KEY_TEMPLATE,
            "OPENAI_BASE_URL": OPENAI_BASE_URL_TEMPLATE,
            "MODEL_NAME": os.getenv("MODEL_NAME", "llm-model"),

            "WORKERS": os.getenv("ENRICH_WORKERS", "16"),
            "QPS": os.getenv("ENRICH_QPS", "1.0"),
            "VERSION_TAG": os.getenv("ENRICH_VERSION_TAG", "enrich-template-v1"),
            "DRY_RUN": os.getenv("DRY_RUN", "false"),
            "MOCK_LLM": os.getenv("MOCK_LLM", "false"),
            "MOCK_DB": os.getenv("MOCK_DB", "false"),
        },
    )

    run_quality >> run_enrich
