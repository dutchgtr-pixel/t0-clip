# Dockerfile.public_db_healing_jobs
#
# Public-release container build for the 8-stage DB-healing pipeline.
#
# SECURITY:
# - No secrets are embedded.
# - Provide PG_DSN and API keys at runtime via secrets/env injection.
#
# Build:
#   docker build -f Dockerfile.public_db_healing_jobs -t marketplace-db-healing-jobs:public .
#
# Run (example):
#   docker run --rm \
#     -e PG_DSN="$PG_DSN" \
#     -e OPENAI_API_KEY="$OPENAI_API_KEY" \
#     -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
#     marketplace-db-healing-jobs:public
#
FROM golang:1.22-bookworm AS go_builder
WORKDIR /src
COPY post_sold_audit.go ./post_sold_audit_public.go
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/post_sold_audit ./post_sold_audit_public.go

FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.public.txt ./requirements.public.txt
RUN pip install --no-cache-dir -r requirements.public.txt

# Jobs (public versions)
COPY quality_control_ai.public.py ./quality_control_ai.public.py
COPY listing_ai_enrich_upserter_public.py ./listing_ai_enrich_upserter_public.py
COPY psa_condition_sync_and_rescore_public.py ./psa_condition_sync_and_rescore_public.py
COPY dedupe_ai_sql_public.py ./dedupe_ai_sql_public.py
COPY battery_refit_llm_v2_public.py ./battery_refit_llm_v2_public.py
COPY gbt_damage.public.py ./gbt_damage.public.py
COPY psa_sold_price_sync.public.py ./psa_sold_price_sync.public.py

COPY entrypoint_8stage_public.sh ./entrypoint_8stage_public.sh
COPY --from=go_builder /out/post_sold_audit ./post_sold_audit

RUN chmod +x /app/entrypoint_8stage_public.sh /app/post_sold_audit

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    MODEL_NAME=llm-model \
    PIPELINE_FAIL_FAST=true

ENTRYPOINT ["/app/entrypoint_8stage_public.sh"]
