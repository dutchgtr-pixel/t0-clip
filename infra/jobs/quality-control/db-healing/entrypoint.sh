#!/usr/bin/env bash
# entrypoint_8stage_public.sh
#
# Public-release entrypoint: runs 8 jobs sequentially.
#
# SECURITY:
# - No secrets are embedded in this file.
# - Provide secrets via environment variables / secret manager.
#
# REQUIRED ENV (inject at runtime):
#   PG_DSN                 Postgres DSN
#   OPENAI_API_KEY or LLM_API_KEY
#   OPENAI_BASE_URL        optional (for OpenAI-compatible gateways)
#
# OPTIONAL ENV (safe defaults):
#   MODEL_NAME             provider-specific model name
#   LOG_LEVEL              INFO|DEBUG|WARNING
#   WORKERS                concurrency for Python LLM jobs
#   QPS                    per-process throttle (if supported by a given job)
#   DRY_RUN                1/0 or true/false
#
# Pipeline controls:
#   PIPELINE_FAIL_FAST          true|false (default: true)
#   PIPELINE_SLEEP_BETWEEN_SEC  integer seconds (default: 0)
#
set -euo pipefail

FAIL_FAST="${PIPELINE_FAIL_FAST:-true}"
SLEEP_BETWEEN="${PIPELINE_SLEEP_BETWEEN_SEC:-0}"

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') | entrypoint | $*"; }

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    log "ERROR: required env var not set: ${name}"
    exit 2
  fi
}

run_stage() {
  local label="$1"; shift
  log "START ${label}"
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "FAIL  ${label} (exit=$rc)"
    if [[ "${FAIL_FAST}" == "true" || "${FAIL_FAST}" == "1" || "${FAIL_FAST}" == "yes" ]]; then
      exit $rc
    fi
  else
    log "OK    ${label}"
  fi
  if [[ "${SLEEP_BETWEEN}" =~ ^[0-9]+$ ]] && [[ "${SLEEP_BETWEEN}" -gt 0 ]]; then
    sleep "${SLEEP_BETWEEN}"
  fi
}

# ----- Required config -----
require_env PG_DSN
if [[ -z "${OPENAI_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  log "ERROR: set OPENAI_API_KEY or LLM_API_KEY (do not hardcode secrets in the repo)."
  exit 2
fi

# Normalize: some scripts accept LLM_API_KEY, some accept OPENAI_API_KEY.
if [[ -z "${OPENAI_API_KEY:-}" && -n "${LLM_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${LLM_API_KEY}"
fi
if [[ -z "${LLM_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export LLM_API_KEY="${OPENAI_API_KEY}"
fi

export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export MODEL_NAME="${MODEL_NAME:-llm-model}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

APP_DIR="${APP_DIR:-/app}"
cd "${APP_DIR}"

# 01) Quality control (LLM-assisted)
run_stage "01_quality_control" python ./quality_control_ai.public.py

# 02) Core enrichment upserter (LLM-assisted)
run_stage "02_enrich_upserter" python ./listing_ai_enrich_upserter_public.py

# 03) Post-sale audit snapshot (Go binary)
run_stage "03_post_sold_audit" ./post_sold_audit --mode run

# 04) PSA condition sync + queue damage rescoring
run_stage "04_psa_condition_sync" python ./psa_condition_sync_and_rescore_public.py

# 05) LLM-assisted de-duplication
export AI_PROVIDER="${AI_PROVIDER:-openai}"
export AI_MODEL="${AI_MODEL:-${MODEL_NAME}}"
export AI_BASE_URL="${AI_BASE_URL:-${OPENAI_BASE_URL:-}}"
run_stage "05_dedupe" python ./dedupe_ai_sql_public.py --limit "${DEDUPE_LIMIT:-0}"

# 06) Battery refit (LLM-assisted)
run_stage "06_battery_refit" python ./battery_refit_llm_v2_public.py

# 07) Damage decider (LLM-assisted)
run_stage "07_damage_decider" python ./gbt_damage.public.py

# 08) PSA sold-price sync
run_stage "08_psa_sold_price_sync" python ./psa_sold_price_sync.public.py

log "PIPELINE COMPLETE"
