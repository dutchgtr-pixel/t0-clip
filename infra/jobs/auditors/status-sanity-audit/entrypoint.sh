#!/usr/bin/env sh
set -euo pipefail

# Public template entrypoint: status sanity audit job (job-mode).
# - No credentials embedded.
# - No target-platform identifiers.
# - All runtime configuration provided via environment variables.

: "${MODE:=run}"                 # init | run | export
: "${PG_DSN:=}"                  # preferred
: "${DB_URL:=}"                  # optional alias
: "${PG_SCHEMA:=marketplace}"

: "${MARKETPLACE_ADAPTER:=mock}" # mock | http-json
: "${MARKETPLACE_BASE_URL:=https://example-marketplace.invalid}"
: "${MARKETPLACE_AUTH_HEADER:=}" # optional secret for adapter=http-json

# Optional behavior knobs (see status_sanity_audit_public.go for full list)
: "${SCOPE:=sold}"
: "${CADENCE_DAYS:=30}"
: "${WORKERS:=16}"
: "${REQUEST_TIMEOUT:=25s}"
: "${REQUEST_RPS:=3.0}"
: "${REQUEST_RPS_MAX:=10.0}"
: "${REQUEST_RPS_MIN:=0.25}"
: "${REQUEST_RPS_STEP:=0.25}"
: "${REQUEST_RPS_DOWN:=0.70}"
: "${BURST_FACTOR:=2.0}"
: "${RETRY_MAX:=4}"
: "${THROTTLE_SLEEP:=3s}"
: "${RETRY_BACKOFF_BASE:=750ms}"
: "${JITTER_MAX:=750ms}"
: "${HEAD_FIRST:=false}"
: "${ALLOW_UI_FALLBACK:=true}"

: "${WRITE_EVENTS:=true}"
: "${APPLY_MODE:=none}"          # none | safe | all
: "${APPLY_CHANGES:=false}"      # alias -> safe
: "${ALLOW_UNSELL:=false}"       # only used when APPLY_MODE=all
: "${ALLOW_REVIVE:=false}"       # only used when APPLY_MODE=all

# DSN resolution (do NOT print DSNs to logs)
if [ -z "${PG_DSN}" ] && [ -n "${DB_URL}" ]; then
  PG_DSN="${DB_URL}"
fi
if [ -z "${PG_DSN}" ]; then
  echo "[status-sanity-audit] ERROR: PG_DSN (or DB_URL) must be set" >&2
  exit 2
fi

export MODE PG_DSN DB_URL PG_SCHEMA \
  MARKETPLACE_ADAPTER MARKETPLACE_BASE_URL MARKETPLACE_AUTH_HEADER \
  SCOPE CADENCE_DAYS WORKERS REQUEST_TIMEOUT REQUEST_RPS REQUEST_RPS_MAX REQUEST_RPS_MIN REQUEST_RPS_STEP REQUEST_RPS_DOWN \
  BURST_FACTOR RETRY_MAX THROTTLE_SLEEP RETRY_BACKOFF_BASE JITTER_MAX HEAD_FIRST ALLOW_UI_FALLBACK \
  WRITE_EVENTS APPLY_MODE APPLY_CHANGES ALLOW_UNSELL ALLOW_REVIVE

exec /usr/local/bin/status-sanity-audit "$@"
