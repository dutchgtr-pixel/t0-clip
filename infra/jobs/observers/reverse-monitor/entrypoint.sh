#!/usr/bin/env sh
set -euo pipefail

# Public template entrypoint for the survival job.
#
# This file intentionally contains:
#   - no credentials
#   - no target-platform identifiers
#   - no browser-fingerprint headers/cookies
#
# Configure the container via environment variables (recommended) or override flags here.

: "${MODE:=diagnose}"                       # test | reverse | repair | audit-status | stale | diagnose
: "${MARKETPLACE_ADAPTER:=mock}"        # mock | http-json
: "${MARKETPLACE_BASE_URL:=https://marketplace.example}"
: "${HTTP_USER_AGENT:=public-template/1.0}"

: "${PG_DSN:=}"                         # required when WRITE_DB=true
: "${PG_SCHEMA:=public}"
: "${GEN_LIST:=1}"                      # space-separated generations/partitions (e.g., "1 2 3")
: "${SCAN_SINCE_DAYS:=30}"
: "${MAX_PROBE:=10000}"

: "${WORKERS:=32}"
: "${REQUEST_RPS:=12}"

: "${RETRY_MAX:=4}"
: "${THROTTLE_SLEEP_MS:=3000}"
: "${JITTER_MS:=150}"

: "${MIN_RPS:=3.0}"
: "${MAX_RPS:=0}"
: "${STEP_UP_RPS:=0.5}"
: "${DOWN_MULT:=0.60}"
: "${BURST_FACTOR:=2.0}"

: "${WRITE_DB:=false}"
: "${WRITE_CSV:=false}"
: "${FRESH_CSV:=false}"
: "${OUT:=/app/out/listings.csv}"
: "${HISTORY_OUT:=/app/out/price_history.csv}"

: "${VERBOSE:=false}"
: "${SLEEP_BETWEEN:=0}"

run_gen() {
  gen="$1"
  echo "[survival] mode=${MODE} generation=${gen} adapter=${MARKETPLACE_ADAPTER}"

  set --     --mode "${MODE}"     --adapter "${MARKETPLACE_ADAPTER}"     --marketplace-base-url "${MARKETPLACE_BASE_URL}"     --user-agent "${HTTP_USER_AGENT}"     --pg-schema "${PG_SCHEMA}"     --generation "${gen}"     --scan-since-days "${SCAN_SINCE_DAYS}"     --max-probe "${MAX_PROBE}"     --workers "${WORKERS}"     --rps "${REQUEST_RPS}"     --retry-max "${RETRY_MAX}"     --throttle-sleep-ms "${THROTTLE_SLEEP_MS}"     --jitter-ms "${JITTER_MS}"     --min-rps "${MIN_RPS}"     --max-rps "${MAX_RPS}"     --step-up-rps "${STEP_UP_RPS}"     --down-mult "${DOWN_MULT}"     --burst-factor "${BURST_FACTOR}"

  if [ "${WRITE_DB}" = "true" ] || [ "${WRITE_DB}" = "1" ]; then
    if [ -z "${PG_DSN}" ]; then
      echo "[survival] ERROR: PG_DSN must be set when WRITE_DB=true" >&2
      exit 2
    fi
    set -- "$@" --write-db --pg-dsn "${PG_DSN}"
  else
    # For non-DB modes (e.g., diagnose), PG_DSN can be omitted.
    if [ -n "${PG_DSN}" ]; then
      set -- "$@" --pg-dsn "${PG_DSN}"
    fi
  fi

  if [ "${WRITE_CSV}" = "true" ] || [ "${WRITE_CSV}" = "1" ]; then
    set -- "$@" --write-csv --out "${OUT}" --history-out "${HISTORY_OUT}"
    if [ "${FRESH_CSV}" = "true" ] || [ "${FRESH_CSV}" = "1" ]; then
      set -- "$@" --fresh-csv
    fi
  fi

  if [ "${VERBOSE}" = "true" ] || [ "${VERBOSE}" = "1" ]; then
    set -- "$@" --verbose
  fi

  /usr/local/bin/survival "$@"
}

# Run one generation only (optional): GEN_ONLY=1
if [ -n "${GEN_ONLY:-}" ]; then
  run_gen "${GEN_ONLY}"
  exit 0
fi

# Run all generations in GEN_LIST
for g in ${GEN_LIST}; do
  run_gen "${g}"
  if [ "${SLEEP_BETWEEN}" != "0" ]; then
    sleep "${SLEEP_BETWEEN}"
  fi
done

exit 0
