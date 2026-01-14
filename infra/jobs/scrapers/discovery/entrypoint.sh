#!/usr/bin/env sh
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat >&2 <<'EOF'
entrypoint modes (set by first CLI arg or JOB env):

  ingest     run the ingest job (default)
  audit      run the audit job (requires PG_DSN)
  pipeline   run ingest and then audit (requires PG_DSN for audit)
  help       show this message

Configuration is via environment variables only.
See README.md and .env.example in the repo.
EOF
}

# ------------------------------
# Defaults (safe only)
# ------------------------------
: "${JOB:=ingest}"
: "${PG_SCHEMA:=public}"
: "${OUT_CSV:=/data/listings.csv}"
: "${SLEEP_BETWEEN:=0}"        # seconds between sweep segments (optional)
: "${SWEEP_QUERIES:=}"         # optional, '|' delimited list of SEARCH_QUERY values

# Do NOT set a default PG_DSN here. If provided at runtime, it may contain secrets.

run_ingest_once() {
  # ingestd reads env vars directly; we avoid echoing secrets.
  if [ -n "${PG_DSN:-}" ]; then
    echo "[entrypoint] ingest: sink=postgres schema=$PG_SCHEMA adapter=${MARKETPLACE_ADAPTER:-mock}"
  else
    echo "[entrypoint] ingest: sink=csv out=$OUT_CSV adapter=${MARKETPLACE_ADAPTER:-mock}"
  fi
  /usr/local/bin/ingestd
}

run_ingest() {
  if [ -n "${SWEEP_QUERIES}" ]; then
    # SWEEP_QUERIES is a '|' delimited list; preserve spaces within each query.
    count="$(printf '%s' "${SWEEP_QUERIES}" | tr '|' '\n' | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
    i=0
    printf '%s' "${SWEEP_QUERIES}" | tr '|' '\n' | sed '/^[[:space:]]*$/d' | while IFS= read -r q; do
      i=$((i+1))
      export SEARCH_QUERY="$q"
      echo "[entrypoint] ingest sweep: segment $i/$count SEARCH_QUERY='<redacted>'"
      run_ingest_once
      if [ "${SLEEP_BETWEEN}" -gt 0 ] && [ "$i" -lt "$count" ]; then
        sleep "${SLEEP_BETWEEN}"
      fi
    done
  else
    run_ingest_once
  fi
}

run_audit() {
  if [ -z "${PG_DSN:-}" ]; then
    die "PG_DSN is required for audit mode (set via secrets or Airflow Connection)."
  fi
  echo "[entrypoint] audit: schema=$PG_SCHEMA adapter=${MARKETPLACE_ADAPTER:-mock} dry_run=${AUDIT_DRY_RUN:-0}"
  /usr/local/bin/auditd
}

MODE="${1:-${JOB}}"
case "$MODE" in
  help|-h|--help)
    usage; exit 0
    ;;
  ingest|"")
    run_ingest
    ;;
  audit)
    run_audit
    ;;
  pipeline)
    run_ingest
    run_audit
    ;;
  *)
    usage
    die "Unknown MODE='$MODE'"
    ;;
esac
