#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# CONFIG
##############################################################################
KEEP_DAYS=${KEEP_DAYS:-14}
GOLD_TABLE="listings.iphone13_listings"
BACKUP_DIR="${BACKUP_DIR:?need BACKUP_DIR}"
STAMP="$(date +'%F_%H-%M-%S')"           # for filenames only
TS="$(date +%s)"                         # epoch seconds – becomes a *value*

##############################################################################
# 0 · Canary check
##############################################################################
echo "🔍  pre-flight: checking $GOLD_TABLE in $PGDATABASE …"
if ! pg_dump -h "$PGHOST" -U "$PGUSER" -s -t "$GOLD_TABLE" "$PGDATABASE" >/dev/null; then
  echo "❌  canary table missing!" >&2
  exit 23
fi
echo "🟢  canary table present"

##############################################################################
# 1 · Dump → gzip
##############################################################################
echo "🔄  running pg_dumpall …"
TMP="$BACKUP_DIR/.tmp_pg_${STAMP}.sql.gz"
pg_dumpall -h "$PGHOST" -U "$PGUSER" | gzip -9 > "$TMP"

##############################################################################
# 2 · Gzip integrity
##############################################################################
echo "🔎  validating gzip …"
gzip -t "$TMP"
echo "🟢  gzip OK"

##############################################################################
# 3 · Promote file
##############################################################################
FINAL="$BACKUP_DIR/pg_dumpall_${STAMP}.sql.gz"
mv "$TMP" "$FINAL"
echo "✅  saved $(basename "$FINAL") ($(du -h "$FINAL" | cut -f1))"

##############################################################################
# 4 · Prune old dumps
##############################################################################
echo "🧹  pruning >${KEEP_DAYS}-day dumps …"
find "$BACKUP_DIR" -maxdepth 1 -type f -name 'pg_dumpall_*.sql.gz' \
     -mtime +"$KEEP_DAYS" -print -delete || true

##############################################################################
# 5 · Volume metrics
##############################################################################
read -r total used avail _ < <(df --output=size,used,avail -B1 "$BACKUP_DIR" | tail -1)

##############################################################################
# 6 · Pushgateway  (NO extra timestamps!)
##############################################################################
cat <<EOF | curl -sf -XPUT -H 'Content-Type: text/plain' \
                 --data-binary @- "http://pushgateway:9091/metrics/job/backup"
backup_success 1
backup_success_timestamp_seconds ${TS}
pgbackup_bytes_total ${total}
pgbackup_bytes_free  ${avail}
EOF

echo "📡  metrics pushed to Pushgateway"
exit 0












