#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psa_sold_price_sync.py

Synchronize post-sale sold_price_snapshot into listings.sold_price
**ONLY for rows where PSA and main disagree**.

Matches your SQL logic where:
  sold_diff = (sold_snap IS NOT NULL AND sold_main IS DISTINCT FROM sold_snap)

So:
  - If PSA has sold_price_snapshot
  - AND iphone_listings.sold_price IS DISTINCT FROM that value
  → we overwrite sold_price from PSA and append an audit JSON entry.

We also log, for EVERY row:
  listing_id, status, sold_main, sold_psa, snap_at, apply_flag

Run this right after:
  REFRESH MATERIALIZED VIEW <POST_SALE_AUDIT_VIEW>;

Environment:
  PG_DSN             e.g. <REDACTED_DSN>
  LOG_LEVEL          (default: INFO)
  WINDOW_HOURS       optional int, only consider PSA snapshots in last N hours
  ONLY_IDS           optional comma-separated listing_ids (e.g. "123,456")
  LIMIT              max rows to process per run (default: 5000)
  APPLY              "false" → dry-run (log only), anything else → apply updates
  PSA_PRICE_VERSION  audit version tag (default: "psa-price-sync-v1")
"""

import os
import sys
import json
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

PG_SCHEMA = (os.getenv("PG_SCHEMA") or "public").strip() or "public"
LISTINGS_TABLE_NAME = (os.getenv("LISTINGS_TABLE") or "listings").strip() or "listings"
AUDIT_TABLE_NAME = (os.getenv("POST_SALE_AUDIT_TABLE") or "post_sale_audit").strip() or "post_sale_audit"
LISTING_ID_COLUMN = (os.getenv("LISTING_ID_COLUMN") or "listing_id").strip() or "listing_id"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _qident(name: str) -> str:
    """Quote a PostgreSQL identifier safely (schema/table/column)."""
    if not _IDENTIFIER_RE.match(name):
        raise SystemExit(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'

SCHEMA   = _qident(PG_SCHEMA)
LISTINGS = f"{SCHEMA}.{_qident(LISTINGS_TABLE_NAME)}"
PSA      = f"{SCHEMA}.{_qident(AUDIT_TABLE_NAME)}"
ID_COL   = _qident(LISTING_ID_COLUMN)

logger = logging.getLogger("psa_sold_price_sync")


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise SystemExit("PG_DSN environment variable is required (provide via environment/secret manager; do not hardcode credentials).")
    return create_engine(dsn, future=True, pool_pre_ping=True)


def _iso(val) -> Optional[str]:
    """Best-effort ISO formatting of timestamps."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    try:
        to_py = getattr(val, "to_pydatetime", None)
        if callable(to_py):
            return to_py().isoformat()
    except Exception:
        pass
    return str(val)


# ──────────────────────────────────────────────────────────────────────────────
# core fetch — ONLY rows where PSA ≠ main
# ──────────────────────────────────────────────────────────────────────────────

def fetch_price_diffs(
    engine: Engine,
    window_hours: Optional[int],
    only_ids: Optional[List[int]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Return rows where latest PSA sold_price_snapshot exists for a SOLD listing
    AND main.sold_price IS DISTINCT FROM PSA.sold_price_snapshot.

    This is the "734 rows" bucket from your SQL:
      sold_diff = (sold_snap IS NOT NULL AND sold_main IS DISTINCT FROM sold_snap)
    """
    params: Dict[str, Any] = {"limit": int(limit)}

    window_clause = ""
    if isinstance(window_hours, int) and window_hours > 0:
        params["cutoff"] = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        window_clause = 'AND a."snapshot_at" >= :cutoff'

    only_clause = ""
    if only_ids:
        params["ids"] = list(map(int, only_ids))
        only_clause = "AND l.{ID_COL} = ANY(:ids)"

    sql = f"""
    WITH latest AS (
      SELECT DISTINCT ON (a.{ID_COL})
             a.{ID_COL} AS listing_id,
             a."snapshot_at"         AS snap_at,
             a."sold_price_snapshot" AS sold_snap
      FROM {PSA} a
      WHERE a."sold_price_snapshot" IS NOT NULL
        AND a."page_ok" = TRUE
        AND a."http_status" BETWEEN 200 AND 299
        {window_clause}
      ORDER BY a.{ID_COL}, a."snapshot_at" DESC
    )
    SELECT
      l.{ID_COL} AS listing_id,
      l."status",
      l."sold_price" AS sold_main,
      x.sold_snap,
      x.snap_at
    FROM latest x
    JOIN {LISTINGS} l USING ({ID_COL})
    WHERE l."status" = 'sold'
      AND l."sold_price" IS DISTINCT FROM x.sold_snap
      {only_clause}
    ORDER BY x.snap_at DESC
    LIMIT :limit
    """

    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text(sql), params).mappings()]
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# apply sync + audit (with detailed logging)
# ──────────────────────────────────────────────────────────────────────────────

def apply_price_sync(
    engine: Engine,
    rows: List[Dict[str, Any]],
    audit_version: str,
    apply: bool,
) -> Dict[str, Any]:
    """
    For each row where PSA and main differ, set sold_price = PSA sold_snap.

    Always overwrite (that's the whole point: they differ).
    Append an audit entry into quality_ai_json.
    Also log listing_id + prices + status for every row, regardless of dry-run or not.
    """
    summary: Dict[str, Any] = {
        "count": len(rows),
        "updated": [],
    }
    if not rows:
        return summary

    with engine.begin() as conn:
        for row in rows:
            fid       = int(row["listing_id"])
            status    = row.get("status")
            sold_main = row.get("sold_main")
            sold_snap = row.get("sold_snap")
            snap_at   = row.get("snap_at")

            if sold_snap is None:
                # Shouldn't happen given WHERE clause, but be safe.
                continue

            # Log the mismatch so you can see WTF is going on.
            logger.info(
                "PRICE_DIFF | fid=%s | status=%s | sold_main=%s | sold_psa=%s | snap_at=%s | apply=%s",
                fid,
                status,
                sold_main,
                sold_snap,
                _iso(snap_at),
                apply,
            )

            summary["updated"].append(fid)

            if not apply:
                # Dry-run: log only, do not touch DB.
                continue

            entry = {
                "version": audit_version,
                "at":      datetime.now(timezone.utc).isoformat(),
                "change":  "sold_price",
                "from":    int(sold_main) if sold_main is not None else None,
                "to":      int(sold_snap),
                "source":  "psa.sold_price_snapshot",
                "snap_at": _iso(snap_at),
                "policy":  "psa_overwrite_diff",
                "status_at_change": status,
            }

            sql_update = f"""
              UPDATE {LISTINGS} AS l
              SET
                "sold_price" = :new_price,
                "quality_ai_json" =
                  CASE
                    WHEN l."quality_ai_json" IS NULL
                      THEN jsonb_build_array(CAST(:entry AS jsonb))
                    WHEN jsonb_typeof(l."quality_ai_json") = 'array'
                      THEN l."quality_ai_json" || CAST(:entry AS jsonb)
                    ELSE jsonb_build_array(l."quality_ai_json") || CAST(:entry AS jsonb)
                  END,
                "quality_ai_at"      = now(),
                "quality_ai_version" = :ver
              WHERE l.{ID_COL} = :fid
            """

            conn.execute(
                text(sql_update),
                {
                    "fid": fid,
                    "new_price": int(sold_snap),
                    "entry": json.dumps(entry, ensure_ascii=False),
                    "ver": audit_version,
                },
            )

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    engine        = get_engine()
    audit_version = os.getenv("PSA_PRICE_VERSION", "psa-price-sync-v1")

    window_env   = os.getenv("WINDOW_HOURS", "").strip()
    window_hours = int(window_env) if window_env.isdigit() else None

    only_ids_env = os.getenv("ONLY_IDS", "").strip()
    if only_ids_env:
        only_ids = [int(x.strip()) for x in only_ids_env.split(",") if x.strip()]
    else:
        only_ids = None

    limit_env = os.getenv("LIMIT", "").strip()
    limit     = int(limit_env) if limit_env.isdigit() else 5000

    apply_env = os.getenv("APPLY", "false").strip().lower()
    apply     = apply_env not in ("0", "false", "no")

    logger.info(
        "Starting PSA→sold_price DIFF sync | version=%s | WINDOW_HOURS=%s | "
        "ONLY_IDS=%s | LIMIT=%s | APPLY=%s",
        audit_version, window_hours, only_ids, limit, apply,
    )

    rows = fetch_price_diffs(
        engine=engine,
        window_hours=window_hours,
        only_ids=only_ids,
        limit=limit,
    )
    logger.info("Fetched %d rows where PSA and main differ", len(rows))

    summary = apply_price_sync(
        engine=engine,
        rows=rows,
        audit_version=audit_version,
        apply=apply,
    )

    logger.info(
        "Done | total_candidates=%d | updated=%d",
        summary.get("count", 0),
        len(summary.get("updated", [])),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

