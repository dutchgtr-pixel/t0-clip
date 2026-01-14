#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psa_condition_sync_and_rescore.py

Synchronize PSA condition scores into iPhone.iphone_listings, append an audit entry,
apply hard-rule damage clamps for cs ∈ {1.0, 0.2}, and queue remaining rows
for LLM damage re-scoring by clearing damage_ai_version.

Run this right after:
  REFRESH MATERIALIZED VIEW "iPhone".post_sold_audit;
and before launching gbt_damage.py (with SKIP_ALREADY_LABELED=1).

Environment:
  PG_DSN               required (provide via environment; do not hardcode credentials)
  LOG_LEVEL            (default: INFO)
  PSA_COND_VERSION     e.g. 'psa-cond-sync-v1' (stamped into quality_ai_version + audit entry)
  EPS                  numeric tolerance for cond diff (default: 0.001)
  WINDOW_HOURS         optional integer; only consider PSA snapshots within last N hours
  ONLY_IDS             optional comma-separated listing_id list to restrict scope
  APPLY                'true' | 'false' (default: true) — when false, no DB writes, only logs
  HARD_RULE_VERSION    version label to stamp when clamping (default: 'rule-cs-hard-v1')
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SCHEMA   = '"iPhone"'
LISTINGS = f'{SCHEMA}.iphone_listings'
PSA      = f'{SCHEMA}.post_sold_audit'


def get_engine() -> Engine:
    dsn = os.getenv("PG_DSN")
    if not dsn:
        raise SystemExit("Set PG_DSN env var (SQLAlchemy Postgres DSN).")
    return create_engine(dsn, future=True, pool_pre_ping=True)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _iso(val) -> Optional[str]:
    if val is None:
        return None
    try:
        if hasattr(val, "to_pydatetime"):
            return val.to_pydatetime().isoformat()
        return val.isoformat()
    except Exception:
        return str(val)


def find_cond_diffs(
    engine: Engine,
    eps: float,
    only_ids: Optional[List[int]] = None,
    window_hours: Optional[int] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Return rows where latest PSA condition_score_snap differs from main condition_score by > eps."""
    params: Dict[str, Any] = {"eps": float(eps), "limit": int(limit)}
    window_clause = ""
    if isinstance(window_hours, int) and window_hours > 0:
        params["cutoff"] = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        window_clause = 'AND a."snapshot_at" >= :cutoff'

    only_clause = ""
    if only_ids:
        params["ids"] = list(map(int, only_ids))
        only_clause = "AND l.listing_id = ANY(:ids)"

    sql = f"""
    WITH latest AS (
      SELECT DISTINCT ON (a.listing_id)
             a.listing_id,
             a."snapshot_at" AS snap_at,
             a."condition_score_snap" AS cond_snap
      FROM {PSA} a
      WHERE a."condition_score_snap" IS NOT NULL
        {window_clause}
      ORDER BY a.listing_id, a."snapshot_at" DESC
    )
    SELECT
      l.listing_id,
      l."condition_score" AS cond_main,
      x.cond_snap          AS cond_snap,
      x.snap_at            AS snap_at
    FROM {LISTINGS} l
    JOIN latest x ON x.listing_id = l.listing_id
    WHERE 1=1
      {only_clause}
      AND x.cond_snap IS NOT NULL
      AND (
        l."condition_score" IS DISTINCT FROM x.cond_snap
        AND (l."condition_score" IS NULL OR ABS(l."condition_score" - x.cond_snap) > :eps)
      )
    ORDER BY ABS(x.cond_snap - COALESCE(l."condition_score", 0)) DESC, x.snap_at DESC
    LIMIT :limit
    """
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text(sql), params).mappings()]
    return rows


def apply_sync_and_queue(
    engine: Engine,
    diffs: List[Dict[str, Any]],
    audit_version: str,
    hard_rule_version: str,
    apply: bool,
) -> Dict[str, Any]:
    """Write condition, append audit, clamp where needed, queue others for LLM. Returns summary dict."""
    summary = {"updated": [], "clamped_mint": [], "clamped_sev3": [], "queued": []}
    if not diffs:
        return summary

    with engine.begin() as conn:
        # Detect optional column presence
        has_damage_ai_at = bool(
            conn.execute(
                text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema='iPhone' AND table_name='iphone_listings' AND column_name='damage_ai_at'
                """)
            ).fetchone()
        )
        ai_at_set_now  = ', "damage_ai_at" = now()'  if has_damage_ai_at else ''
        ai_at_set_null = ', "damage_ai_at" = NULL'   if has_damage_ai_at else ''

        for row in diffs:
            fid       = int(row["listing_id"])
            cond_main = row.get("cond_main")
            cond_snap = float(row["cond_snap"])
            snap_at   = row.get("snap_at")
            summary["updated"].append(fid)

            if not apply:
                continue

            entry = {
                "version": audit_version,
                "at":      datetime.now(timezone.utc).isoformat(),
                "change":  "condition_score",
                "from":    float(cond_main) if (cond_main is not None) else None,
                "to":      float(cond_snap),
                "source":  "psa.condition_score_snap",
                "snap_at": _iso(snap_at),
            }

            # 1) Write condition + append audit entry
            sql_update_cond = f"""
                UPDATE {LISTINGS} AS l
                SET
                  "condition_score"   = :new_cond,
                  "quality_ai_json"   =
                    CASE
                      WHEN l."quality_ai_json" IS NULL
                        THEN jsonb_build_array(CAST(:entry AS jsonb))
                      WHEN jsonb_typeof(l."quality_ai_json") = 'array'
                        THEN l."quality_ai_json" || CAST(:entry AS jsonb)
                      ELSE jsonb_build_array(l."quality_ai_json") || CAST(:entry AS jsonb)
                    END,
                  "quality_ai_at"     = now(),
                  "quality_ai_version"= :ver
                WHERE l."listing_id" = :fid
            """
            conn.execute(
                text(sql_update_cond),
                {"fid": fid, "new_cond": cond_snap, "entry": json.dumps(entry, ensure_ascii=False), "ver": audit_version},
            )

            # 2) Hard clamps
            if abs(cond_snap - 1.0) < 1e-9:
                sql_clamp_mint = f"""
                  UPDATE {LISTINGS}
                  SET
                    "damage_severity"   = 0,
                    "damage_binary"     = 0,
                    "damage_reason_ai"  = 'brand new (cs1.0)',
                    "damage_ai_version" = :hrv{ai_at_set_now}
                  WHERE "listing_id" = :fid
                """
                conn.execute(text(sql_clamp_mint), {"fid": fid, "hrv": hard_rule_version})

                sql_audit_mint = f"""
                  UPDATE {LISTINGS}
                  SET "quality_ai_json" =
                    CASE
                      WHEN "quality_ai_json" IS NULL
                        THEN jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=1.0 → bin0/sev0'))
                      WHEN jsonb_typeof("quality_ai_json")='array'
                        THEN "quality_ai_json" || jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=1.0 → bin0/sev0'))
                      ELSE jsonb_build_array("quality_ai_json") || jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=1.0 → bin0/sev0'))
                    END,
                    "quality_ai_at" = now()
                  WHERE "listing_id" = :fid
                """
                conn.execute(text(sql_audit_mint), {"fid": fid, "ver": audit_version})

            elif abs(cond_snap - 0.2) < 1e-9:
                sql_clamp_sev3 = f"""
                  UPDATE {LISTINGS}
                  SET
                    "damage_severity"   = 3,
                    "damage_binary"     = 1,
                    "damage_reason_ai"  = 'pre-determined sev3 (cs0.2)',
                    "damage_ai_version" = :hrv{ai_at_set_now}
                  WHERE "listing_id" = :fid
                """
                conn.execute(text(sql_clamp_sev3), {"fid": fid, "hrv": hard_rule_version})

                sql_audit_sev3 = f"""
                  UPDATE {LISTINGS}
                  SET "quality_ai_json" =
                    CASE
                      WHEN "quality_ai_json" IS NULL
                        THEN jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=0.2 → bin1/sev3'))
                      WHEN jsonb_typeof("quality_ai_json")='array'
                        THEN "quality_ai_json" || jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=0.2 → bin1/sev3'))
                      ELSE jsonb_build_array("quality_ai_json") || jsonb_build_array(jsonb_build_object('version', :ver, 'at', now(), 'change','damage','note','hard-clamp via cs=0.2 → bin1/sev3'))
                    END,
                    "quality_ai_at" = now()
                  WHERE "listing_id" = :fid
                """
                conn.execute(text(sql_audit_sev3), {"fid": fid, "ver": audit_version})

            # 3) Queue for LLM damage re-score if not clamped
            else:
                sql_queue = f"""
                  UPDATE {LISTINGS}
                  SET
                    "damage_ai_version" = NULL{ai_at_set_null},
                    "quality_ai_json" =
                      CASE
                        WHEN "quality_ai_json" IS NULL
                          THEN jsonb_build_array(jsonb_build_object(
                            'version', :ver, 'at', now(), 'change','condition_score',
                            'queued','damage_rescore_due_to_psa_cond_change', 'snap_at', :snap
                          ))
                        WHEN jsonb_typeof("quality_ai_json")='array'
                          THEN "quality_ai_json" || jsonb_build_array(jsonb_build_object(
                            'version', :ver, 'at', now(), 'change','condition_score',
                            'queued','damage_rescore_due_to_psa_cond_change', 'snap_at', :snap
                          ))
                        ELSE jsonb_build_array("quality_ai_json") || jsonb_build_array(jsonb_build_object(
                          'version', :ver, 'at', now(), 'change','condition_score',
                          'queued','damage_rescore_due_to_psa_cond_change', 'snap_at', :snap
                        ))
                      END,
                    "quality_ai_at" = now()
                  WHERE "listing_id" = :fid
                """
                conn.execute(
                    text(sql_queue),
                    {"fid": fid, "ver": audit_version, "snap": _iso(snap_at)},
                )

    return summary


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("psa_cond_sync")

    engine            = get_engine()
    audit_version     = os.getenv("PSA_COND_VERSION", "psa-cond-sync-v1")
    hard_rule_version = os.getenv("HARD_RULE_VERSION", "rule-cs-hard-v1")
    eps               = float(os.getenv("EPS", "0.001"))
    window_env        = os.getenv("WINDOW_HOURS", "").strip()
    window_hours      = int(window_env) if (window_env.isdigit()) else None
    only_ids_env      = os.getenv("ONLY_IDS", "").strip()
    only_ids          = [int(x) for x in only_ids_env.split(",")] if only_ids_env else None
    apply             = os.getenv("APPLY", "true").lower() != "false"

    logger.info("Starting PSA→Cond sync | EPS=%s | WINDOW_HOURS=%s | ONLY_IDS=%s | APPLY=%s",
                eps, (window_hours if window_hours is not None else "ALL"),
                (only_ids if only_ids else "ALL"), apply)

    diffs = find_cond_diffs(engine, eps, only_ids=only_ids, window_hours=window_hours)
    logger.info("Found %d rows with |psa.condition_score_snap - condition_score| > %s", len(diffs), eps)

    summary = apply_sync_and_queue(engine, diffs, audit_version, hard_rule_version, apply)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))

    if not apply:
        logger.info("DRY_RUN=true — no changes have been written. Re-run with APPLY=true to commit.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)





