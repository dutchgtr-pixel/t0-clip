#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Battery Refit v5.2 — LLM decides, PSA+MAIN eval for metrics, PSA-first write, rich end-of-run metrics

RULE (hard-coded):
• Evaluate BOTH sources for metrics:
  - PSA text (title_snapshot + description_snapshot) ⇒ psa_guess
  - MAIN text (title + description) ⇒ main_guess
• Final write still follows PSA-first policy:
  - If psa_guess exists → final = psa_guess, source="psa"
  - Else if main_guess exists → final = main_guess, source="main"
  - Else → unknown (MAIN ai set to 0 if UNANCHORED_POLICY=zero; PSA snapshot set to 0 if UPDATE_PSA=1)
• Prefer PSA if both explicit and different (PSA wins).

WHAT IT DOES
• Calls LLM separately for PSA and MAIN to compute psa_guess and main_guess.
• Writes ONE final number to:
  - MAIN "iPhone".iphone_listings.battery_pct_fixed_ai  (or 0 if unknown)
  - PSA  "iPhone".post_sold_audit.battery_pct_snapshot (or 0 if unknown & UPDATE_PSA=1)
• Appends one JSON entry per row to "iPhone".post_sold_audit.battery_refit_log (no new columns).

END-OF-RUN METRICS (logs)
• rows_total, psa_text_rows
• final_source: psa / main / unknown
• coverage: psa_anchored, main_after_psa_fail
• psa_unique_adds (PSA created a number when raw/fixed/fixed_ai had none)
• NEW adds overall (final had value, battery_eff had none): new_adds_total / by source
• BOGUS removed (battery_eff had a value 20..100, final is unknown) — excludes NULL→0
• corrections_vs_raw/fixed/fixed_ai (PSA only) with avg |Δ|
• corrections_vs_eff (PSA+MAIN) with avg |Δ|, and changed_vs_eff counts
• psa_overrode_main_conflict (both anchored, different → PSA wins)
• sub80_upgrade_candidates (final<80 & pre_damage_sev∈{0,1} & pre_main_fixed_ai∈{NULL,0})
• unknown_top_reasons (top 5)
• normalizations: emoji_accepts, decimal_rounds, range_lower_bounds

No schema changes. Uses only existing columns.

NEW QUEUING RULES (battery_llm_fetch sentinel):
• Only queue listings whose latest PSA (day=7) meets ALL:
  - http_status BETWEEN 200 AND 299
  - (title_snapshot OR description_snapshot) IS NOT NULL
  - MAIN condition_score <> 1.0 (non-mint)
  - post_sold_audit.battery_llm_fetch IS NULL  ← not fetched yet (sentinel)
• After processing, set battery_llm_fetch='fetched' for that snapshot.
• Always overwrite MAIN battery and processed_by with this tool’s tag.
"""

from __future__ import annotations
import os, sys, json, time, logging, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from openai import OpenAI
from openai.types.chat import ChatCompletion

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs): return x

# ----------------------- Config -----------------------

def env_bool(name: str, default: bool=False) -> bool:
    raw = os.getenv(name)
    if raw is None: return default
    return raw.strip().lower() in ("1","true","yes")

# DB DSNs
PG_DSN = os.getenv("PG_DSN", "").strip()  # required; do not hardcode credentials in repo
PG_DSN_FALLBACKS = os.getenv("PG_DSN_FALLBACKS", "").strip()

# Model creds
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

WORKERS = int(os.getenv("WORKERS", "16"))
DRY_RUN  = env_bool("DRY_RUN", True)

# batching
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
BATCH_PAGE = int(os.getenv("BATCH_PAGE", "0"))
LIMIT      = int(os.getenv("LIMIT", "0"))

MAX_RETRIES      = max(1, int(os.getenv("MAX_RETRIES", "3")))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "0.6"))

# behavior / selection
# NOTE: The sentinel logic (battery_llm_fetch IS NULL) now controls queueing.
#       SKIP_ALREADY_PROCESSED / TARGET_NULL are ignored by selection.
PSA_ONLY                 = env_bool("PSA_ONLY", True)
EXCLUDE_COND_SCORE_EQ_1  = env_bool("EXCLUDE_COND_SCORE_EQ_1", True)
SKIP_ALREADY_PROCESSED   = env_bool("SKIP_ALREADY_PROCESSED", False)  # not used by selection anymore
TARGET_NULL              = env_bool("TARGET_NULL", False)             # not used by selection anymore
RECHECK_NULLIFIED        = env_bool("RECHECK_NULLIFIED", False)
STRICT_NULL_LEGACY       = env_bool("STRICT_NULL_LEGACY", False)

UPDATE_MAIN  = env_bool("UPDATE_MAIN", True)
UPDATE_PSA   = env_bool("UPDATE_PSA",  True)
UNANCHORED_POLICY = os.getenv("UNANCHORED_POLICY", "zero").strip().lower()  # 'zero' or 'null'

# Evaluation mode
EVAL_BOTH = env_bool("EVAL_BOTH", True)  # evaluate both PSA and MAIN for metrics; write remains PSA-first

LISTING_IDS = os.getenv("LISTING_IDS", "").strip()

TAG_BY = "battery_refit_v5_psa"

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ----------------------- Logging -----------------------

logger = logging.getLogger("battery_refit_v5")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ----------------------- Prompt -----------------------

BATTERY_SYSTEM_PROMPT = r"""
You are extracting the **battery health percentage** for a used iPhone.

INPUTS you may receive (PSA-only or MAIN-only per call):
- PSA: post_sold_audit snapshot (title/description)
- MAIN: original listing (title/description)

YOUR TASK
• Output ONE battery-health percent (20..100) **ONLY if** the text clearly refers to battery *health / maximum capacity*.
• If no clear health % appears, return null.

RETURN STRICT JSON ONLY:
{
  "override": true|false,
  "battery_pct": null|20..100,
  "evidence": "<=160 chars minimal quote justifying the number",
  "reason": "<=80 chars e.g., 'explicit %', 'rounded decimal', 'ratio 97/100'"
}

BE PERMISSIVE (use language understanding)
ACCEPT battery health when a 20..100 number is clearly about battery max capacity/health, even without a % sign:
• “Batteri 100%”, “Batteriet er 100%”, “Batteriet er 100”, “Batteri på 95”
• “Batterikapasitet 82%”, “Batterihelsen 93%”, “Maksimal kapasitet 89%”
• “iOS viser 83 %”, “Batteriprosent 94 %”, “Batterinivå 85 % (helse)”
• Decimals “87,7%” → round to nearest int (→ 88). Ranges “80–85%” → lower bound (→ 80).
• Ratios “97/100”, “health 97/100” → interpret as % (→ 97).
• Emoji/symbols “100% 🔋”, “93%🔋”, “🔋 90%” (likely battery health here).
• Abbrev/typos near battery: “BH 92%”, “MAC 97%”, “max cap 89%”, “batt 95%”.
• Spelled-out numerals in context: e.g., “hundre prosent batterihelse” → 100; “nittisyv prosent” → 97.
• “ca / ~ / about 90%” → 90.
• “Se bilde viser 88 %” → accept 88 (seller states a numeric health).

REJECT → null when not health:
• Charging/limits: “charged to 85%”, “optimized 80%”, “limit 80%”, “lades til 85%”
• Condition-only: “100% strøken/mint” (quality, not health)
• Damage %: “batteriet er 90% ødelagt”
• Ambiguous power: “80% strøm kapasitet” without explicit health meaning
• “nytt batteri / new battery” with no number
"""

# ----------------------- DB helpers -----------------------

def _dsn_candidates() -> List[str]:
    cands: List[str] = []
    if PG_DSN: cands.append(PG_DSN.strip())
    if PG_DSN_FALLBACKS:
        for tok in re.split(r"[,\s]+", PG_DSN_FALLBACKS.strip()):
            if tok: cands.append(tok)
    defaults = []  # public release: no embedded DSN defaults
    for d in defaults:
        if d not in cands: cands.append(d)
    # dedupe
    seen=set(); out=[]
    for d in cands:
        if d not in seen:
            out.append(d); seen.add(d)
    return out

def redact_dsn(dsn: str) -> str:
    """Redact credentials in a DSN for safe logging."""
    if not dsn:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit
        p = urlsplit(dsn)
        netloc = p.netloc
        if "@" in netloc:
            _, host = netloc.rsplit("@", 1)
            netloc = f"***@{host}"
        return urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))
    except Exception:
        # Fallback: best-effort credential removal.
        return re.sub(r"//[^@/]+@", "//***@", dsn)

def get_engine() -> Engine:
    dsns = _dsn_candidates()
    if not dsns:
        raise RuntimeError("PG_DSN is required (no DSN defaults are embedded in the public release).")

    last_err: Optional[Exception] = None
    for dsn in dsns:
        try:
            eng = create_engine(dsn, future=True, pool_pre_ping=True)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"DB connect OK: {redact_dsn(dsn)}")
            return eng
        except Exception as e:
            last_err = e
            logger.warning(f"DB connect failed: {redact_dsn(dsn)} | {e}")
            time.sleep(0.1)

    raise RuntimeError(f"All DB connections failed. Last error: {last_err}")

def ensure_psa_columns(engine: Engine) -> None:
    """Ensure PSA has the columns/indexes we rely on."""
    with engine.begin() as conn:
        # Battery refit log column & index (existing behavior)
        conn.execute(text("""
        ALTER TABLE "iPhone".post_sold_audit
        ADD COLUMN IF NOT EXISTS battery_refit_log jsonb
        """))
        conn.execute(text("""
        UPDATE "iPhone".post_sold_audit
        SET battery_refit_log = '[]'::jsonb
        WHERE battery_refit_log IS NULL
        """))
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS post_sold_audit_listing_id_snapshot_at_idx
        ON "iPhone".post_sold_audit (listing_id, snapshot_at DESC)
        """))
        # Sentinel column + helpful index
        conn.execute(text("""
        ALTER TABLE "iPhone".post_sold_audit
        ADD COLUMN IF NOT EXISTS battery_llm_fetch text
        """))
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS psa_llm_fetch_idx
        ON "iPhone".post_sold_audit (day_offset, http_status, battery_llm_fetch)
        """))

def parse_listing_ids() -> List[int]:
    ids: List[int] = []
    raw = LISTING_IDS.strip()
    if not raw: return ids
    for tok in re.split(r"[^0-9]+", raw):
        if tok:
            try: ids.append(int(tok))
            except: pass
    return ids

def fetch_rows(engine: Engine, limit: int = 0, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Queue rows ONLY when latest PSA (day=7) is HTTP 200, PSA has text,
    MAIN condition_score <> 1.0, and PSA battery_llm_fetch IS NULL.
    LISTING_IDS override handled separately.
    """
    conds = []
    # PSA text exists (when PSA_ONLY)
    if PSA_ONLY:
        conds.append("(p.title_snapshot IS NOT NULL OR p.description_snapshot IS NOT NULL)")
    # HTTP OK on PSA
    conds.append("(p.http_status BETWEEN 200 AND 299)")
    # Non-mint
    if EXCLUDE_COND_SCORE_EQ_1:
        conds.append("COALESCE(t.condition_score, -999) <> 1")
    # Sentinel: not fetched yet
    conds.append("(p.battery_llm_fetch IS NULL)")

    # Legacy gates (no-ops for our sentinel flow, but kept for opt-in modes)
    if RECHECK_NULLIFIED:
        conds.append("COALESCE(t.battery_pct_fixed_ai, 0) = 0")
        conds.append("t.battery_pct_fixed_by = :tag_by")
    elif TARGET_NULL:
        conds.append("COALESCE(t.battery_pct_fixed_ai, 0) = 0")
    if STRICT_NULL_LEGACY:
        conds.append("(t.battery_pct IS NULL OR t.battery_pct NOT BETWEEN 20 AND 100)")
        conds.append("(t.battery_pct_fixed IS NULL OR t.battery_pct_fixed NOT BETWEEN 20 AND 100)")

    where_clause = " AND ".join(conds) if conds else "TRUE"

    psa_cte = """
    WITH latest_psa AS (
      SELECT DISTINCT ON (a.listing_id)
             a.listing_id, a.snapshot_at,
             a.http_status,
             a.title_snapshot, a.description_snapshot,
             a.battery_pct_snapshot, a.battery_refit_log,
             a.battery_llm_fetch
      FROM "iPhone".post_sold_audit a
      WHERE a.day_offset = 7
      ORDER BY a.listing_id, a.snapshot_at DESC
    )
    """
    sql = f"""
    {psa_cte}
    SELECT
      t.listing_id, t.title, t.description, t.edited_date,
      t.battery_pct, t.battery_pct_fixed, t.battery_pct_fixed_ai, t.battery_pct_fixed_by,
      t.damage_severity_ai AS pre_damage_sev,
      p.snapshot_at           AS psa_snapshot_at,
      p.title_snapshot        AS psa_title,
      p.description_snapshot  AS psa_desc,
      p.battery_pct_snapshot  AS psa_batt,
      p.battery_refit_log     AS psa_log
    FROM "iPhone".iphone_listings t
    LEFT JOIN latest_psa p ON p.listing_id = t.listing_id
    WHERE {where_clause}
    ORDER BY t.edited_date DESC NULLS LAST, t.generation ASC, t.listing_id ASC
    {{ 'LIMIT :lim' if limit>0 else '' }}
    {{ 'OFFSET :off' if offset>0 else '' }}
    """
    # replace pseudo-template
    if limit>0:
        sql = sql.replace("{ 'LIMIT :lim' if limit>0 else '' }", "LIMIT :lim")
    else:
        sql = sql.replace("{ 'LIMIT :lim' if limit>0 else '' }", "")
    if offset>0:
        sql = sql.replace("{ 'OFFSET :off' if offset>0 else '' }", "OFFSET :off")
    else:
        sql = sql.replace("{ 'OFFSET :off' if offset>0 else '' }", "")

    params: Dict[str, Any] = {"tag_by": TAG_BY}
    if limit>0: params["lim"] = int(limit)
    if offset>0: params["off"] = int(offset)
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings()]

def fetch_rows_by_ids(engine: Engine, ids: List[int]) -> List[Dict[str, Any]]:
    if not ids: return []
    values_sql = ",".join(f"({int(x)})" for x in ids)
    psa_cte = """
    WITH latest_psa AS (
      SELECT DISTINCT ON (a.listing_id)
             a.listing_id, a.snapshot_at,
             a.http_status,
             a.title_snapshot, a.description_snapshot,
             a.battery_pct_snapshot, a.battery_refit_log,
             a.battery_llm_fetch
      FROM "iPhone".post_sold_audit a
      WHERE a.day_offset = 7
      ORDER BY a.listing_id, a.snapshot_at DESC
    )
    """
    sql = f"""
    {psa_cte}
    , ids(id) AS (VALUES {values_sql})
    SELECT
      t.listing_id, t.title, t.description, t.edited_date,
      t.battery_pct, t.battery_pct_fixed, t.battery_pct_fixed_ai, t.battery_pct_fixed_by,
      t.damage_severity_ai AS pre_damage_sev,
      p.snapshot_at           AS psa_snapshot_at,
      p.title_snapshot        AS psa_title,
      p.description_snapshot  AS psa_desc,
      p.battery_pct_snapshot  AS psa_batt,
      p.battery_refit_log     AS psa_log
    FROM "iPhone".iphone_listings t
    JOIN ids ON ids.id = t.listing_id
    LEFT JOIN latest_psa p ON p.listing_id = t.listing_id
    ORDER BY t.edited_date DESC NULLS LAST, t.generation ASC, t.listing_id ASC
    """
    with engine.begin() as conn:
        return [dict(r) for r in conn.execute(text(sql)).mappings()]

# ----------------------- LLM client -----------------------

_CLIENT: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key  = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("\ufeffOPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY / DEEPSEEK_API_KEY required")
        _CLIENT = OpenAI(api_key=key, base_url=base or None)
    return _CLIENT

def _strip_code_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else s

def _call_llm(one_source_title: str, one_source_desc: str) -> Dict[str, Any]:
    client = get_client()
    user_payload = json.dumps({
        "title": (one_source_title or "").strip(),
        "description": (one_source_desc or "").strip(),
    }, ensure_ascii=False)
    last_err = "unknown"
    for attempt in range(1, MAX_RETRIES+1):
        try:
            resp: ChatCompletion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", MODEL_NAME),
                messages=[
                    {"role":"system","content":BATTERY_SYSTEM_PROMPT},
                    {"role":"user","content":user_payload},
                ],
                response_format={"type":"json_object"},
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            payload = resp.choices[0].message.content or ""
            try:
                data = json.loads(_strip_code_fence(payload))
            except Exception:
                data = {}
            override = bool(data.get("override", False))
            pct_raw  = data.get("battery_pct")
            pct: Optional[int] = None
            if pct_raw is not None:
                try:
                    pct = int(round(float(pct_raw)))
                except Exception:
                    pct = None
                if pct is not None and not (20 <= pct <= 100):
                    pct = None
                    override = False
            evidence = (data.get("evidence") or "")[:160]
            reason   = (data.get("reason") or "")[:80]
            return {"override":override, "battery_pct":pct, "evidence":evidence, "reason":reason}
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_BASE_SLEEP * (2**(attempt-1)))
    logger.error(f"LLM failed: {last_err}")
    return {"override":False, "battery_pct":None, "evidence":"", "reason":f"error:{last_err[:40]}"}

def call_battery_eval_both(row: Dict[str, Any]) -> Dict[str, Any]:
    psa_title = (row.get("psa_title") or "").strip()
    psa_desc  = (row.get("psa_desc")  or "").strip()
    main_title= (row.get("title")     or "").strip()
    main_desc = (row.get("description") or "").strip()

    psa_guess = psa_ev = psa_why = None
    main_guess = main_ev = main_why = None

    if psa_title or psa_desc:
        r_psa = _call_llm(psa_title, psa_desc)
        psa_guess = r_psa.get("battery_pct")
        psa_ev    = r_psa.get("evidence") or ""
        psa_why   = r_psa.get("reason") or ""

    if EVAL_BOTH or not (psa_guess is not None):
        if main_title or main_desc:
            r_main = _call_llm(main_title, main_desc)
            main_guess = r_main.get("battery_pct")
            main_ev    = r_main.get("evidence") or ""
            main_why   = r_main.get("reason") or ""

    # Final PSA-first selection
    if psa_guess is not None:
        return {
            "psa_guess": psa_guess, "psa_ev": psa_ev, "psa_why": psa_why,
            "main_guess": main_guess, "main_ev": main_ev, "main_why": main_why,
            "final": psa_guess, "source": "psa",
            "evidence": psa_ev, "reason": psa_why
        }
    if main_guess is not None:
        return {
            "psa_guess": psa_guess, "psa_ev": psa_ev, "psa_why": psa_why,
            "main_guess": main_guess, "main_ev": main_ev, "main_why": main_why,
            "final": main_guess, "source": "main",
            "evidence": main_ev, "reason": main_why
        }
    return {
        "psa_guess": psa_guess, "psa_ev": psa_ev, "psa_why": psa_why,
        "main_guess": main_guess, "main_ev": main_ev, "main_why": main_why,
        "final": None, "source": "none",
        "evidence": (psa_ev or main_ev or ""), "reason": (psa_why or main_why or "no health number")
    }

# ----------------------- Persist -----------------------

def persist_psa_snapshot(engine: Engine, fid: int, snap_at: Optional[str],
                         new_pct: Optional[int], old_psa: Optional[int],
                         log_payload: Dict[str, Any]) -> None:
    if DRY_RUN or not UPDATE_PSA or not snap_at or new_pct is None:
        return
    with engine.begin() as conn:
        conn.execute(text("""
          UPDATE "iPhone".post_sold_audit
          SET battery_pct_snapshot = :v,
              battery_refit_log = COALESCE(battery_refit_log, '[]'::jsonb) ||
                jsonb_build_array(jsonb_build_object(
                    'at', now(),
                    'by', :by,
                    'old', :old,
                    'new', :v,
                    'source', :src,
                    'evidence', :ev,
                    'reason', :why,
                    'psa_guess', :psa_guess,
                    'main_guess', :main_guess,
                    'pre_main_raw', :pre_raw,
                    'pre_main_fixed', :pre_fixed,
                    'pre_main_fixed_ai', :pre_fixed_ai,
                    'pre_damage_sev', :pre_dsev
                ))
          WHERE listing_id = :id AND snapshot_at = :snap
        """), {
            "v": int(new_pct),
            "by": TAG_BY,
            "old": old_psa,
            "src": log_payload.get("source","none"),
            "ev":  log_payload.get("evidence",""),
            "why": log_payload.get("reason",""),
            "psa_guess": log_payload.get("psa_guess"),
            "main_guess": log_payload.get("main_guess"),
            "pre_raw": log_payload.get("pre_main_raw"),
            "pre_fixed": log_payload.get("pre_main_fixed"),
            "pre_fixed_ai": log_payload.get("pre_main_fixed_ai"),
            "pre_dsev": log_payload.get("pre_damage_sev"),
            "id": int(fid),
            "snap": snap_at
        })

def mark_psa_fetched(engine: Engine, fid: int, snap_at: Optional[str]) -> None:
    if DRY_RUN or not snap_at:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE "iPhone".post_sold_audit
            SET battery_llm_fetch = 'fetched'
            WHERE listing_id = :id AND snapshot_at = :snap
        """), {"id": int(fid), "snap": snap_at})

# ----------------------- Runner & Metrics -----------------------

@dataclass
class Outcome:
    action: str   # 'updated' | 'kept' | 'unknown' | 'error'
    old_ai: Optional[int]
    new_ai: Optional[int]
    source: str
    evidence: str
    reason: str
    psa_guess: Optional[int]
    main_guess: Optional[int]
    pre_main_raw: Optional[int]
    pre_main_fixed: Optional[int]
    pre_main_fixed_ai: Optional[int]
    pre_damage_sev: Optional[int]
    psa_had_text: bool
    had_conflict: bool  # both anchored and different

def is_valid_pct(x: Optional[int]) -> bool:
    try:
        return x is not None and 20 <= int(x) <= 100
    except Exception:
        return False

def process_one(engine: Engine, row: Dict[str, Any]) -> Tuple[int, Outcome]:
    fid = int(row["listing_id"])

    pre_main_raw   = row.get("battery_pct")
    pre_main_fixed = row.get("battery_pct_fixed")
    pre_ai_current = row.get("battery_pct_fixed_ai")
    pre_damage_sev = row.get("pre_damage_sev")

    try:
        old_ai_int = int(round(float(pre_ai_current))) if pre_ai_current is not None else None
    except Exception:
        old_ai_int = None

    eval_res = call_battery_eval_both(row)
    psa_guess   = eval_res["psa_guess"]
    main_guess  = eval_res["main_guess"]
    final_val   = eval_res["final"]
    source      = eval_res["source"]
    evidence    = eval_res["evidence"]
    reason      = eval_res["reason"]
    psa_had_txt = bool((row.get("psa_title") or "").strip() or (row.get("psa_desc") or "").strip())
    had_conflict = is_valid_pct(psa_guess) and is_valid_pct(main_guess) and (psa_guess != main_guess) and (source == "psa")

    log_payload = {
        "source": source,
        "evidence": evidence,
        "reason": reason,
        "psa_guess": psa_guess,
        "main_guess": main_guess,
        "pre_main_raw": pre_main_raw,
        "pre_main_fixed": pre_main_fixed,
        "pre_main_fixed_ai": pre_ai_current,
        "pre_damage_sev": pre_damage_sev,
    }

    psa_snap_at = row.get("psa_snapshot_at")
    psa_old_val = row.get("psa_batt")

    # Unknown → write zeros if policy says so, always mark fetched
    if final_val is None:
        if UPDATE_MAIN and not DRY_RUN:
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE "iPhone".iphone_listings
                    SET battery_pct_fixed_ai = :v,
                        battery_pct_fixed_by = :by,
                        battery_pct_fixed_at = now()
                    WHERE listing_id = :id
                """), {"v": (0 if UNANCHORED_POLICY=="zero" else None), "by": TAG_BY, "id": fid})
        if psa_snap_at and UNANCHORED_POLICY == "zero":
            persist_psa_snapshot(engine, fid, psa_snap_at, 0, psa_old_val, log_payload)
        # mark fetched (always, for this snapshot)
        mark_psa_fetched(engine, fid, psa_snap_at)

        logger.info(f"{fid} unknown → MAIN={(0 if UNANCHORED_POLICY=='zero' else 'NULL')} ; PSA={(0 if (psa_snap_at and UNANCHORED_POLICY=='zero') else 'no-change')} | {source} | {evidence!r}")
        return fid, Outcome("unknown", old_ai_int, (0 if UNANCHORED_POLICY=="zero" else None),
                            source, evidence, reason,
                            psa_guess, main_guess,
                            pre_main_raw, pre_main_fixed, pre_ai_current, pre_damage_sev,
                            psa_had_txt, had_conflict)

    # PSA write (if enabled)
    if psa_snap_at:
        persist_psa_snapshot(engine, fid, psa_snap_at, int(final_val), psa_old_val, log_payload)

    # MAIN write — ALWAYS overwrite (force tag + value)
    if UPDATE_MAIN and not DRY_RUN:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE "iPhone".iphone_listings
                SET battery_pct_fixed_ai = :v,
                    battery_pct_fixed_by = :by,
                    battery_pct_fixed_at = now()
                WHERE listing_id = :id
            """), {"v": int(final_val), "by": TAG_BY, "id": fid})

    # mark fetched (always, for this snapshot)
    mark_psa_fetched(engine, fid, psa_snap_at)

    # logging outcome
    if old_ai_int != final_val:
        logger.info(f"{fid} MAIN {old_ai_int} → {final_val} (anchored) src={source} ev={evidence!r}")
        return fid, Outcome("updated", old_ai_int, int(final_val),
                            source, evidence, reason,
                            psa_guess, main_guess,
                            pre_main_raw, pre_main_fixed, pre_ai_current, pre_damage_sev,
                            psa_had_txt, had_conflict)
    else:
        logger.info(f"{fid} MAIN kept {old_ai_int} (forced tag/update) src={source} ev={evidence!r}")
        return fid, Outcome("kept", old_ai_int, int(final_val),
                            source, evidence, reason,
                            psa_guess, main_guess,
                            pre_main_raw, pre_main_fixed, pre_ai_current, pre_damage_sev,
                            psa_had_txt, had_conflict)

# ----------------------- Metrics helpers -----------------------

def add_delta(acc: Dict[str, float], key_sum: str, key_cnt: str, delta: float) -> None:
    acc[key_sum] = acc.get(key_sum, 0.0) + abs(delta)
    acc[key_cnt] = acc.get(key_cnt, 0.0) + 1.0

def print_metrics(m: Dict[str, Any]) -> None:
    logger.info("Metrics (Battery Refit v5.2)")
    logger.info(f"  rows_total={m.get('rows_total',0)} | psa_text_rows={m.get('psa_text_rows',0)}")
    logger.info(f"  final_source: psa={m.get('final_psa',0)} | main={m.get('final_main',0)} | unknown={m.get('final_none',0)}")
    logger.info(f"  coverage: psa_anchored={m.get('psa_anchored',0)} | main_after_psa_fail={m.get('main_after_psa_fail',0)}")
    logger.info(f"  psa_unique_adds={m.get('psa_unique_adds',0)}")
    # New adds vs effective
    logger.info(f"  new_adds: total={m.get('new_adds_total',0)} | psa={m.get('new_adds_psa',0)} | main={m.get('new_adds_main',0)}")
    # Bogus removed (valid eff → unknown), excludes NULL→0
    logger.info(f"  bogus_removed_total={m.get('bogus_removed_total',0)}")
    # Corrections vs raw/fixed/fixed_ai (PSA only)
    cr = m.get("corrections",{})
    def corr_line(tag):
        n = cr.get(f"{tag}_n",0)
        s = cr.get(f"{tag}_sum",0.0)
        return f"{n} (avg |Δ|={(s/n):.2f})" if n else "0"
    logger.info(f"  psa_corrections_vs_raw={corr_line('raw')} | vs_fixed={corr_line('fixed')} | vs_fixed_ai={corr_line('fixed_ai')}")
    # Corrections vs effective and changed counts (both sources)
    eff = m.get("eff_corr",{})
    def eff_line(side):
        n = eff.get(f"{side}_n",0); s = eff.get(f"{side}_sum",0.0)
        return f"{n} (avg |Δ|={(s/n):.2f})" if n else "0"
    logger.info(f"  corrections_vs_eff: psa={eff_line('psa')} | main={eff_line('main')}")
    logger.info(f"  changed_vs_eff: psa={m.get('changed_vs_eff_psa',0)} | main={m.get('changed_vs_eff_main',0)} | total={m.get('changed_vs_eff_total',0)}")
    logger.info(f"  psa_overrode_main_conflict={m.get('psa_overrode_main_conflict',0)}")
    logger.info(f"  sub80_upgrade_candidates={m.get('sub80_upgrade_candidates',0)}")
    # Unknown reasons
    ur = m.get("unknown_reasons",{})
    if ur:
        top = sorted(ur.items(), key=lambda kv: kv[1], reverse=True)[:5]
        logger.info("  unknown_top_reasons:")
        for k,v in top:
            logger.info(f"    - {k} {v}")
    norm = m.get("norms",{})
    logger.info(f"  normalizations: emoji_accepts={norm.get('emoji',0)} | decimal_rounds={norm.get('decimal',0)} | range={norm.get('range',0)}")

# ----------------------- CLI overrides -----------------------

def apply_cli_overrides():
    import argparse
    ap = argparse.ArgumentParser(description="Battery Refit v5.2 — PSA+MAIN eval, PSA-first write, rich metrics")
    ap.add_argument("--api-key",   dest="api_key")
    ap.add_argument("--base-url",  dest="base_url")
    ap.add_argument("--model",     dest="model")
    ap.add_argument("--listing-ids",  dest="listing_ids", help="comma-separated listing IDs")
    ap.add_argument("--batch-size",dest="batch_size", type=int)
    ap.add_argument("--batch-page",dest="batch_page", type=int)
    ap.add_argument("--eval-both", dest="eval_both", action="store_true")
    ap.add_argument("--no-eval-both", dest="no_eval_both", action="store_true")
    args = ap.parse_args()
    if args.api_key:    os.environ["OPENAI_API_KEY"]  = args.api_key
    if args.base_url:   os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.model:      os.environ["MODEL_NAME"]      = args.model
    if args.listing_ids:   os.environ["LISTING_IDS"]        = args.listing_ids
    if args.batch_size is not None: os.environ["BATCH_SIZE"] = str(args.batch_size)
    if args.batch_page is not None: os.environ["BATCH_PAGE"] = str(args.batch_page)
    if args.eval_both: os.environ["EVAL_BOTH"] = "1"
    if args.no_eval_both: os.environ["EVAL_BOTH"] = "0"

# ----------------------- Main -----------------------

def main() -> None:
    apply_cli_overrides()

    key_ok = (os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("\ufeffOPENAI_API_KEY"))
    if not key_ok:
        raise RuntimeError("OPENAI_API_KEY/DEEPSEEK_API_KEY required")

    if BATCH_SIZE > 0:
        limit  = BATCH_SIZE
        offset = BATCH_SIZE * max(0, BATCH_PAGE)
    else:
        limit  = LIMIT
        offset = 0

    logger.info(
        f"Refit v5.2 | model={os.getenv('MODEL_NAME', MODEL_NAME)} | DRY_RUN={DRY_RUN} | PSA_ONLY={PSA_ONLY} | "
        f"EXCLUDE_COND_SCORE_EQ_1={EXCLUDE_COND_SCORE_EQ_1} | UPDATE_MAIN={UPDATE_MAIN} | UPDATE_PSA={UPDATE_PSA} | "
        f"UNANCHORED_POLICY={UNANCHORED_POLICY} | EVAL_BOTH={EVAL_BOTH} | BATCH_SIZE={BATCH_SIZE} | BATCH_PAGE={BATCH_PAGE} | LIMIT={LIMIT}"
    )

    engine = get_engine()
    ensure_psa_columns(engine)

    ids = parse_listing_ids()
    if ids:
        rows = fetch_rows_by_ids(engine, ids)
        logger.info(f"Queued rows: {len(rows)} (LISTING_IDS whitelist)")
    else:
        rows = fetch_rows(engine, limit=limit, offset=offset)
        logger.info(f"Queued rows: {len(rows)} (batch limit={limit} offset={offset})")

    # Metrics accumulators
    m: Dict[str, Any] = {
        "rows_total": 0,
        "psa_text_rows": 0,
        "final_psa": 0, "final_main": 0, "final_none": 0,
        "psa_anchored": 0, "main_after_psa_fail": 0,
        "psa_unique_adds": 0,
        "psa_overrode_main_conflict": 0,
        "corrections": {},
        "unknown_reasons": {},
        "norms": {"emoji":0, "decimal":0, "range":0},
        "sub80_upgrade_candidates": 0,
        "new_adds_total": 0, "new_adds_psa": 0, "new_adds_main": 0,
        "bogus_removed_total": 0,
        "eff_corr": { "psa_sum": 0.0, "psa_n": 0, "main_sum": 0.0, "main_n": 0 },
        "changed_vs_eff_psa": 0, "changed_vs_eff_main": 0, "changed_vs_eff_total": 0,
    }

    def bump_unknown_reason(r: str):
        if not r: r = "unknown"
        m["unknown_reasons"][r] = m["unknown_reasons"].get(r, 0) + 1

    def bump_norms(source: str, reason: str, evidence: str):
        if "🔋" in (evidence or ""):
            m["norms"]["emoji"] += 1
        if "rounded decimal" in (reason or ""):
            m["norms"]["decimal"] += 1
        if "lower bound" in (reason or ""):
            m["norms"]["range"] += 1

    updated = kept = unknown = errors = 0
    t0 = time.time()

    futures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for r in rows:
            futures.append(pool.submit(process_one, engine, r))

        for fut in tqdm(as_completed(futures), total=len(futures), unit="row"):
            try:
                fid, out = fut.result()
                m["rows_total"] += 1
                if out.psa_had_text: m["psa_text_rows"] += 1

                # Final source
                if out.source == "psa":   m["final_psa"]  += 1
                elif out.source == "main":m["final_main"] += 1
                else:                      m["final_none"] += 1

                # Coverage
                if is_valid_pct(out.psa_guess): m["psa_anchored"] += 1
                if (out.psa_guess is None) and is_valid_pct(out.main_guess):
                    m["main_after_psa_fail"] += 1

                # Compute effective pre-run value
                pre_eff = None
                if is_valid_pct(out.pre_main_fixed):
                    pre_eff = int(out.pre_main_fixed)
                elif is_valid_pct(out.pre_main_raw):
                    pre_eff = int(out.pre_main_raw)

                # New adds: final valid while effective absent
                if is_valid_pct(out.new_ai) and (pre_eff is None):
                    m["new_adds_total"] += 1
                    if out.source == "psa": m["new_adds_psa"] += 1
                    elif out.source == "main": m["new_adds_main"] += 1

                # PSA-only corrections vs raw/fixed/fixed_ai
                cr = m["corrections"]
                if out.source == "psa" and is_valid_pct(out.new_ai):
                    if is_valid_pct(out.pre_main_raw) and out.new_ai != int(out.pre_main_raw):
                        add_delta(cr, "raw_sum", "raw_n", out.new_ai - int(out.pre_main_raw))
                    if is_valid_pct(out.pre_main_fixed) and out.new_ai != int(out.pre_main_fixed):
                        add_delta(cr, "fixed_sum", "fixed_n", out.new_ai - int(out.pre_main_fixed))
                    if is_valid_pct(out.pre_main_fixed_ai) and out.new_ai != int(out.pre_main_fixed_ai):
                        add_delta(cr, "fixed_ai_sum", "fixed_ai_n", out.new_ai - int(out.pre_main_fixed_ai))

                # Corrections vs effective & changed counts
                if is_valid_pct(out.new_ai) and (pre_eff is not None) and out.new_ai != pre_eff:
                    m["changed_vs_eff_total"] += 1
                    if out.source == "psa":
                        m["changed_vs_eff_psa"] += 1
                        add_delta(m["eff_corr"], "psa_sum", "psa_n", out.new_ai - pre_eff)
                    elif out.source == "main":
                        m["changed_vs_eff_main"] += 1
                        add_delta(m["eff_corr"], "main_sum", "main_n", out.new_ai - pre_eff)

                if out.had_conflict:
                    m["psa_overrode_main_conflict"] += 1

                if is_valid_pct(out.new_ai) and out.new_ai < 80:
                    prev_ai  = out.pre_main_fixed_ai
                    prev_ai_invalid = (prev_ai is None) or (prev_ai == 0) or (not is_valid_pct(prev_ai))
                    if out.pre_damage_sev in (0,1) and prev_ai_invalid:
                        m["sub80_upgrade_candidates"] += 1

                if out.action == "unknown":
                    bump_unknown_reason(out.reason)
                bump_norms(out.source, out.reason, out.evidence)

                if out.action == "updated": updated += 1
                elif out.action == "kept":  kept += 1
                elif out.action == "unknown": unknown += 1
                else: errors += 1

            except Exception as e:
                errors += 1
                logger.error(f"worker error: {e}")

    dt = time.time() - t0
    rps = (len(rows)/dt) if dt > 0 else 0.0
    logger.info(f"Done. rows={len(rows)} updated={updated} kept={kept} unknown={unknown} errors={errors} | {dt:.2f}s ~ {rps:.2f} r/s")

    # Print metrics
    print_metrics(m)

if __name__ == "__main__":
    main()














