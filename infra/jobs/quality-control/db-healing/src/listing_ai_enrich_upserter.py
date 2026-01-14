#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
listing_ai_enrich_upserter.py — fast, concurrent, evidence-gated LLM enricher (public template)

Purpose
-------
This job reads listing text (title + description) from Postgres and upserts structured, evidence-backed
enrichment fields produced by an OpenAI-compatible LLM into a separate enrichment table.

Public-release changes (sanitization)
-------------------------------------
- Removed any target-platform branding and identifiers.
- Replaced platform-specific identifiers with generic `listing_id`.
- Removed credential-like DSN examples and removed hard-coded DB defaults.
- All runtime configuration comes from environment variables (or safe defaults).

Environment variables (public-safe)
-----------------------------------
Required:
  PG_DSN                SQLAlchemy DSN for Postgres. No default is embedded in code.
  LLM_API_KEY           API key for your OpenAI-compatible provider.

Optional:
  PG_SCHEMA             default: marketplace
  MODEL_NAME            default: (provider-specific; set explicitly in production)
  LLM_BASE_URL          default: empty (use provider default)
  VERSION_TAG           default: enrich-public-v1
  CONF_MIN              default: 0.60
  WORKERS               default: 6
  QPS                   default: 0   (no throttle; set e.g. 1.5 to reduce 429s)
  LOG_LEVEL             default: INFO
  MAX_RETRIES           default: 3
  RETRY_BASE_SLEEP      default: 0.6
  NEW_CLIENT_PER_CALL   default: 0
  INPUT_TABLE           default: listings
  OUTPUT_TABLE          default: listing_ai_enrich

Notes
-----
- This is an enrichment template. It does not scrape any marketplace. It consumes already-stored text.
- The LLM prompt is generic and platform-agnostic.
"""

import os
import sys
import re
import json
import time
import logging
import threading
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from openai import OpenAI

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# ------------------ Helpers ------------------
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def qident(name: str) -> str:
    """Quote and validate an identifier for safe SQL string interpolation."""
    if not name or not _IDENT_RE.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return f'"{name}"'

# ------------------ Config ------------------
PG_DSN = os.getenv("PG_DSN")
PG_SCHEMA = os.getenv("PG_SCHEMA", "marketplace")

INPUT_TABLE = os.getenv("INPUT_TABLE", "listings")
OUTPUT_TABLE = os.getenv("OUTPUT_TABLE", "listing_ai_enrich")

MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
VERSION_TAG = os.getenv("VERSION_TAG", "enrich-public-v1")

CONF_MIN = float(os.getenv("CONF_MIN", "0.60"))
WORKERS = int(os.getenv("WORKERS", "6"))
QPS = float(os.getenv("QPS", "0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "3")))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "0.6"))
NEW_CLIENT_PER_CALL = os.getenv("NEW_CLIENT_PER_CALL", "0").lower() in ("1", "true", "yes")

# OpenAI-compatible auth / routing (generic; supports multiple naming conventions)
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "").strip()

# ------------------ Logging ------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("listing_enrich")

# ------------------ Prompt (generic; no platform fingerprints) ------------------
SYSTEM_PROMPT = r"""
You extract compact, auditable facts from marketplace product listings.

Input JSON:
{"listing_id": <integer>, "modules": [...], "title": "...", "description": "..."}

Text T = title + "\n" + description.

Return STRICT JSON (no prose, no markdown):
{
  "codes":[{"code":string,"value":(string|number|boolean|null),"eid":string},...],
  "kv":{"variant_conf":0..1,"storage_conf":0..1,"negotiability":0..1,"urgency":0..1,"opening_offer_amount":int|null,"lqs_textonly":0..1},
  "risks":{"activation_lock":0..1},
  "evidence":{"spans":[{"eid":string,"field":("title"|"description"),"quote":string,"start":int,"end":int}]},
  "conf":{ field_or_code: 0..1, ... }
}

CODES YOU MAY EMIT (emit ONLY if literally supported by T; otherwise omit):
- "variant_canonical"     → ONLY model family (e.g., "Phone 16 Pro Max"). NO storage ("256GB"/"1TB"/"2TB") or color.
- "storage_gb_fixed_ai"   → one of {64,128,256,512,1024,2048}. (2 TB ⇒ 2048)
- "sale_mode"             → one of {firm|obo|bids|unspecified}.
- "shipping"              → one of {can_ship|pickup_only|both}.
- "payment_constraints"   → one of {normal|mobile_pay_only|cash_only|bank_only|other}.
- "time_window_strict"    → true|false (e.g., "must be collected tonight", "only after 17:00").
- "band_region_5g"        → one of {eu_ok|us_only|unknown}.
- "VAT_invoice"           → true|false.
- "VAT_scheme"            → one of {vat_full|vat_margin|unknown}.
- "us_import"             → true|false.
- "lock_ai"               → one of {locked|unlocked|null}.
- "model_a_code"          → string|null (e.g., "A2849").
- "esim_only"             → true|false.
- "repair_provider"       → one of {manufacturer|authorized|independent|private|unknown}.
- "parts_replaced"        → comma-joined subset of {screen,battery,back_glass,camera,port}. NEVER include "unit".
- "true_tone_missing"     → true|false (explicit True Tone missing).
- "ios_parts_warning"     → comma-joined subset of {display,battery,camera} (OS warnings).
- "owner_type"            → one of {private|company|work_phone|unknown}.
- "first_owner"           → true|false.
- "used_with_case_claim"  → true|false.
- "return_rights"         → one of {none|return_14d|store_policy|unknown}.
- "deferred_payment"      → one of {klarna|installments|none|unknown}.
- "swap_insurance"        → one of {swap|repair|unknown}. ("replacement unit" ⇒ swap; then DO NOT set parts_replaced.)
- "serial_imei_policy"    → one of {provided_public|share_on_dm|refuse|unknown}.
- "inventory_count"       → integer>=0 (explicit multiple units/bundle).
- "seller_role"           → one of {private|pro_reseller|refurb_shop|unknown}.
- "NEGOTIABILITY_CONFLICT"→ true when BOTH a firm-price phrase AND a negotiable/bids phrase are present.

CONFIDENCE & EVIDENCE (MANDATORY):
- For EVERY emitted code AND EVERY kv field, include conf[field] ∈ [0,1].
- For EVERY emitted code, include ONE evidence span (eid) that quotes the exact trigger phrase from T with [start,end) offsets over T.
- If you are not sure, OMIT the code (do NOT guess).

IMPORTANT RULES:
- Emit ONLY codes literally supported by T. If uncertain, omit.
- Each emitted code has exactly one evidence span with correct offsets over T ("title" or "description").
- Provide conf[field] ∈ [0,1] for ALL emitted codes AND kv fields.
- NEVER include storage or color in "variant_canonical".
- NEVER include "unit" in parts_replaced.
"""

# ------------------ Client / Engine ------------------
_client_lock = threading.Lock()
_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if not LLM_API_KEY:
        log.error("LLM_API_KEY (or OPENAI_API_KEY) is required")
        sys.exit(2)

    if NEW_CLIENT_PER_CALL:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)

    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL or None)
    return _client

def get_engine() -> Engine:
    if not PG_DSN:
        log.error("PG_DSN is required")
        sys.exit(2)
    return create_engine(PG_DSN, future=True, pool_pre_ping=True)

def fqn(schema: str, table: str) -> str:
    return f"{qident(schema)}.{qident(table)}"

# ------------------ Fetch ------------------
def fetch_targets(engine: Engine) -> List[Dict[str, Any]]:
    """
    Fetch ONLY rows not enriched yet (missing in OUTPUT_TABLE) and where spam IS NULL.
    NO LIMIT — fetch all such rows.
    """
    listings = fqn(PG_SCHEMA, INPUT_TABLE)
    enrich = fqn(PG_SCHEMA, OUTPUT_TABLE)

    sql = f"""
    SELECT
      l.generation,
      l.listing_id,
      COALESCE(l.title,'')       AS title,
      COALESCE(l.description,'') AS description,
      l.storage_gb               AS storage_scraped
    FROM {listings} l
    LEFT JOIN {enrich} e
      ON e.generation = l.generation AND e.listing_id = l.listing_id
    WHERE l.spam IS NULL
      AND e.listing_id IS NULL
    ORDER BY l.edited_date DESC NULLS LAST
    """
    with engine.begin() as conn:
        rows = [dict(r) for r in conn.execute(text(sql)).mappings()]
    log.info(f"Queued rows: {len(rows)} (missing-only, spam IS NULL; LIMIT=NONE)")
    return rows

# ------------------ Evidence gating & Normalizers (minimal, to satisfy DB) ------------------
def _strip_code_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S | re.I)
    return m.group(1).strip() if m else s

ALLOWED_SALE_MODE = {"firm", "obo", "bids", "unspecified"}
ALLOWED_PAY_CONS = {"normal", "mobile_pay_only", "cash_only", "bank_only", "other"}
ALLOWED_OWNER = {"private", "company", "work_phone", "unknown"}
ALLOWED_RETURNS = {"none", "return_14d", "store_policy", "unknown"}
ALLOWED_BAND5G = {"eu_ok", "us_only", "unknown"}
VALID_STORAGE = {64, 128, 256, 512, 1024, 2048}  # include 2TB
TAIL_STORAGE_RE = re.compile(r"\s*((64|128|256|512|1024|2048)\s*gb|(1|2)\s*tb)\b", re.I)

def _norm_sale_mode(v: Optional[str]) -> str:
    if not v:
        return "unspecified"
    v = str(v).lower().strip()
    if v in ALLOWED_SALE_MODE:
        return v
    if v in {"fixed_price", "fixedprice", "firm_price"}:
        return "firm"
    if v in {"obo", "best_offer", "make_offer"}:
        return "obo"
    if v in {"bids", "auction"}:
        return "bids"
    return "unspecified"

def _norm_payment(v: Optional[str]) -> str:
    if not v:
        return "normal"
    v = str(v).lower().strip()
    if v in ALLOWED_PAY_CONS:
        return v
    # Compatibility aliases
    if v in {"vipps", "vipps_only", "mobilepay", "applepay", "googlepay", "mobile_pay"}:
        return "mobile_pay_only"
    if v in {"cash", "cash only", "cash_only"}:
        return "cash_only"
    if v in {"bank", "bank_transfer", "bank_only"}:
        return "bank_only"
    return "other"

def _norm_owner(v: Optional[str]) -> str:
    if not v:
        return "unknown"
    v = str(v).lower().strip()
    if v in ALLOWED_OWNER:
        return v
    if v in {"person", "private_seller"}:
        return "private"
    if v in {"business", "company", "corporate"}:
        return "company"
    if v in {"work", "job_phone"}:
        return "work_phone"
    return "unknown"

def _norm_returns(v: Optional[str]) -> str:
    if not v:
        return "unknown"
    v = str(v).lower().strip()
    if v in ALLOWED_RETURNS:
        return v
    if v in {"14d", "14-day", "return_14d", "return 14 days"}:
        return "return_14d"
    if v in {"store", "store_policy"}:
        return "store_policy"
    if v in {"none", "no_returns"}:
        return "none"
    # Compatibility alias
    if v in {"angrerett", "angrerett_14d"}:
        return "return_14d"
    return "unknown"

def _norm_band(v: Optional[str]) -> str:
    if not v:
        return "unknown"
    v = str(v).lower().strip()
    if v in ALLOWED_BAND5G:
        return v
    if v in {"us", "usa", "verizon", "at&t", "t-mobile", "mmwave"}:
        return "us_only"
    if v in {"eu", "europe", "nordic"}:
        return "eu_ok"
    return "unknown"

def _norm_storage(x: Any) -> Optional[int]:
    try:
        n = int(x)
        return n if n in VALID_STORAGE else None
    except Exception:
        return None

def _norm_variant_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return TAIL_STORAGE_RE.sub("", s).strip() or s.strip()

def validate_payload(payload: Any, text_len: int) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(_strip_code_fence(payload))
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for k in ("codes", "kv", "risks", "evidence", "conf"):
        payload.setdefault(k, [] if k == "codes" else {})

    spans = {s.get("eid"): s for s in payload.get("evidence", {}).get("spans", []) if isinstance(s, dict)}

    codes_in = payload.get("codes", [])
    kept_codes = []
    for c in codes_in:
        code, eid = c.get("code"), c.get("eid")
        conf = float(payload.get("conf", {}).get(code, 0.0))
        sp = spans.get(eid)
        if conf < CONF_MIN or not sp:
            continue
        s, e = int(sp.get("start", -1)), int(sp.get("end", -1))
        if not (0 <= s < e <= text_len):
            continue
        kept_codes.append(c)
    payload["codes"] = kept_codes

    kv_in = payload.get("kv", {})
    kv_kept = {}
    for k, v in kv_in.items():
        conf = float(payload.get("conf", {}).get(k, 0.0))
        if conf >= CONF_MIN:
            kv_kept[k] = v
    payload["kv"] = kv_kept

    payload["_neg_conflict"] = any(c.get("code") == "NEGOTIABILITY_CONFLICT" for c in kept_codes)

    stats = {
        "n_codes_in": len(codes_in),
        "n_codes_kept": len(kept_codes),
        "n_kv_in": len(kv_in),
        "n_kv_kept": len(kv_kept),
    }
    return payload, stats

def sanitize_params(data: Dict[str, Any], scraped_storage: Optional[int]) -> Dict[str, Any]:
    """Minimal normalization to satisfy DB enums; LLM decides content.

    IMPORTANT: scraped storage always wins. If the scraper already has storage, do NOT write AI storage.
    """
    def pick(code: str, cast=None):
        for c in data.get("codes", []):
            if c.get("code") == code:
                v = c.get("value")
                return cast(v) if cast and v is not None else v
        return None

    kv = data.get("kv", {})
    sale_mode_raw = pick("sale_mode", str)
    pay_raw = pick("payment_constraints", str)
    ship_raw = pick("shipping", str)
    owner_raw = pick("owner_type", str)
    returns_raw = pick("return_rights", str)
    band_raw = pick("band_region_5g", str)

    variant_raw = pick("variant_canonical", str)
    storage_raw = pick("storage_gb_fixed_ai", int)

    # STORAGE RULE: if scraper has storage, we do not write AI storage
    storage_ai = None if (scraped_storage is not None) else _norm_storage(storage_raw)

    # Genericized kv key name: opening_offer_amount (accept legacy opening_offer_nok)
    opening_offer = kv.get("opening_offer_amount")
    if opening_offer is None:
        opening_offer = kv.get("opening_offer_nok")

    return {
        "variant": _norm_variant_name(variant_raw),
        "vconf": kv.get("variant_conf"),
        "storage": storage_ai,
        "sconf": kv.get("storage_conf"),
        "neg": kv.get("negotiability"),
        "urg": kv.get("urgency"),
        "mode": _norm_sale_mode(sale_mode_raw),
        "open_offer": opening_offer,
        "lqs": kv.get("lqs_textonly"),
        "can_ship": (ship_raw in ("can_ship", "both")),
        "pickup_only": (ship_raw == "pickup_only"),
        "pay_const": _norm_payment(pay_raw),
        "time_strict": bool(pick("time_window_strict")),
        "band5g": _norm_band(band_raw),
        "vat_inv": bool(pick("VAT_invoice")),
        "vat_scheme": (pick("VAT_scheme", str) or "unknown"),
        "us_import": bool(pick("us_import")),
        "lock_ai": (pick("lock_ai", str) or "null"),
        "a_code": pick("model_a_code", str),
        "esim_only": bool(pick("esim_only")),
        "rep_provider": (pick("repair_provider", str) or "unknown"),
        "parts": [t.strip() for t in (pick("parts_replaced", str) or "").split(",") if t.strip() and t.strip().lower() not in {"unit"}] or None,
        "tt_missing": bool(pick("true_tone_missing")),
        "ios_warn": [t.strip() for t in (pick("ios_parts_warning", str) or "").split(",") if t.strip()] or None,
        "owner_type": _norm_owner(owner_raw),
        "first_owner": bool(pick("first_owner")),
        "case_claim": bool(pick("used_with_case_claim")),
        "ret_rights": _norm_returns(returns_raw),
        "def_pay": (pick("deferred_payment", str) or "none"),
        "swap_ins": (pick("swap_insurance", str) or "unknown"),
        "imei_policy": (pick("serial_imei_policy", str) or "unknown"),
        "seller_role": (pick("seller_role", str) or "unknown"),
        "neg_conflict": bool(data.get("_neg_conflict", False)),
        "act_risk": float(data.get("risks", {}).get("activation_lock", 0.0)),
    }

# ------------------ UPSERT ------------------
def upsert(engine: Engine, generation: int, listing_id: int, data: Dict[str, Any], scraped_storage: Optional[int]):
    clean = sanitize_params(data, scraped_storage)
    enrich = fqn(PG_SCHEMA, OUTPUT_TABLE)

    sql = f"""
    INSERT INTO {enrich} AS tgt
    (generation, listing_id,
     variant_canonical, variant_conf, storage_gb_fixed_ai, storage_conf,
     negotiability_ai, urgency_ai, sale_mode, opening_offer_amount, lqs_textonly,
     can_ship, pickup_only, payment_constraints, time_window_strict,
     band_region_5g, vat_invoice, vat_scheme, us_import,
     lock_ai, model_a_code, esim_only,
     repair_provider, parts_replaced, true_tone_missing, ios_parts_warning,
     owner_type, first_owner, used_with_case_claim,
     return_rights, deferred_payment,
     swap_insurance, serial_imei_policy, seller_role, negotiability_conflict,
     activation_lock_risk,
     codes_json, conf_json, evidence_json, model, version, updated_at)
    VALUES
    (:gen, :listing_id,
     :variant, :vconf, :storage, :sconf,
     :neg, :urg, :mode, :open_offer, :lqs,
     :can_ship, :pickup_only, :pay_const, :time_strict,
     :band5g, :vat_inv, :vat_scheme, :us_import,
     :lock_ai, :a_code, :esim_only,
     :rep_provider, :parts, :tt_missing, :ios_warn,
     :owner_type, :first_owner, :case_claim,
     :ret_rights, :def_pay,
     :swap_ins, :imei_policy, :seller_role, :neg_conflict,
     :act_risk,
     :codes_json::jsonb, :conf_json::jsonb, :evid_json::jsonb, :model, :ver, now())
    ON CONFLICT (generation, listing_id) DO UPDATE SET
      variant_canonical   = EXCLUDED.variant_canonical,
      variant_conf        = EXCLUDED.variant_conf,
      -- IMPORTANT: never erase an existing AI storage with NULL from a later run
      storage_gb_fixed_ai = COALESCE(tgt.storage_gb_fixed_ai, EXCLUDED.storage_gb_fixed_ai),
      storage_conf        = EXCLUDED.storage_conf,
      negotiability_ai    = EXCLUDED.negotiability_ai,
      urgency_ai          = EXCLUDED.urgency_ai,
      sale_mode           = EXCLUDED.sale_mode,
      opening_offer_amount= EXCLUDED.opening_offer_amount,
      lqs_textonly        = EXCLUDED.lqs_textonly,
      can_ship            = EXCLUDED.can_ship,
      pickup_only         = EXCLUDED.pickup_only,
      payment_constraints = EXCLUDED.payment_constraints,
      time_window_strict  = EXCLUDED.time_window_strict,
      band_region_5g      = EXCLUDED.band_region_5g,
      vat_invoice         = EXCLUDED.vat_invoice,
      vat_scheme          = EXCLUDED.vat_scheme,
      us_import           = EXCLUDED.us_import,
      lock_ai             = EXCLUDED.lock_ai,
      model_a_code        = EXCLUDED.model_a_code,
      esim_only           = EXCLUDED.esim_only,
      repair_provider     = EXCLUDED.repair_provider,
      parts_replaced      = EXCLUDED.parts_replaced,
      true_tone_missing   = EXCLUDED.true_tone_missing,
      ios_parts_warning   = EXCLUDED.ios_parts_warning,
      owner_type          = EXCLUDED.owner_type,
      first_owner         = EXCLUDED.first_owner,
      used_with_case_claim= EXCLUDED.used_with_case_claim,
      return_rights       = EXCLUDED.return_rights,
      deferred_payment    = EXCLUDED.deferred_payment,
      swap_insurance      = EXCLUDED.swap_insurance,
      serial_imei_policy  = EXCLUDED.serial_imei_policy,
      seller_role         = EXCLUDED.seller_role,
      negotiability_conflict = EXCLUDED.negotiability_conflict,
      activation_lock_risk   = EXCLUDED.activation_lock_risk,
      codes_json          = EXCLUDED.codes_json,
      conf_json           = EXCLUDED.conf_json,
      evidence_json       = EXCLUDED.evidence_json,
      model               = EXCLUDED.model,
      version             = EXCLUDED.version,
      updated_at          = now();
    """

    params = {
        "gen": generation,
        "listing_id": listing_id,
        "variant": clean["variant"],
        "vconf": clean["vconf"],
        "storage": clean["storage"],
        "sconf": clean["sconf"],
        "neg": clean["neg"],
        "urg": clean["urg"],
        "mode": clean["mode"],
        "open_offer": clean["open_offer"],
        "lqs": clean["lqs"],
        "can_ship": clean["can_ship"],
        "pickup_only": clean["pickup_only"],
        "pay_const": clean["pay_const"],
        "time_strict": clean["time_strict"],
        "band5g": clean["band5g"],
        "vat_inv": clean["vat_inv"],
        "vat_scheme": clean["vat_scheme"],
        "us_import": clean["us_import"],
        "lock_ai": clean["lock_ai"],
        "a_code": clean["a_code"],
        "esim_only": clean["esim_only"],
        "rep_provider": clean["rep_provider"],
        "parts": clean["parts"],
        "tt_missing": clean["tt_missing"],
        "ios_warn": clean["ios_warn"],
        "owner_type": clean["owner_type"],
        "first_owner": clean["first_owner"],
        "case_claim": clean["case_claim"],
        "ret_rights": clean["ret_rights"],
        "def_pay": clean["def_pay"],
        "swap_ins": clean["swap_ins"],
        "imei_policy": clean["imei_policy"],
        "seller_role": clean["seller_role"],
        "neg_conflict": clean["neg_conflict"],
        "act_risk": clean["act_risk"],
        "codes_json": json.dumps(data.get("codes", []), ensure_ascii=False),
        "conf_json": json.dumps(data.get("conf", {}), ensure_ascii=False),
        "evid_json": json.dumps(data.get("evidence", {}), ensure_ascii=False),
        "model": MODEL_NAME or "(unset)",
        "ver": VERSION_TAG,
    }

    with engine.begin() as conn:
        conn.execute(text(sql), params)

# ------------------ QPS limiter (global) ------------------
class RateLimiter:
    def __init__(self, qps: float):
        self.qps = qps
        self.min_interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
                now = time.monotonic()
            self.next_time = now + self.min_interval

# ------------------ Worker ------------------
def process_one(engine: Engine, limiter: RateLimiter, row: Dict[str, Any]) -> Tuple[int, bool, str]:
    gen = int(row["generation"])
    listing_id = int(row["listing_id"])
    title = row.get("title") or ""
    desc = row.get("description") or ""
    scraped_storage = row.get("storage_scraped")

    body = json.dumps(
        {"listing_id": listing_id, "modules": ["CORE"], "title": title, "description": desc},
        ensure_ascii=False,
    )
    txt_len = len(f"{title}\n{desc}")
    client = get_client()

    err: Optional[str] = None
    t0 = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            limiter.wait()
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": body},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content or "{}"
            payload, stats = validate_payload(raw, txt_len)
            upsert(engine, gen, listing_id, payload, scraped_storage)
            dt = (time.perf_counter() - t0) * 1000
            log.info(
                f"[OK] listing_id={listing_id} gen={gen} | "
                f"codes {stats['n_codes_kept']}/{stats['n_codes_in']} "
                f"kv {stats['n_kv_kept']}/{stats['n_kv_in']} | {dt:.0f} ms"
            )
            return listing_id, True, ""
        except Exception as e:
            err = str(e)
            sleep = (
                (1.5 if any(x in err for x in ("429", "Too Many", "rate", "503", "504")) else RETRY_BASE_SLEEP)
                * (2 ** (attempt - 1))
            )
            time.sleep(sleep)

    # Fail path → write neutral row so downstream never blocks
    upsert(
        engine,
        gen,
        listing_id,
        {"codes": [], "kv": {}, "risks": {}, "evidence": {"spans": []}, "conf": {}, "_neg_conflict": False},
        scraped_storage,
    )
    dt = (time.perf_counter() - t0) * 1000
    log.warning(f"[FAIL] listing_id={listing_id} gen={gen} err={err} | {dt:.0f} ms")
    return listing_id, False, err or "unknown"

# ------------------ Main ------------------
def main() -> None:
    log.info(
        f"Model={MODEL_NAME or '(unset)'} | base_url={'(default)' if not LLM_BASE_URL else LLM_BASE_URL} | "
        f"WORKERS={WORKERS} | QPS={QPS} | CONF_MIN={CONF_MIN} | VERSION_TAG={VERSION_TAG} | LIMIT=NONE"
    )

    if not PG_DSN:
        log.error("PG_DSN is required")
        sys.exit(2)

    if not LLM_API_KEY:
        log.error("LLM_API_KEY (or OPENAI_API_KEY) is required")
        sys.exit(2)
    if not MODEL_NAME:
        log.error("MODEL_NAME is required (set to your provider's chat model name)")
        sys.exit(2)


    engine = get_engine()
    rows = fetch_targets(engine)
    if not rows:
        log.info("No rows to enrich (missing-only, spam IS NULL).")
        return

    limiter = RateLimiter(QPS)
    ok = 0
    fail = 0
    t_all = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process_one, engine, limiter, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="row"):
            try:
                _id, success, _ = fut.result()
                ok += 1 if success else 0
                fail += 0 if success else 1
            except Exception as ex:
                fail += 1
                log.error(f"[EXC] worker crashed: {ex}")

    dt = time.perf_counter() - t_all
    rps = (ok + fail) / dt if dt > 0 else 0.0
    log.info(f"Done. ok={ok} fail={fail} | {dt:.2f}s total ~ {rps:.2f} rows/s")

if __name__ == "__main__":
    main()
