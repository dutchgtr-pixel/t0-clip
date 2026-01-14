#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI de-duplicator for "iPhone".iphone_listings — with live title/desc verification and skip-processed gating.

What it does per duplicated listing_id:
  1) VERIFY: Scrape each row (skip 'removed' rows if desired) and parse title/desc (hydration → JSON-LD → og/h1/title/meta).
     If ALL rows are removed or ALL scrapes return non-2xx: keep the NEWEST row; others spam='duplicate-junk' (no status merge).
  2) DECIDE: Feed VERIFIED title/desc to an LLM (strict JSON). Temporal override: newer consistent row wins if LLM disagrees.
     Bundle is VERY STRICT; mere co-mention is not enough.
  3) APPLY:
     • Non-kept rows → spam='duplicate-junk'
     • Bundle=true → mark kept rows spam='bundled' (NO generation changes)
     • Non-bundle → canonical status merge (upgrade-only): sold > removed > live > older21days
       (fills sold_date/price if present)
  4) AUDIT: Append evidence into quality_ai_json for kept + dropped rows.

This version **skips already processed listings** by default:
  --skip-processed (default True)
  --processed-scope any|exact (default any)
    any   → skip listings with any prior 'dedupe-ai' keep logs (any version)
    exact → skip listings that already have keep logs for THIS --version-tag only
"""

import os, sys, json, re, time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import argparse
import requests

try:
    import psycopg2
    import psycopg2.extras as pg_extras
except Exception:
    psycopg2 = None  # type: ignore
    pg_extras = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SCHEMA = '"iPhone"'
TABLE  = f'{SCHEMA}.iphone_listings'

# ───────────────────── CLI ─────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--pg-dsn", default=os.getenv("PG_DSN"))
ap.add_argument("--only-listing-id", type=str, default=None, help="Restrict processing to a single listing_id")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--dry-run", action="store_true")

# LLM
ap.add_argument("--provider", default=os.getenv("AI_PROVIDER", "deepseek"))
ap.add_argument("--base-url", default=os.getenv("AI_BASE_URL", None))
ap.add_argument("--api-key", default=None)
ap.add_argument("--model", default=os.getenv("AI_MODEL", "deepseek-chat"))

# spam/version/bundle/merge
ap.add_argument("--spam-value", default="duplicate-junk")
ap.add_argument("--version-tag", default="dedupe-ai/v3")
ap.add_argument("--bundle-mode", choices=["spam-bundled","set-gen-0","none"], default=os.getenv("BUNDLE_MODE","spam-bundled"))
ap.add_argument("--bundle-spam", default=os.getenv("BUNDLE_SPAM","bundled"))  # used only if bundle-mode=spam-bundled
ap.add_argument("--merge-status", dest="merge_status", action="store_true", default=True)
ap.add_argument("--no-merge-status", dest="merge_status", action="store_false")

# Scrape settings
ap.add_argument("--scrape", dest="scrape", action="store_true", default=True)
ap.add_argument("--no-scrape", dest="scrape", action="store_false")
ap.add_argument("--scrape-timeout", type=float, default=12.0)
ap.add_argument("--scrape-retries", type=int, default=2)
ap.add_argument("--scrape-user-agent", default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
ap.add_argument("--skip-scrape-removed", action="store_true", default=True)

# Skip-processed gating (prevents re-LLM on same listings)
ap.add_argument("--skip-processed", action="store_true", default=True)
ap.add_argument("--processed-scope", choices=["any","exact"], default="any")

args = ap.parse_args()

# ───────────────────── LLM prompts ─────────────────────
SYSTEM_PROMPT = r"""
You are an expert at identifying the ACTUAL iPhone device(s) being SOLD in a marketplace listing.

You will be given multiple candidate rows for ONE listing_id. These rows are ALTERNATIVE parses of the SAME ad (NOT separate posts). Your job is to pick which row(s) represent the phone(s) actually for sale.

STRICT OUTPUT (JSON ONLY — no prose, no markdown, no code fences):
{
  "keep_indexes": [0, ...],     // at least one index, UNIQUE, SORTED ASC
  "bundle": true|false,         // see STRICT rules below
  "reason": "very short reason (<=200 chars)"
}

HARD CONSTRAINTS:
- Single JSON object only; reason <= 200 chars.

CANONICAL MODEL WHITELIST:
Gen 13: "iPhone 13", "iPhone 13 Mini", "iPhone 13 Pro", "iPhone 13 Pro Max"
Gen 14: "iPhone 14", "iPhone 14 Mini", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max"
Gen 15: "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max"
Gen 16: "iPhone 16", "iPhone 16e", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max"
Gen 17: "iPhone 17", "iPhone 17 Air", "iPhone 17 Plus", "iPhone 17 Pro", "iPhone 17 Pro Max"

IGNORE buyer-intent/swap phrases ("skal kjøpe", "ønsker å kjøpe", "WTB", "byttes i", "oppgraderer til", etc.).

SIGNAL PRIORITY:
1) Structured “chip” blocks clearly naming a model
2) TITLE tokens (e.g., “iPhone 16 Pro”, “17 Air”)
3) DESCRIPTION tokens after removing buyer-intent

DISAMBIGUATION & TEMPORAL STABILIZATION:
- Prefer rows whose TITLE & DESCRIPTION AGREE on a whitelist model for its generation.
- If content changed over time and a NEWER row is internally consistent while another isn’t, prefer the NEWER CONSISTENT row (edited_date, else last_seen).
- If two rows have IDENTICAL title+description (after trivial normalization), prefer the NEWER row.
- If still ambiguous: prefer title over free text.
- If two single-phone rows disagree and the ad doesn’t EXPLICITLY sell multiple phones, this is NOT a bundle (keep one best).

BUNDLE (VERY STRICT):
bundle:true ONLY if the ad explicitly sells multiple phones, via strong words:
"pakke", "pakkepris", "selges samlet", "kun samlet", "samlet pris", "begge", "2 iPhoner", "to telefoner", "selger begge"
(or “both phones”, “as a package”) — mere co-mention is not enough.
If bundle:true, keep >=2 indexes and briefly name both.

TIE-BREAKERS:
live > older21days > sold > removed; then internal consistency; then whitelist coherence; then most recent last_seen; then lowest index.
"""

USER_PROMPT_TEMPLATE = """Decide which rows to KEEP for one listing_id. Rows are 0-based.

Return JSON ONLY:
{{
  "keep_indexes": [<int>, ...],
  "bundle": <true|false>,
  "reason": "<short reason>"
}}

Rows:
{rows_json}
"""

# ───────────────────── LLM client ─────────────────────
def resolve_api(provider: str, base_url: Optional[str], api_key: Optional[str]) -> Tuple[str, str, str]:
    p = (provider or "").lower()
    if p == "openai":
        return (base_url or "https://api.openai.com/v1",
                api_key or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or "",
                "openai")
    if p == "openrouter":
        return (base_url or "https://openrouter.ai/api/v1",
                api_key or os.getenv("OPENROUTER_API_KEY") or "",
                "openrouter")
    if p == "deepseek":
        return (base_url or "https://api.deepseek.com",
                api_key or os.getenv("DEEPSEEK_API_KEY") or "",
                "deepseek")
    return (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key or os.getenv("OPENAI_API_KEY") or "",
            "openai")

def _safe_json_object(content: str) -> Dict[str, Any]:
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    s = content.strip()
    i = s.find('{')
    if i >= 0:
        depth = 0
        for j, ch in enumerate(s[i:], start=i):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[i:j+1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    keep = []
    m = re.search(r'(?:keep[_\s-]?indexes|keepIndexes|keep)\s*:\s*(\[[^\]]*\])', content, re.I|re.S)
    if m:
        try: keep = json.loads(m.group(1))
        except Exception: keep = []
    bm = re.search(r'"bundle"\s*:\s*(true|false)', content, re.I)
    bundle = bool(bm and bm.group(1).lower() == "true")
    rm = re.search(r'"reason"\s*:\s*"(.*?)"', content, re.S)
    reason = rm.group(1) if rm else ""
    return {"keep_indexes": keep or [0], "bundle": bundle, "reason": reason}

def get_llm_client(base_url: str, api_key: str) -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is required (pip install openai)")
    return OpenAI(api_key=api_key, base_url=base_url or None)

def call_llm(client: OpenAI, model: str, rows_for_llm: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows_min = [{
        "index": idx,
        "generation": r.get("generation"),
        "model": r.get("model"),
        "status": r.get("status"),
        "title": r.get("v_title") if r.get("v_title") is not None else (r.get("title") or ""),
        "description": r.get("v_desc") if r.get("v_desc") is not None else (r.get("description") or ""),
        "edited_date": (r.get("edited_date").isoformat() if isinstance(r.get("edited_date"), datetime) else r.get("edited_date")),
        "last_seen": (r.get("last_seen").isoformat() if isinstance(r.get("last_seen"), datetime) else r.get("last_seen")),
    } for idx, r in enumerate(rows_for_llm)]

    user_prompt = USER_PROMPT_TEMPLATE.format(rows_json=json.dumps(rows_min, ensure_ascii=False))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":user_prompt}],
        response_format={"type":"json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    content = ""
    try:
        content = resp.choices[0].message.content or ""
    except Exception:
        if hasattr(resp, "to_dict"):
            content = json.dumps(resp.to_dict(), ensure_ascii=False)

    obj = _safe_json_object(content)
    raw_keep = (obj.get("keep_indexes") or obj.get("keepIndexes") or obj.get("keep") or obj.get("indices") or [])
    if not isinstance(raw_keep, list):
        raw_keep = [raw_keep]

    n = len(rows_for_llm)
    keep_list: List[int] = []
    for x in raw_keep:
        try:
            xi = int(x)
            if 0 <= xi < n:
                keep_list.append(xi)
        except Exception:
            continue
    if not keep_list:
        keep_list = [0]
    return {
        "keep_indexes": sorted(set(keep_list)),
        "bundle": bool(obj.get("bundle", False)),
        "reason": str(obj.get("reason", ""))[:500],
        "_rows_min": rows_min,
    }

# ───────────────────── Scraper (text-based; no bytes-patterns) ─────────────────────
UA = args.scrape_user_agent
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": os.getenv("SCRAPE_ACCEPT_LANGUAGE", "en-US,en;q=0.9"),
    "Upgrade-Insecure-Requests": "1",
})

rx_ld = re.compile(r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>')

rx_og_title  = re.compile(r'(?is)<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']')
rx_meta_desc = re.compile(r'(?is)<meta[^>]+name=["\']description["\'][^>]*content=["\']([^"\']+)["\']')
rx_h1_title  = re.compile(r'(?is)<h1[^>]*>\s*([^<]{3,200})\s*</h1>')
rx_title_tag = re.compile(r'(?is)<title>\s*([^<]{3,200})\s*</title>')
rx_tags      = re.compile(r'(?is)<[^>]+>')

def _html_unescape(s: str) -> str:
    try:
        import html
        return html.unescape(s)
    except Exception:
        return s

def _to_text(b: Optional[bytes]) -> str:
    return "" if b is None else b.decode("utf-8", "ignore")

def _extract_hydration_json_text(html_text: str) -> Optional[str]:
    """Public-release stub.

    Platform-specific hydration extraction has been intentionally removed to avoid
    shipping target-site fingerprints. Title/description verification still works
    via JSON-LD and common meta tags.
    """
    return None


def _hydration_title_desc_text(js_text: str) -> Tuple[str, str]:
    """Best-effort extraction of title/description from an embedded JSON blob.

    Public-release note:
      The original repo had marketplace-specific hydration parsing. For public
      release we use a generic JSON key search only (title/name/description).
    """
    try:
        root = json.loads(js_text)
    except Exception:
        return "", ""

    title, desc = "", ""
    tlen, dlen = 0, 0

    def dfs(v):
        nonlocal title, desc, tlen, dlen
        if isinstance(v, dict):
            for k, vv in v.items():
                kl = str(k).lower()
                if isinstance(vv, str):
                    s = vv.strip()
                    if (kl == "title" or kl == "name") and 3 <= len(s) <= 300:
                        if len(s) > tlen:
                            title, tlen = s, len(s)
                    if kl == "description" and len(s) >= 10:
                        if len(s) > dlen:
                            desc, dlen = s, len(s)
                dfs(vv)
        elif isinstance(v, list):
            for it in v:
                dfs(it)

    dfs(root)
    return _html_unescape(title), _html_unescape(desc)



def _best_from_ld_text(html_text: str) -> Tuple[str,str,bool]:
    matches = rx_ld.findall(html_text)
    best_t, best_d, ok = "", "", False
    for raw in matches:
        try:
            v = json.loads(raw)
        except Exception:
            continue
        def handle(mm):
            nonlocal best_t, best_d, ok
            if not isinstance(mm, dict): return
            name = str(mm.get("name") or "").strip()
            desc = str(mm.get("description") or "").strip()
            if name and len(name) > len(best_t): best_t = name
            if desc and len(desc) > len(best_d): best_d = desc
            ok = ok or bool(name or desc)
        if isinstance(v, list):
            for it in v: handle(it)
        else:
            handle(v)
    return _html_unescape(best_t), _html_unescape(best_d), ok

def _pick_title_desc_from_html_text(html_text: str) -> Tuple[str,str,Dict[str,Any]]:
    ev: Dict[str,Any] = {}
    js = _extract_hydration_json_text(html_text)
    if js:
        t, d = _hydration_title_desc_text(js)
        if t:
            ev["title_src"]="hydration"
            if d: ev["desc_src"]="hydration"
            return t,d,ev
        if d:
            ev["desc_src"]="hydration"
    t2,d2,ok = _best_from_ld_text(html_text)
    if ok and (t2 or d2):
        if t2: ev["title_src"]=ev.get("title_src") or "ld_json"
        if d2: ev["desc_src"]=ev.get("desc_src") or "ld_json"
        return t2 or "", d2 or "", ev
    m = rx_og_title.search(html_text)
    if m:
        ev["title_src"]=ev.get("title_src") or "og:title"
        return m.group(1).strip(),"",ev
    m = rx_h1_title.search(html_text)
    if m:
        ev["title_src"]=ev.get("title_src") or "h1"
        return m.group(1).strip(),"",ev
    m = rx_title_tag.search(html_text)
    if m:
        ev["title_src"]=ev.get("title_src") or "title_tag"
        return m.group(1).strip(),"",ev
    m = rx_meta_desc.search(html_text)
    if m:
        ev["desc_src"]=ev.get("desc_src") or "meta:description"
        return "", m.group(1).strip(), ev
    return "","",ev

def _http_get(url: str, timeout_s: float, retries: int) -> Tuple[int, Optional[str]]:
    if not url: return 0, None
    last_status = 0
    for a in range(retries+1):
        try:
            r = _SESSION.get(url, timeout=timeout_s)
            status = r.status_code
            text   = _to_text(r.content)
            r.close()
            return status, text
        except Exception:
            last_status = 0
            time.sleep(min(0.6*(a+1), 2.0))
    return last_status, None

# ───────────────────── Local normalization & rules ─────────────────────
_STRONG_BUNDLE = [
    r'\bpakkepris\b', r'\bselges\s+samlet\b', r'\bkun\s+samlet\b', r'\bbegge\b',
    r'\b2\s+iphoner\b', r'\bto\s+telefoner\b', r'\bselger\s+begge\b', r'\bsamlet\s+pris\b',
    r'\bboth\s+phones?\b', r'\bas\s+a\s+package\b'
]
_MULTI_PATTERNS = [r'iphone\s+\d{2}.*?(?:\+| og | and | & )\s*iphone\s+\d{2}']
_WTB_MARKERS = [
    r'\bskal\s+kjøpe\b', r'\bønsker\s+å\s+kjøpe\b', r'\bkjøpes\b', r'\bwtb\b',
    r'\blooking\s+to\s+buy\b', r'\bwant\s+to\s+buy\b', r'\bbyttes\s+i\b',
    r'\bbytter\s+til\b', r'\boppgraderer\s+til\b'
]

def looks_like_multi_device(rows: List[Dict[str,Any]], kept: List[int]) -> bool:
    idxs = kept or list(range(len(rows)))
    text = " ".join(((rows[i].get("v_title") or rows[i].get("title") or "") + " " + (rows[i].get("v_desc") or rows[i].get("description") or "")) for i in idxs)
    strong = any(re.search(p, text, re.I) for p in _STRONG_BUNDLE)
    co_mention = any(re.search(p, text, re.I|re.S) for p in _MULTI_PATTERNS)
    return strong or (strong and co_mention)

def _norm(s: Optional[str]) -> str:
    if not s: return ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _is_buyer_intent(title: str, desc: str) -> bool:
    blob = f"{title} {desc}"
    return any(re.search(p, blob, re.I) for p in _WTB_MARKERS)

def _model_key(model: str) -> str:
    return _norm(model)

def _row_text_has_model(row: Dict[str,Any]) -> bool:
    mk = _model_key(row.get("model") or "")
    if not mk: return False
    t = _norm(row.get("v_title") or row.get("title"))
    d = _norm(row.get("v_desc")  or row.get("description"))
    return (mk in t) or (mk in d)

def _row_consistent(row: Dict[str,Any]) -> bool:
    if _is_buyer_intent(row.get("v_title") or row.get("title") or "",
                        row.get("v_desc")  or row.get("description") or ""):
        return False
    return _row_text_has_model(row)

def _dt_of(row: Dict[str,Any]) -> datetime:
    return (row.get("edited_date") or row.get("last_seen") or datetime.min)

def pick_newest_consistent_index(rows: List[Dict[str,Any]]) -> int:
    order = sorted(range(len(rows)), key=lambda i: _dt_of(rows[i]), reverse=True)
    for i in order:
        if _row_consistent(rows[i]):
            return i
    return order[0]

# ───────────────────── DB helpers ─────────────────────
def ensure_audit_columns(cur):
    """Ensure audit columns exist.

    Important:
      - Runtime DDL is hazardous in a busy pipeline because ALTER TABLE requires ACCESS EXCLUSIVE.
      - Even with IF NOT EXISTS, Postgres must still take the lock.
      - Therefore, we *first* check the catalog and return immediately if columns exist.
      - If genuinely missing, we fail fast on locks to avoid wedging the database.
    """
    want = {"quality_ai_json", "quality_ai_at", "quality_ai_version"}
    cur.execute("""
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'iPhone'
        AND table_name   = 'iphone_listings'
        AND column_name IN ('quality_ai_json','quality_ai_at','quality_ai_version')
    """)
    have = {r[0] for r in cur.fetchall()}
    if want.issubset(have):
        return

    # If missing, attempt a single ALTER, but fail fast if locks can't be acquired.
    cur.execute("SET lock_timeout = '2s'")
    cur.execute(f"""
      ALTER TABLE {TABLE}
        ADD COLUMN IF NOT EXISTS quality_ai_json jsonb,
        ADD COLUMN IF NOT EXISTS quality_ai_at timestamptz,
        ADD COLUMN IF NOT EXISTS quality_ai_version text
    """)

def _get_table_columns(cur) -> Set[str]:
    cur.execute("""
      SELECT column_name
      FROM information_schema.columns
      WHERE table_schema = 'iPhone' AND table_name = 'iphone_listings'
    """)
    return {r[0] for r in cur.fetchall()}

def fetch_dup_listings(cur, limit: int, only_listing_id: Optional[str],
                    skip_processed: bool, processed_scope: str, version_tag: str) -> List[int]:
    # Single-LISTING path, respecting skip_processed
    if only_listing_id:
        if skip_processed:
            cur.execute("""
WITH already AS (
  SELECT 1
  FROM "iPhone".iphone_listings l
  WHERE l.listing_id = %s
    AND l.quality_ai_json IS NOT NULL
    AND EXISTS (
      SELECT 1
      FROM jsonb_array_elements(l.quality_ai_json) el
      WHERE el->>'tool' = 'dedupe-ai'
        AND (el->>'action') IN ('keep','keep-bundle')
        AND ( %s = 'any' OR COALESCE(el->>'version','') = %s )
    )
),
is_dup AS (
  SELECT 1
  FROM "iPhone".iphone_listings
  WHERE listing_id = %s
  GROUP BY listing_id
  HAVING COUNT(*) > 1
),
needs_work AS (
  SELECT 1
  FROM "iPhone".iphone_listings
  WHERE listing_id = %s
    AND COALESCE(spam,'') NOT IN ('duplicate-junk','bundled')
)
SELECT CASE
         WHEN EXISTS (SELECT 1 FROM is_dup)
              AND NOT EXISTS (SELECT 1 FROM already)
              AND EXISTS (SELECT 1 FROM needs_work)
         THEN %s
       END AS listing_id
""", (only_listing_id, processed_scope, version_tag, only_listing_id, only_listing_id, only_listing_id))
            row = cur.fetchone()
            return [int(row[0])] if row and row[0] is not None else []
        else:
            cur.execute("""
SELECT listing_id
FROM "iPhone".iphone_listings
WHERE listing_id = %s
GROUP BY listing_id
HAVING COUNT(*) > 1
""", (only_listing_id,))
            r = cur.fetchone()
            return [int(only_listing_id)] if r else []

    # Many-LISTING path with skip gating
    if skip_processed:
        sql = f"""
WITH dups AS (
  SELECT listing_id, MAX(last_seen) AS max_last_seen
  FROM "iPhone".iphone_listings
  GROUP BY listing_id
  HAVING COUNT(*) > 1
),
already AS (
  SELECT DISTINCT listing_id
  FROM "iPhone".iphone_listings l
  WHERE l.quality_ai_json IS NOT NULL
    AND EXISTS (
      SELECT 1
      FROM jsonb_array_elements(l.quality_ai_json) el
      WHERE el->>'tool' = 'dedupe-ai'
        AND (el->>'action') IN ('keep','keep-bundle')
        AND ( %s = 'any' OR COALESCE(el->>'version','') = %s )
    )
),
needs AS (
  SELECT DISTINCT listing_id
  FROM "iPhone".iphone_listings
  WHERE COALESCE(spam,'') NOT IN ('duplicate-junk','bundled')
)
SELECT d.listing_id
FROM dups d
LEFT JOIN already a USING (listing_id)
JOIN needs n USING (listing_id)
WHERE a.listing_id IS NULL
ORDER BY d.max_last_seen DESC
{ 'LIMIT %s' if limit and limit > 0 else '' }
"""
        params = (processed_scope, version_tag) + ((limit,) if limit and limit > 0 else tuple())
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]
    else:
        sql = f"""
SELECT listing_id
FROM "iPhone".iphone_listings
GROUP BY listing_id
HAVING COUNT(*) > 1
ORDER BY MAX(last_seen) DESC
{ 'LIMIT %s' if limit and limit > 0 else '' }
"""
        cur.execute(sql, ((limit,) if limit and limit > 0 else tuple()))
        return [r[0] for r in cur.fetchall()]

def load_group(cur, fid: int, cols: Set[str]) -> List[Dict[str, Any]]:
    base_cols = ["generation","status","model","title","description","edited_date","last_seen","spam","url"]
    if "sold_date" in cols: base_cols.insert(5,"sold_date")
    if "sold_price" in cols:
        insert_at = 6 if "sold_date" in cols else 5
        base_cols.insert(insert_at, "sold_price")
    select_list = ", ".join(base_cols)
    cur.execute(f"""
      SELECT {select_list}
      FROM {TABLE}
      WHERE listing_id=%s
      ORDER BY last_seen DESC NULLS LAST, edited_date DESC NULLS LAST
    """, (fid,))
    colnames = [d.name for d in cur.description]
    return [dict(zip(colnames, row)) for row in cur.fetchall()]

def mark_non_kept(cur, fid: int, keep_gens: List[int], spam_value: str):
    cur.execute(f"""
      UPDATE {TABLE}
      SET spam = %s,
          last_seen = now()
      WHERE listing_id = %s
        AND generation <> ALL(%s)
    """, (spam_value, fid, keep_gens))

def append_quality_log(cur, fid: int, gen: int, entry: Dict[str, Any], version_tag: str):
    entry_json = json.dumps(entry, ensure_ascii=False)
    cur.execute(f"""
      UPDATE {TABLE}
      SET quality_ai_json =
            CASE
              WHEN quality_ai_json IS NULL THEN jsonb_build_array(CAST(%s AS jsonb))
              WHEN jsonb_typeof(quality_ai_json) = 'array' THEN quality_ai_json || CAST(%s AS jsonb)
              ELSE jsonb_build_array(quality_ai_json) || CAST(%s AS jsonb)
            END,
          quality_ai_at = now(),
          quality_ai_version = %s
      WHERE listing_id = %s AND generation = %s
    """, (entry_json, entry_json, entry_json, version_tag, fid, gen))

def apply_bundle_policy(cur, fid: int, keep_gens: List[int], mode: str, bundle_spam: Optional[str]):
    if not keep_gens: return
    if mode == "spam-bundled" and bundle_spam:
        cur.execute(f"""
          UPDATE {TABLE}
          SET spam = %s,
              last_seen = now()
          WHERE listing_id = %s
            AND generation = ANY(%s)
        """, (bundle_spam, fid, keep_gens))
    elif mode == "set-gen-0":
        # Not default (beware PK partitions on (generation,listing_id))
        cur.execute(f"""
          UPDATE {TABLE}
          SET generation = 0,
              model = COALESCE(model, 'BUNDLED'),
              last_seen = now()
          WHERE listing_id = %s
            AND generation = ANY(%s)
        """, (fid, keep_gens))
    # mode "none": do nothing

# ───────────────────── Canonical status merge (non-bundle) ─────────────────────
STATUS_RANK_CANON = {"sold": 3, "removed": 2, "live": 1, "older21days": 0}

def _canon_status(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows: return None
    best = max(rows, key=lambda r: (
        STATUS_RANK_CANON.get((r.get("status") or "").lower(), -1),
        (r.get("last_seen") or r.get("edited_date") or "")
    ))
    st = (best.get("status") or "").lower()
    return st if st else None

def _canon_sold(rows: List[Dict[str, Any]]) -> Tuple[Optional[Any], Optional[Any]]:
    sold_rows = [r for r in rows if (r.get("status") or "").lower() == "sold"]
    if not sold_rows: return (None,None)
    sold_dates = [r.get("sold_date") for r in sold_rows if r.get("sold_date") is not None]
    sold_date = max(sold_dates) if sold_dates else None
    price = None
    if sold_date is not None:
        for r in sold_rows:
            if r.get("sold_date") == sold_date and r.get("sold_price") is not None:
                price = r.get("sold_price"); break
    if price is None:
        for r in sold_rows:
            if r.get("sold_price") is not None:
                price = r.get("sold_price"); break
    return (sold_date, price)

def _escalate_status(cur, fid: int, gen: int, canon: Optional[str],
                     sold_date: Optional[Any], sold_price: Optional[Any],
                     cols: Set[str]):
    if not canon: return
    sold_cols_exist = ("sold_date" in cols) and ("sold_price" in cols)
    if sold_cols_exist:
        cur.execute(f"""
          UPDATE {TABLE} t
          SET status = CASE
                         WHEN %s='sold' THEN 'sold'
                         WHEN %s='removed' AND t.status <> 'sold' THEN 'removed'
                         WHEN %s='live'    AND t.status NOT IN ('sold','removed') THEN 'live'
                         WHEN %s='older21days' AND (t.status IS NULL OR t.status NOT IN ('sold','removed','live')) THEN 'older21days'
                         ELSE t.status
                       END,
              sold_date  = COALESCE(t.sold_date, %s),
              sold_price = COALESCE(t.sold_price, %s),
              last_seen  = GREATEST(t.last_seen, now())
          WHERE t.listing_id=%s AND t.generation=%s
        """, (canon, canon, canon, canon, sold_date, sold_price, int(fid), gen))
    else:
        cur.execute(f"""
          UPDATE {TABLE} t
          SET status = CASE
                         WHEN %s='sold' THEN 'sold'
                         WHEN %s='removed' AND t.status <> 'sold' THEN 'removed'
                         WHEN %s='live'    AND t.status NOT IN ('sold','removed') THEN 'live'
                         WHEN %s='older21days' AND (t.status IS NULL OR t.status NOT IN ('sold','removed','live')) THEN 'older21days'
                         ELSE t.status
                       END,
              last_seen  = GREATEST(t.last_seen, now())
          WHERE t.listing_id=%s AND t.generation=%s
        """, (canon, canon, canon, canon, int(fid), gen))

# ───────────────────── Main ─────────────────────
def main():
    base_url, api_key, provider_tag = resolve_api(args.provider, args.base_url, args.api_key)
    if not api_key:
        print("ERROR: missing API key (use --api-key or set OPENAI_API_KEY/OPENAI_KEY/OPENROUTER_API_KEY/DEEPSEEK_API_KEY)", file=sys.stderr)
        sys.exit(2)
    if OpenAI is None:
        print("ERROR: openai package is required (pip install openai)", file=sys.stderr)
        sys.exit(2)

    client = get_llm_client(base_url, api_key)

    if not args.pg_dsn:
        print("ERROR: missing Postgres DSN (set PG_DSN or pass --pg-dsn)", file=sys.stderr)
        sys.exit(2)

    if psycopg2 is None or pg_extras is None:
        print("ERROR: psycopg2 is required (pip install psycopg2-binary)", file=sys.stderr)
        sys.exit(2)

    conn = psycopg2.connect(args.pg_dsn)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=pg_extras.DictCursor)

    with conn:
        ensure_audit_columns(cur)
    cols = _get_table_columns(cur)
    conn.commit()  # close read txn (avoid idle-in-transaction during network/LLM)

    # target set (SKIP-PROCESSED aware)
    if args.only_listing_id:
        listings = fetch_dup_listings(cur,
                                limit=args.limit,
                                only_listing_id=args.only_listing_id,
                                skip_processed=args.skip_processed,
                                processed_scope=args.processed_scope,
                                version_tag=args.version_tag)
        listing_ids = listings
    else:
        listing_ids = fetch_dup_listings(cur,
                                   limit=args.limit,
                                   only_listing_id=None,
                                   skip_processed=args.skip_processed,
                                   processed_scope=args.processed_scope,
                                   version_tag=args.version_tag)

    conn.commit()  # close read txn before long loop / network work

    if not listing_ids:
        print("No duplicates found.")
        cur.close(); conn.close()
        return

    print(f"Processing {len(listing_ids)} duplicated listings …")
    updated = 0

    for fid in listing_ids:
        try:
            rows = load_group(cur, int(fid), cols)
            conn.commit()  # release read locks before scrape/LLM
            if len(rows) < 2:
                continue

            # VERIFY via scrape (title/desc)
            all_removed = all((str(r.get("status") or "").lower() == "removed") for r in rows)
            all_404 = True
            verify_summary = []

            for r in rows:
                r["v_title"] = None
                r["v_desc"]  = None
                r["v_http"]  = None
                r["v_src"]   = {}
                url = (r.get("url") or "").strip()
                status = str(r.get("status") or "").lower()

                if args.skip_scrape_removed and status == "removed":
                    verify_summary.append({"try": False, "status": status, "url": bool(url)})
                    continue
                if not args.scrape or not url:
                    verify_summary.append({"try": False, "status": status, "url": bool(url)})
                    continue

                http_status, text = _http_get(url, timeout_s=args.scrape_timeout, retries=args.scrape_retries)
                r["v_http"] = http_status
                if text and 200 <= http_status < 300:
                    t, d, ev = _pick_title_desc_from_html_text(text)
                    if t: r["v_title"] = t
                    if d: r["v_desc"]  = d
                    r["v_src"] = ev
                    all_404 = False
                verify_summary.append({"try": True, "http": http_status, "got_title": bool(r["v_title"]), "got_desc": bool(r["v_desc"])})

            # all-removed or all 404/non-2xx → keep newest; others spam
            if all_removed or all_404:
                newest_idx = sorted(range(len(rows)), key=lambda i: _dt_of(rows[i]), reverse=True)[0]
                keep_idxs = [newest_idx]
                keep_gens = sorted({ rows[i]["generation"] for i in keep_idxs })
                drop_gens = sorted({ r["generation"] for k, r in enumerate(rows) if k not in keep_idxs })
                print(f"[listing_id={fid}] ALL-REMOVED/404 fallback → keep_idx={keep_idxs} keep_gens={keep_gens}")
                if args.dry_run:
                    continue
                with conn:
                    if drop_gens:
                        mark_non_kept(cur, int(fid), keep_gens, args.spam_value)
                    for g in keep_gens:
                        append_quality_log(cur, int(fid), g, {
                            "tool":"dedupe-ai","version":args.version_tag,"action":"keep",
                            "reason":"all-removed-or-404 -> newest kept","bundle":False,
                            "provider":provider_tag,"model":args.model,"verify":verify_summary
                        }, args.version_tag)
                    for g in drop_gens:
                        append_quality_log(cur, int(fid), g, {
                            "tool":"dedupe-ai","version":args.version_tag,"action":"mark-spam",
                            "spam":args.spam_value,"reason":"all-removed-or-404 -> newest kept",
                            "provider":provider_tag,"model":args.model,"verify":verify_summary
                        }, args.version_tag)
                updated += 1
                continue

            # Build LLM rows with verified text (v_title/v_desc)
            rows_for_llm = rows

            # Deterministic temporal pick
            newest_idx = pick_newest_consistent_index(rows_for_llm)

            # LLM
            decision = call_llm(client, args.model, rows_for_llm)
            keep_idxs: List[int] = decision["keep_indexes"]
            bundle_claimed: bool = bool(decision.get("bundle", False))
            reason: str = decision.get("reason", "")

            # strict bundle guard
            if bundle_claimed and not looks_like_multi_device(rows_for_llm, keep_idxs):
                status_rank = {"live":3, "older21days":2, "sold":1, "removed":0}
                best_idx = max(
                    keep_idxs,
                    key=lambda i: (status_rank.get((rows_for_llm[i].get("status") or "").lower(), 0),
                                   (rows_for_llm[i].get("last_seen") or rows_for_llm[i].get("edited_date") or ""))
                )
                keep_idxs = [best_idx]
                bundle_claimed = False
                reason = (reason[:150] + " | collapsed: no explicit multi-device wording")[:200]

            # temporal override: newer consistent wins
            if not bundle_claimed and newest_idx not in keep_idxs:
                keep_idxs = [newest_idx]
                reason = (reason[:150] + " | temporal override: newer consistent row wins").strip()[:200]

            keep_gens = sorted({ rows_for_llm[i]["generation"] for i in keep_idxs })
            drop_gens = sorted({ r["generation"] for k, r in enumerate(rows_for_llm) if k not in keep_idxs })

            print(f"[listing_id={fid}] keep_idx={keep_idxs} keep_gens={keep_gens} bundle={bundle_claimed} reason={reason}")

            if args.dry_run:
                continue

            with conn:
                if drop_gens:
                    mark_non_kept(cur, int(fid), keep_gens, args.spam_value)

                for g in keep_gens:
                    append_quality_log(cur, int(fid), g, {
                        "tool":"dedupe-ai","version":args.version_tag,
                        "action":("keep-bundle" if bundle_claimed else "keep"),
                        "reason":reason,"bundle":bundle_claimed,
                        "provider":provider_tag,"model":args.model,
                        "verify":verify_summary,
                        "temporal_override": (not bundle_claimed and newest_idx in keep_idxs)
                    }, args.version_tag)

                for g in drop_gens:
                    append_quality_log(cur, int(fid), g, {
                        "tool":"dedupe-ai","version":args.version_tag,"action":"mark-spam",
                        "spam":args.spam_value,"reason":reason,
                        "provider":provider_tag,"model":args.model,
                        "verify":verify_summary
                    }, args.version_tag)

                if bundle_claimed:
                    # Per decision: mark kept rows as spam='bundled' instead of changing generation.
                    apply_bundle_policy(cur, int(fid), keep_gens, mode=args.bundle_mode, bundle_spam=args.bundle_spam)
                else:
                    if args.merge_status:
                        canon = _canon_status(rows_for_llm)
                        sold_date, sold_price = (None, None)
                        if canon == "sold" and ("sold_date" in cols or "sold_price" in cols):
                            sold_date, sold_price = _canon_sold(rows_for_llm)
                        for g in keep_gens:
                            _escalate_status(cur, int(fid), g, canon, sold_date, sold_price, cols)
                            append_quality_log(cur, int(fid), g, {
                                "tool":"dedupe-ai","version":args.version_tag,"action":"status-merge",
                                "from_statuses":[(r.get("status") or "") for r in rows_for_llm],
                                "to_status": canon,
                                "sold_date": (sold_date.isoformat() if hasattr(sold_date,"isoformat") else sold_date),
                                "sold_price": sold_price
                            }, args.version_tag)

            updated += 1

        except Exception as e:
            conn.rollback()
            print(f"[listing_id={fid}] ERROR: {e}", file=sys.stderr)
            continue

    cur.close()
    conn.close()
    print(f"Done. Updated={updated}/{len(listing_ids)} listings")


if __name__ == "__main__":
    main()























