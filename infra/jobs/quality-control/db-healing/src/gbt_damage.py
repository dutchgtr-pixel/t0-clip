#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gbt_damage_llm_decider.py — LLM-only listing damage scorer (Python 3.10+)

HARD RULES (built-in; no tokens wasted):
- condition_score = 1.0  → damage_binary_ai=0, damage_severity_ai=0, reason 'brand new (cs1.0)'
- condition_score = 0.2  → damage_binary_ai=1, damage_severity_ai=3, reason 'pre-determined sev3 (cs0.2)'
These are stamped in DB before fetching, and the LLM fetch ALWAYS excludes cs in (1.0, 0.2).

• The LLM makes the FINAL decision (bin/sev/reason/lock) for the remaining rows. No deterministic judging. No semantic regex.
• We give the model a BIG, explicit policy prompt + many few-shots.
• We coerce the returned JSON minimally so the pipeline never crashes.
• Results are persisted to Postgres, including the full LLM decision blob (JSONB) in damage_ai_json.

Env:
  PG_DSN, MODEL_NAME, OPENAI_BASE_URL (optional), OPENAI_API_KEY
  WORKERS, CAP_13, CAP_14, CAP_15, CAP_16, CAP_17
  DRY_RUN, BATTERY_REPAIR, LOCK_REPAIR, SKIP_ALREADY_LABELED, VERSION_TAG
  LOG_LEVEL, MAX_RETRIES, RETRY_BASE_SLEEP, RUN_GOLDENS
  INCLUDE_ALL_CS, NEW_CLIENT_PER_CALL

DB:
  Table: <PG_SCHEMA>.<LISTINGS_TABLE>
  Uses: listing_id, generation, title, description, edited_date, condition_score,
        battery_pct, battery_pct_fixed, battery_pct_fixed_ai, operator_lock
  Updates: damage_binary_ai, damage_severity_ai, damage_reason_ai,
           damage_ai_version, damage_ai_at, damage_ai_json (JSONB blob)
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# 3rd-party
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import JSONB

# OpenAI SDK (>=1.0)
from openai import OpenAI
from openai.types.chat import ChatCompletion

# ----------------------- Configuration -----------------------

PG_DSN = (os.getenv("PG_DSN") or "").strip()
if not PG_DSN:
    raise SystemExit("PG_DSN environment variable is required (provide via environment/secret manager; do not hardcode credentials).")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("\ufeffOPENAI_API_KEY")

WORKERS = int(os.getenv("WORKERS", "64"))
CAPS: Dict[int, int] = {
    13: int(os.getenv("CAP_13", "1350")),
    14: int(os.getenv("CAP_14", "1350")),
    15: int(os.getenv("CAP_15", "1350")),
    16: int(os.getenv("CAP_16", "1350")),
    17: int(os.getenv("CAP_17", "1350")),
}

DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
BATTERY_REPAIR = os.getenv("BATTERY_REPAIR", "true").lower() in ("1", "true", "yes")
LOCK_REPAIR = os.getenv("LOCK_REPAIR", "true").lower() in ("1", "true", "yes")
SKIP_ALREADY_LABELED = os.getenv("SKIP_ALREADY_LABELED", "true").lower() in ("1", "true", "yes")
VERSION_TAG = os.getenv("VERSION_TAG", "llm-decider-v1.0")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_RETRIES = max(1, int(os.getenv("MAX_RETRIES", "3")))
RETRY_BASE_SLEEP = float(os.getenv("RETRY_BASE_SLEEP", "0.6"))
RUN_GOLDENS = os.getenv("RUN_GOLDENS", "0").lower() in ("1", "true", "yes")

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# NOTE: we now ALWAYS exclude cs in (1.0, 0.2) from LLM; this flag is kept for compatibility only.
INCLUDE_ALL_CS = os.getenv("INCLUDE_ALL_CS", "0").lower() in ("1", "true", "yes")

NEW_CLIENT_PER_CALL = os.getenv("NEW_CLIENT_PER_CALL", "0").lower() in ("1", "true", "yes")

# Database identifiers (public-release friendly)
PG_SCHEMA = (os.getenv("PG_SCHEMA") or "public").strip() or "public"
LISTINGS_TABLE_NAME = (os.getenv("LISTINGS_TABLE") or "listings").strip() or "listings"
LISTING_ID_COLUMN = (os.getenv("LISTING_ID_COLUMN") or "listing_id").strip() or "listing_id"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def _qident(name: str) -> str:
    """Quote a PostgreSQL identifier safely (schema/table/column)."""
    if not _IDENTIFIER_RE.match(name):
        raise SystemExit(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'

TABLE = f"{_qident(PG_SCHEMA)}.{_qident(LISTINGS_TABLE_NAME)}"
ID_COL = _qident(LISTING_ID_COLUMN)

# Hard-rule stamping version tag (for cs 1.0 / 0.2 updates)
RULE_VERSION = 'rule-cs-hard-v1'

# ----------------------- Logging -----------------------

logger = logging.getLogger("gbt_damage_llm_decider")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(LOG_LEVEL)

def jcompact(d: Any) -> str:
    try:
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(d)

# ----------------------- DECIDER PROMPT (BIG) -----------------------

DECIDER_SYSTEM_PROMPT = r"""
You are the FINAL DECIDER for device listing damage. Read the listing text (title + description) and the numeric context provided:
- cs = condition score (one of 0.9, 0.7, 0.5, 0.0)
- battery_effective = numeric battery capacity (20..100) if known; otherwise null

You MUST apply the rules below EXACTLY. DO NOT infer. Use ONLY literal evidence from the text + the provided cs and battery_effective.

Return STRICT JSON **only** (no prose, no markdown), with EXACT keys and types:

{
  "id": "<listing_id>",                  // string
  "bin": 0 or 1,                      // 0 iff sev=0, else 1
  "sev": 0 or 1 or 2 or 3,            // 0 mint; 1 cosmetic; 2 functional/non-oem/lens-glass/batt<80; 3 device glass or severe panel
  "reason": "<=80 chars string>",     // canonical, see section H
  "lock": "locked"|"unlocked"|"null", // carrier/SIM lock if literally stated; else "null"
  "meta": {                           // TRUE only if literally supported by the text
    "no_wear_global": true|false,     // explicit whole-device denial; see A2 phrases
    "protector_only": true|false,     // protector is damaged AND device glass is fine
    "glass": true|false,              // device front/back glass cracked (NOT protector; NOT lens)
    "back_glass": true|false,         // back glass cracked (text says back/bakglass)
    "lens_glass": true|false,         // camera lens cover cracked
    "panel_severe": true|false,       // green/purple line, ghosting/burn-in, black blotch
    "light_panel": true|false,        // dead pixel / tiny dot
    "charging": true|false,           // lader ikke / må vri / dårlig kontakt / loose/wobbly (present now)
    "non_oem": true|false,            // part replaced: screen/battery/backglass/camera/port/etc. (NOT insurance swap alone)
    "battery_clamp": true|false       // ONLY if battery_effective < 80 (strict)
  }
}


================================================================
READ-TO-END & CONTRADICTIONS (CRITICAL)
----------------------------------------------------------------
• Always read the ENTIRE title + description before deciding. Do NOT stop early on “mint / ingen skader / ripefri” claims.
• If later text contradicts earlier mint/no-wear statements (e.g., faults, numeric battery <80, “må vri for å lade”, non-OEM parts, lens-glass crack),
  the CONTRADICTION WINS. Mint/no-wear must be ignored when any higher-priority evidence appears.

PRIORITY LADDER (highest → lowest). Pick the HIGHEST present; that sets sev/bin/reason. Never use mint words if any higher item is present.
1) Device glass or severe panel (e.g., “knust/skjerm sprukket/sprekk”, grønn/purple line, ghosting/burn-in, black blotch)
   → sev=3; bin=1; reason one of {"knust skjerm","knust bak","panel grønn linje","panel ghosting","panel black blotch"};
     meta.glass/back_glass/panel_severe accordingly.

2) Functional faults PRESENT NOW (charging issues “må vri/dårlig kontakt/lader ikke”, Face ID virker ikke, kamera/mik/earpiece/høyttaler defekt,
   SIM/nett problem “modulfeil (nett)”)
   → sev=2; bin=1; canonical reason (e.g., "ladeport kinkig","face id virker ikke","modulfeil (nett)"); set matching meta.* (charging=true, etc.).

3) Lens-glass cracked (“linse/linseglass/kameraglass sprukket/knust”)
   → sev=2; bin=1; reason "linseglass sprukket"; meta.lens_glass=true.

E3. App/service ban (device-level ban/lockout):
    Tokens: "sperret", "utestengt", "bannet/banned", "blokkert/blocked", "suspendert/suspended"
    Context: mentions of a named app/service (e.g., Snapchat, TikTok, BankID) saying the device is banned/blocked.
    → sev=2, reason "app/service ban".


4) Non-OEM/part replaced explicitly (screen/battery/backglass/camera/port/etc.) — NOT insurance swap alone
   → sev=2; bin=1; reason "non-oem <part>"; meta.non_oem=true.

5) Numeric battery clamp: battery_effective < 80 (strict)
   → sev=2; bin=1; reason "batt<80"; meta.battery_clamp=true. (Do NOT clamp at 80..100.)

6) Explicit cosmetic wear only (no faults) (e.g., “små riper/merker/hakk/bruksspor” with a surface)
   → sev=1; bin=1; short cosmetic reason.

7) Otherwise (no higher-priority evidence): follow normal cs defaults (e.g., cs=0.9 → sev=0 mint; cs=0.7/0.0 can mint ONLY with global no-wear + no faults).

Tie-handling:
• If multiple items appear, choose the highest in the ladder as the primary reason; set meta flags for all that apply.
• If any item in 1–5 appears anywhere, IGNORE mint/no-wear phrasing and do NOT use mint words in the reason.
• bin = 0 iff sev=0; otherwise 1.


================================================================
A) CONDITION-SCORE CLAMPS (final defaults; override ONLY with literal evidence)
----------------------------------------------------------------
A1. cs = 0.9 → default sev=0 (mint) unless there is EXPLICIT cosmetic wear or ANY fault.
    • Words like "strøken / som ny / pent brukt / i god stand / ser ut som ny / ikke synlig brukt" are GENERIC.
      They do NOT count as explicit wear or faults and MUST NOT demote cs=0.9 to sev=1.
    • Generic/spec-only text at cs=0.9 STILL MINTS (sev=0) unless explicit wear/faults exist.

A2. cs = 0.7 or cs = 0.0 → baseline sev=1.
    You may set sev=0 ONLY with an explicit GLOBAL no-wear denial AND no faults. Accepted global phrases (case-insensitive):
      "ingen riper", "uten riper", "ingen skader", "uten skader", "ingen merker", "uten merker",
      "ripefri", "no scratches", "no damage", "verken riper eller skader",
      "ingen synlige riper", "ingen synlige merker", "ingen synlige riper eller merker",
      "ingen synlige skader", "skadefri"
    NOT global (no minting at cs=0.7/0.0):
      • hedged: "tilnærmet / nesten / så godt som / omtrent ripefri"
      • component-only: "ingen riper i skjermen / i rammen / på baksiden"

A3. cs = 0.5 → baseline sev=2 (“cs0.5 baseline”). Elevate to sev=3 ONLY for device-glass cracks or severe panel.

================================================================
B) GENERIC/BOILERPLATE/SPEC TEXT (IMPORTANT)
----------------------------------------------------------------
Treat copied specs/boilerplate (“Apple iPhone <model> … passer for brukere som ønsker … høy ytelse / stor lagringsplass”) as NON-EVIDENCE.
• At cs=0.7 this MUST NOT mint → use sev=1 ("cs0.7 baseline (no neg-wear)") unless A2 global proof exists.
• At cs=0.9, generic/spec NEVER demotes → default remains mint (sev=0) unless explicit wear/faults exist.

================================================================
C) SURFACE / TYPE MAPPING (device vs lens vs panel)
----------------------------------------------------------------
C1. Device glass crack (“knust/knus/knuss/sprukket/sprekk/crack/cracked”; normalize typos like “kunst skjerm”, “skrerm”):
    - front: reason "knust skjerm" → sev=3
    - back:  reason "knust bak"    → sev=3
    Set meta.glass=true; set meta.back_glass=true if back side is explicitly indicated.

C2. Lens glass crack (kameralinse/linseglass/kameraglass) → sev=2, reason "linseglass sprukket", meta.lens_glass=true.

C3. Panel severe (grønn/purple line, ghosting/innbrenning/burn-in, black blotch) → sev=3, reason "panel grønn linje"/"panel ghosting"/"panel black blotch", meta.panel_severe=true.

C4. Light panel (dead pixel/dot) → sev=2, reason "død piksel" (or "lett panelblemme" if needed), meta.light_panel=true.

================================================================
D) PROTECTOR-ONLY (priority rule)
----------------------------------------------------------------
Protector (skjermbeskytter/panserglass/beskyttelsesglass/glassfilm) explicitly damaged AND device glass is fine:
  meta.protector_only=true → sev=0, bin=0, reason "protector-only".
Do NOT set meta.glass or meta.lens_glass when protector-only.
Merely stating “brukt med skjermbeskytter/deksel” is NOT protector-only.

================================================================
E) FUNCTIONAL FAULTS (sev=2) — CANONICAL REASONS
----------------------------------------------------------------
E1. Charging (present now: lader ikke / må vri / dårlig kontakt / loose/wobbly) → sev=2, reason "ladeport kinkig", meta.charging=true.

E2. Modules/features → sev=2, pick ONE canonical reason:
    "face id virker ikke" / "kamera defekt" / "mik defekt" / "earpiece defekt" / "høyttaler defekt" / "sim-feil" / "modulfeil (nett)"
    Use "modulfeil (nett)" for network issues like “mobilnettet virker ikke / ingen nett” (not an operator lock).

================================================================
F) NON_OEM (repairs) — NOT insurance swap
----------------------------------------------------------------
Set meta.non_oem=true if text explicitly says a part was replaced/repaired (screen/battery/backglass/camera/port/etc.), even if “original Apple part”.
Insurance swap/bytte-enhet/erstatningsenhet ALONE is NOT non_oem.
Canonical reasons: "non-oem skjerm" / "non-oem batteri" / "non-oem bakglass" / "non-oem kamera" / "non-oem port".
Any non_oem=true ⇒ sev ≥ 2 (unless device glass/panel severe pushes to 3).

================================================================
G) BATTERY CLAMP (numeric only; STRICT)
----------------------------------------------------------------
meta.battery_clamp=true ONLY if battery_effective < 80. Then sev ≥ 2, reason MAY be "batt<80".
80..100 is NOT a clamp. Do NOT clamp at 80, 81..89. Ignore vague “good/normal battery”.
Never write reasons like “batt<85 / batt<87”.

================================================================
H) REASON (<=80 chars; canonical)
----------------------------------------------------------------
Mint:
  • "mint @ cs0.9"  • "mint @ cs0.7"  • "mint @ cs0.0"
Protector-only:
  • "protector-only"
Baselines:
  • "cs0.7 baseline (no neg-wear)" • "cs0.0 baseline (no neg-wear)" • "cs0.5 baseline"
Cosmetic (sev=1): short explicit snippet, e.g.:
  • "små riper i skjerm" • "normale bruksmerker i rammen" • "svake riper i skjermen"
Functional / faults (sev=2/3):
  • "ladeport kinkig" • "face id virker ikke" • "kamera defekt" • "mik defekt" • "earpiece defekt" • "høyttaler defekt" • "sim-feil" • "modulfeil (nett)"
  • "non-oem skjerm" / "non-oem batteri" / "non-oem bakglass" / "non-oem kamera" / "non-oem port"
  • "linseglass sprukket" • "død piksel" • "knust skjerm" • "knust bak" • "panel grønn linje" • "panel ghosting" • "panel black blotch"
cs=0.5 cosmetic/fault reasons may end with " + cs0.5" if helpful.

At sev=1 NEVER include mint/no-wear words (mint/uten riper/ripefri/ingen skader/strøken/som ny).

================================================================
I) LOCK (carrier/SIM)
----------------------------------------------------------------
"locked" only when text explicitly states SIM/carrier lock (“låst til [operator]”, sim-locked/carrier locked).
"unlocked" only when explicitly "ulåst / factory unlocked / open for all operators".
Else "null". Lock is orthogonal to severity — never affects sev/bin/reason.

================================================================
J) BIN
----------------------------------------------------------------
bin = 0 IFF sev = 0; otherwise 1.

================================================================
K) BORDERLINE HANDLING (owner preferences)
----------------------------------------------------------------
K1. cs=0.9 de-minimis cosmetic (LENIENT): tiny hedged wear like “minimalt med bruksmerker”, “svært få/veldig lite synlige bruksmerker”, “knapt synlig” with no surface named
    → MAY still mint sev=0 at cs=0.9 (if no faults).
K2. cs=0.7 contradiction leniency: “global no-wear” + trivial hedged bruksmerker and NO faults → mint sev=0 is acceptable.

================================================================
L) NORMALIZATION (typos/aliases)
----------------------------------------------------------------
• Crack → device glass: knust/knus/knuss/sprukket/sprekk/crack/cracked; typos like “kunst skjerm”, “skrerm” ⇒ device glass.
• Lens cover: kameralinse/linseglass/kameraglass (sprukket/knust/hakk) ⇒ lens_glass.
• Protector terms: skjermbeskytter/panserglass/beskyttelsesglass/glassfilm.
• Hedged: tilnærmet/nesten/så godt som/omtrent ripefri ⇒ NOT global.
• Wear tokens to recognize as cosmetic: riper/små riper/mikroriper/overflateriper/merker/hakk/skrammer/bulker/bruksspor/bruksmerker (+ locations).
• Charging: ladeport kinkig/må vri/dårlig kontakt/lader ikke (present now).
• eSIM-only (USA) is capability, not damage.

================================================================
M) POST-DECISION QA TRAPS (AUTO-CORRECT DRIFT)
----------------------------------------------------------------
Apply these BEFORE returning JSON:

M1. cs=0.9 but sev=1 AND no faults AND no explicit wear → set sev=0, bin=0, reason "mint @ cs0.9".

M2. cs in {0.7, 0.0} minted sev=0 BUT missing global proof (no_wear_global!=true OR no quoted phrase OR scope not global) OR any fault present
    → set sev=1, bin=1, reason "cs0.[7|0] baseline (no neg-wear)".

M3. protector_only=true AND (any other wear OR any fault) → cannot mint: sev = max(sev, (1 if only cosmetic else 2+)) and reason must NOT be "protector-only".

M4. Battery reason hygiene: If battery_effective ≥ 80, ensure meta.battery_clamp=false and reason does NOT mention battery clamp.

M5. Replacements: If text says a part was replaced, ensure meta.non_oem=true and sev ≥ 2 (even if “original Apple part”), unless it’s clearly a clean swap unit.

================================================================
N) OUTPUT COMPLIANCE CHECK (internal; do not print)
----------------------------------------------------------------
Before you output, SELF-CHECK (do not print this checklist):
[ ] JSON only; exact keys; types correct; reason ≤ 80 chars.
[ ] cs=0.9 → sev=0 unless explicit wear/faults.
[ ] cs=0.7/0.0 minted to 0 only with global no-wear phrase + no faults.
[ ] No fake battery clamp: clamp ONLY if battery_effective < 80.
[ ] Protector-only purity respected.
[ ] Canonical reason chosen (no mint words at sev=1).
[ ] lock set ONLY if explicitly stated; else "null".
[ ] bin = (sev==0 ? 0 : 1).

================================================================
FEW-SHOTS (targeted)
----------------------------------------------------------------
INPUT:
id:X
cs: 0.7
battery_effective: 90
title: iPhone 14 Pro
description:
Pent brukt, ingen synlige skader. Fungerer som den skal.
EXPECTED:
{"id":"X","bin":0,"sev":0,"reason":"mint @ cs0.7","lock":"null",
 "meta":{"no_wear_global":true,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 95
title: iPhone 15 Pro
description:
Apple iPhone 15 Pro mobiltelefon med 128 GB lagringskapasitet. Enheten har en svart farge og er designet for brukere som ønsker høy ytelse.
EXPECTED:
{"id":"X","bin":1,"sev":1,"reason":"cs0.7 baseline (no neg-wear)","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.9
battery_effective: 100
title: iPhone 15 Pro
description:
Minimalt med bruksmerker. Ellers strøken. (Ingen spesifikk flate nevnt.)
EXPECTED:
{"id":"X","bin":0,"sev":0,"reason":"mint @ cs0.9","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.9
battery_effective: 98
title: iPhone 16 Pro Max
description:
Mobilnettet virker ikke i Norge, fungerer i andre land.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"modulfeil (nett)","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 99
title: iPhone 14
description:
Skjermbeskytteren er sprukket; skjermen under strøken.
EXPECTED:
{"id":"X","bin":0,"sev":0,"reason":"protector-only","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":true,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 95
title: iPhone 14
description:
Må vri for å lade.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"ladeport kinkig","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":true,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.9
battery_effective: 86
title: iPhone 15 Pro
description:
Bare generisk/spec tekst, ingen slitasje omtalt.
EXPECTED:
{"id":"X","bin":0,"sev":0,"reason":"mint @ cs0.9","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 86
title: iPhone 15 Pro
description:
Litt merker i rammen (se bilder). Batteri 86 %.
EXPECTED:
{"id":"X","bin":1,"sev":1,"reason":"cs0.7 baseline (no neg-wear)","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 86
title: iPhone 14 Pro Max
description:
Ingen riper eller skader.
EXPECTED:
{"id":"X","bin":0,"sev":0,"reason":"mint @ cs0.7","lock":"null",
 "meta":{"no_wear_global":true,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.5
battery_effective: 95
title: iPhone 14
description:
Noen riper i skjerm. Ellers normal bruk.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"riper i skjerm + cs0.5","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}


INPUT:
id:X
cs: 0.7
battery_effective: 79
title: iPhone 13 Pro – Strøken, ingen skader
description:
Absolutt ingen riper eller skader. Batterikapasitet 79 %. Ellers alt fungerer.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"batt<80","lock":"null",
 "meta":{"no_wear_global":true,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":true}
}

INPUT:
id:X
cs: 0.7
battery_effective: 95
title: iPhone 14 Pro – Som ny, ingen riper
description:
Ingen skader. Må vri for å lade av og til, ellers fin.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"ladeport kinkig","lock":"null",
 "meta":{"no_wear_global":true,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":true,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.7
battery_effective: 96
title: iPhone 14 – Ripefri
description:
Ripefri. Liten sprekk i skjermen i hjørnet (se bilde).
EXPECTED:
{"id":"X","bin":1,"sev":3,"reason":"knust skjerm","lock":"null",
 "meta":{"no_wear_global":true,"protector_only":false,"glass":true,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":false,"battery_clamp":false}
}

INPUT:
id:X
cs: 0.9
battery_effective: 88
title: iPhone 15 Pro – Strøken
description:
Skjermen er byttet. Alt fungerer. Ingen riper.
EXPECTED:
{"id":"X","bin":1,"sev":2,"reason":"non-oem skjerm","lock":"null",
 "meta":{"no_wear_global":false,"protector_only":false,"glass":false,"back_glass":false,"lens_glass":false,"panel_severe":false,"light_panel":false,"charging":false,"non_oem":true,"battery_clamp":false}
}
"""

# Battery extraction (STRICT, anchored-only) — upgraded
BATTERY_SYSTEM_PROMPT = r"""
Goal: Return a numeric battery-health percentage ONLY when it is explicitly supported by seller text. Be conservative but DO NOT miss clear numeric anchors.

Return STRICT JSON ONLY:
{"override": true|false, "battery_pct": null|20..100, "reason":"<=80 chars"}

You may set override=true IFF one of these is present:
A) A number 20..100 followed by % / prosent / percent / pct (spaces allowed), e.g., "88 %", "88%", "80 prosent".
B) A number 20..100 that is directly attached to a BATTERY term within ~3 tokens, even WITHOUT a % sign. Accept typos and English/Norwegian forms:
   battery/batteri/bateri, batterihelse, batteritilstand, (maksimal|maks) kapasitet, capacity, health, maximum capacity.
   Examples that MUST be accepted: "Bateri 87", "Batterikapasitet 80", "battery at 69 of health", "batteri kapasitet på 80".
C) Comparators:
   - "< 80", "under 80", "below 80" near a BATTERY term ⇒ set battery_pct to (80-1)=79 (to reflect strictly under 80 without guessing higher).
   - ">= 80", "over 80" ⇒ keep override=false (not exact) unless a second explicit % appears.

Never infer 100 from adjectives/labels. Phrases like "Batteritilstand er på maksimal kapasitet" WITHOUT a number are boilerplate → not evidence.

Normalization:
- Round to int in [20..100]. Allowed forms: "88%", "88 %", "80 prosent".
- Accept decimal commas/dots by rounding: "87,5%" → 88.
- If a range like "80–85%" appears, choose the LOWER bound (80) conservatively.
- Ignore cycles, mAh/Wh, time/duration, qualitative words (god/normal/very good), prices, storage (GB), camera MP.

Self-check before output (do not print):
[ ] I can point to a nearby battery phrase AND an integer 20..100.
[ ] If output=100, I literally saw "100%" / "100 prosent".
[ ] If I used a comparator like "under 80", I returned 79.

Few-shots:
INPUT: "Batteritilstand er på maksimal kapasitet. Ingen riper."
EXPECTED: {"override": false, "battery_pct": null, "reason": "no numeric anchor"}

INPUT: "Maksimal kapasitet: 88 %."
EXPECTED: {"override": true, "battery_pct": 88, "reason": "explicit 88%"}

INPUT: "Bateri 87"
EXPECTED: {"override": true, "battery_pct": 87, "reason": "number near battery term"}

INPUT: "batteri kapasitet på 80"
EXPECTED: {"override": true, "battery_pct": 80, "reason": "number near battery term"}

INPUT: "battery at 69 of health"
EXPECTED: {"override": true, "battery_pct": 69, "reason": "english near battery"}

INPUT: "batterikapasitet under 80 prosent."
EXPECTED: {"override": true, "battery_pct": 79, "reason": "under 80% => 79"}
"""

PROMPT_HASH = hashlib.sha256(DECIDER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]

# ----------------------- OpenAI client -----------------------

_CLIENT: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required")
        if OPENAI_BASE_URL:
            _CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        else:
            _CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return _CLIENT

# ----------------------- DB helpers -----------------------

def get_engine() -> Engine:
    return create_engine(PG_DSN, future=True, pool_pre_ping=True)

def ensure_columns(engine: Engine) -> None:
    # No DDL here to avoid locks. Assume required columns (incl. damage_ai_json) already exist.
    return

def stamp_hard_rule_labels(engine: Engine) -> None:
    """
    Auto-labels cs=1.0 and cs=0.2 rows BEFORE any LLM work.
    This is idempotent-ish; feel free to run every time.
    """
    if DRY_RUN:
        logger.info("DRY_RUN: skipping hard-rule stamping")
        return
    with engine.begin() as conn:
        res1 = conn.execute(text(f"""
UPDATE {TABLE}
SET  damage_binary_ai   = 0,
     damage_severity_ai = 0,
     damage_reason_ai   = 'brand new (cs1.0)',
     damage_ai_version  = :ver,
     damage_ai_at       = now(),
     damage_ai_json     = NULL
WHERE condition_score = 1.0
"""), {"ver": RULE_VERSION})

        res2 = conn.execute(text(f"""
UPDATE {TABLE}
SET  damage_binary_ai   = 1,
     damage_severity_ai = 3,
     damage_reason_ai   = 'pre-determined sev3 (cs0.2)',
     damage_ai_version  = :ver,
     damage_ai_at       = now(),
     damage_ai_json     = NULL
WHERE condition_score = 0.2
"""), {"ver": RULE_VERSION})

    logger.info("Hard-rule stamp: cs=1.0 and cs=0.2 applied.")

def fetch_rows(engine: Engine) -> List[Dict[str, Any]]:
    # ALWAYS skip cs in (1.0, 0.2) so LLM never touches them.
    cond_filter = "AND (condition_score IS NULL OR condition_score NOT IN (1.0, 0.2))"
    skip_clause = "AND damage_ai_version IS NULL" if SKIP_ALREADY_LABELED else ""
    sql = f"""
WITH caps(gen, cap) AS (
  VALUES (13, :cap13), (14, :cap14), (15, :cap15), (16, :cap16), (17, :cap17)
),
base AS (
  SELECT
    {ID_COL} AS listing_id, generation, title, description, edited_date, condition_score,
    battery_pct, battery_pct_fixed, battery_pct_fixed_ai, operator_lock
  FROM {TABLE}
  WHERE spam IS NULL
    AND generation IN (13,14,15,16,17)
    {cond_filter}
    {skip_clause}
),
ranked AS (
  SELECT b.*, row_number() OVER (PARTITION BY generation ORDER BY edited_date DESC) rn
  FROM base b JOIN caps c ON c.gen=b.generation
)
SELECT
  listing_id, generation, title, description, edited_date, condition_score,
  battery_pct, battery_pct_fixed, battery_pct_fixed_ai, operator_lock
FROM ranked r
JOIN caps c ON c.gen=r.generation
WHERE r.rn <= c.cap
ORDER BY generation, edited_date DESC;
"""
    params = {
        "cap13": CAPS[13],
        "cap14": CAPS[14],
        "cap15": CAPS[15],
        "cap16": CAPS[16],
        "cap17": CAPS[17],
    }
    with engine.begin() as conn:
        return [dict(x) for x in conn.execute(text(sql), params).mappings()]

def update_damage(engine: Engine, listing_id: int, out: Dict[str, Any], llm_json: Optional[Dict[str, Any]] = None) -> None:
    sql = f"""
UPDATE {TABLE}
SET damage_binary_ai = :bin,
    damage_severity_ai = :sev,
    damage_reason_ai = :reason,
    damage_ai_version = :ver,
    damage_ai_at = :ts
    {", damage_ai_json = :llm_json" if llm_json is not None else ""}
WHERE {ID_COL} = :id
"""
    params = {
        "bin": int(out["bin"]),
        "sev": int(out["sev"]),
        "reason": out["reason"],
        "ver": VERSION_TAG,
        "ts": datetime.now(timezone.utc),
        "id": int(listing_id),
    }
    stmt = text(sql)
    if llm_json is not None:
        params["llm_json"] = llm_json
        stmt = stmt.bindparams(bindparam("llm_json", type_=JSONB))
    if DRY_RUN:
        return
    with engine.begin() as conn:
        conn.execute(stmt, params)

def persist_battery_fixed_ai(engine: Engine, listing_id: int, pct: int) -> None:
    if DRY_RUN:
        return
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {TABLE} SET battery_pct_fixed_ai = :v WHERE {ID_COL} = :id"),
            {"v": int(pct), "id": int(listing_id)},
        )

def persist_operator_lock_locked(engine: Engine, listing_id: int) -> None:
    if not LOCK_REPAIR or DRY_RUN:
        return
    with engine.begin() as conn:
        conn.execute(
            text(f"UPDATE {TABLE} SET operator_lock = 'locked' WHERE {ID_COL} = :id"),
            {"id": int(listing_id)},
        )

# ----------------------- Utilities -----------------------

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def coalesce_battery(row: Dict[str, Any]) -> Optional[float]:
    for k in ("battery_pct_fixed_ai", "battery_pct_fixed", "battery_pct"):
        v = _to_float(row.get(k))
        if v is not None and 20.0 <= v <= 100.0:
            return float(v)
    return None

def battery_suspicious(row: Dict[str, Any]) -> bool:
    raw = _to_float(row.get("battery_pct"))
    fix = _to_float(row.get("battery_pct_fixed"))
    vals = [v for v in (raw, fix) if v is not None]
    if not vals:
        return True
    for v in vals:
        if (1.0 <= v <= 70.0) or (v < 10.0):
            return True
    eff = coalesce_battery(row)
    return eff is None

# ----------------------- LLM DECIDER -----------------------

def _strip_code_fence(payload: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", payload, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else payload

def _coerce_decider_json(obj: Any, fallback_id: int) -> Dict[str, Any]:
    """Coerce model output into the exact schema; minimal defaults if missing."""
    if isinstance(obj, str):
        try:
            obj = json.loads(_strip_code_fence(obj))
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    data = {
        "id": str(obj.get("id", fallback_id)),
        "sev": int(obj.get("sev", 1)),
        "reason": str(obj.get("reason", "cs0.7 baseline (no neg-wear)"))[:80],
        "lock": obj.get("lock", "null") if obj.get("lock") in ("locked","unlocked","null") else "null",
        "meta": obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    }
    data["bin"] = 0 if data["sev"] == 0 else 1
    return data

def call_decider_llm(
    listing_id: int,
    cs: Optional[float],
    battery_effective: Optional[float],
    title: str,
    description: str
) -> Dict[str, Any]:
    client = get_client() if not NEW_CLIENT_PER_CALL else OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    user = (
        f"id: {listing_id}\n"
        f"cs: {'null' if cs is None else cs}\n"
        f"battery_effective: {'null' if battery_effective is None else int(round(battery_effective))}\n"
        f"title: {title or ''}\n"
        f"description:\n{(description or '').strip()}"
    )
    last_err = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp: ChatCompletion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": DECIDER_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            payload = resp.choices[0].message.content or ""
            return _coerce_decider_json(payload, fallback_id=listing_id)
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
    logger.error(f"decider failed for id={listing_id}: {last_err}; fallback to baseline sev=1")
    return {"id": str(listing_id), "bin": 1, "sev": 1, "reason": "cs0.7 baseline (no neg-wear)", "lock":"null", "meta": {}}

def battery_extract_llm(row: Dict[str, Any]) -> Optional[int]:
    """
    STRICT, anchored-only extractor. Uses ONLY title+description (no DB battery fields)
    to avoid biasing the model. Returns int in [20..100] or None.
    """
    client = get_client() if not NEW_CLIENT_PER_CALL else OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    title = row.get("title") or ""
    desc = (row.get("description") or "").strip()
    user = f"title: {title}\n" f"description:\n{desc}"
    last_err = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp: ChatCompletion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": BATTERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
            )
            payload = resp.choices[0].message.content or ""
            try:
                data = json.loads(_strip_code_fence(payload))
            except Exception:
                last_err = "not json"
                raise ValueError(last_err)
            override = bool(data.get("override", False))
            pct = data.get("battery_pct", None)
            if not override or pct is None:
                return None
            val = int(round(float(pct)))
            if 20 <= val <= 100:
                return val
            last_err = f"bad pct {pct}"
        except Exception as e:
            last_err = str(e)
        time.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
        logger.warning(f"battery_extract_llm failed: {last_err}")
    return None

# ----------------------- Public API -----------------------

@dataclass
class FinalDecision:
    bin: int
    sev: int
    reason: str
    needs_review: bool

def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
    fid = int(row["listing_id"])
    title = row.get("title") or ""
    description = row.get("description") or ""

    # cs
    try:
        cs: Optional[float] = float(row.get("condition_score")) if row.get("condition_score") is not None else None
    except Exception:
        cs = None

    # HARD SKIP guard (belt-and-suspenders)
    if cs in (1.0, 0.2):
        logger.info(f"{fid} SKIP (policy): condition_score={cs}")
        return {"bin": 0, "sev": 0, "reason": "skipped (policy)"}, {"id": str(fid), "bin": 0, "sev": 0, "reason": "skipped (policy)", "meta": {}}

    # battery (optional repair)
    battery_eff = coalesce_battery(row)
    if BATTERY_REPAIR and battery_suspicious(row):
        extracted = battery_extract_llm(row)  # STRICT extractor
        battery_eff = float(extracted) if extracted is not None else battery_eff

    # LLM makes final decision
    dec = call_decider_llm(fid, cs, battery_eff, title, description)
    logger.info(f"FINAL id={fid} bin={dec['bin']} sev={dec['sev']} reason={dec['reason']}")

    return {"bin": dec["bin"], "sev": dec["sev"], "reason": dec["reason"]}, dec

# ----------------------- Worker pipeline -----------------------

def _run_golden_tests_smoke() -> None:
    """Optional smoke tests for common edge cases. Only runs if RUN_GOLDENS=1."""
    samples = [
        # device crack vs negation
        {"cs":0.7,"title":"iPhone 13","desc":"noen sprek på forskjerm","expect_sev":3,"reason_has":"knust"},
        {"cs":0.7,"title":"iPhone 13","desc":"ikke knust glass. noen bruksmerker","expect_sev":1,"reason_has":"baseline"},
        # cs0.7 mint requires global phrase
        {"cs":0.7,"title":"iPhone 14","desc":"Strøken. Brukt med deksel.","expect_sev":1,"reason_has":"baseline"},
        {"cs":0.7,"title":"iPhone 14","desc":"Ingen riper eller skader. Brukt med deksel.","expect_sev":0,"reason_has":"mint @ cs0.7"},
        # protector-only strict
        {"cs":0.9,"title":"iPhone 14","desc":"Sprukket skjermbeskytter, skjermen under strøken.","expect_sev":0,"reason_has":"protector"},
        # non-oem examples
        {"cs":0.9,"title":"iPhone 16 Pro","desc":"Nytt glass i front. Alt annet strøkent.","expect_sev":2,"reason_has":"non-oem"},
        # panel
        {"cs":0.7,"title":"iPhone X","desc":"Grønn linje midt på skjermen.","expect_sev":3,"reason_has":"panel"},
        {"cs":0.7,"title":"iPhone X","desc":"En død piksel midt på skjermen.","expect_sev":2,"reason_has":"død piksel"},
        # charging
        {"cs":0.7,"title":"iPhone 13","desc":"Må vri for å lade, ellers fin.","expect_sev":2,"reason_has":"ladeport"},
        # battery <80
        {"cs":0.9,"title":"iPhone 13","desc":"Batterihelse 79%.","expect_sev":2,"reason_has":"batt<"},
    ]
    for i, s in enumerate(samples, 1):
        fake_row = {
            "listing_id": 990000+i, "title": s["title"], "description": s["desc"], "condition_score": s["cs"],
            "battery_pct": None, "battery_pct_fixed": None, "battery_pct_fixed_ai": None
        }
        out, dec = score_row(fake_row)
        if out["sev"] != s["expect_sev"] or s["reason_has"].lower() not in out["reason"].lower():
            raise AssertionError(f"GOLDEN FAIL case#{i}: got sev={out['sev']}, reason={out['reason']}")

def process_one(engine: Engine, row: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    fid = int(row["listing_id"])
    try:
        # Guard against policy rows (even if they slip in — they shouldn't)
        try:
            cs: Optional[float] = float(row.get("condition_score")) if row.get("condition_score") is not None else None
        except Exception:
            cs = None
        if cs in (1.0, 0.2):
            logger.info(f"{fid} SKIP (policy): condition_score={cs}")
            return fid, None, None

        # battery repair path
        battery_eff = coalesce_battery(row)
        if BATTERY_REPAIR and battery_suspicious(row):
            extracted = battery_extract_llm(row)
            if extracted is not None:
                persist_battery_fixed_ai(engine, fid, extracted)
                battery_eff = float(extracted)
                logger.info(f"{fid} battery_pct_fixed_ai -> {extracted}")

        title = row.get("title") or ""
        description = row.get("description") or ""

        # Decide in LLM
        dec = call_decider_llm(fid, cs, battery_eff, title, description)

        if dec.get("lock") == "locked" and LOCK_REPAIR:
            persist_operator_lock_locked(engine, fid)

        # Persist
        out = {"bin": dec["bin"], "sev": dec["sev"], "reason": dec["reason"]}
        meta_blob = {
            "model": MODEL_NAME,
            "version": VERSION_TAG,
            "prompt_hash": PROMPT_HASH,
            "cs": cs,
            "battery_effective": None if battery_eff is None else int(round(battery_eff)),
            "decision": dec
        }
        update_damage(engine, fid, out, llm_json=meta_blob)

        logger.info(f"{fid}\tbin={out['bin']}\tsev={out['sev']}\treason={out['reason']}")
        return fid, out, None

    except Exception as e:
        try:
            update_damage(engine, fid, {"bin": 1, "sev": 3, "reason": f"error:{e.__class__.__name__}"[:80]})
        except Exception:
            pass
        logger.error(f"{fid} ERROR: {e}")
        return fid, None, str(e)

# ----------------------- CLI -----------------------

def main() -> None:
    logger.info(f"Model: {MODEL_NAME} | Workers: {WORKERS} | DRY_RUN={DRY_RUN} | VERSION_TAG={VERSION_TAG}")
    engine = get_engine()
    ensure_columns(engine)

    # 1) Stamp hard rules (cs 1.0 / 0.2) so those rows are finalized and never sent to LLM
    stamp_hard_rule_labels(engine)

    # 2) Optional golden-smoke tests
    if RUN_GOLDENS:
        try:
            _run_golden_tests_smoke()
            logger.info("Golden tests: PASS")
        except Exception as ge:
            logger.error(f"Golden tests: FAIL: {ge}")
            sys.exit(2)

    # 3) Fetch rows (ALWAYS excludes cs 1.0 / 0.2)
    rows = fetch_rows(engine)
    logger.info(f"Queued rows: {len(rows)} | Caps: {CAPS}")

    t0 = time.time()
    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process_one, engine, row) for row in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="row"):
            _fid, out, err = fut.result()
            if err is None and out is not None:
                ok += 1
            else:
                fail += 1

    dt = time.time() - t0
    rps = (len(rows) / dt) if dt > 0 else 0.0
    logger.info(f"Done. ok={ok} fail={fail} | {dt:.2f}s total ~ {rps:.2f} rows/s")

if __name__ == "__main__":
    main()





























