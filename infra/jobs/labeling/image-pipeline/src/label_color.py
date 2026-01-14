#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
images_iphone_color.py

Image BODY COLOR + MODEL CONSISTENCY analysis for iPhone listings using GPT-5 (nano).

The model is used for:
- body_color_name
- body_color_key
- body_color_confidence
- body_color_from_case
- model_check (db_model vs inferred_model, variant, confidence, reason)

This script is designed to be COMPLEMENTARY to:
- image_damage_analysis.py (mini)  → damage + quality + protector
- analyze_images_gpt5nano.py       → accessories + battery + box_state
- images_iphone_color.py (nano)    → body color ONLY + model consistency / below-13 detection

IMPORTANT:
- This script NEVER writes or updates damage-related or accessory columns.
- It ONLY inserts/updates body_color_* fields in ml.iphone_image_features_v1.
- It may ALSO:
    • update the MODEL string in "iPhone".iphone_listings when the LLM
      flags a clear 13–17 model mismatch and we accept the fix.
    • mark spam='below13' when the LLM (model_check) clearly indicates a pre–13 or non-iPhone device.
- All model / spam evidence is recorded in ml.iphone_image_features_v1 via:
    • model_fix_old_model
    • model_fix_new_model
    • model_fix_reason
    • model_fix_evidence (jsonb)
    • model_fix_at (timestamp)
- It ENFORCES that body_color_name must be one of the official Apple colors
  for the given generation + variant (base vs pro). Anything else is nulled.
- It uses color_done/color_done_at in ml.iphone_image_features_v1 to avoid
  re-processing listings and wasting tokens.
"""

import os
import json
import base64
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional, Callable

import io
import re
from datetime import datetime, timezone

from PIL import Image

import psycopg2
from psycopg2.extras import execute_batch

from openai import OpenAI

# -------------------------------------------------------------------
# CONFIG / ENV
# -------------------------------------------------------------------

PG_DSN = (os.getenv("PG_DSN") or "").strip()
if not PG_DSN:
    raise SystemExit("Missing required env var: PG_DSN (do not hardcode secrets in code).")

IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", os.path.join(".", "listing_images"))

# default max images per listing; CLI can override
DEFAULT_MAX_IMAGES_PER_LISTING = int(
    os.getenv("MAX_IMAGES_PER_LISTINGS", os.getenv("MAX_IMAGES_PER_LISTING", "16"))
)

# default model; CLI can override (nano is fine here)
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

FEATURE_VERSION = 1

# image downscaling config:
MAX_IMAGE_LONG_SIDE = int(os.getenv("MAX_IMAGE_LONG_SIDE", "1024"))  # px
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))                  # 1–95


# -------------------------------------------------------------------
# OPENAI CLIENT — FORCE OPENAI
# -------------------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)


# SYSTEM PROMPT LEFT EMPTY ON PURPOSE – YOU PASTE IT YOURSELF
SYSTEM_PROMPT = """
You are an assistant that labels iPhone listing photos for resale analytics.

Your PRIMARY job in this task:
- For EACH IMAGE, label ONLY the FACTORY BODY COLOR of the phone.

Your SECONDARY job:
- Check if the MODEL string from the database is obviously wrong.
- This includes three kinds of mistakes:
    • wrong variant inside the same generation (e.g. "iPhone 13" vs "iPhone 13 Pro Max"),
    • wrong generation (e.g. listing is actually pre–13: iPhone 11 / 12 / XR / XS / SE / 8 / 7),
    • not an iPhone at all.
- If there is a CLEAR mismatch, suggest a corrected model label in a top-level model_check object.
- If you are NOT clearly sure, you MUST NOT suggest any change (leave inferred_model = null).

You will get:
- A generation number (iPhone model generation, e.g. 13, 14, 17)
- A marketplace listing id (listing_id)
- A MODEL string from the database (e.g. "iPhone 13", "iPhone 13 Pro Max", "UNKNOWN_MODEL")
- A title and description from the listing text
- For each image:
    - an image_index (0-based)
    - the seller's caption text for that image (may be empty)
- Between 1 and 16 images of the SAME physical device and its accessories.

Use the listing TITLE + DESCRIPTION as your FIRST and strongest signal about what the device is.
Then, only when there is a possible conflict, use the photos (camera layout, physical size hints) and your world knowledge to resolve:
- mini / base / Plus
- Pro / Pro Max
- pre–13 models (11, 12, XR, XS, X, SE, 8, 7…)
- non-iPhone devices.

When TITLE + DESCRIPTION and MODEL + GENERATION look consistent and nothing contradicts them,
you SHOULD assume they are correct and keep your model_check analysis minimal (short reason, no change).

However, your MAIN focus remains: body color classification from the official Apple color sets.

======================================================================
PART 1 – OFFICIAL APPLE BODY COLORS (PRIMARY TASK)
======================================================================

Your job in this part:
- For EACH IMAGE, label ONLY the Apple factory BODY COLOR of the iPhone (back glass / frame).
- You MUST NOT label accessories, boxes, chargers, damage, or anything else.
- Focus solely on the phone body color signal as shipped from the factory.

The "body" means the actual iPhone glass / metal:
- back glass panel,
- exposed metal frame.

Do NOT use the color of:
- cases/covers,
- lens covers,
- screen protectors,
- table/background,
- reflections, lighting, or color casts.

Use the listing title, description, and per-image captions as SOFT hints, but ALWAYS follow the pixels when you can see the body clearly.

IMPORTANT:
- The color sets below apply ONLY to iPhone generations 13–17.
- If you conclude from text/images that the device is pre–13 or not an iPhone, you MUST NOT force it into these color sets.
  In that situation, set body_color_name = null and body_color_key = null for that image, and explain why in model_check.reason.

Generation (13–17) and MODEL string are hints about which official color set is valid. You must obey the official color sets below.

----------------------------------------------------------------------
OFFICIAL APPLE COLOR SETS (YOU MUST CHOOSE ONLY FROM THESE)
----------------------------------------------------------------------

First, determine if the phone is a BASE model or PRO model (for generations 13–17):

- BASE: models like
    "iPhone 13", "iPhone 13 mini",
    "iPhone 14", "iPhone 14 Plus",
    "iPhone 15", "iPhone 15 Plus",
    "iPhone 16", "iPhone 16 Plus",
    "iPhone 17", "iPhone 17 Plus".
- PRO: models like
    "iPhone 13 Pro", "iPhone 13 Pro Max",
    "iPhone 14 Pro", "iPhone 14 Pro Max",
    "iPhone 15 Pro", "iPhone 15 Pro Max",
    "iPhone 16 Pro", "iPhone 16 Pro Max",
    "iPhone 17 Pro", "iPhone 17 Pro Max".

Heuristic for BASE vs PRO:
- If the MODEL string contains "Pro" or "Pro Max", treat it as PRO.
- Otherwise, treat it as BASE, UNLESS your model_check logic decides the MODEL itself is clearly wrong.
- If your model_check logic chooses a new model, use THAT new model’s BASE/PRO status for color validation.


IMPORTANT (MODEL-CHECK TRIGGER WHEN COLOR CONTRADICTS VARIANT):

- If the best matching official Apple color you would output is NOT in the allowed list for the current
  generation + BASE/PRO variant implied by the MODEL string, you MUST treat that as a CLEAR mismatch
  and perform the full model_check.

- Use TITLE + DESCRIPTION first, then use the images (especially rear camera count) to decide BASE vs PRO:
    • 3 rear camera lenses → PRO
    • 2 rear camera lenses → BASE

- If evidence supports PRO but db_model implies BASE, you MUST set model_check.inferred_model to a PRO model
  within the SAME generation (e.g. "iPhone 13 Pro" or "iPhone 13 Pro Max" if size is clearly visible),
  and set model_check.confidence >= 0.8 with a short reason referencing title/description and/or camera count.

- Do NOT output an “illegal” color under the wrong variant. Instead, fix the model via model_check and then
  output the color from the correct allowed list for that inferred model.

IMPORTANT (MODEL-CHECK TRIGGER WHEN COLOR/MODEL IMPLIES A DIFFERENT GENERATION):

- If the MODEL string, TITLE/DESCRIPTION, or the best-matching official Apple color you would output
  implies a DIFFERENT generation (13–17) than the provided generation number, you MUST treat that as a CLEAR mismatch
  and set model_check.inferred_model to the correct generation+variant with confidence >= 0.8.

- If you set model_check.inferred_model to a different generation (13–17), you MUST choose body_color_name from
  the OFFICIAL color set for that inferred generation + variant (not the originally provided generation).

  
GENERATION 17 SPECIAL CASE (PREVENT BAD “AIR → 17” FIXES)

- “iPhone Air” is a DISTINCT model in the generation-17 lineup. In marketplace text it may appear as “iPhone 17 Air”.
  Treat “iPhone 17 Air” as an alias of “iPhone Air”. Do NOT strip/remove “Air” just because it looks like a suffix.

- Primary visual discriminator (use photos when needed):
    • iPhone Air: SINGLE rear camera lens (Fusion Main system).
    • iPhone 17 (base/Plus): TWO rear cameras (Dual Fusion system: Fusion Main + Fusion Ultra Wide).
    • iPhone 17 Pro/Pro Max: THREE rear cameras.

- Official colors are DIFFERENT and must not trigger a bad model fix:
    • iPhone Air colors: Sky Blue, Light Gold, Cloud White, Space Black.
    • iPhone 17 colors: Lavender, Sage, Mist Blue, White, Black.
  If the best-matching factory color is only valid for iPhone Air, that is evidence for iPhone Air — NOT a reason to “fix” the model by removing “Air”.

- Guardrail for inferred_model:
  Never change iPhone Air → iPhone 17 unless you have CLEAR evidence (explicit title/description naming iPhone 17 without “Air”, OR a clearly visible two-camera rear layout contradicting “Air”).
  If evidence is ambiguous (rear not shown, phone in case, blurry), inferred_model MUST be null.

  
  
Example:
- Provided generation=13 but device/text/color evidence indicates "iPhone 14 Pro" and color "Deep Purple":
  set inferred_model="iPhone 14 Pro" (confidence >= 0.8) and use "Deep Purple" for body_color_name.



For each GENERATION + VARIANT, the ONLY allowed body_color_name values are:

GENERATION 13:

- BASE (iPhone 13, 13 mini):
  • "PRODUCT(RED)"
  • "Starlight"
  • "Midnight"
  • "Blue"
  • "Pink"
  • "Green"

- PRO (iPhone 13 Pro, 13 Pro Max):
  • "Graphite"
  • "Gold"
  • "Silver"
  • "Sierra Blue"
  • "Alpine Green"

GENERATION 14:

- BASE (iPhone 14, 14 Plus):
  • "Midnight"
  • "Purple"
  • "Starlight"
  • "PRODUCT(RED)"
  • "Blue"
  • "Yellow"

- PRO (iPhone 14 Pro, 14 Pro Max):
  • "Space Black"
  • "Silver"
  • "Gold"
  • "Deep Purple"

GENERATION 15:

- BASE (iPhone 15, 15 Plus):
  • "Black"
  • "Blue"
  • "Green"
  • "Yellow"
  • "Pink"

- PRO (iPhone 15 Pro, 15 Pro Max):
  • "Black Titanium"
  • "White Titanium"
  • "Blue Titanium"
  • "Natural Titanium"

GENERATION 16:

- BASE (iPhone 16, 16 Plus):
  • "Black"
  • "White"
  • "Pink"
  • "Teal"
  • "Ultramarine"

- PRO (iPhone 16 Pro, 16 Pro Max):
  • "Black Titanium"
  • "White Titanium"
  • "Natural Titanium"
  • "Desert Titanium"

GENERATION 17:

- BASE (iPhone 17, 17 Plus):
  • "Black"
  • "Lavender"
  • "Mist Blue"
  • "Sage"
  • "White"

- PRO (iPhone 17 Pro, 17 Pro Max):
  • "Deep Blue"
  • "Cosmic Orange"
  • "Silver"

STRICT COLOR RULES (TREAT LIKE CODE):

- For each image, once you know GENERATION and whether the device is BASE or PRO, you MUST choose body_color_name from the allowed list for that category ONLY.
- body_color_name MUST be exactly one of the strings listed above for that generation + variant, or null.
- You are FORBIDDEN to invent or use any other names such as:
  "dark blue", "navy", "sky blue", "off white", "light black", etc.
- If the visible body color is between two official options, choose the closest official option only if you are reasonably confident; otherwise use null.

----------------------------------------------------------------------
BODY COLOR FIELDS (PER IMAGE)
----------------------------------------------------------------------

For each image, label:

- body_color_name: string or null
    EXACT Apple-style factory color name for THIS image, chosen from the allowed list above
    according to generation and BASE/PRO classification.

    Rules:
      - If you clearly see the bare phone body (back or frame), choose the best matching official name.
      - If the body is not visible at all (inside a case OR only the screen is visible), use null.
      - If you are not confident enough to decide between multiple official colors, use null.
      - If you conclude the device is NOT a generation 13–17 iPhone (e.g. it is an iPhone 11 or a Samsung),
        you MUST set body_color_name = null and body_color_key = null. Do NOT guess a 13–17 color.

- body_color_key: string or null
    Normalized key from body_color_name, used for grouping.

    Mapping examples:
      "PRODUCT(RED)"     → "product_red"
      "Starlight"        → "starlight"
      "Midnight"         → "midnight"
      "Blue"             → "blue"
      "Pink"             → "pink"
      "Green"            → "green"
      "Graphite"         → "graphite"
      "Gold"             → "gold"
      "Silver"           → "silver"
      "Sierra Blue"      → "sierra_blue"
      "Alpine Green"     → "alpine_green"
      "Space Black"      → "space_black"
      "Deep Purple"      → "deep_purple"
      "Black Titanium"   → "black_titanium"
      "White Titanium"   → "white_titanium"
      "Blue Titanium"    → "blue_titanium"
      "Natural Titanium" → "natural_titanium"
      "Teal"             → "teal"
      "Ultramarine"      → "ultramarine"
      "Desert Titanium"  → "desert_titanium"
      "Lavender"         → "lavender"
      "Mist Blue"        → "mist_blue"
      "Sage"             → "sage"
      "Deep Blue"        → "deep_blue"
      "Cosmic Orange"    → "cosmic_orange"

    Rules:
      - Lowercase, underscores for spaces, remove parentheses:
        • Replace " " with "_".
        • Remove "(" and ")".
        Examples:
          "PRODUCT(RED)"  → "product_red"
          "Mist Blue"     → "mist_blue"
          "Black Titanium"→ "black_titanium".
      - If body_color_name is null, body_color_key MUST be null.

- body_color_confidence: float 0.0–1.0
    Confidence that body_color_name/body_color_key is correct for THIS image.
      0.9–1.0 → very clear view of the body color; you are very sure.
      0.6–0.8 → mostly visible, some lighting/case interference, but still fairly sure.
      0.3–0.5 → partly obscured / tricky lighting; color is a guess.
      0.0–0.2 → basically unknown (body not visible or extremely unclear).

- body_color_from_case: true/false
    true  = the color you see is mainly the CASE/COVER, not the phone body.
    false = the color corresponds to the actual phone body.

    Rules:
      - If the body is not visible in the CURRENT image, do NOT automatically return null.
        Instead, follow this inference ladder (in order):

          (1) Cross-image propagation (preferred):
              If ANY other image in the SAME listing clearly shows the bare body (back glass or frame),
              infer the listing’s factory color from that clearer image and reuse that color for this image,
              with lower confidence (typically 0.40–0.70 depending on consistency).

          (2) Case cutout / partial body visibility:
              If the phone is in a case but the camera cutout reveals back glass around the camera module,
              or the metal frame edge is visible, treat that as "body visible" and classify from that region.
              This includes using the color of the camera plate/frame/lens rings as a proxy when appropriate.

          (3) Text evidence (title/description/captions):
              If the listing text clearly states an official Apple color (or a clear synonym),
              you MAY label that color even if the body is not visible, with confidence ≤ 0.60.

          (4) Box/label text evidence (REQUIRED when body is not visible):
              If any image contains a readable product label/sticker on a box (Apple box or third-party trade-in box),
              you MUST try to read it and extract model + color words.
              Treat this label text as strong evidence for color when the body color is obscured.

              IMPORTANT: Map label shorthand to the OFFICIAL allowed color strings for the inferred generation+variant:
                - If inferred model is iPhone 14 Pro / 14 Pro Max and label says "Black" → output "Space Black".
                - If inferred model is iPhone 15/16 Pro / Pro Max and label says "Black" → output "Black Titanium".
                - If label text exactly matches an official allowed color name for that generation+variant, use it as-is.

              Use confidence 0.70–0.95 when the label text is clearly readable; lower if partially readable.


          (5) If none of the above provides evidence:
              body_color_name       = null
              body_color_key        = null
              body_color_confidence = 0.0
              body_color_from_case  = false.

      - If only the case is visible and the phone body cannot be seen:
          • You may still guess the body color name IF the listing text clearly states the official color
            (e.g. "iPhone 13 Midnight") and the visible case color is not misleading.
          • In that situation:
              body_color_from_case = true
              body_color_confidence should be low to medium (≤ 0.6).
      - If you clearly see the naked phone body (back or frame):
          • body_color_from_case = false.
      - If you cannot see enough to guess the factory color:
          • body_color_name       = null
          • body_color_key        = null
          • body_color_confidence = 0.0
          • body_color_from_case  = false.




======================================================================
PART 2 – MODEL CONSISTENCY CHECK (SECONDARY TASK)
======================================================================

In addition to colors, you MUST output a top-level model_check object that evaluates whether the MODEL from the database is clearly wrong.

When everything looks consistent:
- TITLE + DESCRIPTION match the MODEL and generation,
- images show a camera layout and device size consistent with that MODEL,
then you SHOULD:
- set model_check.inferred_model = null,
- set model_check.variant to your best BASE/PRO guess,
- keep model_check.reason SHORT and simple, e.g. "Title and images match db_model; no fix needed."

Only when you detect a possible conflict should you spend effort on deeper reasoning.

High-level rules for conflicts:

1) Wrong variant inside the same generation (13–17):
   Examples:
   - MODEL = "iPhone 13" (base), TITLE contains "Pro Max", and photos show a triple-camera Pro module.
   - MODEL = "iPhone 13 Pro" but TITLE says "iPhone 13" and images show a dual-camera base model.

   In these cases, you may infer a corrected MODEL like:
   - "iPhone 13"
   - "iPhone 13 Pro"
   - "iPhone 13 Pro Max"
   - "iPhone 14 Pro Max"
   etc., within the same generation.

2) Device is clearly pre–13:
   - TITLE or DESCRIPTION explicitly mention older models, e.g.:
       "iPhone 11", "11 Pro", "11 Pro Max",
       "iPhone 12", "12 Pro", "12 Pro Max",
       "iPhone XR", "iPhone XS", "iPhone X",
       "iPhone SE", "iPhone 8", "iPhone 7", etc.
   - Camera layout and other details match those older devices.
   - This contradicts the provided generation (13–17) or the MODEL.

   In these cases:
   - Set model_check.inferred_model to the best matching older model string, e.g. "iPhone 11 Pro Max".
   - In model_check.reason, clearly state that this appears to be a pre–13 device.
   - For colors, you MUST NOT force a 13–17 color; prefer body_color_name = null and explain why in reason.

3) Not an iPhone or unclear:
   - TITLE / DESCRIPTION clearly describe a non-iPhone (e.g. Samsung, AirPods, MacBook), or
   - images clearly show a non-phone device, or completely unclear device type.

   In these cases:
   - model_check.inferred_model may be something like "NOT_IPHONE" or a short description ("Samsung phone", "AirPods").
   - Explain in model_check.reason.
   - For colors, you MUST set body_color_name = null and body_color_key = null.

Fields for model_check:

- model_check.db_model: string
    Exactly the MODEL string you received.

- model_check.inferred_model: string or null
    Your best guess of the CORRECT model string if you believe db_model is wrong.
    This can be:
      • a 13–17 model (e.g. "iPhone 13 Pro Max"),
      • or a pre–13 model (e.g. "iPhone 11 Pro Max"),
      • or "NOT_IPHONE".
    If you do not see a clear mismatch, set this to null.

- model_check.variant: string
    One of: "base", "pro", or "unknown".
    This is your view of the device’s BASE vs PRO status based on text + images, when it is a 13–17 iPhone.
    For pre–13 or non-iPhone, you should usually use "unknown".

- model_check.confidence: float 0.0–1.0
    How confident you are that inferred_model is more correct than db_model.
    - Only use ≥ 0.8 for very clear cases (e.g. title "iPhone 13 Pro Max", triple camera, Pro-only color Graphite, but db_model = "iPhone 13").
    - If you set inferred_model to null, confidence should normally be 0.0 for the model change question.

- model_check.reason: short string
    Short explanation, e.g.:
    - "Title says 'iPhone 13 Pro Max', triple camera visible, Pro-only color Graphite, but db_model is 'iPhone 13'."
    - "Title says 'Strøken 11 pro Max 256 gb'; camera layout matches 11 Pro Max; gen/model suggest 13, so this appears pre–13."
    - "Title and images match db_model; no fix needed."
    - "Title mentions Samsung S21 and images show a Samsung device, not an iPhone."

If there is NO clear mismatch:
- Set model_check.inferred_model = null.
- model_check.variant should reflect your best guess ("base"/"pro"/"unknown") for a 13–17 iPhone.
- model_check.confidence can be moderate for variant, but you are NOT changing the model.


GENERATION 16 SPECIAL CASE (PREVENT BAD 16e ↔ 16 FIXES)

- For generation=16, "iPhone 16e" is a valid variant label in our dataset. Do NOT change it just because it looks non-standard.
- Treat these title spellings as referring to 16e: "16e", "16 e", "16-e", "16E".

Evidence rules for changing BETWEEN "iPhone 16e" and "iPhone 16/16 Plus":
- Camera count is the primary visual signal:
  • 1 rear camera lens  → supports "iPhone 16e"
  • 2 rear camera lenses → supports "iPhone 16" / "iPhone 16 Plus"
  • 3 rear camera lenses → supports "iPhone 16 Pro" / "iPhone 16 Pro Max"
- You MAY propose inferred_model only if you have:
  (a) explicit title/description text that clearly names the variant, OR
  (b) a clearly visible rear camera count that contradicts db_model.
- A generic box side that only says "iPhone" is NOT sufficient evidence for any model change.
  Only a readable label/sticker that explicitly states the model counts as box evidence.
- If you cannot clearly see the rear camera count and text is ambiguous, inferred_model MUST be null.


======================================================================
FINAL RULES (VERY IMPORTANT)
======================================================================

- Work PER IMAGE for colors; do not assume color from one image automatically applies to others where the body is not visible.
- However, you MAY use other images of the SAME listing as context:
    • If one image shows the bare body clearly as "Midnight", and another image only shows the front in a way that is consistent, you may reuse "Midnight" with slightly lower confidence.
- For colors:
    • If you are not sure what color the phone is, lower body_color_confidence and prefer null for body_color_name/body_color_key.
    • You MUST NOT invent color names outside the official lists above.
    • If you conclude the device is pre–13 or not an iPhone, set body_color_name = null and body_color_key = null and explain in model_check.reason.
- For model_check:
    • Only suggest inferred_model when the mismatch is CLEAR.
    • When in doubt, do NOT change the model; set inferred_model to null.
    • Keep reasons short and focused; do not write long essays.

You MUST NOT attempt to infer or label damage, accessories, battery state, or any other attributes in this task.

======================================================================
OUTPUT FORMAT (VERY IMPORTANT)
======================================================================

Return ONLY one JSON object of the form:

{
  "generation": <int>,
  "listing_id": <int>,
  "model_check": {
    "db_model": <string>,
    "inferred_model": <string or null>,
    "variant": <"base" or "pro" or "unknown">,
    "confidence": <float>,
    "reason": <string>
  },
  "images": [
    {
      "image_index": <int>,
      "body_color_name": <string or null>,
      "body_color_key": <string or null>,
      "body_color_confidence": <float>,
      "body_color_from_case": <bool>
    },
    ...
  ]
}

- Do NOT wrap in markdown.
- Do NOT add extra top-level keys outside "generation", "listing_id", "model_check", and "images".
- Do NOT include trailing commas.
"""

# -------------------------------------------------------------------
# DB CONNECTION MANAGEMENT (SINGLE LONG-LIVED CONN + METRICS)
# -------------------------------------------------------------------

_DB_CONN: Optional[psycopg2.extensions.connection] = None
_DB_STATS: Dict[str, int] = {
    "connects": 0,       # how many times we opened a new connection
    "reconnects": 0,     # how many times we had to reconnect after an error
    "cursor_calls": 0,   # how many logical DB operations we did
}


def _get_conn() -> psycopg2.extensions.connection:
    """
    Get (or open) the global DB connection. Reused for the entire script run.
    """
    global _DB_CONN

    if _DB_CONN is None or getattr(_DB_CONN, "closed", 0):
        _DB_CONN = psycopg2.connect(PG_DSN)
        _DB_CONN.autocommit = False
        _DB_STATS["connects"] += 1
        print(
            f"[DB] opened new connection connects={_DB_STATS['connects']} "
            f"reconnects={_DB_STATS['reconnects']}"
        )

    return _DB_CONN


def with_db_cursor(fn: Callable[[psycopg2.extensions.connection, psycopg2.extensions.cursor], Any]) -> Any:
    """
    Run a DB operation with a cursor on the global connection.

    - Retries once if the connection is broken (OperationalError / InterfaceError),
      then re-raises if it still fails.
    - Commits on success.
    - Updates _DB_STATS for basic observability.
    """
    global _DB_CONN
    last_exc: Optional[BaseException] = None

    for attempt in (1, 2):
        conn = None
        cur = None
        try:
            conn = _get_conn()
            cur = conn.cursor()
            _DB_STATS["cursor_calls"] += 1
            result = fn(conn, cur)
            conn.commit()
            return result
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_exc = e
            print(
                f"[DB] {type(e).__name__} on attempt {attempt}, reconnecting… "
                f"(connects={_DB_STATS['connects']} reconnects={_DB_STATS['reconnects']})"
            )
            try:
                if conn is not None and not getattr(conn, "closed", 0):
                    conn.close()
            except Exception:
                pass
            _DB_CONN = None
            _DB_STATS["reconnects"] += 1
        except Exception:
            # Non-connection error: rollback so the connection can be reused.
            try:
                if conn is not None and not getattr(conn, "closed", 0):
                    conn.rollback()
            except Exception:
                pass
            raise

        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DB operation failed after reconnect attempts")


def close_db_conn():
    """
    Close the global DB connection at the end of the script (best-effort).
    """
    global _DB_CONN
    if _DB_CONN is not None and not getattr(_DB_CONN, "closed", 0):
        try:
            _DB_CONN.close()
        except Exception:
            pass


# -------------------------------------------------------------------
# OFFICIAL APPLE COLOR MAP (ENFORCEMENT)
# -------------------------------------------------------------------

# Canonical official colors per generation + variant (base/pro)
ALLOWED_COLORS: Dict[int, Dict[str, List[str]]] = {
    13: {
        "base": [
            "PRODUCT(RED)",
            "Starlight",
            "Midnight",
            "Blue",
            "Pink",
            "Green",
        ],
        "pro": [
            "Graphite",
            "Gold",
            "Silver",
            "Sierra Blue",
            "Alpine Green",
        ],
    },
    14: {
        "base": [
            "Midnight",
            "Purple",
            "Starlight",
            "PRODUCT(RED)",
            "Blue",
            "Yellow",
        ],
        "pro": [
            "Space Black",
            "Silver",
            "Gold",
            "Deep Purple",
        ],
    },
    15: {
        "base": [
            "Black",
            "Blue",
            "Green",
            "Yellow",
            "Pink",
        ],
        "pro": [
            "Black Titanium",
            "White Titanium",
            "Blue Titanium",
            "Natural Titanium",
        ],
    },
    16: {
        "base": [
            "Black",
            "White",
            "Pink",
            "Teal",
            "Ultramarine",
        ],
        "pro": [
            "Black Titanium",
            "White Titanium",
            "Natural Titanium",
            "Desert Titanium",
        ],
    },
    17: {
        "base": [
            "Black",
            "Lavender",
            "Mist Blue",
            "Sage",
            "White",
        ],
        "pro": [
            "Deep Blue",
            "Cosmic Orange",
            "Silver",
        ],
    },
}


def canonical_key(name: str) -> str:
    """
    Normalize Apple-style color names to a grouping key:
    - lowercase
    - spaces → underscores
    - remove parentheses
    """
    if not name:
        return ""
    key = name.strip().lower()
    key = key.replace("(", "").replace(")", "")
    key = key.replace(" ", "_")
    return key


def get_variant(model_label: str) -> str:
    """
    Decide BASE vs PRO from the model string.

    If model_label contains "pro" (case-insensitive), treat as "pro".
    Otherwise treat as "base".
    """
    if not model_label:
        return "base"
    ml = model_label.lower()
    return "pro" if "pro" in ml else "base"


def validate_and_normalize_color(
    generation: int,
    model_label: str,
    body_color_name: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Enforce that body_color_name is one of the official Apple colors
    for the given generation + variant (base/pro).

    - If name is not in the allowed set, return (None, None).
    - If name is valid, return (name, canonical_key(name)).
    """
    if body_color_name is None:
        return None, None

    name = body_color_name.strip()
    if not name:
        return None, None

    variant = get_variant(model_label)
    allowed_for_gen = ALLOWED_COLORS.get(generation, {})
    allowed_list = allowed_for_gen.get(variant, [])

    if name not in allowed_list:
        # INVALID / hallucinated color → drop it
        print(
            f"[WARN] Invalid body_color_name='{name}' for gen={generation} "
            f"variant={variant} model='{model_label}'. Allowed={allowed_list}"
        )
        return None, None

    # Valid official color
    key = canonical_key(name)
    return name, key


# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------

def get_candidate_listings(limit_listings: int) -> List[Tuple[int, int]]:
    """
    Return (generation, listing_id) pairs that:
    - are eligible listings,
    - have at least 1 image in iphone_image_assets, and
    - have NOT yet had body_color_* processed (color_done = false).

    This prevents re-running the color analysis on listings
    that already have color data, so we don't burn tokens twice.
    """

    def _run(conn, cur):
        cur.execute(
            """
            WITH eligible AS (
                SELECT generation, listing_id
                FROM "iPhone".iphone_listings
                WHERE COALESCE(status,'') IN ('live','sold','older21days')
                  AND spam IS NULL
                  AND url IS NOT NULL
            )
            SELECT e.generation, e.listing_id
            FROM eligible e
            JOIN "iPhone".iphone_image_assets a
              ON a.generation = e.generation
             AND a.listing_id    = e.listing_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM ml.iphone_image_features_v1 f
                WHERE f.generation      = e.generation
                  AND f.listing_id         = e.listing_id
                  AND f.feature_version = %s
                  AND f.color_done      = TRUE
            )
            GROUP BY e.generation, e.listing_id
            ORDER BY e.generation, e.listing_id
            LIMIT %s;
            """,
            (FEATURE_VERSION, limit_listings),
        )
        return cur.fetchall()

    rows = with_db_cursor(_run)
    return rows


def get_listing_context(gen: int, listing_id: int) -> Dict[str, Any]:
    """
    Fetch text-side metadata (title, description, condition_score, model).

    For SOLD rows:
      - Prefer PSA snapshots (title_snapshot, description_snapshot) when they
        are longer / more informative than the main scraped title/description.

    For LIVE / OLDER21DAYS:
      - Use the main scraped title/description only.
    """

    def _run(conn, cur):
        cur.execute(
            """
            SELECT
                COALESCE(l.title, '')            AS title_main,
                COALESCE(l.description, '')      AS description_main,
                l.condition_score,
                COALESCE(l.model, '')            AS model_main,
                COALESCE(l.status, '')           AS status_main,
                COALESCE(psa.title_snapshot, '') AS title_psa,
                COALESCE(psa.description_snapshot, '') AS description_psa
            FROM "iPhone".iphone_listings l
            LEFT JOIN LATERAL (
                SELECT
                    title_snapshot,
                    description_snapshot
                FROM "iPhone".post_sold_audit p
                WHERE p.listing_id = l.listing_id
                ORDER BY p.snapshot_at DESC
                LIMIT 1
            ) psa ON TRUE
            WHERE l.generation = %s
              AND l.listing_id    = %s;
            """,
            (gen, listing_id),
        )
        return cur.fetchone()

    row = with_db_cursor(_run)

    if not row:
        return {
            "title": "",
            "description": "",
            "condition_score": None,
            "model": "",
        }

    (
        title_main,
        description_main,
        condition_score,
        model_main,
        status_main,
        title_psa,
        description_psa,
    ) = row

    # Start with main scraped values
    title = title_main or ""
    description = description_main or ""

    # For SOLD rows, prefer PSA snapshots when they look richer
    if status_main == "sold":
        if title_psa and len(title_psa) > len(title):
            title = title_psa
        if description_psa and len(description_psa) > len(description):
            description = description_psa

    return {
        "title": title,
        "description": description,
        "condition_score": float(condition_score) if condition_score is not None else None,
        "model": model_main or "",
        "status": status_main,
        "title_main": title_main,
        "desc_main": description_main,
        "title_psa": title_psa,
        "desc_psa": description_psa,
    }


def get_images_for_listing(
    gen: int, listing_id: int, max_images: int
) -> List[Tuple[int, str, str]]:
    """
    Return list of (image_index, full_path, caption_text) for this listing,
    ordered by image_index, up to max_images.
    """

    def _run(conn, cur):
        cur.execute(
            """
            SELECT image_index, storage_path, COALESCE(caption_text, '')
            FROM "iPhone".iphone_image_assets
            WHERE generation = %s AND listing_id = %s
            ORDER BY image_index
            LIMIT %s;
            """,
            (gen, listing_id, max_images),
        )
        return cur.fetchall()

    rows = with_db_cursor(_run)

    images: List[Tuple[int, str, str]] = []
    for idx, storage_path, caption in rows:
        full_path = os.path.join(IMAGE_ROOT_DIR, storage_path)
        if os.path.isfile(full_path):
            images.append((idx, full_path, caption))
        else:
            print(f"[WARN] Missing file on disk for listing_id: {listing_id} idx={idx}: {full_path}")
    return images


def encode_image_base64(path: str) -> str:
    """
    Load an image, downscale so the longest side is at most MAX_IMAGE_LONG_SIDE,
    re-encode as JPEG with JPEG_QUALITY, and return base64-encoded bytes.
    """
    img = Image.open(path)

    # Convert to RGB if needed
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downscale if needed
    w, h = img.size
    long_side = max(w, h)
    if long_side > MAX_IMAGE_LONG_SIDE:
        scale = MAX_IMAGE_LONG_SIDE / float(long_side)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Encode as JPEG in memory
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    data = buf.read()

    # Base64 encode
    return base64.b64encode(data).decode("ascii")


# -------------------------------------------------------------------
# BELOW-13 / NOT-IPHONe SPAM HANDLING (LLM-DRIVEN)
# -------------------------------------------------------------------

def mark_spam_below13(
    gen: int,
    listing_id: int,
    ctx: Dict[str, Any],
    source: str,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Mark spam='below13' in iphone_listings and record evidence
    in ml.iphone_image_features_v1.model_fix_* for this listing.

    Returns True if the spam mark succeeded (or was already below13).
    """
    title = ctx.get("title") or ""
    description = ctx.get("description") or ""

    def _run(conn, cur):
        # Fetch current model + spam for evidence
        cur.execute(
            """
            SELECT model, spam
            FROM "iPhone".iphone_listings
            WHERE generation = %s
              AND listing_id    = %s
            FOR UPDATE;
            """,
            (gen, listing_id),
        )
        row = cur.fetchone()
        if not row:
            return False

        old_model, old_spam = row

        # Update spam to 'below13'
        cur.execute(
            """
            UPDATE "iPhone".iphone_listings
            SET spam = 'below13'
            WHERE generation = %s
              AND listing_id    = %s;
            """,
            (gen, listing_id),
        )

        # Build evidence JSON
        ev: Dict[str, Any] = {
            "type": "spam_below13",
            "at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "reason": reason,
            "generation": gen,
            "listing_id": listing_id,
            "title": title,
            "description": description,
            "old_spam": old_spam,
            "new_spam": "below13",
            "old_model": old_model,
        }
        if extra:
            ev["extra"] = extra

        evidence_json = json.dumps(ev, ensure_ascii=False)

        # Record evidence on all feature rows for this listing.
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET model_fix_old_model = COALESCE(model_fix_old_model, %s),
                model_fix_new_model = COALESCE(model_fix_new_model, %s),
                model_fix_reason    = 'spam_below13',
                model_fix_evidence  = %s::jsonb,
                model_fix_at        = now()
            WHERE generation      = %s
              AND listing_id         = %s
              AND feature_version = %s;
            """,
            (
                old_model,
                old_model,
                evidence_json,
                gen,
                listing_id,
                FEATURE_VERSION,
            ),
        )

        return True

    try:
        ok = with_db_cursor(_run)
        if ok:
            print(
                f"[SPAM-BELOW13] gen={gen} listing_id={listing_id} "
                f"→ spam='below13' (source={source})"
            )
        return ok
    except Exception as e:
        print(f"[WARN] spam below13 mark failed gen={gen} listing_id={listing_id} → {e}")
        return False


def maybe_mark_below13_from_llm(
    gen: int,
    listing_id: int,
    data: Dict[str, Any],
    inferred_model: str,
    confidence: float,
) -> bool:
    """
    Below13 / non-iPhone mark based on LLM model_check.inferred_model.
    Does NOT change model; only spam + evidence.
    """
    ctx = data.get("_ctx") or {}
    model_check = data.get("model_check") or {}
    reason = model_check.get("reason") or ""
    extra = {
        "inferred_model": inferred_model,
        "confidence": confidence,
        "model_check": model_check,
    }
    return mark_spam_below13(
        gen=gen,
        listing_id=listing_id,
        ctx=ctx,
        source="llm_model_check",
        reason=reason,
        extra=extra,
    )


# -------------------------------------------------------------------
# MODEL FIX HELPER (uses model_check + new model_fix_* columns)
# -------------------------------------------------------------------

def maybe_apply_model_fix(
    gen: int,
    listing_id: int,
    data: Dict[str, Any],
    inferred_model: str,
    model_check: Dict[str, Any],
) -> None:
    """
    Apply a 13–17 model fix based on model_check.inferred_model
    (same generation, e.g. 13 base → 13 Pro / Pro Max) and record evidence.

    This mutates `data` in-place to:
      - update data["model_label"] to the new model when a fix is applied
      - set data["_model_fix_old_model"], data["_model_fix_new_model"]
    """
    ctx = data.get("_ctx") or {}
    db_model_ctx = ctx.get("model", "") or data.get("model_label", "") or ""
    db_model_llm = model_check.get("db_model") or ""
    variant = model_check.get("variant") or "unknown"
    confidence_raw = model_check.get("confidence", 0.0)
    reason = model_check.get("reason") or ""

    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    def norm(s: str) -> str:
        return (s or "").strip().lower()

    old_model = db_model_ctx or db_model_llm or ""
    if norm(old_model) == norm(inferred_model):
        return

    # Ensure LLM's db_model matches what we actually used (or is empty)
    if db_model_llm and norm(db_model_llm) != norm(db_model_ctx):
        print(
            f"[MODEL-FIX-SKIP] gen={gen} listing_id={listing_id} "
            f"db_model mismatch ctx='{db_model_ctx}' llm='{db_model_llm}'"
        )
        return

    evidence = {
        "model_check": model_check,
        "ctx_model": db_model_ctx,
        "ctx_title": ctx.get("title", ""),
        "ctx_description": ctx.get("description", ""),
    }

    def _run(conn, cur):
        # 1) Update the listings table (source of truth)
        cur.execute(
            """
            UPDATE "iPhone".iphone_listings
            SET model = %s
            WHERE generation = %s
              AND listing_id    = %s;
            """,
            (inferred_model, gen, listing_id),
        )

        # 2) Record the fix on all image rows for this listing / feature_version
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET model_fix_old_model = %s,
                model_fix_new_model = %s,
                model_fix_reason    = %s,
                model_fix_evidence  = %s::jsonb,
                model_fix_at        = now()
            WHERE generation      = %s
              AND listing_id         = %s
              AND feature_version = %s;
            """,
            (
                old_model,
                inferred_model,
                reason or "model_check auto-correction",
                json.dumps(evidence, ensure_ascii=False),
                gen,
                listing_id,
                FEATURE_VERSION,
            ),
        )

    with_db_cursor(_run)

    print(
        f"[MODEL-FIX] gen={gen} listing_id={listing_id} "
        f"'{old_model}' → '{inferred_model}' "
        f"(variant={variant}, conf={confidence:.2f})"
    )

    # Update in-memory context + data so color validation uses the new model
    ctx["model"] = inferred_model
    data["_ctx"] = ctx
    data["model_label"] = inferred_model
    data["_model_fix_old_model"] = old_model
    data["_model_fix_new_model"] = inferred_model


def insert_generation_fix_audit_row(
    cur,
    *,
    old_gen: int,
    new_gen: int,
    listing_id: int,
    old_model: str,
    new_model: str,
    status: str,   # 'applied' | 'skipped' | 'failed'
    reason: str,
    evidence: Dict[str, Any],
) -> None:
    cur.execute(
        """
        INSERT INTO ml.listing_model_generation_fix_audit (
            run_id,
            actor,
            old_generation,
            new_generation,
            listing_id,
            old_model,
            new_model,
            status,
            reason,
            evidence
        )
        VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s::jsonb
        );
        """,
        (
            RUN_ID,
            AUDIT_ACTOR,
            int(old_gen),
            int(new_gen),
            int(listing_id),
            (old_model or None),
            (new_model or None),
            status,
            (reason or None),
            json.dumps(evidence, ensure_ascii=False),
        ),
    )


def maybe_apply_generation_fix(
    old_gen: int,
    new_gen: int,
    listing_id: int,
    data: Dict[str, Any],
    inferred_model: str,
    model_check: Dict[str, Any],
) -> None:
    """
    Apply a 13–17 generation correction when model_check.inferred_model implies a different generation.

    Updates (in one transaction):
      - "iPhone".iphone_listings.generation + model
      - "iPhone".iphone_image_assets.generation
      - ml.iphone_image_features_v1.generation (moves any existing feature rows)

    Also sets:
      - data["_effective_generation"] = new_gen
      - data["model_label"] = inferred_model
    so color validation/inserts use the corrected generation immediately.

    Writes an audit row to:
      - ml.listing_model_generation_fix_audit
    """
    if new_gen == old_gen:
        return

    ctx = data.get("_ctx") or {}
    old_model = (ctx.get("model", "") or data.get("model_label", "") or "").strip()

    run_id = os.getenv("RUN_ID", None)
    actor = "images_iphone_color.py"

    evidence = {
        "type": "generation_fix",
        "old_generation": old_gen,
        "new_generation": new_gen,
        "old_model": old_model,
        "new_model": inferred_model,
        "model_check": model_check,
        "ctx_title": ctx.get("title", ""),
        "ctx_description": ctx.get("description", ""),
        "at": datetime.now(timezone.utc).isoformat(),
    }

    def _insert_audit(cur, status: str, reason: str, ev: Dict[str, Any]) -> None:
        cur.execute(
            """
            INSERT INTO ml.listing_model_generation_fix_audit (
                run_id,
                actor,
                old_generation,
                new_generation,
                listing_id,
                old_model,
                new_model,
                status,
                reason,
                evidence
            )
            VALUES (
                %s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s::jsonb
            );
            """,
            (
                run_id,
                actor,
                int(old_gen),
                int(new_gen),
                int(listing_id),
                (old_model or None),
                (inferred_model or None),
                status,
                (reason or None),
                json.dumps(ev, ensure_ascii=False),
            ),
        )

    def _run(conn, cur):
        # Lock the source listing row
        cur.execute(
            """
            SELECT 1
            FROM "iPhone".iphone_listings
            WHERE generation = %s AND listing_id = %s
            FOR UPDATE;
            """,
            (old_gen, listing_id),
        )
        if not cur.fetchone():
            _insert_audit(cur, "skipped", "missing_source", evidence)
            return ("missing_source", 0)

        # Prevent collisions if a row already exists at the target generation
        cur.execute(
            """
            SELECT 1
            FROM "iPhone".iphone_listings
            WHERE generation = %s AND listing_id = %s;
            """,
            (new_gen, listing_id),
        )
        if cur.fetchone():
            _insert_audit(cur, "skipped", "target_exists", evidence)
            return ("target_exists", 0)

        # 1) Move listing to new generation + update model
        cur.execute(
            """
            UPDATE "iPhone".iphone_listings
            SET generation = %s,
                model      = %s
            WHERE generation = %s
              AND listing_id    = %s;
            """,
            (new_gen, inferred_model, old_gen, listing_id),
        )

        # 2) Move image assets
        cur.execute(
            """
            UPDATE "iPhone".iphone_image_assets
            SET generation = %s
            WHERE generation = %s
              AND listing_id    = %s;
            """,
            (new_gen, old_gen, listing_id),
        )

        # 3) Move any existing feature rows
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET generation = %s
            WHERE generation = %s
              AND listing_id    = %s;
            """,
            (new_gen, old_gen, listing_id),
        )

        # 4) Record evidence on feature rows (if any exist now)
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET model_fix_old_model = %s,
                model_fix_new_model = %s,
                model_fix_reason    = %s,
                model_fix_evidence  = %s::jsonb,
                model_fix_at        = now()
            WHERE generation      = %s
              AND listing_id         = %s
              AND feature_version = %s;
            """,
            (
                old_model,
                inferred_model,
                "generation_fix",
                json.dumps(evidence, ensure_ascii=False),
                new_gen,
                listing_id,
                FEATURE_VERSION,
            ),
        )

        # 5) Audit trail row (applied)
        _insert_audit(cur, "applied", "generation_fix_from_model_check", evidence)

        return ("ok", 1)

    try:
        status, ok = with_db_cursor(_run)
    except Exception as e:
        # Best-effort: reset broken/aborted connection and write a FAILED audit row.
        fail_ev = dict(evidence)
        fail_ev["error"] = f"{type(e).__name__}: {e}"

        try:
            close_db_conn()
        except Exception:
            pass

        def _audit_fail(conn, cur):
            _insert_audit(cur, "failed", "exception_in_generation_fix", fail_ev)

        try:
            with_db_cursor(_audit_fail)
        except Exception:
            pass

        print(f"[GEN-FIX-FAIL] old_gen={old_gen} new_gen={new_gen} listing_id={listing_id} → {e}")
        return

    if status != "ok":
        print(f"[GEN-FIX-SKIP] old_gen={old_gen} new_gen={new_gen} listing_id={listing_id} reason={status}")
        return

    print(f"[GEN-FIX] listing_id={listing_id} generation {old_gen}→{new_gen} model '{old_model}'→'{inferred_model}'")

    # Update in-memory state so validation uses the corrected generation/model immediately
    ctx["model"] = inferred_model
    data["_ctx"] = ctx
    data["model_label"] = inferred_model
    data["_effective_generation"] = new_gen




def handle_model_check(gen: int, listing_id: int, data: Dict[str, Any]) -> None:
    """
    Use model_check to:
      - auto-fix 13–17 models, OR
      - mark spam='below13' for pre–13 or non-iPhone devices.
    """
    model_check = data.get("model_check") or {}
    if not model_check:
        return

    inferred_model = model_check.get("inferred_model")
    confidence_raw = model_check.get("confidence", 0.0)

    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0

    if not inferred_model or not isinstance(inferred_model, str):
        return

    inferred_model = inferred_model.strip()
    if not inferred_model or confidence < 0.8:
        return

    # Decide based on digits in inferred_model
    digits = re.findall(r"\d+", inferred_model.lower())

    if not digits:
        maybe_mark_below13_from_llm(gen, listing_id, data, inferred_model, confidence)
        return

    try:
        n = int(digits[0])
    except Exception:
        maybe_mark_below13_from_llm(gen, listing_id, data, inferred_model, confidence)
        return

    if n < 13:
        maybe_mark_below13_from_llm(gen, listing_id, data, inferred_model, confidence)
    elif 13 <= n <= 17:
        if n != gen:
            maybe_apply_generation_fix(gen, n, listing_id, data, inferred_model, model_check)
        else:
            maybe_apply_model_fix(gen, listing_id, data, inferred_model, model_check)



# -------------------------------------------------------------------
# OPENAI CALL (BODY COLOR + MODEL CHECK)
# -------------------------------------------------------------------

def call_model_for_listing(
    gen: int,
    listing_id: int,
    images: List[Tuple[int, str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Call the chosen GPT model with all images for one listing.
    Returns parsed JSON dict as described in the body color + model_check system prompt.
    """
    if not images:
        raise ValueError(f"No images for listing_id: {listing_id}")

    # Fetch listing metadata (title + description + condition_score + model),
    # using PSA text for SOLD rows if available.
    ctx = get_listing_context(gen, listing_id)
    title = ctx["title"]
    description = ctx["description"]
    condition_score = ctx["condition_score"]
    model_label = ctx.get("model", "")

    # Build image inputs as data URLs
    input_images: List[Dict[str, Any]] = []
    for image_index, path, _caption in images:
        b64 = encode_image_base64(path)
        data_url = f"data:image/jpeg;base64,{b64}"
        input_images.append(
            {
                "type": "input_image",
                "image_url": data_url,
            }
        )

    # Build a textual block listing image indices and captions
    caption_lines = []
    for idx, _path, caption in images:
        if caption:
            caption_lines.append(f"- image_index {idx} CAPTION: {caption}")
        else:
            caption_lines.append(f"- image_index {idx} CAPTION: (none)")

    captions_block = "\n".join(caption_lines)

    user_text = (
        f"Listing context:\n"
        f"- generation: {gen}\n"
        f"- listing_id: {listing_id}\n"
        f"- MODEL: {model_label}\n"
        f"- CONDITION_SCORE: {condition_score if condition_score is not None else 'null'}\n"
        f"- TITLE: {title}\n"
        f"- DESCRIPTION: {description}\n\n"
        f"Image captions (per image, may be empty):\n"
        f"{captions_block}\n\n"
        f"You will receive {len(images)} images for this listing.\n"
        f"For each image, you MUST label ONLY the body color fields defined in the system prompt.\n"
        f"In addition, you MUST output the top-level model_check object as defined in the system prompt.\n\n"
        "Image indices (in order):\n"
        + "\n".join(f"- image_index {idx}: file {os.path.basename(path)}" for idx, path, _cap in images)
        + "\n\nNow output the JSON object."
    )

    print(
        f"[DEBUG] OpenAI call listing_id: {listing_id} imgs={len(images)} model={model_name}"
    )

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                        *input_images,
                    ],
                },
            ],
        )
    except Exception as e:
        print(f"[DEBUG] OpenAI exception type={type(e).__name__} listing_id: {listing_id}")
        print(f"[DEBUG] OpenAI exception detail={repr(e)}")
        traceback.print_exc()
        raise RuntimeError(f"OpenAI call failed for listing_id: {listing_id}: {e}") from e

    # Log the model the API says it actually used
    try:
        api_model = getattr(response, "model", None)
    except Exception:
        api_model = None
    print(f"[MODEL] requested={model_name} api_response_model={api_model} listing_id: {listing_id}")

    # Extract JSON text from the *message* item, not the reasoning item
    try:
        msg = None
        for item in response.output:
            if hasattr(item, "content"):
                msg = item
                break
        if msg is None:
            raise RuntimeError("No message item with content found in response.output")

        text_out = msg.content[0].text
    except Exception as e:
        raise RuntimeError(
            f"Unexpected response format for listing_id: {listing_id}: {e}\n{response}"
        ) from e

    try:
        data = json.loads(text_out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON for listing_id: {listing_id}: {e}\n{text_out}"
        ) from e

    # attach context used for this call
    data["_ctx"] = ctx

    # grab token usage if available
    usage_info: Dict[str, Optional[int]] = {}
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_info = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception:
        usage_info = {}

    if usage_info:
        data["_usage"] = usage_info

    # Basic sanity check
    if data.get("listing_id") != listing_id or data.get("generation") != gen:
        print(
            f"[WARN] JSON meta mismatch for listing_id: {listing_id}: "
            f"got listing_id={data.get('listing_id')} generation={data.get('generation')}"
        )

    # Attach model_label so insert step can validate with it
    data["model_label"] = model_label

    return data


# -------------------------------------------------------------------
# SUMMARY LINE FOR LOGGING
# -------------------------------------------------------------------

def summarize_listing(gen: int, listing_id: int, data: Dict[str, Any], inserted_count: int) -> str:
    """
    Build a one-line summary string for logging, based on the JSON
    returned by the model for this listing.
    """
    imgs = data.get("images") or []
    n_images = len(imgs)

    # choose best color by highest confidence
    best_color_name: Optional[str] = None
    best_conf: float = -1.0
    for img in imgs:
        name = img.get("body_color_name")
        conf_raw = img.get("body_color_confidence")
        try:
            conf = float(conf_raw) if conf_raw is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        if name and conf >= best_conf:
            best_color_name = str(name)
            best_conf = conf

    if best_color_name is None:
        color_str = "unknown"
    else:
        display_conf = best_conf
        if display_conf > 1.0:
            display_conf = display_conf / 100.0
        if display_conf > 0.0:
            color_str = f"{best_color_name} ({display_conf:.2f})"
        else:
            color_str = best_color_name

    # token usage
    usage = data.get("_usage") or {}
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    tot_tok = usage.get("total_tokens")
    if tot_tok is not None:
        tokens_str = f"{tot_tok} (in={in_tok}, out={out_tok})"
    else:
        tokens_str = "unknown"

    # model fix summary (if any)
    fix_old = data.get("_model_fix_old_model")
    fix_new = data.get("_model_fix_new_model")
    if fix_old and fix_new:
        fix_str = f" model_fix='{fix_old}'→'{fix_new}'"
    else:
        fix_str = ""

    return (
        f"listing_id: {listing_id} "
        f"imgs={n_images} "
        f"body_color={color_str} "
        f"tokens={tokens_str} "
        f"inserted={inserted_count}"
        f"{fix_str}"
    )


# -------------------------------------------------------------------
# INSERT + MARK COLOR_DONE
# -------------------------------------------------------------------

def insert_features_from_json(
    gen: int, listing_id: int, data: Dict[str, Any]
) -> int:
    """
    Insert/update rows in ml.iphone_image_features_v1 from LLM JSON.

    IMPORTANT:
    - This script ONLY writes body_color_* fields.
    - It NEVER touches damage, accessories, battery, etc.
    - It enforces Apple-official colors by generation + model.
    """
    images = data.get("images") or []
    if not images:
        print(f"[WARN] No images array in JSON for listing_id: {listing_id}")
        return 0

    model_label = data.get("model_label", "")

    rows: List[Tuple[Any, ...]] = []
    for img in images:
        image_index = img["image_index"]

        # Raw outputs from LLM
        body_color_name = img.get("body_color_name")
        if body_color_name is not None and not isinstance(body_color_name, str):
            body_color_name = str(body_color_name)

        # Enforce allowed Apple colors and recompute key
        body_color_name, body_color_key = validate_and_normalize_color(
            generation=gen,
            model_label=model_label,
            body_color_name=body_color_name,
        )

        body_color_conf_raw = img.get("body_color_confidence")
        try:
            body_color_confidence = float(body_color_conf_raw)
        except (TypeError, ValueError):
            body_color_confidence = 0.0

        body_color_from_case = bool(img.get("body_color_from_case", False))

        # If color invalid → all null/zero
        if body_color_name is None:
            body_color_key = None
            body_color_confidence = 0.0
            body_color_from_case = False

        rows.append(
            (
                gen,
                listing_id,
                image_index,
                FEATURE_VERSION,
                body_color_name,
                body_color_key,
                body_color_confidence,
                body_color_from_case,
            )
        )

    def _run(conn, cur):
        execute_batch(
            cur,
            """
            INSERT INTO ml.iphone_image_features_v1 (
                generation,
                listing_id,
                image_index,
                feature_version,
                body_color_name,
                body_color_key,
                body_color_confidence,
                body_color_from_case
            )
            VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s
            )
            ON CONFLICT (generation, listing_id, image_index, feature_version) DO UPDATE
            SET body_color_name       = EXCLUDED.body_color_name,
                body_color_key        = EXCLUDED.body_color_key,
                body_color_confidence = EXCLUDED.body_color_confidence,
                body_color_from_case  = EXCLUDED.body_color_from_case,
                created_at            = LEAST(ml.iphone_image_features_v1.created_at, now())
            ;
            """,
            rows,
        )

    with_db_cursor(_run)
    return len(images)


def mark_color_done(gen: int, listing_id: int) -> None:
    """
    Mark color_done/color_done_at in ml.iphone_image_features_v1
    for this (generation, listing_id). This is the processed marker so we don't
    run the color script twice on the same listing.
    """

    def _run(conn, cur):
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET color_done    = TRUE,
                color_done_at = now()
            WHERE generation      = %s
              AND listing_id         = %s
              AND feature_version = %s;
            """,
            (gen, listing_id, FEATURE_VERSION),
        )

    with_db_cursor(_run)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch BODY COLOR + MODEL CHECK image analysis for iPhone listings using GPT-5 nano."
    )
    ap.add_argument(
        "--limit-listings",
        type=int,
        default=10,
        help="Number of listings to label (default: 10)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"OpenAI model name (default: {DEFAULT_MODEL_NAME})",
    )
    ap.add_argument(
        "--max-images-per-listing",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_LISTING,
        help=f"Max images per listing to send to the model (default: {DEFAULT_MAX_IMAGES_PER_LISTING})",
    )

    args = ap.parse_args()

    model_name = args.model
    max_images_per_listing = args.max_images_per_listing

    print("===================================================")
    print("[CONFIG] PG_DSN              = <set>")
    print(f"[CONFIG] IMAGE_ROOT          = {IMAGE_ROOT_DIR}")
    print(f"[CONFIG] model               = {model_name}")
    print(f"[CONFIG] limit_listings      = {args.limit_listings}")
    print(f"[CONFIG] max_images          = {max_images_per_listing}")
    print(f"[CONFIG] MAX_IMAGE_LONG_SIDE = {MAX_IMAGE_LONG_SIDE}")
    print(f"[CONFIG] JPEG_QUALITY        = {JPEG_QUALITY}")
    print("===================================================")

    candidates = get_candidate_listings(args.limit_listings)
    print(f"[INFO] Found {len(candidates)} candidate listings to analyze.")

    total_listings_processed = 0
    total_images_processed = 0

    for gen, listing_id in candidates:
        imgs = get_images_for_listing(gen, listing_id, max_images_per_listing)
        if not imgs:
            print(f"[SKIP] gen={gen} listing_id={listing_id} → No disk images found.")
            continue

        print(
            f"[RUN] gen={gen} listing_id={listing_id} → {len(imgs)} images to analyze using model={model_name}"
        )

        try:
            data = call_model_for_listing(gen, listing_id, imgs, model_name)

            # LLM model_check → spam or model_fix
            try:
                handle_model_check(gen, listing_id, data)
            except Exception as mc_e:
                print(f"[WARN] gen={gen} listing_id={listing_id} handle_model_check failed → {mc_e}")

            # Use corrected generation if model_check inferred a new generation
            effective_gen = int(data.get("_effective_generation", gen))

            inserted_count = insert_features_from_json(effective_gen, listing_id, data)

            if inserted_count > 0:
                mark_color_done(effective_gen, listing_id)

            total_listings_processed += 1
            total_images_processed += inserted_count

            summary_line = summarize_listing(effective_gen, listing_id, data, inserted_count)
            print(f"[OK] {summary_line}")

            # Every 10 successful listings, log DB health metrics
            if total_listings_processed % 10 == 0:
                print(
                    f"[DB][HEALTH] after {total_listings_processed} listings: "
                    f"connects={_DB_STATS['connects']} "
                    f"reconnects={_DB_STATS['reconnects']} "
                    f"cursor_calls={_DB_STATS['cursor_calls']}"
                )

        except Exception as e:
            print(f"[ERR] gen={gen} listing_id={listing_id} → {e}")

    print("===================================================")
    print(f"[SUMMARY] listings_processed = {total_listings_processed}")
    print(f"[SUMMARY] images_processed   = {total_images_processed}")
    print("===================================================")

    close_db_conn()



if __name__ == "__main__":
    main()