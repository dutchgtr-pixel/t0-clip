#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_accessories_analysis.py

Image ACCESSORY analysis for iPhone listings using GPT-5 (nano).

The model is used ONLY for:
- accessories / box / cases / charger / earbuds / receipt
- case_count
- box_state_level
- battery_screenshot
- battery_health_pct_img

This script is designed to be COMPLEMENTARY to image_damage_analysis.py:
- image_damage_analysis.py (mini) owns:
    • photo_quality_level, background_clean_level, is_stock_photo
    • visible_damage_level, damage_summary_text
    • has_screen_protector, damage_on_protector_only
- image_accessories_analysis.py (nano) owns:
    • has_box, has_charger_brick, has_cable, has_charger
    • has_earbuds, has_case, has_screen_guard
    • has_other_accessory, has_receipt, case_count
    • battery_screenshot, battery_health_pct_img
    • box_state_level

IMPORTANT:
- This script NEVER writes or updates damage-related columns.
- It ONLY inserts/updates accessory & battery/box_state columns in ml.iphone_image_features_v1.
- It uses accessories_done/accessories_done_at in ml.iphone_image_features_v1
  to avoid re-processing listings and burning tokens.
"""

import os
import json
import base64
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional, Callable

import io
from PIL import Image

import psycopg2
from psycopg2.extras import execute_batch

from openai import OpenAI

# -------------------------------------------------------------------
# CONFIG / ENV
# -------------------------------------------------------------------

PG_DSN = os.getenv("PG_DSN")  # required; no default in public release
IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", "./images")

# default max images per listing; CLI can override
DEFAULT_MAX_IMAGES_PER_LISTING = int(
    os.getenv("MAX_IMAGES_PER_LISTINGS", os.getenv("MAX_IMAGES_PER_LISTING", "16"))
)

# default model; CLI can override (nano by default)
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")

FEATURE_VERSION = 1

# image downscaling config:
MAX_IMAGE_LONG_SIDE = int(os.getenv("MAX_IMAGE_LONG_SIDE", "1024"))  # px
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))                  # 1–95

# -------------------------------------------------------------------
# OPENAI CLIENT (FORCE CORRECT BASE + KEY)
# -------------------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

print("[DEBUG] Using base_url:", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

# SYSTEM PROMPT LEFT EMPTY ON PURPOSE
SYSTEM_PROMPT = """
You are an assistant that labels iPhone listing photos for resale analytics, with a focus on:

- accessories and extras (box, charger, cable, earbuds, cases, etc.),
- battery screenshots and battery health percentage from images,
- packaged screen protectors (unapplied),
- box presence and box state (sealed vs opened).

You will get:
- A generation number (iPhone model generation, e.g. 13, 14, 17)
- A marketplace listing id (listing_id)
- A title and description from the listing text
- For each image:
    - an image_index (0-based)
    - the seller's caption text for that image (may be empty)
- Between 1 and 16 images of the SAME physical device and its accessories.

Use the listing title, description, and per-image captions as hints about what objects are shown
(phone body, case, box, cable, earbuds, extra accessories, receipts, battery settings screen, etc.),
but ALWAYS follow what you actually see in the photos. Prefer fewer false positives over guessing.

Apply the following rules and output format EXACTLY:

ACCESSORIES & EXTRAS (PER IMAGE):

- has_box: true/false
    true if a retail iPhone box is visible (3D cardboard package that can close around a phone, with artwork/Apple logo and real depth).
    Rules:
      - Thin shell-shaped objects with camera cut-out are CASES, not boxes.
      - A flat slab under/next to the phone with camera cut-out is almost always a CASE, not a box.
      - If you see only phone + case + protector + cable on a table and no obvious cardboard packaging → has_box = false.
      - If you are not clearly seeing a cardboard retail phone package → has_box = false.

- has_charger_brick: true/false
    true if a wall charger / power adapter block is visible.
    Rules:
      - Set to true when you see:
          • a solid rectangular/square power adapter with plug prongs / wall-plug shape, OR
          • a retail adapter box clearly showing a power adapter (e.g. "USB-C 20W Power Adapter" box).
      - Do NOT assume a brick from a phone box alone.

- has_cable: true/false
    true if a charging cable is visible.
    Rules:
      - Set has_cable = true if you see:
          • a cable coiled in a box insert or tray,
          • a cable coiled or lying next to/on the phone/box,
          • a cable partially off-frame but clearly attached to phone or brick,
          • a retail cable box clearly showing a charging cable.
      - If it clearly looks like a charging lead or a box advertising a cable, set has_cable = true.
      - A cable alone is NOT a charger brick.

- has_charger: true/false
    Derived: true if EITHER a cable OR a charger brick is visible:
      has_charger = (has_cable OR has_charger_brick).

- has_earbuds: true/false
    true if any earphones (wired or wireless) are visible.
    Includes:
      - loose earbuds, AirPods / AirPods Pro-style cases,
      - a clear retail earphone box (e.g. "EarPods with Lightning Connector" showing earphones).
    Rules:
      - Set has_earbuds = true if you see an AirPods-style case, wired EarPods, Beats-style earbuds, or a box that clearly shows earphones.
      - Do NOT set has_earbuds = true for random objects that are not clearly earphones.

- has_case: true/false
    true if at least one phone case/cover is visible (empty or with phone inside).
    A "case" means:
      - shell shaped like the phone,
      - plastic/silicone/leather,
      - wraps around sides with camera cut-out.
    Rules:
      - Set has_case = true if:
          • phone is next to a shell-shaped cover,
          • phone is on top of a shell-shaped cover,
          • you see a spare case by itself,
          • phone is inside a visible case,
          • you see a retail box clearly showing a phone case.
      - If an object might be either a thin box or a case, treat it as CASE:
          • has_case = true, has_box = false.

- has_screen_guard: true/false
    true if you see PACKAGED/unapplied screen protectors:
      - glass/film sheet clearly meant for the screen,
      - film pack, "1/2" wipes with typical protector packaging.
    This is about protectors not yet applied to the phone (packaging/loose sheets), not protectors already on the screen.

- has_other_accessory: true/false
    true if you see other phone accessories in THIS image, e.g.:
      - power bank / battery pack,
      - car mount/holder,
      - MagSafe ring/wallet,
      - wireless charging pad,
      - hubs/dongles clearly part of the bundle.
    If you are unsure what an object is and it does not clearly look like a standard box/cable/brick/case/earbuds/screen-guard package, you may leave has_other_accessory = false (prefer fewer false positives).

- has_receipt: true/false
    true only if a purchase receipt / invoice / order confirmation is clearly visible (paper or on-screen).
    Do NOT count random pieces of paper or manuals.

- case_count: integer ≥ 0
    Number of separate phone cases you see in THIS image.
    Rules:
      - phone next to one shell-shaped cover → case_count = 1.
      - two different cases (e.g. black and clear) visible → case_count = 2.
      - only extra objects are cable + glass protector and no case shell → case_count = 0.
      - if you are sure you see exactly one case → case_count = 1.

BATTERY / SYSTEM INFO (PER IMAGE):

- battery_screenshot: true/false
    true if the image shows an iOS battery health/charging screen or similar battery-status page (any language, light or dark mode).
    Typical signs:
      - navigation title like "Batteritilstand og lading" / "Battery Health & Charging",
      - a row labeled "Maksimal kapasitet" / "Maximum Capacity" with a percentage (e.g. "86 %", "99 %"),
      - a toggle like "Optimalisert batterilading" / "Optimized Battery Charging".
    If you clearly see such a battery settings page with a capacity percentage row, set battery_screenshot = true.

- battery_health_pct_img: integer 50–100 or null
    If a clear battery health percentage is visible on that screen, extract ONLY that percentage as an integer (e.g. "86 %" → 86).
    STRICT RULES:
      - The value MUST be a plain integer between 50 and 100 inclusive.
      - NEVER output decimals, strings, or values outside 50–100. If you see anything like "3", "120", "0", "999", or non-numeric text → use null instead.
      - Only read the percentage from the BATTERY HEALTH / MAXIMUM CAPACITY row (or local-language equivalent) of the battery health screen.
      - Ignore all other percentages (e.g. battery charge level, storage usage, discount % in text, etc.).
      - If you are not 100% sure you see a valid capacity % in that row (clear digits + % sign), set battery_health_pct_img = null.
      - If the percentage is unreadable, cut off, too blurry, or simply not present → battery_health_pct_img = null.
      - If capacity shown is below 80%, still extract that number as a normal integer; downstream logic handles low values.

BOX STATE:

- box_state_level: 0–2 or null
    0 = no box visible in this image.
    1 = box visible but clearly opened/used (or state unclear).
    2 = box appears sealed/unopened (intact pull tabs, factory seal).
    Rules:
      - If has_box = false, box_state_level MUST be 0.
      - If has_box = true and you cannot see any seal/pull tab, choose 1 (opened) by default.
      - If unsure but box is visible, choose 1.

FINAL RULES:

- Work PER IMAGE; do not assume accessories/protector presence from one image carries to another unless you clearly see them.
- Count accessories (cases, charger, etc.) per image; we aggregate later.
- If you see a shell-shaped cover, set has_case = true.
- If an object might be either a case or a box, treat it as a CASE and keep has_box = false.
- If you clearly see a cable or cable box, set has_cable = true.
- If you clearly see an adapter brick or adapter box, set has_charger_brick = true.

OUTPUT FORMAT (VERY IMPORTANT):

Return ONLY one JSON object:

{
  "generation": <int>,
  "listing_id": <int>,
  "images": [
    {
      "image_index": <int>,
      "has_box": <bool>,
      "has_charger_brick": <bool>,
      "has_cable": <bool>,
      "has_charger": <bool>,
      "has_earbuds": <bool>,
      "has_case": <bool>,
      "has_screen_guard": <bool>,
      "has_other_accessory": <bool>,
      "has_receipt": <bool>,
      "case_count": <int>,
      "battery_screenshot": <bool>,
      "battery_health_pct_img": <int or null>,
      "box_state_level": <int 0-2 or null>
    },
    ...
  ]
}

- Do NOT wrap in markdown.
- Do NOT add extra keys or comments.
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

    if not PG_DSN:
        raise RuntimeError("PG_DSN is not set. Set the PG_DSN environment variable to connect to Postgres.")

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
    - Rolls back on any non-connection error so we don't leave the
      transaction in an aborted state.
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
            # Connection-level problem. Close and clear, then retry once.
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
            # Any other DB error (constraint, syntax, etc.): rollback and propagate.
            if conn is not None:
                try:
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
# DB HELPERS
# -------------------------------------------------------------------


def get_candidate_listings(limit_listings: int) -> List[Tuple[int, int]]:
    """
    Return (generation, listing_id) pairs that:
    - are eligible listings,
    - have at least 1 image in iphone_image_assets, and
    - have NOT yet had accessories/battery/box_state processed
      (accessories_done = false) in ml.iphone_image_features_v1.

    This prevents re-running the accessories analysis on listings
    that already have accessories data, so we don't burn tokens twice.
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
                  AND f.accessories_done = TRUE
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
    Fetch text-side metadata (title, description, condition_score, model)
    so we can give the model global context.

    We DO NOT send any text-based damage decisions to the model.
    """

    def _run(conn, cur):
        cur.execute(
            """
            SELECT
                COALESCE(title, '')       AS title,
                COALESCE(description, '') AS description,
                condition_score,
                COALESCE(model, '')       AS model
            FROM "iPhone".iphone_listings
            WHERE generation = %s
              AND listing_id    = %s;
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

    title, description, condition_score, model = row

    return {
        "title": title,
        "description": description,
        "condition_score": float(condition_score) if condition_score is not None else None,
        "model": model,
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
# OPENAI CALL
# -------------------------------------------------------------------


def call_model_for_listing(
    gen: int,
    listing_id: int,
    images: List[Tuple[int, str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Call the chosen GPT model with all images for one listing.
    Returns parsed JSON dict as described in the accessories system prompt.
    """
    if not images:
        raise ValueError(f"No images for listing_id: {listing_id}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    # Fetch listing metadata (title + description + condition_score + model)
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
        f"For each image, you MUST label ONLY:\n"
        f"- accessories and extras (box, charger brick, cable, charger, earbuds, cases, packaged screen protectors, other accessories, receipts),\n"
        f"- case_count,\n"
        f"- battery_screenshot and battery_health_pct_img (if visible),\n"
        f"- box_state_level.\n"
        f"Do NOT attempt to decide damage severity or body color in this task.\n\n"
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

    # Extract JSON text from the *message* item (not any reasoning)
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

    # battery info: all images where a battery screenshot + pct exists
    battery_hits = [
        (img.get("image_index"), img.get("battery_health_pct_img"))
        for img in imgs
        if img.get("battery_screenshot") and img.get("battery_health_pct_img") is not None
    ]
    battery_str = "none"
    if battery_hits:
        idx, pct = battery_hits[0]
        battery_str = f"{pct}%@img{idx}"

    # accessory summary
    has_box_any = any(img.get("has_box") for img in imgs)
    has_charger_any = any(img.get("has_charger") for img in imgs)
    has_brick_any = any(img.get("has_charger_brick") for img in imgs)
    has_earbuds_any = any(img.get("has_earbuds") for img in imgs)
    total_cases = sum((img.get("case_count") or 0) for img in imgs)

    box_str = "Y" if has_box_any else "N"
    chg_str = "Y" if has_charger_any else "N"
    brick_str = "Y" if has_brick_any else "N"
    earbuds_str = "Y" if has_earbuds_any else "N"

    # token usage
    usage = data.get("_usage") or {}
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    tot_tok = usage.get("total_tokens")
    if tot_tok is not None:
        tokens_str = f"{tot_tok} (in={in_tok}, out={out_tok})"
    else:
        tokens_str = "unknown"

    return (
        f"listing_id: {listing_id} "
        f"imgs={n_images} "
        f"battery={battery_str} "
        f"box={box_str} "
        f"charger={chg_str} "
        f"brick={brick_str} "
        f"earbuds={earbuds_str} "
        f"cases={total_cases} "
        f"tokens={tokens_str} "
        f"inserted={inserted_count}"
    )


# -------------------------------------------------------------------
# INSERT + MARK ACCESSORIES_DONE
# -------------------------------------------------------------------


def _sanitize_battery_pct(raw_value: Any) -> Optional[int]:
    """
    Sanitize battery_health_pct_img from LLM:

    - Accept only numeric values in a safe % range (e.g. 50–100).
    - Anything else returns None so we NEVER violate DB check constraints.
    """
    if raw_value is None:
        return None
    try:
        pct = float(raw_value)
    except (TypeError, ValueError):
        return None
    # Adjust this range to match your CHECK constraint if needed.
    if 50.0 <= pct <= 100.0:
        return int(round(pct))
    return None


def insert_features_from_json(
    gen: int, listing_id: int, data: Dict[str, Any]
) -> int:
    """
    Insert/update rows in ml.iphone_image_features_v1 from LLM JSON.

    IMPORTANT:
    - This script ONLY writes accessory/battery/box_state fields.
    - It NEVER touches damage-related or color columns, so it is safe to run
      after image_damage_analysis.py and before/after a separate color script.
    """
    images = data.get("images") or []
    if not images:
        print(f"[WARN] No images array in JSON for listing_id: {listing_id}")
        return 0

    rows: List[Tuple[Any, ...]] = []
    for img in images:
        image_index = img["image_index"]

        has_box = img.get("has_box")
        has_charger_brick = img.get("has_charger_brick")
        has_cable = img.get("has_cable")
        has_charger = img.get("has_charger")
        has_earbuds = img.get("has_earbuds")
        has_case = img.get("has_case")
        has_screen_guard = img.get("has_screen_guard")
        has_other_accessory = img.get("has_other_accessory")
        has_receipt = img.get("has_receipt")
        case_count = img.get("case_count")
        battery_screenshot = img.get("battery_screenshot")
        battery_health_pct_img = _sanitize_battery_pct(img.get("battery_health_pct_img"))
        box_state_level = img.get("box_state_level")

        rows.append(
            (
                gen,
                listing_id,
                image_index,
                FEATURE_VERSION,
                has_box,
                has_charger_brick,
                has_cable,
                has_charger,
                has_earbuds,
                has_case,
                has_screen_guard,
                has_other_accessory,
                has_receipt,
                case_count,
                battery_screenshot,
                battery_health_pct_img,
                box_state_level,
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
                has_box,
                has_charger_brick,
                has_cable,
                has_charger,
                has_earbuds,
                has_case,
                has_screen_guard,
                has_other_accessory,
                has_receipt,
                case_count,
                battery_screenshot,
                battery_health_pct_img,
                box_state_level
            )
            VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s
            )
            ON CONFLICT (generation, listing_id, image_index, feature_version) DO UPDATE
            SET has_box                = EXCLUDED.has_box,
                has_charger_brick      = EXCLUDED.has_charger_brick,
                has_cable              = EXCLUDED.has_cable,
                has_charger            = EXCLUDED.has_charger,
                has_earbuds            = EXCLUDED.has_earbuds,
                has_case               = EXCLUDED.has_case,
                has_screen_guard       = EXCLUDED.has_screen_guard,
                has_other_accessory    = EXCLUDED.has_other_accessory,
                has_receipt            = EXCLUDED.has_receipt,
                case_count             = EXCLUDED.case_count,
                battery_screenshot     = EXCLUDED.battery_screenshot,
                battery_health_pct_img = EXCLUDED.battery_health_pct_img,
                box_state_level        = EXCLUDED.box_state_level,
                created_at             = LEAST(ml.iphone_image_features_v1.created_at, now())
            ;
            """,
            rows,
        )

    with_db_cursor(_run)
    return len(images)


def mark_accessories_done(gen: int, listing_id: int) -> None:
    """
    Mark accessories_done/accessories_done_at in ml.iphone_image_features_v1
    for this (generation, listing_id). This is the processed marker so we don't
    run the accessories script twice on the same listing.
    """

    def _run(conn, cur):
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET accessories_done    = TRUE,
                accessories_done_at = now()
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
        description="Batch ACCESSORY image analysis for iPhone listings using GPT-5 nano."
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
    print(f"[CONFIG] PG_DSN set          = {bool(PG_DSN)}")
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
            print(f"[SKIP] gen={gen} listing_id: {listing_id} → No disk images found.")
            continue

        print(
            f"[RUN] gen={gen} listing_id: {listing_id} → {len(imgs)} images to analyze using model={model_name}"
        )

        try:
            data = call_model_for_listing(gen, listing_id, imgs, model_name)
            inserted_count = insert_features_from_json(gen, listing_id, data)

            if inserted_count > 0:
                mark_accessories_done(gen, listing_id)

            total_listings_processed += 1
            total_images_processed += inserted_count

            summary_line = summarize_listing(gen, listing_id, data, inserted_count)
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
            print(f"[ERR] gen={gen} listing_id: {listing_id} → {e}")

    print("===================================================")
    print(f"[SUMMARY] listings_processed = {total_listings_processed}")
    print(f"[SUMMARY] images_processed   = {total_images_processed}")
    print("===================================================")

    close_db_conn()


if __name__ == "__main__":
    main()

