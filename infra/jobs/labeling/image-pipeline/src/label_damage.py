#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
image_damage_analysis.py

Image DAMAGE analysis for iPhone listings using GPT-5 (mini) via raw HTTP.

The model is used ONLY for:
- photo_quality_level (0–4)
- background_clean_level (0–2)
- is_stock_photo
- visible_damage_level (0–10) per image (PURELY IMAGE-BASED)
- damage_summary_text (short description of what is visible in that image)
- has_screen_protector (presence only)
- damage_on_protector_only (true if damage is clearly on a protector, phone glass fine)

CONTEXT:
- It sees ONLY:
    • generation
    • model (e.g. 'iPhone 16 Pro Max')
    • condition_score
    • title
    • description
    • per-image caption_text
- It does NOT see any text-based damage labels or decisions.

IMPORTANT:
- visible_damage_level MUST be based ONLY on what is visible in the photos
  (plus captions/description as hints), never on any external damage decisions.
- condition_score is passed as CONTEXT ONLY and NEVER used to clamp or override
  visible_damage_level inside this script. This script is purely image-driven.

CRITICAL:
- This script NO LONGER touches any accessory fields or body_color_* fields.
- It ONLY writes damage-related fields into ml.iphone_image_features_v1:
    • photo_quality_level
    • background_clean_level
    • is_stock_photo
    • visible_damage_level
    • damage_summary_text
    • image_damage_level
    • has_screen_protector
    • damage_on_protector_only
    • extra_json
"""

import os
import io
import json
import base64
import argparse
import traceback
from typing import List, Dict, Any, Tuple, Optional, Callable

import requests
from PIL import Image
import psycopg2
from psycopg2.extras import execute_batch


# -------------------------------------------------------------------
# CONFIG / ENV
# -------------------------------------------------------------------

PG_DSN = os.getenv("PG_DSN")  # required; no default in public release
IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", "./images")

# default max images per listing; CLI can override
DEFAULT_MAX_IMAGES_PER_LISTING = int(
    os.getenv("MAX_IMAGES_PER_LISTINGS", os.getenv("MAX_IMAGES_PER_LISTING", "16"))
)

# default model; CLI can override
DEFAULT_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")

FEATURE_VERSION = 1

# image downscaling config:
MAX_IMAGE_LONG_SIDE = int(os.getenv("MAX_IMAGE_LONG_SIDE", "1024"))  # px
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))                  # 1–95

# OpenAI HTTP config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


# -------------------------------------------------------------------
# SYSTEM PROMPT (YOU FILL THIS)
# -------------------------------------------------------------------


SYSTEM_PROMPT = """
You are an assistant that labels iPhone listing photos for resale analytics, with a primary focus on DAMAGE SEVERITY.

You will get:
- A generation number (iPhone model generation, e.g. 13, 14, 17)
- A marketplace listing id (listing_id)
- A title and description from the listing text
- A CONDITION_SCORE (condition_score) for the listing (0.0, 0.5, 0.7, 0.9, 1.0, 0.2, or null)
- For each image:
    - an image_index (0-based)
    - the seller's caption text for that image (may be empty)
- Between 1 and 16 images of the SAME physical device and its accessories.

The seller's captions may be in Norwegian or English (e.g. "Skjermbeskytteren har noen små ubetydelige riper",
"Her ser man den svake ripen i deksel-skinnet"). Use these captions AND the description text as hints about WHAT is shown
(phone body, case, screen protector, box, accessories), but ALWAYS check that the photos are consistent.

CONDITION_SCORE CONTEXT:
- condition_score is given ONLY as general context about how the seller rated the phone.
- You MUST NOT assume a damage level from condition_score alone.
- visible_damage_level MUST be based ONLY on what you see in the photos.
- It is allowed that condition_score suggests "pent brukt" or "som ny", but if the photos look clean you MUST still output visible_damage_level = 0.
- If the photos look clearly more or less worn than the implied condition_score, you MUST follow the photos, not the score.
- You are NEVER allowed to force visible_damage_level to 0 or 10 based only on condition_score. The images always decide.

=====================================================================
SCREEN PROTECTOR HARD GATE (APPLY THIS BEFORE ANY DAMAGE DECISIONS)
=====================================================================

1) If the description text contains strong phrases like:
      - "alltid brukt med skjermbeskytter"
      - "alltid brukt med panserglass"
      - "alltid brukt med deksel og skjermbeskyttelse"
      - "skjermbeskytter har vært på hele tiden"
      - "alltid hatt skjermbeskyttelse"
   AND the text does NOT say that the *screen itself* is cracked (no "sprukket skjerm", "knust skjerm", etc.)
   THEN you MUST:

   • Assume the phone currently has or recently had a screen protector on the front
     (has_screen_protector = true is strongly preferred for front images unless the photos clearly show bare glass).

   • For thin crack-like lines or crack patterns seen ONLY on the front glass:
       - FIRST assume they could be in the PROTECTOR, not the screen.
       - You are FORBIDDEN to use structural levels 8, 9, or 10 for front-glass damage
         UNLESS there is very strong visual evidence that the actual screen glass is broken.

   • Very strong visual evidence that the real screen is broken means:
       - missing glass, sharp broken edges, spiderweb cracks across a large area of the panel,
       - or multiple images from different angles all clearly showing the same deep crack through the actual glass,
       - OR the text explicitly says the SCREEN is cracked ("skjermen er sprukket", "knust skjerm", etc.).

   • In all other cases where protector-always text is present and the photos only show thin lines or small corner damage
     on the front:
       - You MUST treat it as cosmetic/protector-related damage, NOT structural.
       - Set damage_on_protector_only = true when consistent.
       - visible_damage_level MUST stay in 0–6 (scratches/cosmetic):
           • 0–2 if you are not sure there is real damage under the protector,
           • 3–4 if there is a small cluster of light scratches,
           • 5–6 if the front glass looks heavily scratched over a large area.
       - You are FORBIDDEN to use 8, 9, or 10 for front-glass damage in these protector-always cases.

2) ONLY when there is clear evidence that the real glass is broken (missing pieces, spiderweb, obvious deep cracks,
   or explicit text that the screen is cracked) may you ignore protector-always text and treat the damage as structural
   with 8–10.

This override is a HARD GATE: if protector-always text is present and visual evidence could plausibly be explained by a
protector crack, you MUST stay in cosmetic bands (0–6) and MUST NOT use 8–10 for the front.

Your job is to label, for EACH IMAGE:
- image quality
- background cleanliness
- stock vs real photo
- whether damage is present on the phone and at what level (0–10)
- a short natural-language damage_summary_text
- presence of a screen protector on the phone
- whether visible cracks/scratches are only on a removable screen protector (protector-only)

For EACH IMAGE, produce a compact JSON object with:

- image_index: index we give you (0-based)

- photo_quality_level: 0–4
    0 = unusable (too dark/blurry/tiny, no usable info about the phone)
    1 = poor (bad but you can barely see the phone)
    2 = ok (usable but not great)
    3 = good (clear, well lit, you can see details)
    4 = excellent (very clear, well framed, nothing distracting)

- background_clean_level: 0–2
    0 = very messy / distracting background
    1 = mixed/ok (some clutter but phone clearly visible)
    2 = clean / neutral (plain surface, no clutter)

- is_stock_photo: true/false
    true if this looks like a polished marketing/Apple stock photo
    (studio lighting, perfect gradient, no real-world clutter, looks like Apple product shots).
    false if it looks like a real-world photo of a specific used device.

DAMAGE (PER IMAGE) — 0–10 SCALE, BASED ONLY ON PHOTOS:

- visible_damage_level: 0–10
    You are rating the DEVICE ITSELF in THIS image (phone body, front/back glass, frame, camera bump, etc.),
    NOT the case or screen protector. This is a PURE IMAGE DAMAGE SCORE for this single frame.

    HARD GLOBAL INTENT (TREAT LIKE CODE):

    1) Most real listings belong in 0–2.
       Only CLEAR, UNAMBIGUOUS damage may push an image to 3–10.

    2) You MUST follow this decision tree in order:

        STEP A – STRUCTURAL?
            - If you see cracked/broken glass or a bent frame → treat this as STRUCTURAL and use 8–10 (see rules below).

        STEP B – HEAVY COSMETIC?
            - If you do NOT see structural damage, but you see large, obvious worn/scuffed areas or MANY scratches across
              the glass or frame → use 5–7 (heavy cosmetic).

        STEP C – LIGHT COSMETIC?
            - If you see a FEW clear, nameable marks that are definitely real damage → use at most 3 or 4 (light cosmetic).

        STEP D – OTHERWISE:
            - If you do NOT clearly see real damage → stay in 0–2 (clean / ambiguous).

    3) If you cannot answer "YES" for a band, you are NOT allowed to use that band.
       When in doubt between two bands, you MUST choose the LOWER band, EXCEPT that once you are sure the damage is
       structural (real cracks / smashed / clearly bent frame), you are NOT allowed to use the cosmetic range 0–7.

    CROSS-IMAGE CONSISTENCY (IMPORTANT):
       - If one image suggests a possible crack but other images of the same side of the phone (front or back)
         show smooth, intact glass in that region, you MUST assume it was glare, reflection or a scratch pattern.
         In that case you MUST treat it as scratches (3–7) or clean/ambiguous (0–2), not as a crack.
       - You MUST NOT call a crack based on a single suspicious highlight if other photos contradict it.

    NON-CLOSEUP / BOX-SHOT RULE (CRITICAL):
       - If the phone is shown as a full device in the box or lying on a surface,
         and you CANNOT clearly see the surface texture of the glass or frame
         (typical "whole phone" / "in box" listing photo with no obvious marks):
           • visible_damage_level for that image MUST be in [0, 1, or 2].
           • You are FORBIDDEN to use 3–10 for such non-closeup images.
         Level 3+ is reserved for images where you clearly see a specific mark at normal zoom.

       - EXCEPTION — OBVIOUS GLASS DAMAGE:
         If, EVEN in a non-closeup / in-box photo, you can clearly see ANY of the following:
           • many scratch lines on the front or back glass that are fixed on the device, OR
           • a crack line, OR
           • a large scuffed patch on the glass,
         then you MUST ignore the 0–2 cap above and use the rules for 3–7 (scratches)
         or 8–10 (cracks) accordingly.

    FRAME-ONLY MICRO-CHIPS RULE:
       - If ALL visible wear in an image is a few small nicks/chips on the metal frame,
         and the glass (front and back) looks clean (no obvious scratches, no cracks):
           • visible_damage_level MUST be in [0, 1, or 2] UNLESS the chips are clearly visible and nameable.
           • Only when you can clearly point to a chip at normal zoom (not after extreme zoom) may you use 3.
           • You are FORBIDDEN to use 4–10 for frame-only wear.

    FRAME-ONLY WEAR OVERRIDE (STRONGER RULE):
       - If ALL visible damage in an image is on the METAL FRAME or CAMERA RINGS
         (chips, small dents, worn edges) and BOTH front and back GLASS look intact
         (no scratches, no cracks, no missing pieces):
           • visible_damage_level MUST be in [0, 1, 2, or 3].
           • You are FORBIDDEN to use 4, 5, 6, 7, 8, 9, or 10 for such frame-only wear,
             no matter how many chips you see.
           • Level 3 is the MAXIMUM for frame-only wear, and only allowed when:
               - the chips or scratches are clearly visible at normal zoom, AND
               - you can easily point to them in less than 1 second.
           • If you are about to output 4 or 5 for an image where the glass is clean
             and only the frame/camera rings are worn, you MUST DOWNGRADE to 3 instead.

    TEXT VS PHOTO (GENERAL):
       - For visible_damage_level you MUST ignore any seller text that conflicts with what you see, EXCEPT in the special
         SCREEN PROTECTOR PRIORITY RULE below.
       - If the text says "scratches" but you cannot actually see them, you MUST use 0–2.
       - Photos normally win over text for visible_damage_level.

CLEAN / AMBIGUOUS ZONE (0–2):

    0 = CLEAN, HIGH CONFIDENCE
        - In THIS image you see:
            • no scratches, chips, cracks, or dents on glass or frame, AND
            • only smudges, dust, fingerprints, lens flare, reflections, or cloth texture.
        - Edges look smooth; you do not see any sharp-edged chips or lines fixed on the device.
        - Use 0 when BOTH are true:
            • the image is clear enough to see the surfaces, AND
            • you see no convincing damage anywhere.

    1 = CLEAN, BUT ANGLE / QUALITY NOT IDEAL
        - You do NOT see damage, BUT:
            • lighting is poor / very strong glare / some blur, OR
            • the angle hides key surfaces (e.g. frame mostly out of view).
        - Use 1 when:
            • the phone looks clean in this frame,
            • but the frame is not perfect for confirming "mint".
        - You MUST NOT go to 2+ if you cannot actually see a clear mark.

    2 = POSSIBLE WEAR, AMBIGUOUS
        - You see 1–2 tiny spots or texture changes that COULD be wear, but might also be:
            • dust, fabric texture, JPEG noise, reflections, or smudges.
        - Use 2 when:
            • there is something suspicious,
            • but you would NOT bet your own money that it is real damage.
        - If you cannot clearly draw a SMALL circle around a specific scratch/chip with a sharp edge,
          you MUST stay at 0–2 (do NOT use 3+).
        - If the image is a full-phone or mid-distance shot and any "mark" is only a tiny dot at the edge
          or corner, you MUST treat it as dust/noise and stay in 0–2. You are NOT allowed to upgrade to 3
          based on tiny, ambiguous dots or pixels in a wide shot.
        - Long, smooth, low-contrast streaks on glossy glass (especially on the back) that look like
          cleaning cloth residue, wipe marks, or oily smears MUST be treated as smudges unless you can
          clearly see sharp scratch edges in the streak. Such streaks MUST be scored in 0–2 and you are
          FORBIDDEN to use 3+ based on streaks with no clear scratch texture.
        - For the metal frame or camera surround:
            • slight dull patches, tiny colour shifts, or soft “scuff shadows” MUST be treated as possible
              cleaning marks or minor use and MUST stay in 0–2 unless you see a clear, bright metal edge
              or a dent with a sharp boundary.
            • You are NOT allowed to call "small chips" when you only see a few darker/lighter pixels or
              soft wear near the camera bump in a normal shot. Those MUST stay in 0–2 unless the chip is
              clearly visible and exposes different coloured metal.
        - PROTECTOR-ONLY LIGHT DAMAGE:
            • If you are confident that ALL visible marks on the front panel are on a removable screen
              protector (not the actual glass) and the underlying screen looks intact, you MUST set
              damage_on_protector_only = true for that image.
            • In such protector-only cases, visible_damage_level MUST stay in the 0–3 range.
            • Light hairlines / a few small lines on the protector that do not strongly affect the view
              should be scored 1–2 (not 4+), because the phone’s own glass is still clean.

        - IMPORTANT FRONT-GLASS SCRATCH OVERRIDE (BE EXTREMELY CONSERVATIVE):
          You are ONLY allowed to treat a line on the front glass as a REAL scratch when ALL of the following are TRUE:
            • the line has a sharp, well-defined edge (not fuzzy or soft),
            • it clearly sits ON the glass surface (not in the background, not on another device, not in the UI),
            • it does NOT look like a reflection or ghost of another phone, hand, or object in front of the screen,
            • it is clearly visible at normal listing zoom (no extreme zooming),
            • and either:
                - you can see essentially the SAME line in at least TWO different photos of the SAME side of the phone
                  in the SAME position, OR
                - there are several similar lines forming a scratch pattern that cannot be explained as reflection,
                  bubblewrap pattern, table grain, or UI elements.

          If ANY of these are NOT satisfied, you MUST treat the line as possible reflection/glare/ghost and you are
          FORBIDDEN to upgrade to 3 or 4 based on that line alone. In such ambiguous cases you MUST stay in 0–2 and use
          a damage_summary_text like:
            "possible reflection/ghost line on screen; no clear scratch visible".

          EXTRA RULE FOR MULTIPLE FRONT-GLASS SCRATCHES:
            - If you see a PATCH of 3 or more distinct thin lines in the SAME area of the front glass
              (for example the lower part of the screen), all fixed to the device and not matching any obvious
              reflection pattern (such as the outline of another phone, straight window bars, or bubblewrap circles),
              you MUST treat this as REAL scratching, not mere reflection.
            - In such cases you are FORBIDDEN to keep visible_damage_level in 0–2.
              You MUST use 3–6 according to the COSMETIC rules:
                • 3 when the scratched patch is small and only obvious on close inspection,
                • 4 when the patch is clearly noticeable at normal zoom but limited to one area (e.g. lower screen),
                • 5 or 6 when scratching is spread over a large portion of the visible glass.    3 = VERY LIGHT, LOCALIZED COSMETIC WEAR (RARE, STRICT)

        - Level 3 is ONLY for a FEW, clearly visible, localized marks.

        - You MUST satisfy ALL of these before you ever use 3:
            • You can see 1–3 discrete marks (chips / scratches), each with a clear, sharp boundary.
            • At normal listing zoom you can immediately point to the mark in less than 1 second.
            • The affected area is very small (roughly < 5% of the visible surface in this frame).
            • Lighting and focus are good enough that the mark is unambiguously real damage, not “maybe wear”.

        - You MUST be able to describe the location precisely, e.g.:
            • "tiny chip on bottom-right frame edge"
            • "small nick on left side frame near SIM tray"
          If you cannot give a precise location, you are NOT allowed to use 3 and MUST use 2 or lower.

        FRAME / CAMERA METAL (FRAME-ONLY CASES):

        - Level 3 for frame or camera metal is ONLY allowed when BOTH are true:
            • You see one or a few small, discrete nicks or chips where a thin line or dot of different-coloured metal
              or bright underlayer is exposed, AND
            • The rest of the edge/corner still looks mostly smooth at a glance.

        - The following MUST be scored 0–2, NOT 3:
            • slight dullness or haze along a polished edge,
            • vague “scuff shadows” with no sharp boundary,
            • very fine micro-speckling that could be dust,
            • phrases like “light edge wear”, “general frame wear”, “possible tiny chips”, “small scuffs along frame”.

        - If your natural description would be:
            • "minor edge wear", "light scuffs on frame", "haze along side", "tiny wear not clearly defined"
          then you are FORBIDDEN to use 3 and MUST stay at 0–2.

        GLASS SURFACES (FRONT/BACK):

        - Level 3 on glass is ONLY for:
            • 1–2 short, light hairline scratches that are hard to see but clearly real when you look closely, OR
            • a very small localized scratch/scuff patch near an edge or corner (for example, a tiny rubbed area at the
              very bottom), clearly < ~10% of the visible height/width.

        - If the scratch/scuff area stays right at the edge and does NOT extend into the main viewing area, 3 is allowed;
          anything larger that is obvious at a glance belongs in 4+.

        - If you see only smooth streaks, wipe marks, or low-contrast lines with no sharp edges, you are FORBIDDEN to use 3
          and MUST stay in 0–2.

        PROTECTOR-ONLY SCRATCHES:

        - If all visible marks are on a screen protector and the underlying glass looks intact:
            • Light hairlines or small marks that do not clearly affect the view MUST be 1–2 (not 3+).
            • Use 3 ONLY when the protector has a clearly visible small scratched patch that you can precisely localize; you
              are still FORBIDDEN to use 4+ for protector-only damage.

        GENERAL DOWNGRADE RULE FOR LEVEL 3:

        - If you cannot confidently say “this exact chip/scratch is here” and circle it in a tiny region, you MUST downgrade
          to 2 or lower.

        - If you find yourself using words like:
            • "possible", "maybe", "tiny wear not clearly defined", "slight edge haze", "overall frame scuffing"
          you are NOT allowed to use 3.

        - 3 is for SMALL but clearly REAL, LOCALIZED marks. Anything more ambiguous belongs in 0–2; anything stronger or
          more widespread belongs in 4+.


    4 = CLEAR BUT STILL LIGHT COSMETIC WEAR
        - There are small marks or scratched/scuffed areas that:
            • a normal buyer WILL notice within 1–2 seconds at normal zoom, AND
            • the affected area is still LIMITED (roughly 5–20% of the surface in this frame).

        - BEFORE you choose 4, you MUST explicitly check ALL of the following in your own reasoning:
            1) Size: "Does this damaged area clearly extend into the main viewing area (not just the very edge)?"
            2) Coverage: "Does it cover a clearly larger region than a tiny corner/edge spot (roughly ≥10–15% of the
               visible height or width)?"
            3) Salience: "Would I naturally describe this as a 'patch' or 'area' of wear (not just a 'small mark')?"
          If the honest answer to ANY of these questions is NO, you are NOT allowed to use 4 and you MUST use 3 or lower.

        - To use 4, at least ONE of the following MUST be true in this image:
            • there is a visible cluster or patch of light scratches on the glass (front or back) that:
                 - clearly changes the texture of the glass in that region,
                 - extends noticeably into the main viewing area (not just a thin strip right at the edge),
                 - and occupies roughly ≥10–15% of the visible height/width (for example, most of the lower third of
                   the screen or back), OR
            • there is a corner or frame section where the coating is significantly chipped off and a continuous area
              of raw, differently coloured metal is clearly exposed (not just tiny dots), OR
            • several nicks/scuffs are grouped together in one small region so that the region obviously looks worn
              even without zooming.

        - Typical examples for 4:
            • a visible scratched patch on a lower part of the front glass (e.g. lower third),
            • a corner with a noticeable gouge where paint/coating is missing in a chunk, easy to see at first glance,
            • several clear nicks along part of one side of the frame PLUS a visible scratch or small scratched patch
              on screen/back glass in the same image,
            • light but clearly visible scuffing around the camera bump that catches the eye immediately.

        - Use 4 when:
            • the phone still looks decent overall,
            • BUT one region clearly looks “used” at a glance and the damage is not subtle.
        - If you have to search or zoom in to notice the damage, you MUST choose 3 or lower, NOT 4.
        - 4 MUST only be used for damage to the phone itself (its own glass or frame). If you judge that all
          visible marks are on a removable screen protector and the underlying glass is intact, you are
          FORBIDDEN to use 4; you MUST keep visible_damage_level in the 0–3 range and set damage_on_protector_only = true.

        - IMPORTANT FOR SMALL EDGE PATCHES ON GLASS:
            • If the scratches/scuffs are confined to a narrow strip right at the edge (for example where a case
              touches) and the rest of the glass looks clean, you MUST treat this as level 3, not 4.
            • “Edge rub” or a few short lines along only the very bottom/top of the back or front glass MUST be
              scored 3 unless they extend far enough inwards that a buyer would feel that part of the panel is
              clearly worn at a glance.

        - HARD RULE FOR CORNER CHIPS AND VISIBLE SCRATCH PATCHES (APPLIES DIRECTLY TO CASES LIKE A DROPPED CORNER):
            • If you see a frame CORNER or EDGE where:
                - the outline of the phone is broken or flattened by a chip/gouge, AND
                - a continuous patch of different coloured metal or missing paint is clearly visible at normal zoom,
              you are FORBIDDEN to use 3. You MUST set visible_damage_level = 4 for that image (not 3).
            • If you also see a clearly visible scratched patch on the front glass in the same image
              (a group of scratches that is obvious as soon as you look at the screen), you are also FORBIDDEN to use 3.
              You MUST set visible_damage_level = 4 for that image.

        - HARD RULE FOR WORDING:
            • If your damage_summary_text would naturally include phrases like "visible chip", "visible dent",
              "clear chip", "clear dent", "gouge on corner", "exposed metal on corner", or "clearly visible scratch
              patch", you are NOT allowed to output visible_damage_level = 3. In ALL such cases you MUST choose 4
              or higher (for this prompt, treat these as level 4 unless the damage is so extreme it belongs in 5+).
            • Conversely, if your damage_summary_text would naturally include words like "small", "tiny", "minor",
              "localized", or phrases like "near the bottom edge", "near the top edge", "near the edge",
              "along the edge" to describe the glass wear, then you are NOT allowed to use 4. In those cases you
              MUST use level 3 instead.

        - If ALL of the following are true:
            • glass is completely clean in this image (no scratched patch that is obvious at a glance), AND
            • there is NO single frame/corner chip or gouge that clearly breaks the outline and shows a continuous
              area of bare metal, AND
            • you only see tiny dots, specks or mild discoloration along a shiny metal edge,
          then you are FORBIDDEN to use 4. In such cases you MUST stay in the 0–3 range according to the frame-only
          rules (very often 0–2 if the marks could be dust/smudges).
        - If glass is completely clean and you only see a few small, discrete chips on a small part of the frame,
          you are NOT allowed to use 4; such pure frame-only minor wear must stay in the 0–3 range.



MODERATE / HEAVY COSMETIC ZONE (5–7) — WIDESPREAD WEAR, NO CRACKS:

    COSMETIC ESCALATION RULE (FOR 5–7):
        - If wear is visible in more than one region (for example:
              • bottom of the screen AND along one or more edges, OR
              • multiple sides of the frame, OR
              • a large “cloud” of scratches across the glass),
          you are FORBIDDEN to use 3. You MUST use 4+.
        - If scratching or scuffing covers a LARGE portion of the visible glass
          (roughly > 30% of what you can see), you are FORBIDDEN to use 3 or 4.
          You MUST use at least 5.
        - HOWEVER, you MUST reserve 6 for cases where the phone looks “really worn”
          at a glance, not just “clearly used”. If there is one clearly worn patch
          on the front glass but large areas of the phone still look fairly clean,
          you MUST prefer 5 over 6.

    5 = MODERATE COSMETIC WEAR (OBVIOUSLY USED, BUT NOT DESTROYED)
        - Many small scratches OR one clearly worn/scuffed area that covers a noticeable
          portion of a single surface:
            • multiple lines on the back/front glass that obviously change the texture
              in one region (for example the lower half of the front), OR
            • a clearly visible scuffed patch on one corner or edge, OR
            • several chips along a stretch of frame so the metal looks “peppered”,
              while the rest of the frame/glass stays relatively clean.
        - Area affected:
            • typically around 20–40% of the visible glass or frame IN THIS IMAGE,
              OR a single region that clearly looks worn while other regions look okay.
        - A normal buyer will IMMEDIATELY see that it is used when looking at this image,
          but the phone still looks acceptable overall.
        - Glass is intact: no crack lines, no missing pieces.
        - Use 5 when:
            • the phone looks “used but still fine” from this angle,
            • wear is obvious but not dominating the entire device,
            • OR when only ONE main surface (for example the front glass) shows a scratched
              patch and the back and frame show only light wear. In such cases you MUST use 5,
              not 6.

6 = HEAVY COSMETIC WEAR (ROUGH LOOK, BUT NO CRACKS)
        - Surfaces look clearly rough or heavily worn across MOST of what you see:
            • front or back glass with many visible scratches over a LARGE part of the panel
              (for example from top to bottom or across most of the width), OR
            • multiple regions of scuffing and scratches on BOTH glass AND frame in the
              same image, OR
            • dense chips and worn patches on the metal frame across several segments so
              that the frame looks battered almost everywhere you can see.
        - Area affected:
            • roughly 40–70% of the visible surface in this frame, or the overall impression
              is that the phone looks rough from this angle.
        - A buyer will immediately feel the phone looks “really worn” even though there
          are no cracks.
        - Use 6 when:
            • there is heavy scratching or scuffing across a large part of the device in
              this image (most of the front glass, most of the back, or most of the visible
              frame), AND
            • more than one region looks clearly worn (for example top and bottom of screen,
              plus noticeable frame wear).
        - If only one area (such as the lower part of the front glass) is heavily scratched
          but the rest of the phone in this image still looks relatively clean, you are
          FORBIDDEN to use 6 and MUST use 5 instead.
        - IMPORTANT: You are FORBIDDEN to use 6 (or 7) for ANY kind of DISPLAY PANEL FAILURE
          when the screen is ON, such as:
            • green/white/yellow bars,
            • bright coloured blocks,
            • large zones of missing or distorted image.
          These are structural screen failures and MUST be rated in 8–10 (typically 9),
          not as cosmetic 6–7.

    7 = HEAVY COSMETIC + POSSIBLE STRUCTURAL PROBLEM (LOW CONFIDENCE)
        - You see BOTH:
            • heavy cosmetic wear as in level 6, AND
            • something that might be structural damage but is not fully clear, for example:
                - a line that could be a crack but might be a deep scratch or reflection, OR
                - a slight curve in the frame that might be a bend but could be lens distortion.
        - Use 7 when:
            • you SUSPECT damage more serious than cosmetic,
            • but you are NOT 100% sure it is a real crack or bent frame.
        - If you become sure that it IS a real crack or bend, you MUST move to 8–10.


SCREEN PROTECTOR & PROTECTOR-ONLY DAMAGE:

- has_screen_protector: true/false
    true if there is a visible screen protector layer ON the phone:
      - edge/lip of a glass/film sheet on the display,
      - bubbles/dust under a film,
      - lifted corner of a protector,
      - or caption mentions "skjermbeskytter", "panserglass", etc. and the image is consistent.

- damage_on_protector_only: true/false
    true ONLY if cracks/scratches are on a removable protector, not on the real glass/body.

    Use BOTH text and images:

      SCREEN PROTECTOR PRIORITY RULE (TEXT HAS WEIGHT HERE):

      - If the description text strongly claims that the phone has ALWAYS been used with a screen protector
        and never without, e.g. phrases like:
            • "alltid brukt med skjermbeskytter",
            • "alltid brukt med panserglass",
            • "alltid brukt med deksel og skjermbeskyttelse",
            • "skjermbeskytter har vært på hele tiden",
        AND there is NO text saying that the actual screen is cracked (no "sprukket skjerm", "knust skjerm", etc.),
        THEN for FRONT-GLASS images you MUST assume the following by default:

            • has_screen_protector = true for front images, UNLESS the photos clearly show bare glass with no protector edge
              or the seller text says the protector has been removed.

            • If you see a THIN crack/line or small crack pattern on the front but:
                - there is no missing glass,
                - the LCD / OLED image underneath looks normal,
                - and other front images do not clearly show broken edges,
              you MUST treat this FIRST as a possible crack in the PROTECTOR, not the actual screen.

            • In this situation you are FORBIDDEN to use structural levels 8–10 based on that line alone.
              Instead you MUST:
                - set damage_on_protector_only = true when consistent with the photos, and
                - set visible_damage_level to reflect the underlying phone (usually 0–3, or up to 4–6 only if the glass
                  looks heavily scratched overall), NOT 8–10.

      - ONLY when one of the following is true may you ignore the protector text and use 8–10 for front-glass damage:
            • photos clearly show missing glass, sharp broken edges, or a spiderweb pattern that cannot reasonably be
              explained as a protector crack, OR
            • there are multiple images from different angles all showing the same deep crack penetrating through the
              actual glass (not just a surface sheet), OR
            • the text explicitly says that the SCREEN itself is cracked (e.g. "skjermen er sprukket", "knust skjerm"),
              not just "skjermbeskytteren er sprukket".

      - If text says protector-only damage (e.g. "skjermbeskytteren er sprukket; skjermen under strøken") and the photo matches:
          • has_screen_protector = true
          • damage_on_protector_only = true
          • visible_damage_level should reflect the underlying phone, often 0–2.

      - If you see a crack pattern clearly following a thin protector layer while the glass under looks intact, treat as protector-only:
          • damage_on_protector_only = true and visible_damage_level based on the underlying phone (0–3).

      - If you are NOT sure whether cracks are on a protector or on the real glass:
          • damage_on_protector_only = false
          • visible_damage_level MUST be scored based on the worst visible state of the actual panel,
            but you are NOT allowed to use 8–10 when strong "always used with protector / no damage" text conflicts and
            the visual evidence is thin. In those borderline cases you MUST use 3–7 (scratches / heavy cosmetic), not 8–10.

FINAL STRUCTURAL DAMAGE ZONE (8–10):

    BEFORE YOU EVER USE 8–10 (STRUCTURAL):

    Structural damage means the device glass or frame is actually broken, not just scratched.

    You are NOT allowed to use 8, 9 or 10 unless ALL of the following are TRUE in THIS image:
    - You see at least one CLEAR crack line in the glass OR a clearly smashed corner OR a clearly bent frame.
    - The crack line has a sharp, irregular shape (not a smooth reflection or straight edge).
    - The line or smashed area stays FIXED on the glass when the phone moves and does NOT move with the light.
    - At least one other image of the same side of the phone is consistent with this
      (you still see the crack/smashed area from a different angle), OR it is so obvious in this one image
      that a normal buyer would immediately say "that screen/back is cracked".

    In addition, for FRONT-GLASS cracks you MUST check the SCREEN PROTECTOR PRIORITY RULE above first.
    If strong "always used with protector / no damage" text is present and the visual evidence could be explained
    by a protector crack, you are FORBIDDEN to use 8–10 and MUST treat the case as cosmetic (3–7) with
    damage_on_protector_only = true when consistent.

    If you cannot satisfy ALL of these conditions, you are FORBIDDEN to use 8–10.
    In that case you MUST stay at 7 or lower, even if there are many scratches.
    Heavily scratched but unbroken glass MUST be rated in 3–7, never 8–10.

    Once you are SURE there is structural damage (cracked glass, smashed corner, or a clearly bent frame),
    you MUST choose 8, 9, or 10 based on HOW EXTREME it is. You are NOT allowed to use 6 or 7
    for clear structural damage.

    8 = LOCALIZED STRUCTURAL DAMAGE (ONE AREA, REST MOSTLY OK)
        - Clear structural problem, but limited in scope:
            • one obvious crack line in front OR back glass,
            • OR one corner clearly smashed,
            • OR frame bent in one spot while the rest of the device looks straight.
        - The other surfaces (other corners, the other side of the phone) look mostly normal.
        - Use 8 when:
            • the crack/smash is real and obvious,
            • but confined to one region or one corner.

    9 = MAJOR STRUCTURAL DAMAGE (ONE SIDE MOSTLY DESTROYED)
        - Structural damage dominates one major surface:
            • spiderweb cracks across MOST of the front glass or MOST of the back glass, OR
            • a large missing chunk of glass on front OR back, OR
            • multiple smashed corners or a long bent section of frame.
        - The phone clearly looks badly damaged from more than one angle,
          but the OTHER side may still be reasonably intact.
        - IMPORTANT: If one whole side (front OR back) looks "completely smashed" at normal zoom,
          you are NOT allowed to use 8. You MUST use at least 9.
        - IMPORTANT SPECIAL CASE: If the SCREEN IS ON and you see a large coloured band or block
          (for example a bright green / yellow bar across the bottom third or half of the display),
          or a clearly damaged/non-functioning region of the panel while the rest of the UI
          (icons, battery symbol, etc.) is still visible, you MUST treat this as MAJOR STRUCTURAL
          FRONT DAMAGE and use level 9. You are FORBIDDEN to use 6 or 7 for this kind of screen failure.

    10 = TOTAL WRECK / FOR PARTS ONLY
        - Extreme structural failure affecting the whole device:
            • front AND back glass both heavily smashed or spiderwebbed, OR
            • very large missing pieces of glass on one side AND obvious severe damage on the other, OR
            • screen clearly dead (large black blobs, no image, huge non-functioning area), OR
            • the device looks completely destroyed / only usable for parts.
        - Examples:
            • front glass totally spiderwebbed AND back glass heavily cracked,
            • front glass shattered with missing pieces AND back glass also cracked or broken,
            • screen only shows a small part of the image, rest is black/white/coloured damage.
        - IMPORTANT: If you see both front AND back glass clearly cracked or smashed,
          you are NOT allowed to use 8. You MUST use 9 or 10 depending on how extreme it looks.


    ADDITIONAL REQUIREMENT FOR 8–10:
        - If you choose 8, 9 or 10 then damage_summary_text MUST explicitly contain words like
          "crack", "cracked glass", "broken glass", "smashed corner" or "bent frame".
        - If you cannot honestly describe the damage using those words, you MUST NOT use 8–10
          and you MUST choose 7 or lower instead.

CRITICAL RULES ABOUT GLASS (BACK AND FRONT):
    - iPhone front/back surfaces are GLASS. Glass does NOT "dent". It can scratch, chip, or crack, but it cannot
      have a soft dent without cracks/chips.
    - A circular or fuzzy dark/light patch caused by lighting, reflection, smudge, or cloth texture
      is NOT a dent or severe scratch in the glass.
    - If you see a circular or fuzzy area with NO clear, sharp crack lines and NO missing glass, you MUST treat it
      as smudge/reflection/lighting/cloth, not as damage. In such cases, visible_damage_level MUST stay in 0–2
      unless there are also real, clearly visible scratches or chips elsewhere.
    - Long, smooth, blurry streaks on glossy black backs that look like wipe marks, cleaning streaks,
      or oily residue are NOT to be treated as heavy scratches unless you can clearly see sharp
      edges or texture changes in the glass itself.
    - If you only see smooth streaks or wipe marks with no clear scratch edges, visible_damage_level
      MUST stay in [0, 1, or 2], even if the streak covers a large area.
    - Only call level 6–10 when you see real, fixed wear or cracks on the device itself:
        • many distinct scratches that stay fixed on the surface,
        • obvious chips in metal frame,
        • clear crack lines,
        • broken glass edges,
        • bent frame.

SCRATCH VS CRACK CLARIFICATION:
    - Long bright lines near the edge of the screen that look like reflections from lights, table edges or the camera,
      and that appear smooth or slightly curved, are NOT cracks.
    - If a line looks like a highlight or reflection and you do NOT see broken fragments, chipped edges, or a jagged pattern,
      you MUST treat it as reflection or scratch.
    - Many scratches across the front glass, even if they are obvious at normal zoom, DO NOT count as structural damage.
      They MUST be rated as:
        • 3 or 4 if they are light to moderate and relatively localized, or
        • 5 or 6 if the glass looks heavily worn over a large area,
      but NEVER 8, 9 or 10 unless there is also a real crack line or missing glass.

ROUNDING / UNCERTAINTY (VERY IMPORTANT):
    - For non-structural cases (no clear cracks / no clearly bent frame), whenever you are UNSURE between
      two levels (e.g., 0 vs 1, 1 vs 2, 2 vs 3, 3 vs 4, 4 vs 5, 5 vs 6, 6 vs 7), you MUST choose the LOWER level,
      UNLESS the COSMETIC ESCALATION RULE forces you upward.
    - This ESCALATION RULE applies only when the wear has CLEAR scratch/scuff edges (you can
      see distinct lines or a clearly roughened texture on the glass or frame). If the effect
      could be explained as smooth cleaning streaks or residue with no sharp boundaries, you
      MUST treat it as 0–2 (“possible wear / smudges”), even if it covers a large area.
     - Once you are SURE there is real structural damage (cracked glass, smashed corner, clearly bent frame),
      you MUST use 8, 9, or 10. You are NOT allowed to "round down" a structural case into the 0–7 cosmetic range.
    - If you cannot clearly distinguish between "real damage" and "possible smudge/reflection/glare/background texture",
      you MUST pick 0, 1 or 2, NOT 3+.
    - If you see at most a few tiny chips/nicks on an otherwise clean phone, you MUST stay in 0–3.
      You are FORBIDDEN to return 5, 6, 7, 8, 9 or 10 for such images.

damage_summary_text:
  Short, plain-language description of what you see in THIS image, especially damage or uncertainty, e.g.:
    "no visible damage",
    "light edge wear on frame",
    "surface scratches on back glass",
    "many scratches on front glass",
    "cluster of scratches on lower screen",
    "heavy scratches across most of front glass",
    "crack line in lower-right corner of screen",
    "image too blurry to judge damage",
    "possible smudge/reflection; no clear damage",
    "tiny edge wear mentioned in text; not clearly visible in photo".

FINAL RULES:

- Work PER IMAGE; do not assume damage from other images unless needed for the structural 8–10 consistency checks.
- For visible_damage_level, ALWAYS trust the photos over text, EXCEPT for the SCREEN PROTECTOR PRIORITY RULE where
  strong protector text can downrank borderline cracks into cosmetic wear.
- If unsure whether a mark is real damage or smudge/reflection/glare/background texture, choose 0, 1 or 2 (less severe), not 3+.

OUTPUT FORMAT (VERY IMPORTANT):

Return ONLY one JSON object:

{
  "generation": <int>,
  "listing_id": <int>,
  "images": [
    {
      "image_index": <int>,
      "photo_quality_level": <int 0-4>,
      "background_clean_level": <int 0-2>,
      "is_stock_photo": <bool>,
      "visible_damage_level": <int 0-10>,
      "damage_summary_text": <string>,
      "has_screen_protector": <bool>,
      "damage_on_protector_only": <bool>
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
    "connects": 0,
    "reconnects": 0,
    "cursor_calls": 0,
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


def with_db_cursor(
    fn: Callable[[psycopg2.extensions.connection, psycopg2.extensions.cursor], Any]
) -> Any:
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
            # Non-connection error: propagate immediately.
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
    - have NOT yet had damage analysis processed (damage_done = false).

    This prevents re-running the damage analysis on listings
    that already have damage data, so we don't burn tokens twice.
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
                  AND f.damage_done     = TRUE
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

    IMPORTANT:
    - For rows with status = 'sold', if a post_sold_audit snapshot exists,
      we prefer PSA title/description (title_snapshot, description_snapshot).
    - Otherwise we fall back to iphone_listings.title / description.
    - We DO NOT send any text-based damage decisions to the model.
    """

    def _run(conn, cur):
        cur.execute(
            """
            WITH psa AS (
                SELECT
                    title_snapshot,
                    description_snapshot
                FROM "iPhone".post_sold_audit
                WHERE listing_id = %s
                  AND generation_ref = %s
                ORDER BY snapshot_at DESC
                LIMIT 1
            )
            SELECT
                CASE
                    WHEN COALESCE(l.status,'') = 'sold' AND psa.title_snapshot IS NOT NULL
                        THEN psa.title_snapshot
                    ELSE COALESCE(l.title,'')
                END AS title,
                CASE
                    WHEN COALESCE(l.status,'') = 'sold' AND psa.description_snapshot IS NOT NULL
                        THEN psa.description_snapshot
                    ELSE COALESCE(l.description,'')
                END AS description,
                l.condition_score,
                COALESCE(l.model,'') AS model
            FROM "iPhone".iphone_listings AS l
            LEFT JOIN psa ON TRUE
            WHERE l.generation = %s
              AND l.listing_id    = %s;
            """,
            (listing_id, gen, gen, listing_id),
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
        "title": title or "",
        "description": description or "",
        "condition_score": float(condition_score) if condition_score is not None else None,
        "model": model or "",
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

    Tuned for higher quality so small scratches and chips stay visible,
    while still capping resolution.
    """
    img = Image.open(path)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    long_side = max(w, h)
    if long_side > MAX_IMAGE_LONG_SIDE:
        scale = MAX_IMAGE_LONG_SIDE / float(long_side)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    buf.seek(0)
    data = buf.read()

    return base64.b64encode(data).decode("ascii")


# -------------------------------------------------------------------
# OPENAI CALL VIA RAW HTTP
# -------------------------------------------------------------------

def call_model_for_listing(
    gen: int,
    listing_id: int,
    images: List[Tuple[int, str, str]],
    model_name: str,
) -> Dict[str, Any]:
    """
    Call the OpenAI /v1/responses endpoint with all images for one listing,
    using raw HTTP (requests).

    Returns parsed JSON dict as described in the system prompt.
    """
    if not images:
        raise ValueError(f"No images for listing_id: {listing_id}")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment")

    ctx = get_listing_context(gen, listing_id)
    title = ctx["title"]
    description = ctx["description"]
    condition_score = ctx["condition_score"]
    model_label = ctx.get("model", "")

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
        f"For each image, you MUST base visible_damage_level ONLY on what you see in that photo, "
        f"using the caption + TITLE + DESCRIPTION + CONDITION_SCORE only as hints.\n"
        f"If a phone looks clean in the photos at normal zoom, you MUST set visible_damage_level = 0.\n"
        f"In any borderline case between two levels (0 vs 1, 1 vs 2, 4 vs 5, etc.), you MUST choose the LOWER level "
        f"for non-structural damage, and you MUST use 8–10 for clear structural damage as described in the system prompt.\n\n"
        "Image indices (in order):\n"
        + "\n".join(f"- image_index {idx}: file {os.path.basename(path)}" for idx, path, _cap in images)
        + "\n\nNow output the JSON object."
    )

    print(f"[DEBUG] OpenAI HTTP call listing_id: {listing_id} imgs={len(images)} model={model_name}")

    payload: Dict[str, Any] = {
        "model": model_name,
        "input": [
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
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    url = f"{OPENAI_API_BASE}/responses"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as e:
        print(f"[DEBUG] HTTP exception type={type(e).__name__} listing_id: {listing_id}")
        print(f"[DEBUG] HTTP exception detail={repr(e)}")
        traceback.print_exc()
        raise RuntimeError(f"HTTP call to OpenAI failed for listing_id: {listing_id}: {e}") from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenAI HTTP {resp.status_code} for listing_id: {listing_id}: {resp.text}"
        )

    data = resp.json()

    api_model = data.get("model")
    print(f"[MODEL] requested={model_name} api_response_model={api_model} listing_id: {listing_id}")

    msg: Optional[Dict[str, Any]] = None
    for item in data.get("output", []):
        if item.get("type") == "message":
            msg = item
            break

    if msg is None:
        raise RuntimeError(
            f"No 'message' item with content found in OpenAI response for listing_id: {listing_id}: {data}"
        )

    contents = msg.get("content", [])
    if not contents:
        raise RuntimeError(
            f"No content in 'message' item for listing_id: {listing_id}: {data}"
        )

    first = contents[0]

    if isinstance(first, dict):
        if first.get("type") == "output_text":
            text_obj = first.get("text")
            if isinstance(text_obj, dict):
                text_out = text_obj.get("value", "")
            else:
                text_out = text_obj or ""
        else:
            text_out = first.get("text") or first.get("value") or ""
    else:
        text_out = str(first)

    if not text_out:
        raise RuntimeError(
            f"Empty text content in OpenAI response for listing_id: {listing_id}: {data}"
        )

    try:
        data_json = json.loads(text_out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from OpenAI text for listing_id: {listing_id}: {e}\n{text_out}"
        ) from e

    usage = data.get("usage") or {}
    if usage:
        data_json["_usage"] = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    return data_json


# -------------------------------------------------------------------
# SUMMARY LINE FOR LOGGING
# -------------------------------------------------------------------

def summarize_listing(
    gen: int, listing_id: int, data: Dict[str, Any], inserted_count: int
) -> str:
    """
    Build a one-line summary string for logging, based on the JSON
    returned by the model for this listing.
    """
    imgs = data.get("images") or []
    n_images = len(imgs)

    vis_levels = [
        img.get("visible_damage_level")
        for img in imgs
        if img.get("visible_damage_level") is not None
    ]
    max_vis = max(vis_levels) if vis_levels else None
    dmg_str = "none" if max_vis is None else str(max_vis)

    prot_count = sum(1 for img in imgs if img.get("has_screen_protector"))
    prot_only_count = sum(1 for img in imgs if img.get("damage_on_protector_only"))

    stock_count = sum(1 for img in imgs if img.get("is_stock_photo"))
    real_count = n_images - stock_count

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
        f"max_vis_dmg={dmg_str} "
        f"protector_imgs={prot_count} "
        f"protector_only_imgs={prot_only_count} "
        f"real/stock={real_count}/{stock_count} "
        f"tokens={tokens_str} "
        f"inserted={inserted_count}"
    )


# -------------------------------------------------------------------
# INSERT INTO ml.iphone_image_features_v1 (DAMAGE-ONLY)
# -------------------------------------------------------------------

def insert_features_from_json(
    gen: int, listing_id: int, data: Dict[str, Any]
) -> int:
    """
    Insert rows into ml.iphone_image_features_v1 from LLM JSON.

    DAMAGE-ONLY VERSION:
    - ONLY writes:
        • photo_quality_level
        • background_clean_level
        • is_stock_photo
        • visible_damage_level
        • damage_summary_text
        • image_damage_level
        • has_screen_protector
        • damage_on_protector_only
        • extra_json
    - DOES NOT touch any accessory fields or body_color_* fields.
    """
    images = data.get("images") or []
    if not images:
        print(f"[WARN] No images array in JSON for listing_id: {listing_id}")
        return 0

    rows: List[Tuple[Any, ...]] = []

    for img in images:
        image_index = img["image_index"]

        vis_lvl = img.get("visible_damage_level")
        dmg_summary = img.get("damage_summary_text")

        rows.append(
            (
                gen,
                listing_id,
                image_index,
                FEATURE_VERSION,
                img.get("photo_quality_level"),
                img.get("background_clean_level"),
                img.get("is_stock_photo"),
                vis_lvl,
                dmg_summary,
                None,  # image_damage_level (reserved)
                img.get("has_screen_protector"),
                img.get("damage_on_protector_only"),
                json.dumps(img, ensure_ascii=False),
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
                photo_quality_level,
                background_clean_level,
                is_stock_photo,
                visible_damage_level,
                damage_summary_text,
                image_damage_level,
                has_screen_protector,
                damage_on_protector_only,
                extra_json
            )
            VALUES (
                %s,%s,%s,%s,
                %s,%s,%s,
                %s,%s,%s,
                %s,%s,%s
            )
            ON CONFLICT (generation, listing_id, image_index, feature_version) DO UPDATE
            SET photo_quality_level      = EXCLUDED.photo_quality_level,
                background_clean_level   = EXCLUDED.background_clean_level,
                is_stock_photo           = EXCLUDED.is_stock_photo,
                visible_damage_level     = EXCLUDED.visible_damage_level,
                damage_summary_text      = EXCLUDED.damage_summary_text,
                image_damage_level       = EXCLUDED.image_damage_level,
                has_screen_protector     = EXCLUDED.has_screen_protector,
                damage_on_protector_only = EXCLUDED.damage_on_protector_only,
                extra_json               = EXCLUDED.extra_json,
                created_at               = LEAST(ml.iphone_image_features_v1.created_at, now())
            ;
            """,
            rows,
        )

    with_db_cursor(_run)
    return len(images)


def mark_damage_done(gen: int, listing_id: int) -> None:
    """
    Mark damage_done / damage_done_at in ml.iphone_image_features_v1
    for this (generation, listing_id). This is the processed marker so we don't
    run the damage script twice on the same listing.
    """

    def _run(conn, cur):
        cur.execute(
            """
            UPDATE ml.iphone_image_features_v1
            SET damage_done    = TRUE,
                damage_done_at = now()
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
        description="Batch image DAMAGE analysis for iPhone listings using GPT-5 mini (damage 0–10 + protector flags)."
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
                mark_damage_done(gen, listing_id)

            total_listings_processed += 1
            total_images_processed += inserted_count

            summary_line = summarize_listing(gen, listing_id, data, inserted_count)
            print(f"[OK] {summary_line}")

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
