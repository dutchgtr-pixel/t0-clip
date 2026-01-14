#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scrape_images_playwright.py (public-safe)

Batch image extraction + download for marketplace listings using Playwright.

Sanitization goals for public release:
- No embedded credentials (PG_DSN must be provided via environment variable).
- No target-platform identifiers or selectors. All scraping is generic and configurable.
- Uses listing_id (not platform-specific identifiers) throughout.

Required environment variables:
- PG_DSN: Postgres DSN (do NOT hardcode secrets in the script)

Optional environment variables:
- PG_SCHEMA: DB schema (default: iPhone)
- LISTINGS_TABLE: listings table (default: iphone_listings)
- IMAGE_ASSETS_TABLE: image assets table (default: iphone_image_assets)
- LISTING_ID_COLUMN: primary listing id column (default: listing_id)
- LISTING_URL_COLUMN: listing URL column (default: url)
- LISTING_STATUS_COLUMN: status column (default: status)
- LISTING_SPAM_COLUMN: spam marker column (default: spam)
- LISTING_IMAGES_SCRAPED_AT_COLUMN: scrape-stamp column (default: images_scraped_at)

- IMAGE_ROOT_DIR: where to store downloaded images (default: ./listing_images)
- PLAYWRIGHT_BATCH_SIZE: DB batch size (default: 50)
- PLAYWRIGHT_MAX_IMAGES: max images per listing (default: 16)
- PLAYWRIGHT_GALLERY_IMG_SELECTOR: CSS selector for listing images (default: img)
- ALLOWED_IMAGE_HOSTS: comma-separated host allowlist for image URLs (default: empty = allow all)
- HIGH_RES_URL_REWRITE_FROM / HIGH_RES_URL_REWRITE_TO: optional URL rewrite to prefer hi-res variants
  (default: disabled)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
import requests
from playwright.sync_api import sync_playwright


# -----------------------------
# Config (env)
# -----------------------------

PG_DSN = (os.getenv("PG_DSN") or "").strip()
if not PG_DSN:
    raise SystemExit("Missing required env var: PG_DSN (do not hardcode secrets in code).")

PG_SCHEMA = os.getenv("PG_SCHEMA", "iPhone")
LISTINGS_TABLE = os.getenv("LISTINGS_TABLE", "iphone_listings")
IMAGE_ASSETS_TABLE = os.getenv("IMAGE_ASSETS_TABLE", "iphone_image_assets")

LISTING_ID_COLUMN = os.getenv("LISTING_ID_COLUMN", "listing_id")
LISTING_URL_COLUMN = os.getenv("LISTING_URL_COLUMN", "url")
LISTING_STATUS_COLUMN = os.getenv("LISTING_STATUS_COLUMN", "status")
LISTING_SPAM_COLUMN = os.getenv("LISTING_SPAM_COLUMN", "spam")
LISTING_IMAGES_SCRAPED_AT_COLUMN = os.getenv(
    "LISTING_IMAGES_SCRAPED_AT_COLUMN", "images_scraped_at"
)

IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", os.path.join(".", "listing_images"))

BATCH_SIZE = int(os.getenv("PLAYWRIGHT_BATCH_SIZE", "50"))
MAX_IMAGES_PER_LISTING = int(os.getenv("PLAYWRIGHT_MAX_IMAGES", "16"))

GALLERY_IMG_SELECTOR = os.getenv("PLAYWRIGHT_GALLERY_IMG_SELECTOR", "img")

_ALLOWED_HOSTS_RAW = os.getenv("ALLOWED_IMAGE_HOSTS", "")
ALLOWED_IMAGE_HOSTS = {h.strip().lower() for h in _ALLOWED_HOSTS_RAW.split(",") if h.strip()}

HIGH_RES_URL_REWRITE_FROM = os.getenv("HIGH_RES_URL_REWRITE_FROM", "").strip()
HIGH_RES_URL_REWRITE_TO = os.getenv("HIGH_RES_URL_REWRITE_TO", "").strip()


# -----------------------------
# DB helpers
# -----------------------------

def _conn():
    return psycopg2.connect(PG_DSN)


def _id(name: str) -> sql.Identifier:
    # psycopg2.sql.Identifier safely quotes/escapes.
    if not name or not isinstance(name, str):
        raise ValueError("Empty identifier")
    return sql.Identifier(name)


def get_pending_rows(limit: int) -> List[Tuple[int, int, str]]:
    """
    Returns rows: (generation, listing_id, url)
    """
    conn = _conn()
    cur = conn.cursor()

    q = sql.SQL(
        """
        SELECT {gen_col}, {id_col}, {url_col}
          FROM {schema}.{tbl}
         WHERE {images_scraped_at_col} IS NULL
           AND COALESCE({status_col}, '') IN ('live','sold','older21days')
           AND {spam_col} IS NULL
           AND {url_col} IS NOT NULL
         ORDER BY {gen_col}, {id_col}
         LIMIT %s;
        """
    ).format(
        gen_col=_id("generation"),
        id_col=_id(LISTING_ID_COLUMN),
        url_col=_id(LISTING_URL_COLUMN),
        schema=_id(PG_SCHEMA),
        tbl=_id(LISTINGS_TABLE),
        images_scraped_at_col=_id(LISTING_IMAGES_SCRAPED_AT_COLUMN),
        status_col=_id(LISTING_STATUS_COLUMN),
        spam_col=_id(LISTING_SPAM_COLUMN),
    )

    cur.execute(q, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_single_row(listing_id: int) -> List[Tuple[int, int, str]]:
    """Fetch a single listing row by listing_id (manual test mode)."""
    conn = _conn()
    cur = conn.cursor()

    q = sql.SQL(
        """
        SELECT {gen_col}, {id_col}, {url_col}
          FROM {schema}.{tbl}
         WHERE {id_col} = %s
         ORDER BY {gen_col}
         LIMIT 1;
        """
    ).format(
        gen_col=_id("generation"),
        id_col=_id(LISTING_ID_COLUMN),
        url_col=_id(LISTING_URL_COLUMN),
        schema=_id(PG_SCHEMA),
        tbl=_id(LISTINGS_TABLE),
    )

    cur.execute(q, (listing_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return []
    return [row]


def mark_images_scraped(generation: int, listing_id: int) -> None:
    """Stamp images_scraped_at so we don't retry the listing in future runs."""
    conn = _conn()
    cur = conn.cursor()

    q = sql.SQL(
        """
        UPDATE {schema}.{tbl}
           SET {images_scraped_at_col} = now()
         WHERE {gen_col} = %s AND {id_col} = %s;
        """
    ).format(
        schema=_id(PG_SCHEMA),
        tbl=_id(LISTINGS_TABLE),
        images_scraped_at_col=_id(LISTING_IMAGES_SCRAPED_AT_COLUMN),
        gen_col=_id("generation"),
        id_col=_id(LISTING_ID_COLUMN),
    )

    cur.execute(q, (generation, listing_id))
    conn.commit()
    cur.close()
    conn.close()


def upsert_image_assets(generation: int, listing_id: int, saved_entries: Sequence[Tuple[int, str, str]]) -> None:
    """
    saved_entries: list of (image_index, source_url, caption_text)
    Writes into IMAGE_ASSETS_TABLE with ON CONFLICT UPDATE.

    Expected columns in IMAGE_ASSETS_TABLE:
      - generation (int)
      - listing_id (int/bigint)
      - image_index (int)
      - source_url (text)
      - caption_text (text)
      - mime_type (text)
      - storage_path (text)
      - updated_at (timestamptz) (optional; used by UPDATE)
    """
    if not saved_entries:
        return

    conn = _conn()
    cur = conn.cursor()

    params = []
    for idx, url, caption in saved_entries:
        storage_path = f"{listing_id}/{idx}.jpg"
        params.append(
            (
                generation,
                listing_id,
                idx,
                url,
                caption,
                "image/jpeg",
                storage_path,
            )
        )

    q = sql.SQL(
        """
        INSERT INTO {schema}.{tbl} (
            generation,
            {id_col},
            image_index,
            source_url,
            caption_text,
            mime_type,
            storage_path
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (generation, {id_col}, image_index) DO UPDATE
        SET source_url   = EXCLUDED.source_url,
            caption_text = EXCLUDED.caption_text,
            mime_type    = EXCLUDED.mime_type,
            storage_path = EXCLUDED.storage_path,
            updated_at   = now();
        """
    ).format(
        schema=_id(PG_SCHEMA),
        tbl=_id(IMAGE_ASSETS_TABLE),
        id_col=_id(LISTING_ID_COLUMN),
    )

    execute_batch(cur, q, params)
    conn.commit()
    cur.close()
    conn.close()


# -----------------------------
# Image URL selection helpers
# -----------------------------

def _is_allowed_host(url: str) -> bool:
    if not ALLOWED_IMAGE_HOSTS:
        return True
    host = (urlparse(url).netloc or "").lower()
    return host in ALLOWED_IMAGE_HOSTS


def choose_best_from_srcset(srcset: str) -> Optional[str]:
    """
    Pick the best candidate URL from a srcset string.

    This implementation is platform-agnostic:
    - Prefer the largest width (e.g., "1600w") if present.
    - Else prefer the largest density (e.g., "2x") if present.
    """
    if not srcset:
        return None

    candidates: List[Tuple[float, str]] = []

    for part in srcset.split(","):
        part = part.strip()
        if not part:
            continue

        bits = part.split()
        url = bits[0].strip()

        # srcset URLs can be relative
        if url.startswith("//"):
            url = "https:" + url

        if not url.lower().startswith(("http://", "https://")):
            # Leave relative URLs for later urljoin() in the caller.
            pass

        if len(bits) >= 2:
            desc = bits[1].strip().lower()
        else:
            desc = ""

        score = 0.0
        if desc.endswith("w"):
            try:
                score = float(int(desc[:-1]))
            except Exception:
                score = 0.0
        elif desc.endswith("x"):
            try:
                score = float(desc[:-1]) * 10000.0
            except Exception:
                score = 0.0

        candidates.append((score, url))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def rewrite_high_res_url(uri: str) -> str:
    """
    Optional high-res URL rewrite.

    If HIGH_RES_URL_REWRITE_FROM and HIGH_RES_URL_REWRITE_TO are set, the URL is
    rewritten using string replacement. Otherwise, the URL is returned unchanged.
    """
    if not uri:
        return uri
    if HIGH_RES_URL_REWRITE_FROM and HIGH_RES_URL_REWRITE_TO:
        return uri.replace(HIGH_RES_URL_REWRITE_FROM, HIGH_RES_URL_REWRITE_TO)
    return uri


def download_image(url: str, listing_id: int, idx: int) -> Optional[str]:
    os.makedirs(os.path.join(IMAGE_ROOT_DIR, str(listing_id)), exist_ok=True)
    try:
        resp = requests.get(url, timeout=20)
    except Exception as e:
        print(f"[IMG] download error listing_id={listing_id} idx={idx} url={url}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[IMG] download error listing_id={listing_id} idx={idx} http={resp.status_code}")
        return None

    path = os.path.join(IMAGE_ROOT_DIR, str(listing_id), f"{idx}.jpg")
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


# -----------------------------
# Generic 404 detection
# -----------------------------

def is_html_404(page) -> bool:
    """
    Best-effort detection of an HTML 404 page when the server returns HTTP 200.

    This is intentionally generic (no platform-specific phrases).
    """
    try:
        h1 = (page.text_content("h1") or "").strip().lower()
        title = (page.title() or "").strip().lower()

        if h1 in {"404", "not found", "page not found"}:
            return True
        if "404" in title and ("not found" in title or "404" in h1):
            return True

        body = (page.text_content("body") or "").lower()
        if "404" in body and ("not found" in body or "page not found" in body):
            return True
    except Exception:
        return False
    return False


# -----------------------------
# Scrape listing images (generic)
# -----------------------------

def _normalize_url(maybe_url: str, base_url: str) -> Optional[str]:
    if not maybe_url:
        return None
    u = maybe_url.strip()
    if not u:
        return None
    if u.startswith("//"):
        u = "https:" + u
    if u.lower().startswith(("http://", "https://")):
        return u
    # relative
    return urljoin(base_url, u)


def scrape_listing_images(page, url: str, generation: int, listing_id: int) -> Tuple[int, bool]:
    """
    Visits a listing URL and extracts a small set of image URLs + captions.

    Returns:
      (saved_count, is_404)
    """
    resp = page.goto(url, wait_until="domcontentloaded")
    status_code = None
    try:
        if resp is not None:
            status_code = resp.status
    except Exception:
        status_code = None

    # 404 detection
    if status_code in (404, 410) or is_html_404(page):
        print(f"[SCRAPE] 404 listing_id={listing_id} url={url}")
        return 0, True

    # Give client-side apps a moment to render.
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    # Try to wait for any images (selector configurable).
    try:
        page.wait_for_selector(GALLERY_IMG_SELECTOR, timeout=8000)
    except Exception:
        pass

    img_elements = page.query_selector_all(GALLERY_IMG_SELECTOR) or []

    entries: List[Tuple[str, str]] = []
    seen_filenames = set()

    for img in img_elements:
        try:
            srcset = img.get_attribute("srcset") or ""
            src = img.get_attribute("src") or ""

            best = choose_best_from_srcset(srcset) if srcset else (src.strip() or None)
            best = _normalize_url(best or "", page.url)
            if not best:
                continue

            best = rewrite_high_res_url(best)
            if not best.lower().startswith(("http://", "https://")):
                continue

            if not _is_allowed_host(best):
                continue

            filename = os.path.basename(urlparse(best).path)
            if not filename or filename in seen_filenames:
                continue
            seen_filenames.add(filename)

            caption = (img.get_attribute("alt") or "").strip()

            # Suppress obviously-generic alt text
            if caption.lower() in {"image", "thumbnail", "photo", "gallery"}:
                caption = ""

            if not caption:
                # best-effort figcaption lookup
                try:
                    caption = (
                        img.evaluate(
                            """el => {
                                const fig = el.closest('figure');
                                if (!fig) return null;
                                const cap = fig.querySelector('figcaption');
                                return cap ? cap.innerText : null;
                            }"""
                        )
                        or ""
                    ).strip()
                except Exception:
                    caption = ""

            entries.append((best, caption))
            if len(entries) >= MAX_IMAGES_PER_LISTING:
                break
        except Exception:
            continue

    if not entries:
        return 0, False

    saved_entries_for_db: List[Tuple[int, str, str]] = []
    saved = 0

    for idx, (u, caption) in enumerate(entries):
        if idx >= MAX_IMAGES_PER_LISTING:
            break
        path = download_image(u, listing_id, idx)
        if not path:
            continue
        saved_entries_for_db.append((idx, u, caption))
        saved += 1

    if saved_entries_for_db:
        upsert_image_assets(generation, listing_id, saved_entries_for_db)

    return saved, False


# -----------------------------
# Batch runner
# -----------------------------

def run_batch(rows: Sequence[Tuple[int, int, str]]) -> None:
    if not rows:
        print("Playwright: no pending listings. Done.")
        return

    print(f"Playwright: processing {len(rows)} listings...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for gen, listing_id, url in rows:
            print(f"[RUN] gen={gen} listing_id={listing_id} url={url}")
            try:
                saved, is_404 = scrape_listing_images(page, url, gen, listing_id)
                if saved > 0:
                    print(f"[OK] saved={saved} listing_id={listing_id}")
                    mark_images_scraped(gen, listing_id)
                elif is_404:
                    print(f"[WARN] 404/no images listing_id={listing_id} (marking scraped)")
                    mark_images_scraped(gen, listing_id)
                else:
                    # No images found but not a 404. Mark scraped to avoid infinite retries.
                    print(f"[WARN] no images extracted listing_id={listing_id} (marking scraped)")
                    mark_images_scraped(gen, listing_id)
            except Exception as e:
                print(f"[ERR] listing_id={listing_id}: {e}")
            time.sleep(0.5)

        browser.close()


def main_single(listing_id: int) -> None:
    rows = get_single_row(listing_id)
    if not rows:
        print(f"No row found for listing_id={listing_id}")
        return
    run_batch(rows)


def main_loop() -> None:
    while True:
        rows = get_pending_rows(BATCH_SIZE)
        if not rows:
            print("Playwright: no pending listings. Done.")
            break
        run_batch(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-listing-id",
        type=int,
        help="Run scraper only for this listing_id (manual test mode)",
    )
    args = parser.parse_args()

    if args.single_listing_id:
        main_single(args.single_listing_id)
    else:
        main_loop()
