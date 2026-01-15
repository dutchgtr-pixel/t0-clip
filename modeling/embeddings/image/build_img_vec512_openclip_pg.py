#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_img_vec512_openclip_pg_public.py

Compute CLIP-style image embeddings for marketplace listings:

- reads image paths from an image-asset inventory table (storage_path)
- loads actual image files from IMAGE_ROOT_DIR
- selects top-K images per listing using per-image quality/meta flags
  (exclude stock photos and battery screenshots by default)
- computes per-image embeddings (OpenCLIP) in GPU/CPU batches
- aggregates to ONE embedding per listing (mean of L2-normalized image vectors, then L2 normalize)
- upserts into an embeddings table as pgvector (vector(512))

Benchmark mode included: measures images/sec on YOUR machine.
No global time estimates are made here because disk + decode dominates.

SECURITY (public release):
- No secrets are embedded in this script.
- Provide the database connection string via env var PG_DSN.
- Do not commit real paths or DSNs into the repository.

Quick examples (no secrets embedded):

  # PowerShell (read secrets from a local file / secret manager)
  #   $env:PG_DSN = (Get-Content .\secrets\PG_DSN.txt -Raw)
  #   $env:IMAGE_ROOT_DIR = ".\listing_images"
  #   python .\build_img_vec512_openclip_pg_public.py --benchmark-images 2000 --limit-listings 5000 --dry-run
  #
  # Bash
  #   export PG_DSN="$(cat ./secrets/pg_dsn.txt)"
  #   export IMAGE_ROOT_DIR="./listing_images"
  #   python ./build_img_vec512_openclip_pg_public.py --limit-listings 50000 --topk 5 --batch-size 96 --num-workers 6

"""

from __future__ import annotations

import os
import time
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm
from PIL import Image

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

import torch
import open_clip


# --------------------------
# ENV (PUBLIC RELEASE SAFE)
# --------------------------

PG_DSN = (os.getenv("PG_DSN") or "").strip()
if not PG_DSN:
    raise SystemExit("Missing required env var: PG_DSN (do not hardcode secrets in code).")

IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", os.path.join(".", "listing_images"))
FEATURE_VERSION = int(os.getenv("IMAGE_FEATURE_VERSION", "1"))

# Optional: allow schema/table customization without code edits.
# Identifiers are strictly validated to avoid SQL injection via env vars.
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def qident(name: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


def qname(schema: str, table: str) -> str:
    return f"{qident(schema)}.{qident(table)}"


LISTINGS_SCHEMA = os.getenv("LISTINGS_SCHEMA", "iPhone")
LISTINGS_TABLE = os.getenv("LISTINGS_TABLE", "iphone_listings")

ASSETS_SCHEMA = os.getenv("ASSETS_SCHEMA", "iPhone")
ASSETS_TABLE = os.getenv("ASSETS_TABLE", "iphone_image_assets")

ML_SCHEMA = os.getenv("ML_SCHEMA", "ml")
IMAGE_FEATURES_TABLE = os.getenv("IMAGE_FEATURES_TABLE", "iphone_image_features_v1")
VECTORS_TABLE = os.getenv("VECTORS_TABLE", "img_vec512_v1")

LISTINGS_FQN = qname(LISTINGS_SCHEMA, LISTINGS_TABLE)
ASSETS_FQN = qname(ASSETS_SCHEMA, ASSETS_TABLE)
IMAGE_FEATURES_FQN = qname(ML_SCHEMA, IMAGE_FEATURES_TABLE)
VECTORS_FQN = qname(ML_SCHEMA, VECTORS_TABLE)


# --------------------------
# DATA STRUCTURES
# --------------------------

@dataclass(frozen=True)
class ListingKey:
    listing_id: int
    edited_date_iso: str  # keep string for hashing; convert only for DB writes


@dataclass
class ImgRow:
    listing_idx: int
    image_index: int
    full_path: str


# --------------------------
# DB
# --------------------------

def db_connect():
    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False
    return conn


def fetch_todo_assets(
    conn,
    limit_listings: int,
    force: bool,
    source: str,
    model_rev: str,
    pca_rev: str,
) -> List[Dict[str, Any]]:
    """
    Returns per-image rows for listings to process. Includes join to image feature flags.

    We return many rows, later grouped by (listing_id, edited_date).
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if force:
            cur.execute(
                f"""
                WITH todo AS (
                  SELECT l.generation, l.listing_id, l.edited_date
                  FROM {LISTINGS_FQN} l
                  WHERE l.edited_date IS NOT NULL
                    AND l.url IS NOT NULL
                    AND l.spam IS NULL
                  ORDER BY l.generation, l.listing_id
                  LIMIT %s
                )
                SELECT
                  t.generation,
                  t.listing_id,
                  t.edited_date,
                  a.image_index,
                  a.storage_path,
                  COALESCE(f.is_stock_photo,false)     AS is_stock_photo,
                  COALESCE(f.battery_screenshot,false) AS battery_screenshot,
                  f.photo_quality_level,
                  f.background_clean_level
                FROM todo t
                JOIN {ASSETS_FQN} a
                  ON a.generation=t.generation AND a.listing_id=t.listing_id
                LEFT JOIN {IMAGE_FEATURES_FQN} f
                  ON f.generation=t.generation
                 AND f.listing_id=t.listing_id
                 AND f.image_index=a.image_index
                 AND f.feature_version=%s
                ORDER BY t.generation, t.listing_id, a.image_index;
                """,
                (limit_listings, FEATURE_VERSION),
            )
        else:
            cur.execute(
                f"""
                WITH todo AS (
                  SELECT l.generation, l.listing_id, l.edited_date
                  FROM {LISTINGS_FQN} l
                  WHERE l.edited_date IS NOT NULL
                    AND l.url IS NOT NULL
                    AND l.spam IS NULL
                    AND NOT EXISTS (
                      SELECT 1
                      FROM {VECTORS_FQN} v
                      WHERE v.listing_id=l.listing_id
                        AND v.edited_date=l.edited_date
                        AND v.source=%s AND v.model_rev=%s AND v.pca_rev=%s
                    )
                  ORDER BY l.generation, l.listing_id
                  LIMIT %s
                )
                SELECT
                  t.generation,
                  t.listing_id,
                  t.edited_date,
                  a.image_index,
                  a.storage_path,
                  COALESCE(f.is_stock_photo,false)     AS is_stock_photo,
                  COALESCE(f.battery_screenshot,false) AS battery_screenshot,
                  f.photo_quality_level,
                  f.background_clean_level
                FROM todo t
                JOIN {ASSETS_FQN} a
                  ON a.generation=t.generation AND a.listing_id=t.listing_id
                LEFT JOIN {IMAGE_FEATURES_FQN} f
                  ON f.generation=t.generation
                 AND f.listing_id=t.listing_id
                 AND f.image_index=a.image_index
                 AND f.feature_version=%s
                ORDER BY t.generation, t.listing_id, a.image_index;
                """,
                (source, model_rev, pca_rev, limit_listings, FEATURE_VERSION),
            )
        return cur.fetchall()


def upsert_listing_vectors(
    conn,
    rows: List[Tuple[int, str, str, str, str, str, int]],
):
    """
    rows: (listing_id, edited_date_iso, source, model_rev, pca_rev, vec_txt, n_images)
    """
    if not rows:
        return

    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {VECTORS_FQN} (
              listing_id, edited_date, source, model_rev, pca_rev, vec, n_images, updated_at
            )
            VALUES %s
            ON CONFLICT (listing_id, edited_date, source, model_rev, pca_rev)
            DO UPDATE SET
              vec       = EXCLUDED.vec,
              n_images  = EXCLUDED.n_images,
              updated_at= now();
            """,
            rows,
            template="(%s, %s::timestamptz, %s, %s, %s, %s::vector, %s, now())",
            page_size=2000,
        )
    conn.commit()


# --------------------------
# IMAGE SELECTION
# --------------------------

def score_image(r: Dict[str, Any]) -> float:
    """Higher is better. This is only to pick which images to embed."""
    pq = r.get("photo_quality_level")
    bg = r.get("background_clean_level")
    pq = float(pq) if pq is not None else 0.0
    bg = float(bg) if bg is not None else 0.0
    # Weight quality strongly; background mildly.
    return 10.0 * pq + 3.0 * bg


def select_topk_images_per_listing(
    raw_rows: List[Dict[str, Any]],
    topk: int,
    exclude_stock: bool,
    exclude_screenshots: bool,
) -> Tuple[List[ListingKey], List[ImgRow]]:
    """
    Groups rows by (listing_id, edited_date) and selects top-K images.

    Returns:
      - listing_keys: indexable list
      - image_tasks: flattened list of ImgRow(listing_idx, image_index, full_path)
    """
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        listing_id = int(r["listing_id"])
        edited_date_iso = str(r["edited_date"])
        groups.setdefault((listing_id, edited_date_iso), []).append(r)

    listing_keys: List[ListingKey] = []
    image_tasks: List[ImgRow] = []

    for (listing_id, edited_date_iso), rows in groups.items():
        # Build candidates
        cand = []
        for rr in rows:
            if exclude_stock and bool(rr.get("is_stock_photo", False)):
                continue
            if exclude_screenshots and bool(rr.get("battery_screenshot", False)):
                continue

            storage_path = rr.get("storage_path")
            if not storage_path:
                continue
            full_path = os.path.join(IMAGE_ROOT_DIR, storage_path)
            if not os.path.isfile(full_path):
                continue

            cand.append((score_image(rr), int(rr["image_index"]), full_path))

        # Fallback: if exclusions remove everything, fall back to first images
        if not cand:
            fallback = []
            for rr in rows:
                storage_path = rr.get("storage_path")
                if not storage_path:
                    continue
                full_path = os.path.join(IMAGE_ROOT_DIR, storage_path)
                if os.path.isfile(full_path):
                    fallback.append((0.0, int(rr["image_index"]), full_path))
            cand = fallback

        if not cand:
            continue

        cand.sort(key=lambda x: x[0], reverse=True)
        chosen = cand[:topk]

        listing_idx = len(listing_keys)
        listing_keys.append(ListingKey(listing_id=listing_id, edited_date_iso=edited_date_iso))

        for _, image_index, full_path in chosen:
            image_tasks.append(ImgRow(listing_idx=listing_idx, image_index=image_index, full_path=full_path))

    return listing_keys, image_tasks


# --------------------------
# TORCH DATASET
# --------------------------

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[ImgRow], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        # Use RGB always
        img = Image.open(it.full_path).convert("RGB")
        x = self.preprocess(img)
        return x, it.listing_idx


def format_vec_pgvector(v: np.ndarray) -> str:
    # v must be 1D float32, length 512
    # pgvector text format: [0.1,0.2,...]
    # Use compact formatting to reduce payload.
    return "[" + ",".join(f"{float(x):.6g}" for x in v.tolist()) + "]"


def _mask(s: str) -> str:
    """Best-effort masking for logs (does not guarantee anonymity)."""
    if not s:
        return "<empty>"
    # Never print full DSNs; show only length for debugging.
    return f"<set len={len(s)}>"


# --------------------------
# MAIN
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-listings", type=int, default=50000)
    ap.add_argument("--topk", type=int, default=5, help="Images per listing to embed (default 5)")
    ap.add_argument("--exclude-stock", action="store_true", default=True)
    ap.add_argument("--include-stock", dest="exclude_stock", action="store_false")
    ap.add_argument("--exclude-screenshots", action="store_true", default=True)
    ap.add_argument("--include-screenshots", dest="exclude_screenshots", action="store_false")
    ap.add_argument("--force", action="store_true", help="Overwrite existing rows for this (source,model_rev,pca_rev)")

    ap.add_argument("--model-name", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--source", default="listing_images_topk5_v1")
    ap.add_argument("--pca-rev", default="")

    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")

    ap.add_argument("--benchmark-images", type=int, default=0,
                    help="If >0: run inference on first N images and report throughput. No DB writes.")

    ap.add_argument("--dry-run", action="store_true", help="Do everything but skip DB writes.")

    args = ap.parse_args()

    model_rev = f"open_clip_{args.model_name}_{args.pretrained}"

    print("===================================================")
    print("[CONFIG] PG_DSN          =", _mask(PG_DSN))
    print("[CONFIG] IMAGE_ROOT_DIR  =", IMAGE_ROOT_DIR)
    print("[CONFIG] source          =", args.source)
    print("[CONFIG] model_rev       =", model_rev)
    print("[CONFIG] pca_rev         =", args.pca_rev)
    print("[CONFIG] limit_listings  =", args.limit_listings)
    print("[CONFIG] topk            =", args.topk)
    print("[CONFIG] exclude_stock   =", args.exclude_stock)
    print("[CONFIG] exclude_shots   =", args.exclude_screenshots)
    print("[CONFIG] device          =", args.device)
    print("[CONFIG] batch_size      =", args.batch_size)
    print("[CONFIG] num_workers     =", args.num_workers)
    print("[CONFIG] fp16            =", args.fp16)
    print("[CONFIG] benchmark_images=", args.benchmark_images)
    print("[CONFIG] tables          =",
          f"listings={LISTINGS_FQN} assets={ASSETS_FQN} image_features={IMAGE_FEATURES_FQN} vectors={VECTORS_FQN}")
    print("===================================================")

    # Load todo rows from DB
    conn = db_connect()
    raw = fetch_todo_assets(
        conn,
        limit_listings=args.limit_listings,
        force=args.force,
        source=args.source,
        model_rev=model_rev,
        pca_rev=args.pca_rev,
    )
    if not raw:
        print("[INFO] No rows to process (already embedded or no eligible listings).")
        conn.close()
        return

    listing_keys, image_tasks = select_topk_images_per_listing(
        raw_rows=raw,
        topk=args.topk,
        exclude_stock=args.exclude_stock,
        exclude_screenshots=args.exclude_screenshots,
    )

    if not listing_keys or not image_tasks:
        print("[INFO] After filtering, no valid images found on disk.")
        conn.close()
        return

    if args.benchmark_images and args.benchmark_images > 0:
        image_tasks = image_tasks[: int(args.benchmark_images)]
        # reduce listing set to only those referenced in the benchmark subset
        used_listing_ids = sorted({it.listing_idx for it in image_tasks})
        id_remap = {old: new for new, old in enumerate(used_listing_ids)}
        listing_keys = [listing_keys[old] for old in used_listing_ids]
        for it in image_tasks:
            it.listing_idx = id_remap[it.listing_idx]

    print(f"[INFO] listings={len(listing_keys)} images_selected={len(image_tasks)}")

    # Build OpenCLIP model
    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    model.eval()
    model.to(device)

    if args.fp16 and device.type == "cuda":
        # OpenCLIP supports autocast; keep model weights in fp16 for speed
        model.half()

    ds = ImgDataset(image_tasks, preprocess)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dim = 512
    sums = np.zeros((len(listing_keys), dim), dtype=np.float32)
    counts = np.zeros((len(listing_keys),), dtype=np.int32)

    # Inference
    t0 = time.perf_counter()
    n_done = 0

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else None

    with torch.no_grad():
        for xb, listing_idx in tqdm(dl, desc="embed", unit="batch"):
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True)
                listing_idx = listing_idx.to(device, non_blocking=True)
                if autocast_ctx is not None:
                    with autocast_ctx():
                        feats = model.encode_image(xb)
                else:
                    feats = model.encode_image(xb)
            else:
                feats = model.encode_image(xb.to(device))

            feats = feats.float()
            feats = feats / torch.clamp(feats.norm(dim=-1, keepdim=True), min=1e-12)

            feats_np = feats.detach().cpu().numpy()
            idx_np = listing_idx.detach().cpu().numpy().astype(np.int64)

            for i in range(feats_np.shape[0]):
                li = int(idx_np[i])
                sums[li] += feats_np[i]
                counts[li] += 1

            n_done += feats_np.shape[0]

    t1 = time.perf_counter()
    dt = max(t1 - t0, 1e-9)
    ips = n_done / dt
    print(f"[BENCH] images_processed={n_done} seconds={dt:.3f} images_per_sec={ips:.3f}")

    # Aggregate listing vectors
    out_rows: List[Tuple[int, str, str, str, str, str, int]] = []

    for li, lk in enumerate(listing_keys):
        c = int(counts[li])
        if c <= 0:
            continue
        v = sums[li] / float(c)
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm <= 0:
            continue
        v = (v / nrm).astype(np.float32, copy=False)
        vec_txt = format_vec_pgvector(v)
        out_rows.append((int(lk.listing_id), lk.edited_date_iso, args.source, model_rev, args.pca_rev, vec_txt, c))

    print(f"[INFO] listing_vectors_ready={len(out_rows)}")

    # Benchmark mode: never write
    if args.benchmark_images and args.benchmark_images > 0:
        print("[INFO] benchmark mode: skipping DB writes.")
        conn.close()
        return

    if args.dry_run:
        print("[INFO] dry-run: skipping DB writes.")
        conn.close()
        return

    upsert_listing_vectors(conn, out_rows)
    conn.close()
    print("[OK] upsert complete.")


if __name__ == "__main__":
    main()
