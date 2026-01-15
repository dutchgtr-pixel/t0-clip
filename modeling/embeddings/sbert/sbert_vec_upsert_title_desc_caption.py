#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""sbert_vec_upsert_title_desc_cap_PUBLIC.py

Public-release (sanitized) embedding upsert pipeline.

What it does
------------
1) Builds sentence embeddings from merged listing text:
      title + "\n\n" + description + "\n\n" + aggregated image captions
2) Encodes with a Transformer + mean pooling (SentenceTransformers wrapper)
3) Projects embeddings to a smaller dimension D using PCA (D ∈ {64, 128, 256})
4) Upserts the projected vectors into Postgres (pgvector) tables:
      ml.sbert_vec64_v1   when --pca-dim 64
      ml.sbert_vec128_v1  when --pca-dim 128
      ml.sbert_vec256_v1  when --pca-dim 256

Sanitization changes vs. the internal/original script
-----------------------------------------------------
- Standardized the primary key column name to listing_id.
- Removed dataset-specific schema/table names. Source tables are configurable via CLI.
- Kept the algorithm and DB schema pattern intact.

Expected source tables (defaults; override via CLI)
---------------------------------------------------
Listings table (default: public.listings):
  - listing_id        BIGINT (or similar)
  - edited_at         TIMESTAMPTZ  (t0 / snapshot time)
  - title             TEXT
  - description       TEXT
  - media_scraped_at  TIMESTAMPTZ  (optional; used for "refresh if captions arrived later")
  - spam_flag         (nullable; used only for optional spam filtering)

Image assets table (default: public.listing_image_assets):
  - listing_id        BIGINT
  - image_index       INT
  - caption_text      TEXT

Security note
-------------
This script dynamically inserts identifiers (table/column names) into SQL.
For safety, identifiers are validated against conservative regexes and must be
trusted configuration values (do not pass untrusted user input).

"""  # noqa: E501

from __future__ import annotations

import os
import re
import sys
import time
import pickle
import hashlib
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from sentence_transformers import SentenceTransformer, models
from sklearn.decomposition import IncrementalPCA


ALLOWED_DIMS = (64, 128, 256)

# Conservative identifier validators (avoid SQL injection via identifiers)
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_COL_RE   = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")  # column or unqualified identifier


def _validate_table_ident(name: str, arg: str) -> str:
    if not name or not _TABLE_RE.match(name):
        raise ValueError(
            f"Invalid {arg}='{name}'. Expected schema.table or table with only letters/digits/_."
        )
    return name


def _validate_col_ident(name: str, arg: str) -> str:
    if not name or not _COL_RE.match(name):
        raise ValueError(f"Invalid {arg}='{name}'. Expected a simple identifier (letters/digits/_).")
    return name


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def vec_to_pgvector_literal(v: np.ndarray) -> str:
    # pgvector text input format: '[1,2,3]'
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def detect_device(cli_device: str) -> str:
    if cli_device and cli_device != "auto":
        return cli_device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def require_cuda_if_requested(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled torch build."
            )
    except Exception as e:
        raise RuntimeError(f"CUDA requested but not available: {e}") from e


def build_merged_text(title: str, desc: str, captions: str) -> str:
    title = (title or "").strip()
    desc = (desc or "").strip()
    captions = (captions or "").strip()

    parts: List[str] = []
    if title:
        parts.append(title)
    if desc:
        parts.append(desc)
    if captions:
        parts.append(captions)

    return "\n\n".join(parts).strip()


def build_transformer_meanpool_embedder(model_id: str, device: str, max_seq_len: int) -> SentenceTransformer:
    """
    Build a SentenceTransformer model from a HuggingFace transformer + MEAN pooling.

    This works for many transformer checkpoints that are not packaged as native
    sentence-transformers models.
    """
    tr = models.Transformer(model_id, max_seq_length=max_seq_len)
    pool = models.Pooling(
        word_embedding_dimension=tr.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    return SentenceTransformer(modules=[tr, pool], device=device)


def resolve_table_and_indexes(pca_dim: int) -> Tuple[str, str, str]:
    if pca_dim not in ALLOWED_DIMS:
        raise ValueError(f"--pca-dim must be one of {ALLOWED_DIMS}, got {pca_dim}")
    table = f"ml.sbert_vec{pca_dim}_v1"
    idx_lookup = f"sbert_vec{pca_dim}_v1_lookup"
    idx_listing = f"sbert_vec{pca_dim}_v1_listing_edit"
    return table, idx_lookup, idx_listing


def ddl_for_dim(pca_dim: int, table: str, idx_lookup: str, idx_listing: str) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS {table} (
  listing_id     bigint      NOT NULL,
  edited_at      timestamptz NOT NULL,
  source         text        NOT NULL,
  model_rev      text        NOT NULL,
  pca_rev        text        NOT NULL,
  text_sha1      text        NOT NULL,
  vec            vector({pca_dim}) NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (listing_id, edited_at, source, model_rev, pca_rev)
);

CREATE INDEX IF NOT EXISTS {idx_lookup}
  ON {table} (source, model_rev, pca_rev, edited_at DESC);

CREATE INDEX IF NOT EXISTS {idx_listing}
  ON {table} (listing_id, edited_at DESC);
"""


def select_candidates_sql(
    table: str,
    listings_table: str,
    image_assets_table: str,
    col_listing_id: str,
    col_edited_at: str,
    col_title: str,
    col_description: str,
    col_media_scraped_at: str,
    col_spam_flag: str,
    col_caption_text: str,
    col_image_index: str,
) -> str:
    """
    Candidate selection:
      - Missing embedding rows (no matching row in {table})
      - Refresh rows where media_scraped_at > embedding.created_at (captions arrived later)

    We also explicitly exclude rows where merged text is empty to prevent "can't embed"
    infinite loops.

    Note: If your schema does not have media_scraped_at, you can set --media-scraped-at-col
    to a column that is always NULL; refresh selection will effectively be disabled.
    """
    spam_filter = f"( %(include_spam)s OR l.{col_spam_flag} IS NULL )" if col_spam_flag else "TRUE"

    return f"""
WITH caps AS (
  SELECT
      {col_listing_id} AS listing_id,
      string_agg(NULLIF(btrim({col_caption_text}), ''), ' | ' ORDER BY {col_image_index}) AS captions
  FROM {image_assets_table}
  GROUP BY {col_listing_id}
),
base AS (
  SELECT DISTINCT ON (l.{col_listing_id}, l.{col_edited_at})
      l.{col_listing_id}                 AS listing_id,
      l.{col_edited_at}                  AS edited_at,
      l.{col_media_scraped_at}           AS media_scraped_at,
      COALESCE(l.{col_title},'')         AS title,
      COALESCE(l.{col_description},'')   AS description,
      COALESCE(c.captions,'')            AS captions,
      s.text_sha1                        AS existing_sha1,
      s.created_at                       AS existing_created_at
  FROM {listings_table} l
  LEFT JOIN caps c
    ON c.listing_id = l.{col_listing_id}
  LEFT JOIN {table} s
    ON s.listing_id = l.{col_listing_id}
   AND s.edited_at  = l.{col_edited_at}
   AND s.source     = %(source)s
   AND s.model_rev  = %(model_rev)s
   AND s.pca_rev    = %(pca_rev)s
  WHERE
      l.{col_listing_id} IS NOT NULL
      AND l.{col_edited_at} IS NOT NULL
      AND {spam_filter}

      -- prevent infinite loops on empty text rows:
      AND (
        COALESCE(l.{col_title},'') <> ''
        OR COALESCE(l.{col_description},'') <> ''
        OR COALESCE(c.captions,'') <> ''
      )

      AND (
        s.listing_id IS NULL
        OR (
          l.{col_media_scraped_at} IS NOT NULL
          AND s.created_at IS NOT NULL
          AND s.created_at < l.{col_media_scraped_at}
        )
      )
  ORDER BY l.{col_listing_id}, l.{col_edited_at}, l.{col_media_scraped_at} DESC NULLS LAST
)
SELECT *
FROM base
ORDER BY edited_at ASC, listing_id ASC
LIMIT %(lim)s;
"""


def upsert_sql(table: str) -> str:
    return f"""
INSERT INTO {table}
  (listing_id, edited_at, source, model_rev, pca_rev, text_sha1, vec)
VALUES %s
ON CONFLICT (listing_id, edited_at, source, model_rev, pca_rev)
DO UPDATE SET
  text_sha1  = EXCLUDED.text_sha1,
  vec        = EXCLUDED.vec,
  created_at = now();
"""


def touch_unchanged_sql(table: str) -> str:
    """
    If a row is selected only because media_scraped_at > embedding.created_at,
    but the merged text hash is unchanged, we "touch" created_at to be >= media_scraped_at
    so the row stops getting selected forever.
    """
    return f"""
WITH v(listing_id, edited_at, source, model_rev, pca_rev, media_scraped_at) AS (VALUES %s)
UPDATE {table} s
   SET created_at = GREATEST(now(), COALESCE(v.media_scraped_at, now()))
  FROM v
 WHERE s.listing_id=v.listing_id
   AND s.edited_at=v.edited_at
   AND s.source=v.source
   AND s.model_rev=v.model_rev
   AND s.pca_rev=v.pca_rev;
"""


def save_pca(pca: IncrementalPCA, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(pca)
    path.write_bytes(blob)
    return hashlib.sha1(blob).hexdigest()[:12]


def load_pca(path: Path) -> Tuple[IncrementalPCA, str]:
    blob = path.read_bytes()
    pca = pickle.loads(blob)
    return pca, hashlib.sha1(blob).hexdigest()[:12]


def fit_pca(
    conn,
    embedder: SentenceTransformer,
    device: str,
    pca_dim: int,
    enc_batch: int,
    fit_limit: int,
    include_spam: bool,
    listings_table: str,
    image_assets_table: str,
    col_listing_id: str,
    col_edited_at: str,
    col_title: str,
    col_description: str,
    col_spam_flag: str,
    col_caption_text: str,
    col_image_index: str,
) -> IncrementalPCA:
    """Fit an IncrementalPCA model on a sample of merged texts from the DB."""
    native_dim = embedder.get_sentence_embedding_dimension()

    spam_filter = f"( %s OR l.{col_spam_flag} IS NULL )" if col_spam_flag else "TRUE"

    sql = f"""
    WITH caps AS (
      SELECT
          {col_listing_id} AS listing_id,
          string_agg(NULLIF(btrim({col_caption_text}), ''), ' | ' ORDER BY {col_image_index}) AS captions
      FROM {image_assets_table}
      GROUP BY {col_listing_id}
    )
    SELECT
      COALESCE(l.{col_title},'') || E'\\n\\n' ||
      COALESCE(l.{col_description},'') || E'\\n\\n' ||
      COALESCE(c.captions,'')
    FROM {listings_table} l
    LEFT JOIN caps c
      ON c.listing_id = l.{col_listing_id}
    WHERE
      l.{col_listing_id} IS NOT NULL
      AND l.{col_edited_at} IS NOT NULL
      AND {spam_filter}
    ORDER BY l.{col_edited_at} DESC
    LIMIT %s;
    """

    with conn.cursor() as cur:
        # Two parameters: include_spam flag (for spam_filter) and fit_limit
        cur.execute(sql, (include_spam, fit_limit) if col_spam_flag else (fit_limit,))
        texts = [(r[0] or "").strip() for r in cur.fetchall()]

    texts = [t for t in texts if t]
    if not texts:
        raise RuntimeError("No texts found for PCA fit")

    pca = IncrementalPCA(n_components=pca_dim, batch_size=2048)

    for i in range(0, len(texts), 2048):
        batch = texts[i : i + 2048]
        emb = embedder.encode(
            batch,
            batch_size=enc_batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device,
        )
        emb = np.asarray(emb, dtype=np.float32).reshape(len(batch), native_dim)
        pca.partial_fit(emb)

    return pca


def run_loop(
    conn,
    embedder: SentenceTransformer,
    pca: IncrementalPCA,
    device: str,
    pca_dim: int,
    table: str,
    source: str,
    model_rev: str,
    pca_rev: str,
    db_batch: int,
    enc_batch: int,
    max_rows: int,
    sleep_s: float,
    include_spam: bool,
    listings_table: str,
    image_assets_table: str,
    col_listing_id: str,
    col_edited_at: str,
    col_title: str,
    col_description: str,
    col_media_scraped_at: str,
    col_spam_flag: str,
    col_caption_text: str,
    col_image_index: str,
    touch_unchanged: bool = True,
) -> int:
    total = 0

    sel_sql = select_candidates_sql(
        table=table,
        listings_table=listings_table,
        image_assets_table=image_assets_table,
        col_listing_id=col_listing_id,
        col_edited_at=col_edited_at,
        col_title=col_title,
        col_description=col_description,
        col_media_scraped_at=col_media_scraped_at,
        col_spam_flag=col_spam_flag,
        col_caption_text=col_caption_text,
        col_image_index=col_image_index,
    )
    ins_sql = upsert_sql(table)
    t_sql = touch_unchanged_sql(table)

    ins_template = "(%s, %s, %s, %s, %s, %s, %s::vector)"
    touch_template = "(%s, %s, %s, %s, %s, %s)"

    while True:
        if max_rows and total >= max_rows:
            break

        lim = min(db_batch, max_rows - total) if max_rows else db_batch

        with conn.cursor() as cur:
            cur.execute(
                sel_sql,
                {
                    "source": source,
                    "model_rev": model_rev,
                    "pca_rev": pca_rev,
                    "lim": lim,
                    "include_spam": include_spam,
                },
            )
            rows = cur.fetchall()

        if not rows:
            if sleep_s and sleep_s > 0:
                print("[OK] no candidates; sleeping...", flush=True)
                time.sleep(sleep_s)
                continue
            break

        # de-dupe inside batch by (listing_id, edited_at)
        unique: Dict[Tuple[int, Any], Tuple[str, str]] = {}
        touch_rows: List[Tuple[Any, ...]] = []

        skipped_empty = 0
        skipped_unchanged = 0

        # rows = (listing_id, edited_at, media_scraped_at, title, description, captions, old_sha1, old_created)
        for (listing_id, edited_at, media_at, title, desc, captions, old_sha1, old_created) in rows:
            merged = build_merged_text(title, desc, captions)
            if not merged:
                skipped_empty += 1
                continue

            new_sha1 = sha1_text(merged)

            # unchanged text: if it was selected due to refresh condition, TOUCH it so it stops reselecting forever
            if (old_sha1 or "") == new_sha1:
                skipped_unchanged += 1
                if touch_unchanged and old_created is not None and media_at is not None:
                    touch_rows.append((int(listing_id), edited_at, source, model_rev, pca_rev, media_at))
                continue

            key = (int(listing_id), edited_at)
            prev = unique.get(key)
            if prev is None or len(merged) > len(prev[0]):
                unique[key] = (merged, new_sha1)

        # First: touch unchanged refresh rows (cheap DB update)
        touched = 0
        if touch_rows:
            with conn.cursor() as cur:
                execute_values(cur, t_sql, touch_rows, template=touch_template, page_size=1000)
            conn.commit()
            touched = len(touch_rows)

        # If nothing to embed/upsert this batch, do NOT loop forever
        if not unique:
            print(
                f"[OK] batch no-op (touched_unchanged={touched} sql_rows={len(rows)} "
                f"skipped_empty={skipped_empty} skipped_unchanged={skipped_unchanged})",
                flush=True,
            )
            if sleep_s and sleep_s > 0:
                time.sleep(sleep_s)
                continue

            # If we touched something, the next SELECT should drop them; loop once more.
            if touched > 0:
                continue

            # Otherwise, we cannot make progress (should be rare now that SQL excludes empty text)
            break

        items = list(unique.items())
        texts = [v[0] for (_k, v) in items]
        keys = [k for (k, _v) in items]
        sha1s = [v[1] for (_k, v) in items]

        emb_native = embedder.encode(
            texts,
            batch_size=enc_batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device,
        )
        emb_proj = pca.transform(np.asarray(emb_native, dtype=np.float32))
        if emb_proj.shape[1] != pca_dim:
            raise RuntimeError(f"PCA produced dim={emb_proj.shape[1]}, expected {pca_dim}")

        recs = []
        for ((lid, ed), txt_sha1, vecp) in zip(keys, sha1s, emb_proj):
            recs.append((lid, ed, source, model_rev, pca_rev, txt_sha1, vec_to_pgvector_literal(vecp)))

        with conn.cursor() as cur:
            execute_values(cur, ins_sql, recs, template=ins_template, page_size=1000)
        conn.commit()

        total += len(recs)
        print(
            f"[OK] dim={pca_dim} upserted={len(recs)} total={total} "
            f"(sql_rows={len(rows)} touched_unchanged={touched} skipped_empty={skipped_empty} skipped_unchanged={skipped_unchanged}) "
            f"table={table} source={source} model_rev={model_rev} pca_rev={pca_rev}",
            flush=True,
        )

        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    return total


def main() -> None:
    ap = argparse.ArgumentParser()

    # DB / output
    ap.add_argument("--pg-dsn", default=os.getenv("PG_DSN", ""), help="Postgres DSN")
    ap.add_argument("--ensure-table", action="store_true", help="Create schema/table/indexes if missing")

    ap.add_argument("--source", default=os.getenv("SBERT_SOURCE", "title_desc_cap"))
    ap.add_argument("--model", default=os.getenv("SBERT_MODEL", "bert-base-uncased"))
    ap.add_argument("--model-rev", default=os.getenv("SBERT_MODEL_REV", "meanpool_v1"))

    ap.add_argument("--pca-dim", type=int, default=int(os.getenv("SBERT_PCA_DIM", "64")))
    ap.add_argument("--pca-path", default=os.getenv("SBERT_PCA_PATH", ""))
    ap.add_argument("--fit-pca", action="store_true", help="Fit PCA and overwrite --pca-path")
    ap.add_argument("--fit-only", action="store_true", help="Fit PCA then exit (no backfill)")
    ap.add_argument("--pca-fit-limit", type=int, default=int(os.getenv("SBERT_PCA_FIT_LIMIT", "30000")))

    ap.add_argument("--db-batch", type=int, default=int(os.getenv("SBERT_DB_BATCH", "2000")))
    ap.add_argument("--enc-batch", type=int, default=int(os.getenv("SBERT_ENC_BATCH", "64")))
    ap.add_argument("--max-rows", type=int, default=int(os.getenv("SBERT_MAX_ROWS", "0")), help="0=unlimited")
    ap.add_argument("--sleep-s", type=float, default=float(os.getenv("SBERT_SLEEP_S", "0")))

    ap.add_argument("--device", default=os.getenv("SBERT_DEVICE", "auto"), choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-seq-len", type=int, default=int(os.getenv("SBERT_MAX_SEQ_LEN", "256")))

    ap.add_argument("--include-spam", action="store_true", help="Embed spam rows too (default: exclude spam)")

    ap.add_argument(
        "--no-touch-unchanged",
        action="store_true",
        help="Disable touching unchanged refresh rows (not recommended).",
    )

    # Source schema (configurable identifiers)
    ap.add_argument("--listings-table", default=os.getenv("LISTINGS_TABLE", "public.listings"))
    ap.add_argument("--image-assets-table", default=os.getenv("IMAGE_ASSETS_TABLE", "public.listing_image_assets"))

    ap.add_argument("--listing-id-col", default=os.getenv("LISTING_ID_COL", "listing_id"))
    ap.add_argument("--edited-at-col", default=os.getenv("EDITED_AT_COL", "edited_at"))
    ap.add_argument("--title-col", default=os.getenv("TITLE_COL", "title"))
    ap.add_argument("--description-col", default=os.getenv("DESCRIPTION_COL", "description"))
    ap.add_argument("--media-scraped-at-col", default=os.getenv("MEDIA_SCRAPED_AT_COL", "media_scraped_at"))
    ap.add_argument("--spam-flag-col", default=os.getenv("SPAM_FLAG_COL", "spam_flag"))

    ap.add_argument("--caption-text-col", default=os.getenv("CAPTION_TEXT_COL", "caption_text"))
    ap.add_argument("--image-index-col", default=os.getenv("IMAGE_INDEX_COL", "image_index"))

    args = ap.parse_args()

    if not args.pg_dsn:
        print("ERROR: --pg-dsn (or PG_DSN) is required", file=sys.stderr)
        raise SystemExit(2)

    if args.pca_dim not in ALLOWED_DIMS:
        print(f"ERROR: --pca-dim must be one of {ALLOWED_DIMS}", file=sys.stderr)
        raise SystemExit(2)

    if not args.pca_path:
        print("ERROR: --pca-path is required (store PCA model on disk)", file=sys.stderr)
        raise SystemExit(2)

    # Validate identifiers
    listings_table = _validate_table_ident(args.listings_table, "--listings-table")
    image_assets_table = _validate_table_ident(args.image_assets_table, "--image-assets-table")

    col_listing_id = _validate_col_ident(args.listing_id_col, "--listing-id-col")
    col_edited_at = _validate_col_ident(args.edited_at_col, "--edited-at-col")
    col_title = _validate_col_ident(args.title_col, "--title-col")
    col_description = _validate_col_ident(args.description_col, "--description-col")
    col_media_scraped_at = _validate_col_ident(args.media_scraped_at_col, "--media-scraped-at-col")
    col_spam_flag = _validate_col_ident(args.spam_flag_col, "--spam-flag-col") if args.spam_flag_col else ""

    col_caption_text = _validate_col_ident(args.caption_text_col, "--caption-text-col")
    col_image_index = _validate_col_ident(args.image_index_col, "--image-index-col")

    device = detect_device(args.device)
    require_cuda_if_requested(device)

    table, idx_lookup, idx_listing = resolve_table_and_indexes(args.pca_dim)

    print(f"[CONFIG] dim={args.pca_dim} table={table}", flush=True)
    print(
        f"[CONFIG] model={args.model} model_rev={args.model_rev} source={args.source} "
        f"device={device} include_spam={args.include_spam}",
        flush=True,
    )
    print(
        f"[CONFIG] listings_table={listings_table} image_assets_table={image_assets_table} "
        f"t0_col={col_edited_at} id_col={col_listing_id}",
        flush=True,
    )
    print(
        f"[CONFIG] pca_path={args.pca_path} fit_pca={args.fit_pca} fit_only={args.fit_only} "
        f"db_batch={args.db_batch} enc_batch={args.enc_batch} max_rows={args.max_rows} sleep_s={args.sleep_s}",
        flush=True,
    )

    conn = psycopg2.connect(args.pg_dsn)

    if args.ensure_table:
        ddl = ddl_for_dim(args.pca_dim, table, idx_lookup, idx_listing)
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
        print("[OK] ensured schema/table/indexes", flush=True)

    embedder = build_transformer_meanpool_embedder(args.model, device=device, max_seq_len=args.max_seq_len)
    print("[OK] sentence embedding dim:", embedder.get_sentence_embedding_dimension(), flush=True)

    pca_path = Path(args.pca_path)
    if args.fit_pca or not pca_path.exists():
        pca = fit_pca(
            conn=conn,
            embedder=embedder,
            device=device,
            pca_dim=args.pca_dim,
            enc_batch=args.enc_batch,
            fit_limit=args.pca_fit_limit,
            include_spam=args.include_spam,
            listings_table=listings_table,
            image_assets_table=image_assets_table,
            col_listing_id=col_listing_id,
            col_edited_at=col_edited_at,
            col_title=col_title,
            col_description=col_description,
            col_spam_flag=col_spam_flag,
            col_caption_text=col_caption_text,
            col_image_index=col_image_index,
        )
        pca_rev = save_pca(pca, pca_path)
        print(f"[OK] fitted PCA-{args.pca_dim} saved to {pca_path} pca_rev={pca_rev}", flush=True)
    else:
        pca, pca_rev = load_pca(pca_path)
        print(f"[OK] loaded PCA from {pca_path} pca_rev={pca_rev}", flush=True)

    if args.fit_only:
        print("[DONE] fit-only requested; exiting.", flush=True)
        conn.close()
        return

    wrote = run_loop(
        conn=conn,
        embedder=embedder,
        pca=pca,
        device=device,
        pca_dim=args.pca_dim,
        table=table,
        source=args.source,
        model_rev=args.model_rev,
        pca_rev=pca_rev,
        db_batch=args.db_batch,
        enc_batch=args.enc_batch,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else 0,
        sleep_s=float(args.sleep_s),
        include_spam=bool(args.include_spam),
        listings_table=listings_table,
        image_assets_table=image_assets_table,
        col_listing_id=col_listing_id,
        col_edited_at=col_edited_at,
        col_title=col_title,
        col_description=col_description,
        col_media_scraped_at=col_media_scraped_at,
        col_spam_flag=col_spam_flag,
        col_caption_text=col_caption_text,
        col_image_index=col_image_index,
        touch_unchanged=(not bool(args.no_touch_unchanged)),
    )

    print(f"[DONE] dim={args.pca_dim} upserted={wrote} table={table} model_rev={args.model_rev} pca_rev={pca_rev}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
