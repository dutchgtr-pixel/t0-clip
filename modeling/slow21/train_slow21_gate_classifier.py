#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Public release note
# -----------------------------------------------------------------------------
# This file has been sanitized for public release:
# - Platform-specific identifiers have been removed or generalized.
# - The canonical primary key is now `listing_id` (formerly platform-specific IDs).
# - No credentials are embedded; runtime configuration is expected via environment variables.
# See: REDACTION_REPORT_SURVIVAL_MODEL.md for details.
# -----------------------------------------------------------------------------

"""
train_xgb_aft_duration_pg.py — AFT with strict, leak-safe anchors @ t₀ and 7d-aware classification.

Upgrades:
• Leak-safe anchors at t₀: model × storage × CS × SEV (no gen mixing) + 30/60d blend → ptv_anchor_strict_t0.
• Robust AFT (no exp()), monotone constraints, boundary-aware isotonic, JSON-safe metrics.
• Silent training (no per-iter spam).
• NEW: Persist eval-time predictions for SOLD rows into ml.predictions_eval_sold_v1
  so we can analyze how wrong/right the model was on real sales.
• NEW: Optional Optuna hyperparameter tuning (50 trials by default) with multi-metric objective.
• NEW: Top-20 feature importances (gain %) logged at the end.
"""

import os, json, math, pickle, argparse, pathlib, datetime as dt, copy, hashlib, time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import psycopg, psycopg.rows
from sklearn.isotonic import IsotonicRegression
import optuna
import re
from psycopg import sql as psql


# ---------------- CLI ----------------

# -----------------------------------------------------------------------------
# Feature contracts / relation safety helpers
# -----------------------------------------------------------------------------

_REL_PART_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

def _validate_relname(relname: str) -> tuple[str, str]:
    """Validate and split a relation name like 'schema.table'.

    We intentionally restrict to simple identifiers to prevent SQL injection
    when relation names are provided via CLI args.
    """
    if not isinstance(relname, str) or not relname:
        raise ValueError("relname must be a non-empty string")
    parts = relname.split(".")
    if len(parts) == 1:
        schema, name = "public", parts[0]
    elif len(parts) == 2:
        schema, name = parts
    else:
        raise ValueError(f"Invalid relation name: {relname!r} (expected 'table' or 'schema.table')")
    if not _REL_PART_RE.match(schema) or not _REL_PART_RE.match(name):
        raise ValueError(f"Invalid relation name: {relname!r} (unsafe identifier)")
    return schema, name


def _rel_ident(relname: str) -> psql.Composed:
    schema, name = _validate_relname(relname)
    return psql.SQL(".").join([psql.Identifier(schema), psql.Identifier(name)])


def get_relation_columns(conn, relname: str) -> list[str]:
    """Return column names (ordinal order) for a schema-qualified relation.

    IMPORTANT:
      We use pg_catalog instead of information_schema because in many Postgres
      setups information_schema.columns does NOT list MATERIALIZED VIEW columns.
      pg_catalog works for tables, views, and materialized views.

    We still apply strict identifier validation to stay injection-safe.
    """
    schema, name = _validate_relname(relname)

    sql = """
        SELECT a.attname
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s
          AND c.relname = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
    """

    with conn.cursor() as cur:
        cur.execute(sql, (schema, name))
        return [r[0] for r in cur.fetchall()]
def load_feature_set_hash(conn, feature_set_name: str) -> str:
    """Return the expected features_hash for a feature_set."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT features_hash
            FROM ml.feature_set
            WHERE name = %(name)s
            """,
            {"name": feature_set_name},
        )
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Feature set not found in ml.feature_set: {feature_set_name!r}")
    return row[0]


def load_feature_contract_spec(conn, feature_set_name: str) -> list[tuple[str, str]]:
    """Return ordered (feature_name, expr_sql) for a feature_set."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT fc.feature_name, fc.expr_sql
            FROM ml.feature_contract fc
            JOIN ml.feature_set fs ON fs.set_id = fc.set_id
            WHERE fs.name = %(name)s
            ORDER BY fc.ordinal
            """,
            {"name": feature_set_name},
        )
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"No feature contract rows found for feature set: {feature_set_name!r}")
    return [(r[0], r[1]) for r in rows]


def assert_feature_contract(conn, feature_set_name: str) -> None:
    """Fail fast if the DB relation no longer matches the contract."""
    with conn.cursor() as cur:
        cur.execute("SELECT ml.assert_feature_contract(%(name)s)", {"name": feature_set_name})
        _ = cur.fetchone()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="tom_features_v1")
    ap.add_argument("--model-key",   default=None)
    ap.add_argument("--outdir",      required=True)

    # Postgres session safety/performance knobs (applied by the script)
    ap.add_argument("--pg_lock_timeout", default=os.getenv("PG_LOCK_TIMEOUT", "5s"),
                    help="Postgres lock_timeout for this session (e.g. 5s). Helps avoid hanging on locks.")
    ap.add_argument("--pg_statement_timeout", default=os.getenv("PG_STATEMENT_TIMEOUT", "20min"),
                    help="Postgres statement_timeout for this session (e.g. 20min). Prevents runaway queries.")
    ap.add_argument("--pg_jit", default=os.getenv("PG_JIT", "off"), choices=["on", "off"],
                    help="Enable/disable Postgres JIT for this session (default off for predictability).")
    ap.add_argument("--pg_application_name", default=os.getenv("PG_APPLICATION_NAME", "train_aft"),
                    help="Postgres application_name for this session (helps debug in pg_stat_activity).")
    ap.add_argument("--cert_max_age_hours", type=int, default=int(os.getenv("CERT_MAX_AGE_HOURS", "24")),
                    help="Max allowed age (hours) for certified feature stores before training is allowed.")
    ap.add_argument("--skip_cert_checks", action="store_true",
                    help="Skip audit.require_certified_strict preflight checks (not recommended).")
    ap.add_argument("--use_stock_features", action="store_true", help="Attach live-inventory stock features in Python (no DB objects): stock_n_sm4_gen_sbucket, stock_n_sm4_gen, stock_share_sbucket.")

    # Eval window & censored threshold
    ap.add_argument("--eval_days",           type=int,   default=7)
    ap.add_argument("--censored_min_days",   type=int,   default=7)
    ap.add_argument("--censored_inactive_min_active_days", type=float, default=1.0,
                    help="For inclusion as a censored row (sold_event=false), require that the first inactive meta_edited_at occurs at least this many days after edited_date (t0). 0 disables the filter.")
    ap.add_argument("--censored_inactive_filter_mode", type=str, default="drop_early_inactive",
                    choices=["drop_early_inactive", "off"],
                    help="How to apply inactive-table hygiene to the censored set: drop_early_inactive removes rows that were marked inactive too soon after t0; off keeps current behavior.")


    # SVAL/CAL window (recent SOLD slice immediately before eval window)
    ap.add_argument(
        "--sval_days",
        "--cal_days",
        type=int,
        default=None,
        help=(
            "Days of SOLD data immediately BEFORE the eval window to use as the calibration/early-stopping slice (sval). "
            "If omitted, keeps the current default behavior: cal_from_days = max(eval_days + 14, 21) "
            "(about 14 days when eval_days>=7)."
        ),
    )

    # Training/weights/calibration
    ap.add_argument("--winsor_q",            type=float, default=0.9995)
    ap.add_argument("--half_life_days",      type=float, default=150.0)
    
    # WOE anchor (supervised prior) — governance + reproducibility
    ap.add_argument(
        "--woe_folds",
        type=int,
        default=5,
        help=(
            "Time-based folds for OUT-OF-FOLD WOE anchor scoring on TRAIN SOLD rows. "
            "Use 1 or --disable_woe_oof to fall back to single final mapping (less strict)."
        ),
    )
    ap.add_argument(
        "--woe_eps",
        type=float,
        default=0.5,
        help="Additive smoothing (eps) for WOE tables. Must match the SQL live scorer.",
    )
    ap.add_argument(
        "--woe_band_schema_version",
        type=int,
        default=1,
        help="WOE banding schema version (stored in ml.woe_anchor_model_registry_v1.band_schema_version).",
    )
    ap.add_argument(
        "--disable_woe_oof",
        action="store_true",
        help="Disable OOF WOE scoring for train sold rows (uses final mapping for all rows).",
    )
    ap.add_argument(
        "--disable_woe_persist",
        action="store_true",
        help="Do not persist WOE anchor artifacts to Postgres (even if --no_db_writes is NOT set).",
    )
    ap.add_argument("--slow_tail_weight",    type=float, default=2.0)  # extra weight on SOLD >7d (see xgb_aft_train_predict)
    ap.add_argument("--very_slow_tail_weight", type=float, default=1.0,
                    help="Extra weight for SOLD durations >21d (504h). Use 1.0 to disable.")

    ap.add_argument("--boundary_focus_k",    type=float, default=0.6)
    ap.add_argument("--boundary_focus_sigma",type=float, default=10.0)
    ap.add_argument("--seed",                type=int,   default=42)

    # XGBoost params (mapped from LGBM knobs)
    ap.add_argument("--learning_rate",       type=float, default=0.05)
    ap.add_argument("--n_estimators",        type=int,   default=1200)
    ap.add_argument("--num_leaves",          type=int,   default=63)     # -> max_leaves
    ap.add_argument("--min_data_in_leaf",    type=int,   default=10)     # -> min_child_weight (heuristic)
    ap.add_argument("--feature_fraction",    type=float, default=0.95)   # -> colsample_bytree
    ap.add_argument("--bagging_fraction",    type=float, default=0.90)   # -> subsample
    ap.add_argument("--lambda_l2",           type=float, default=1.5)    # -> reg_lambda
    ap.add_argument("--max_bin",             type=int,   default=256)
    ap.add_argument("--early_stopping_rounds", type=int, default=100)

    # AFT distribution
    ap.add_argument("--aft_dist",            type=str,   default="logistic",
                    choices=["logistic","normal","extreme"])
    ap.add_argument("--aft_scale",           type=float, default=1.0)

    # Anchors / region feature
    ap.add_argument("--use_strict_anchor",   action="store_true",
                    help="Add ptv_anchor_strict_t0 + anchor thickness")
    ap.add_argument("--geo_col",             type=str,   default="location_city")
    ap.add_argument("--region_map_csv",      type=str,   default="")
    ap.add_argument("--region_shrink_k",     type=float, default=150.0)

    # --- Geo mapping v4 (DB-backed) ---
    ap.add_argument(
        "--features_view",
        default="ml.tom_features_v1_enriched_ai_clean_mv",
        help=(
            "Feature source view/MV. Use your geo-enriched read view (e.g. "
            "ml.tom_features_v1_enriched_ai_clean_read_v) to include geo columns without "
            "touching the underlying matviews."
        ),
    )

    # --- External feature blocks (governed by feature contracts) ---
    ap.add_argument(
        "--image_features_view",
        default="ml.iphone_image_features_unified_v1",
        help="Vision features view (joined by generation,listing_id).",
    )
    ap.add_argument(
        "--image_feature_set",
        default="iphone_image_unified_v1_model",
        help="ml.feature_set name that defines the contract for --image_features_view.",
    )
    ap.add_argument("--image_chunk_size", type=int, default=20000)

    ap.add_argument(
        "--fusion_features_view",
        default="ml.v_damage_fusion_features_v2_scored",
        help="Damage fusion scored view (joined by listing_id).",
    )
    ap.add_argument(
        "--fusion_feature_set",
        default="damage_fusion_v2_scored_model",
        help="ml.feature_set name that defines the contract for --fusion_features_view.",
    )
    ap.add_argument("--fusion_chunk_size", type=int, default=20000)
    ap.add_argument(
        "--disable_fusion",
        action="store_true",
        help="If set, do not join damage fusion feature block (ablation / debugging).",
    )

    ap.add_argument(
        "--device_meta_features_view",
        default="ml.iphone_device_meta_encoded_v1",
        help="Encoded device meta view (generation/model-variant/color). Joined by generation,listing_id.",
    )
    ap.add_argument(
        "--device_meta_feature_set",
        default="iphone_device_meta_encoded_v1_model",
        help="ml.feature_set name that defines the contract for --device_meta_features_view.",
    )
    ap.add_argument("--device_meta_chunk_size", type=int, default=20000)
    ap.add_argument(
        "--disable_device_meta",
        action="store_true",
        help="If set, do not join device meta feature block (ablation / debugging).",
    )

    # Trainer-derived feature store (DB)
    ap.add_argument(
        "--trainer_derived_features_view",
        default="ml.trainer_derived_features_v1_mv",
        help=(
            "Trainer-derived feature store SOURCE relation (view or materialized view). "
            "Default is the precomputed MV (ml.trainer_derived_features_v1_mv) for performance. "
            "The script still fails closed by requiring the CERTIFIED entrypoint "
            "(ml.trainer_derived_feature_store_t0_v1_v) once before reading. "
            "If disabled or empty, trainer-derived features are computed in Python."
        ),
    )
    ap.add_argument(
        "--trainer_derived_chunk_size",
        type=int,
        default=25000,
        help="Chunk size for fetching trainer-derived store features.",
    )
    ap.add_argument(
        "--disable_trainer_derived_store",
        action="store_true",
        help="If set, compute trainer-derived features in Python instead of joining the DB store.",
    )

    ap.add_argument(
        "--geo_mode",
        default="off",
        choices=["off", "ohe", "priors", "both"],
        help=(
            "How to incorporate geo mapping columns. "
            "'ohe' adds one-hot encoded geo categories; "
            "'priors' adds smoothed numeric priors learned from the training sold events; "
            "'both' enables both."
        ),
    )
    ap.add_argument(
        "--geo_dim_view",
        default="ml.geo_dim_super_metro_v4_current_v",
        help=(
            "Fallback geo dimension view used to attach geo columns by listing_id when "
            "--features_view does not already contain them."
        ),
    )
    ap.add_argument(
        "--geo_cols",
        default="region_geo,super_metro_v4_geo,pickup_metro_30_200_geo,geo_match_method",
        help="Comma-separated geo categorical columns to one-hot encode when geo_mode includes 'ohe'.",
    )
    ap.add_argument(
        "--geo_prior_cols",
        default="region_geo,super_metro_v4_geo",
        help="Comma-separated geo categorical columns to compute smoothed priors when geo_mode includes 'priors'.",
    )
    ap.add_argument(
        "--geo_ohe_min_n",
        type=int,
        default=50,
        help=(
            "Per geo column, pool categories with < geo_ohe_min_n training examples into '__rare__' "
            "before one-hot encoding."
        ),
    )
    ap.add_argument(
        "--geo_prior_k",
        type=float,
        default=None,
        help="Shrinkage strength for geo priors. If omitted, defaults to --region_shrink_k.",
    )

    # Optuna
    ap.add_argument("--optuna-trials",       type=int,   default=50,
                    help="Number of Optuna trials for hyperparameter tuning (0 = disable)")
    
    # ---- SBERT vec{64,128,256} ----
    ap.add_argument(
        "--use_sbert_vec",
        "--use_sbert_vec64",
        action="store_true",
        help="Join ml.sbert_vec{dim}_v1 and expand vec(dim) into numeric features.",
    )
    ap.add_argument("--sbert_dim", type=int, default=64, choices=[64, 128, 256])
    ap.add_argument("--sbert_source", default="title_desc_cap")
    ap.add_argument("--sbert_model_rev", default="nb-bert-base_meanpool_v1")
    ap.add_argument("--sbert_pca_rev", default="")
    ap.add_argument("--sbert_prefix", default="")
    ap.add_argument("--sbert_table", default="")


    ap.add_argument("--sbert_use_prototypes", action="store_true",
                    help="Derive prototype cosine-sim features from SBERT embeddings (fit on TRAIN events only).")
    ap.add_argument("--sbert_proto_min_n", type=int, default=50,
                    help="Min rows required in train_events to create a prototype.")
    ap.add_argument("--sbert_drop_raw", action="store_true",
                    help="After deriving prototype features, drop raw sbert{dim}_XX columns (keeps *_present unless you drop it yourself).")



    # Dev
    ap.add_argument("--train_limit",         type=int,   default=None)
    ap.add_argument("--no_db_writes",        action="store_true")

    # Segment / generation filter
    ap.add_argument(
        "--gen-filter",
        type=int,
        default=None,
        help="If set, train only on rows where generation == this value (e.g. 13, 14, 15, 16, 17).",
    )

    return ap.parse_args()



# ---------------- Utils ----------------
NUM_BLOCK_EXACT  = {"listing_id","postal_code","id","listing_id","tise_id"}
NUM_BLOCK_SUBSTR = ("sold","duration","label","target","bucket","pred","actual","ground","gate","uuid","url","zip","plz")

# Explicit allowlist for good "sold"-ish features we actually WANT
ALLOW_NUM_FEATURES = {
    "delta_vs_sold_median_30d",
    "ptv_sold_30d",
    "ptv_ask_day",
    "delta_vs_ask_median_day",
}



# ---------------------------------------------------------------------------
# Optional Orderbook (OB) feature MV (used if not already present in features_view)
# ---------------------------------------------------------------------------
OB_FEATURES_VIEW = "ml.tom_ob_features_v1_mv"
OB_FEATURE_COLS: List[str] = [
    "ob_k_active_30d_sm",
    "ob_median_price_30d_sm",
    "ob_lr_vs_median_30d_sm",
    "ob_lr_vs_min_30d_sm",
    "ob_delta_vs_median_30d_sm",
    "ob_price_pctl_30d_sm",
    "ob_log_p90_p10_30d_sm",
    "ob_lr_over_spread_30d_sm",
    "ob_has_comp_30d_sm",
]

def log(msg: str):
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}][train_aft] {msg}", flush=True)

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Postgres session + certification helpers
# -----------------------------

def apply_pg_session_settings(conn, args) -> None:
    """
    Apply per-session Postgres settings so training doesn't hang silently on locks
    or run-away queries.
    """
    from psycopg import sql  # <-- THIS fixes NameError

    lock_to = getattr(args, "pg_lock_timeout", None)
    stmt_to = getattr(args, "pg_statement_timeout", None)
    app_name = getattr(args, "pg_application_name", None)
    jit = getattr(args, "pg_jit", None)

    with conn.cursor() as cur:
        if app_name:
            cur.execute(
                sql.SQL("SET application_name TO {}").format(sql.Literal(str(app_name)))
            )
        if lock_to:
            cur.execute(
                sql.SQL("SET lock_timeout TO {}").format(sql.Literal(str(lock_to)))
            )
        if stmt_to:
            cur.execute(
                sql.SQL("SET statement_timeout TO {}").format(sql.Literal(str(stmt_to)))
            )
        if jit in ("on", "off"):
            cur.execute(sql.SQL("SET jit TO {}").format(sql.SQL(jit)))

    conn.commit()



def relation_exists(conn, rel: str) -> bool:
    """Return True if schema-qualified relation exists in the current DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (rel,))
        return cur.fetchone()[0] is not None


def audit_require_certified_strict(conn, rel: str, max_age: Optional[str] = None) -> None:
    """
    Enforce certification on a feature store entrypoint.

    Supports both 1-arg and (rel, max_age) overloads; falls back automatically.
    """
    with conn.cursor() as cur:
        if max_age:
            try:
                cur.execute(
                    "SELECT audit.require_certified_strict(%s, %s::interval)",
                    (rel, max_age),
                )
                cur.fetchone()
                return
            except Exception as e:
                # If the 2-arg overload doesn't exist, fall back to 1-arg.
                msg = str(e)
                if "require_certified_strict" not in msg or "does not exist" not in msg:
                    raise

        cur.execute("SELECT audit.require_certified_strict(%s)", (rel,))
        cur.fetchone()


def preflight_cert_checks(conn, args) -> None:
    """
    Fail fast if any required feature store is not certified.

    This is intentionally cheap (no data reads), and avoids spending minutes
    loading features only to fail later.
    """
    if getattr(args, "skip_cert_checks", False):
        log("[cert] skip_cert_checks=True; skipping preflight certification checks")
        return

    max_age_hours = int(getattr(args, "cert_max_age_hours", 24))
    max_age = f"{max_age_hours} hours"

    checks: list[tuple[str, str]] = []

    # Base feature store (socio/market enriched)
    if getattr(args, "features_view", None):
        checks.append(("features_view", "ml.socio_market_feature_store_t0_v1_v"))

    # Geo store
    if getattr(args, "geo_mode", "both") != "off":
        checks.append(("geo_dim_view", "ml.geo_feature_store_t0_v1_v"))

    # Vision/image store
    if getattr(args, "image_features_view", None):
        checks.append(("image_features_view", "ml.vision_feature_store_t0_v1_v"))

    # Fusion store
    if getattr(args, "fusion_features_view", None):
        checks.append(("fusion_features_view", "ml.fusion_feature_store_t0_v1_v"))

    # Device meta store
    if getattr(args, "device_meta_features_view", None):
        checks.append(("device_meta_features_view", "ml.device_meta_store_t0_v1_v"))

    # Trainer-derived store (this one was the historical long pole)
    if getattr(args, "trainer_derived_features_view", None) and not getattr(args, "disable_trainer_derived_store", False):
        checks.append(("trainer_derived_features_view", "ml.trainer_derived_feature_store_t0_v1_v"))

    # De-dupe + enforce
    seen: set[str] = set()
    for label, rel in checks:
        if rel in seen:
            continue
        seen.add(rel)
        if not relation_exists(conn, rel):
            log(f"[cert] WARN: missing cert entrypoint (skipping): {rel} ({label})")
            continue
        audit_require_certified_strict(conn, rel, max_age=max_age)
        log(f"[cert] OK: {rel} ({label}) max_age={max_age}")

def ensure_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def winsorize_upper(x: np.ndarray, q: float) -> np.ndarray:
    if not (0 < q < 1):
        return x
    cap = np.nanquantile(x, q)
    return np.clip(x, 0, cap)

# ---- Numerics safety ----
# AFT uses log(time) internally; 0 hours will create -inf.
EPS_TIME_H = 1e-3  # 3.6 seconds

# Upper bound for right-censoring (keep huge, but finite float32-safe)
AFT_UB_CENS = 1e9


# ------------------------
# Stock (live-inventory) features
# ------------------------

def _sbucket_from_storage_gb(storage_gb: pd.Series) -> pd.Series:
    """Best-effort sbucket replication used across the SQL feature stores.

    Mirrors the common SQL CASE logic:
      - >= 900  -> 1024
      - >= 500  -> 512
      - >= 250  -> 256
      - >= 120  -> 128
      - else    -> storage_gb
    """
    s = pd.to_numeric(storage_gb, errors="coerce")
    b = np.where(
        s >= 900, 1024,
        np.where(
            s >= 500, 512,
            np.where(
                s >= 250, 256,
                np.where(s >= 120, 128, s),
            ),
        ),
    )
    return pd.Series(b, index=storage_gb.index).fillna(-1).astype("int32")


def _compute_stock_counts_by_group(
    df: pd.DataFrame,
    intervals: pd.DataFrame,
    *,
    group_cols: list[str],
    t0_ns_col: str,
) -> pd.Series:
    """Compute active-interval counts at t0 for each row in df, per group.

    intervals must contain:
      - group_cols
      - start_ns (int64)
      - end_ns   (int64), inclusive

    Count at t0 is:
      #starts <= t0  minus  #ends < t0

    This matches an inclusive interval [start, end].
    """
    out = pd.Series(0, index=df.index, dtype="int32")
    if df.empty or intervals.empty:
        return out

    # Precompute sorted boundary arrays per group (fast lookups)
    cache: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for key, sub in intervals.groupby(group_cols, sort=False):
        starts = sub["start_ns"].to_numpy(dtype="int64", copy=True)
        ends = sub["end_ns"].to_numpy(dtype="int64", copy=True)
        if len(starts) == 0:
            continue
        starts.sort()
        ends.sort()
        cache[key] = (starts, ends)

    if not cache:
        return out

    for key, idx in df.groupby(group_cols, sort=False).groups.items():
        se = cache.get(key)
        if se is None:
            continue
        starts, ends = se
        t0s = df.loc[idx, t0_ns_col].to_numpy(dtype="int64", copy=False)
        # inclusive end: subtract only those that ended strictly before t0
        counts = np.searchsorted(starts, t0s, side="right") - np.searchsorted(ends, t0s, side="left")
        out.loc[idx] = counts.astype("int32", copy=False)

    return out


def attach_stock_features_python(
    conn,
    df: pd.DataFrame,
    *,
    t0_col: str = "edited_date",
    geo_dim_view: str = "ml.geo_dim_super_metro_v4_t0_train_v",
    meta_view: str = "ml.socio_market_feature_store_train_v",
    listings_table: str = '"iPhone".iphone_listings',
) -> pd.DataFrame:
    """Attach live-inventory ('stock') features computed at t0.

    Adds exactly three features:
      - stock_n_sm4_gen_sbucket
      - stock_n_sm4_gen
      - stock_share_sbucket = stock_n_sm4_gen_sbucket / NULLIF(stock_n_sm4_gen, 0)

    No DB objects (views/MVs) are created; we only read from existing tables.
    """
    need = {"generation", "listing_id", t0_col}
    missing = need - set(df.columns)
    if missing:
        log(f"[stock] skipped (missing columns): {sorted(missing)}")
        out = df.copy()
        out["stock_n_sm4_gen_sbucket"] = 0
        out["stock_n_sm4_gen"] = 0
        out["stock_share_sbucket"] = np.nan
        return out

    out = df.copy()

    # Ensure segmentation cols
    if "super_metro_v4_geo" not in out.columns:
        out["super_metro_v4_geo"] = "unknown"
    out["super_metro_v4_geo"] = out["super_metro_v4_geo"].fillna("unknown").astype(str)

    if "sbucket" not in out.columns:
        if "storage_gb" in out.columns:
            out["sbucket"] = _sbucket_from_storage_gb(out["storage_gb"])
        else:
            out["sbucket"] = -1
    out["sbucket"] = pd.to_numeric(out["sbucket"], errors="coerce").fillna(-1).astype("int32")

    # Ensure t0 is UTC datetime for stable epoch conversion
    out[t0_col] = ensure_datetime_utc(out[t0_col])

    gens = (
        pd.to_numeric(out["generation"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not gens:
        out["stock_n_sm4_gen_sbucket"] = 0
        out["stock_n_sm4_gen"] = 0
        out["stock_share_sbucket"] = np.nan
        return out

    min_t0 = out[t0_col].min()
    max_t0 = out[t0_col].max()

    q = f"""
    WITH life AS (
      SELECT
        generation,
        listing_id,
        MIN(first_seen) AS first_seen,
        MAX(last_seen)  AS last_seen,
        MIN(sold_date) FILTER (WHERE sold_date IS NOT NULL) AS sold_date_min
      FROM {listings_table}
      WHERE generation = ANY(%(gens)s::int[])
        AND first_seen IS NOT NULL
        AND last_seen  IS NOT NULL
        AND first_seen <= %(max_t0)s
        AND last_seen  >= %(min_t0)s
      GROUP BY 1,2
    ),
    latest_flags AS (
      SELECT DISTINCT ON (generation, listing_id)
        generation,
        listing_id,
        spam,
        is_bidding
      FROM {listings_table}
      WHERE generation = ANY(%(gens)s::int[])
      ORDER BY generation, listing_id, last_seen DESC NULLS LAST
    ),
    geo AS (
      SELECT listing_id, super_metro_v4_geo
      FROM {geo_dim_view}
    ),
    meta AS (
      -- Latest known sbucket for each listing (stable enough for segmentation)
      SELECT DISTINCT ON (fs.generation, fs.listing_id)
        fs.generation,
        fs.listing_id,
        fs.sbucket
      FROM {meta_view} fs
      JOIN life l USING (generation, listing_id)
      WHERE fs.edited_date IS NOT NULL
      ORDER BY fs.generation, fs.listing_id, fs.edited_date DESC
    )
    SELECT
      l.generation,
      l.listing_id,
      COALESCE(g.super_metro_v4_geo, 'unknown') AS super_metro_v4_geo,
      COALESCE(m.sbucket, -1) AS sbucket,
      l.first_seen,
      CASE
        WHEN l.sold_date_min IS NOT NULL THEN LEAST(l.sold_date_min, l.last_seen)
        ELSE l.last_seen
      END AS live_end
    FROM life l
    JOIN latest_flags lf USING (generation, listing_id)
    LEFT JOIN geo  g USING (listing_id)
    LEFT JOIN meta m USING (generation, listing_id)
    WHERE lf.spam IS NULL
      AND (lf.is_bidding IS DISTINCT FROM TRUE);
    """

    try:
        intervals = pd.read_sql_query(
            q,
            conn,
            params={"gens": gens, "min_t0": min_t0, "max_t0": max_t0},
        )
    except Exception as e:
        log(f"[stock] interval query failed; disabling stock features: {type(e).__name__}: {e}")
        out["stock_n_sm4_gen_sbucket"] = 0
        out["stock_n_sm4_gen"] = 0
        out["stock_share_sbucket"] = np.nan
        return out

    if intervals.empty:
        log(f"[stock] no intervals returned (gens={gens})")
        out["stock_n_sm4_gen_sbucket"] = 0
        out["stock_n_sm4_gen"] = 0
        out["stock_share_sbucket"] = np.nan
        return out

    # Normalize types
    intervals["generation"] = pd.to_numeric(intervals["generation"], errors="coerce").fillna(-1).astype("int32")
    intervals["listing_id"] = pd.to_numeric(intervals["listing_id"], errors="coerce").fillna(-1).astype("int64")
    intervals["super_metro_v4_geo"] = intervals["super_metro_v4_geo"].fillna("unknown").astype(str)
    intervals["sbucket"] = pd.to_numeric(intervals["sbucket"], errors="coerce").fillna(-1).astype("int32")

    intervals["first_seen"] = ensure_datetime_utc(intervals["first_seen"])
    intervals["live_end"] = ensure_datetime_utc(intervals["live_end"])

    # Attach per-listing t0 (for start adjustment; fixes small t0 vs first_seen skew)
    key_map = (
        out[["generation", "listing_id", t0_col]]
        .drop_duplicates(subset=["generation", "listing_id"])
        .rename(columns={t0_col: "_t0"})
        .copy()
    )
    intervals = intervals.merge(key_map, on=["generation", "listing_id"], how="left")

    intervals["start"] = intervals["first_seen"]
    m = intervals["_t0"].notna()
    if m.any():
        # allow t0 to slightly precede first_seen (scrape lag)
        intervals.loc[m, "start"] = intervals.loc[m, ["first_seen", "_t0"]].min(axis=1)

    intervals["start_ns"] = intervals["start"].astype("int64")
    intervals["end_ns"] = intervals["live_end"].astype("int64")

    # Drop invalid intervals
    intervals = intervals.loc[intervals["end_ns"] >= intervals["start_ns"]].copy()
    if intervals.empty:
        out["stock_n_sm4_gen_sbucket"] = 0
        out["stock_n_sm4_gen"] = 0
        out["stock_share_sbucket"] = np.nan
        return out

    out["_t0_ns"] = out[t0_col].astype("int64")

    # --- Compute stock counts ---
    out["stock_n_sm4_gen"] = _compute_stock_counts_by_group(
        out,
        intervals,
        group_cols=["super_metro_v4_geo", "generation"],
        t0_ns_col="_t0_ns",
    )

    out["stock_n_sm4_gen_sbucket"] = _compute_stock_counts_by_group(
        out,
        intervals,
        group_cols=["super_metro_v4_geo", "generation", "sbucket"],
        t0_ns_col="_t0_ns",
    )

    out["stock_share_sbucket"] = (
        out["stock_n_sm4_gen_sbucket"].astype("float32")
        / out["stock_n_sm4_gen"].replace(0, np.nan).astype("float32")
    )

    out.drop(columns=["_t0_ns"], inplace=True, errors="ignore")

    log(
        f"[stock] attached: n={len(out)} "
        f"avg(stock_n_sm4_gen)={float(out['stock_n_sm4_gen'].mean()):.2f} "
        f"avg(stock_n_sm4_gen_sbucket)={float(out['stock_n_sm4_gen_sbucket'].mean()):.2f}"
    )

    return out

def add_missing_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Your pipeline median-imputes NaNs. That erases the "missingness is signal" effect.
    These flags restore that signal and usually improves performance for image/AI fields.
    """
    for c in cols:
        if c in df.columns:
            df[f"{c}__missing"] = pd.isna(df[c]).astype("int8")
    return df





def time_decay_weights(
    edited: pd.Series,
    half_life_days: float,
    ref_ts: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(edited), dtype=np.float32)
    t = ensure_datetime_utc(edited)
    if ref_ts is None:
        ref = t.max()
    else:
        ref = pd.Timestamp(ref_ts)
        if ref.tzinfo is None:
            ref = ref.tz_localize("UTC")
        else:
            ref = ref.tz_convert("UTC")
    age_days = (ref - t).dt.total_seconds() / 86400.0
    age_days = np.clip(age_days, 0, None)
    w = np.power(0.5, age_days / half_life_days).astype(np.float32)
    m = np.nanmean(w)
    return (w/m) if m else w

def boundary_focus_weights(y_hours: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """
    Put extra calibration/selection weight near the 21-day decision boundary (504h).

    This replaces the old 24h/7d (168h) bumps: the pipeline is now optimized for the
    21-day (504h) classifier behavior.

    Args:
      y_hours: true durations in hours.
      k: bump magnitude (0 disables).
      sigma: bump width in hours (must be >0).
    """
    if not k or k <= 0 or sigma <= 0:
        return np.ones_like(y_hours, dtype=np.float32)
    y = np.asarray(y_hours, dtype=float)
    THRESH = 21.0 * 24.0  # 504h
    bump = np.exp(-((np.abs(y - THRESH) / float(sigma)) ** 2.0))
    w = 1.0 + float(k) * bump
    m = np.nanmean(w)
    return (w / m).astype(np.float32) if m else w.astype(np.float32)

def encode_numeric(df: pd.DataFrame, median_from: Optional[pd.DataFrame]=None) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(np.int8)
    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    keep = []
    for c in num_cols:
        lc = c.lower()
        if c in NUM_BLOCK_EXACT:
            continue
        # if it matches blocked substring AND is NOT explicitly allowed → skip
        if any(s in lc for s in NUM_BLOCK_SUBSTR) and c not in ALLOW_NUM_FEATURES:
            continue
        keep.append(c)
    X = d[keep].replace([np.inf,-np.inf], np.nan)
    med_src = median_from[keep].median(numeric_only=True) if (median_from is not None and set(keep).issubset(median_from.columns)) else X.median(numeric_only=True)
    X = X.fillna(med_src)
    return X, keep

def mae_med_p90(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ae = np.abs(y_true - y_pred)
    if len(ae) == 0:
        return {"mae": float("nan"), "med": float("nan"), "p90": float("nan")}
    return {
        "mae": float(ae.mean()),
        "med": float(np.median(ae)),
        "p90": float(np.percentile(ae, 90.0))
    }

def fast72_metrics(y_true_hours: np.ndarray, y_pred_hours: np.ndarray) -> Dict[str, Any]:
    """
    7-day (168h) classification metrics. Internally uses fast72/slow72 keys.
    """
    n = len(y_true_hours)
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "fast72": {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "slow72": {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "confusion": {"tp_fast": 0, "fp_fast": 0, "tn_fast": 0, "fn_fast": 0},
            "pred_counts": {"pred_fast": 0, "pred_slow": 0},
            "true_counts": {"true_fast": 0, "true_slow": 0},
        }
    THRESH = 168.0  # 7 days in hours
    true_fast = (y_true_hours <= THRESH)
    pred_fast = (y_pred_hours <= THRESH)
    tp = int(np.sum(true_fast & pred_fast))
    fp = int(np.sum(~true_fast & pred_fast))
    tn = int(np.sum(~true_fast & ~pred_fast))
    fn = int(np.sum(true_fast & ~pred_fast))
    acc = (tp + tn) / n if n else float("nan")

    def prf(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else float("nan")
        rec  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec > 0 and rec > 0) else (0.0 if (prec == 0 or rec == 0) else float("nan"))
        return prec, rec, f1

    p_fast, r_fast, f1_fast = prf(tp, fp, fn)
    tp_s = tn
    fp_s = fn
    fn_s = fp
    p_slow, r_slow, f1_slow = prf(tp_s, fp_s, fn_s)

    return {
        "n": n,
        "accuracy": acc,
        "fast72": {"precision": p_fast, "recall": r_fast, "f1": f1_fast},
        "slow72": {"precision": p_slow, "recall": r_slow, "f1": f1_slow},
        "confusion": {"tp_fast": tp, "fp_fast": fp, "tn_fast": tn, "fn_fast": fn},
        "pred_counts": {"pred_fast": int(np.sum(pred_fast)), "pred_slow": int(np.sum(~pred_fast))},
        "true_counts": {"true_fast": int(np.sum(true_fast)), "true_slow": int(np.sum(~true_fast))},
    }

def canonicalize_fast72_for_json(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert fast72_metrics() output into a schema with ['fast']['f1'] etc.

    So in JSON we can always do:
      metrics->'fast7d_classification'->'fast'->>'f1'
    """
    if not isinstance(m, dict):
        return m
    return {
        "n": m.get("n", 0),
        "accuracy": m.get("accuracy"),
        "fast":  m.get("fast72", {}),
        "slow":  m.get("slow72", {}),
        "confusion": m.get("confusion", {}),
        "pred_counts": m.get("pred_counts", {}),
        "true_counts": m.get("true_counts", {}),
    }

def classify_at_threshold(y_true_hours: np.ndarray,
                          y_pred_hours: np.ndarray,
                          thresh_hours: float) -> Dict[str, Any]:
    n = len(y_true_hours)
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "fast": {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "slow": {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "confusion": {"tp_fast": 0, "fp_fast": 0, "tn_fast": 0, "fn_fast": 0},
            "pred_counts": {"pred_fast": 0, "pred_slow": 0},
            "true_counts": {"true_fast": 0, "true_slow": 0},
        }
    true_fast = (y_true_hours <= thresh_hours)
    pred_fast = (y_pred_hours <= thresh_hours)
    tp = int(np.sum(true_fast & pred_fast))
    fp = int(np.sum(~true_fast & pred_fast))
    tn = int(np.sum(~true_fast & ~pred_fast))
    fn = int(np.sum(true_fast & ~pred_fast))
    acc = (tp + tn) / n if n else float("nan")

    def prf(tp_, fp_, fn_):
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else float("nan")
        rec  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else float("nan")
        f1   = 2 * prec * rec / (prec + rec) if (prec > 0 and rec > 0) else (0.0 if (prec == 0 or rec == 0) else float("nan"))
        return prec, rec, f1

    p_fast, r_fast, f1_fast = prf(tp, fp, fn)
    tp_s = tn
    fp_s = fn
    fn_s = fp
    p_slow, r_slow, f1_slow = prf(tp_s, fp_s, fn_s)

    return {
        "n": n,
        "accuracy": acc,
        "fast": {"precision": p_fast, "recall": r_fast, "f1": f1_fast},
        "slow": {"precision": p_slow, "recall": r_slow, "f1": f1_slow},
        "confusion": {"tp_fast": tp, "fp_fast": fp, "tn_fast": tn, "fn_fast": fn},
        "pred_counts": {"pred_fast": int(np.sum(pred_fast)), "pred_slow": int(np.sum(~pred_fast))},
        "true_counts": {"true_fast": int(np.sum(true_fast)), "true_slow": int(np.sum(~true_fast))},
    }


# ---------------- Slow21 (504h) focus helpers ----------------
SLOW21_H = 21.0 * 24.0  # 504h (21 days)
LT10_H   = 10.0 * 24.0  # 240h (10 days) — used for collateral/sacrifice caps

# Hard cap on fast-seller (<10d) false-SLOW rate when optimizing the 21d classifier.
# Kept internal on purpose (no new CLI flag allowed).
FP_LT10_CAP_DEFAULT = 0.05

def slow21_sacrifice_rates(
    y_true_hours: np.ndarray,
    y_pred_hours: np.ndarray,
    slow_thresh_hours: float = SLOW21_H,
    lt10_thresh_hours: float = LT10_H,
) -> Dict[str, Any]:
    """
    Collateral metrics for the 21d classifier at `slow_thresh_hours` (default 504h).

    We treat "predicting SLOW" as: pred_hours > slow_thresh_hours.

    SAC_LT10 penalizes hurting very fast sellers (<10d):
      FP_LT10 = count(y < 240 AND pred > 504)
      SAC_LT10 = FP_LT10 / N_LT10

    SAC_MID penalizes hurting mid sellers (10–21d):
      FP_MID = count(240 <= y < 504 AND pred > 504)
      SAC_MID = FP_MID / N_MID
    """
    y = np.asarray(y_true_hours, dtype=float)
    p = np.asarray(y_pred_hours, dtype=float)
    pred_slow = (p > float(slow_thresh_hours))

    m_lt10 = (y < float(lt10_thresh_hours))
    m_mid  = (y >= float(lt10_thresh_hours)) & (y < float(slow_thresh_hours))

    n_lt10 = int(np.sum(m_lt10))
    n_mid  = int(np.sum(m_mid))

    fp_lt10 = int(np.sum(m_lt10 & pred_slow))
    fp_mid  = int(np.sum(m_mid  & pred_slow))

    sac_lt10 = (fp_lt10 / n_lt10) if n_lt10 > 0 else 0.0
    sac_mid  = (fp_mid  / n_mid)  if n_mid  > 0 else 0.0

    return {
        "fp_lt10": fp_lt10,
        "n_lt10": n_lt10,
        "sac_lt10": float(sac_lt10),
        "fp_mid": fp_mid,
        "n_mid": n_mid,
        "sac_mid": float(sac_mid),
    }


def slow21_focus_report(
    y_true_hours: np.ndarray,
    y_pred_hours: np.ndarray,
    slow_thresh_hours: float = SLOW21_H,
    lt10_thresh_hours: float = LT10_H,
) -> Dict[str, Any]:
    """Convenience wrapper: classifier@504 + collateral SAC metrics + SLOW confusion."""
    cls = classify_at_threshold(y_true_hours, y_pred_hours, float(slow_thresh_hours))
    sac = slow21_sacrifice_rates(y_true_hours, y_pred_hours, float(slow_thresh_hours), float(lt10_thresh_hours))

    # classify_at_threshold reports fast confusion (tp_fast/fp_fast/tn_fast/fn_fast).
    # Convert to SLOW confusion:
    conf = cls.get("confusion", {}) if isinstance(cls, dict) else {}
    tp_fast = int(conf.get("tp_fast", 0))
    fp_fast = int(conf.get("fp_fast", 0))
    tn_fast = int(conf.get("tn_fast", 0))
    fn_fast = int(conf.get("fn_fast", 0))

    confusion_slow = {
        "tp_slow": tn_fast,  # true slow predicted slow
        "fp_slow": fn_fast,  # true fast predicted slow
        "fn_slow": fp_fast,  # true slow predicted fast
        "tn_slow": tp_fast,  # true fast predicted fast
    }

    out = {
        "threshold_hours": float(slow_thresh_hours),
        "slow": cls.get("slow", {}),
        "fast": cls.get("fast", {}),
        "confusion_slow": confusion_slow,
        "sacrifice": sac,
    }
    return out

# -----------------------------
# WOE / log-odds additive anchor for SLOW21 (duration > 504h)
# -----------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_logit(p: float, eps: float = 1e-12) -> float:
    # Clamp to avoid infinities
    p2 = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p2 / (1.0 - p2)))


def _woe_map_from_series(
    cat: pd.Series,
    y: pd.Series,
    w: pd.Series,
    *,
    eps: float,
    tot_pos: float,
    tot_neg: float,
) -> Dict[Any, float]:
    """
    Computes WOE(category) = log( P(cat|pos) / P(cat|neg) ) with additive smoothing.
    Missing values are mapped to a sentinel so lookup is stable.
    """
    cat2 = cat.astype("object").where(cat.notna(), "__MISSING__")
    y2 = y.astype(int)
    w2 = w.astype(float)

    pos_w = (w2 * y2).astype(float)
    neg_w = (w2 * (1 - y2)).astype(float)

    g = (
        pd.DataFrame({"cat": cat2, "pos_w": pos_w, "neg_w": neg_w})
        .groupby("cat", sort=False, dropna=False)[["pos_w", "neg_w"]]
        .sum()
    )

    out: Dict[Any, float] = {}
    if tot_pos <= 0.0 or tot_neg <= 0.0:
        for k in g.index.tolist():
            out[k] = 0.0
        return out

    for k, row in g.iterrows():
        num = (float(row["pos_w"]) + eps) / tot_pos
        den = (float(row["neg_w"]) + eps) / tot_neg
        out[k] = float(np.log(num / den))
    return out


def _woe_table_from_series(
    cat: pd.Series,
    y: pd.Series,
    w: pd.Series,
    *,
    eps: float,
    tot_pos: float,
    tot_neg: float,
) -> pd.DataFrame:
    """Return a per-category WOE table including weighted pos/neg sums.

    This is used to persist WOE artifacts to Postgres for auditability and
    production inference.
    """
    cat2 = cat.astype("object").where(cat.notna(), "__MISSING__")
    y2 = y.astype(int)
    w2 = w.astype(float)

    pos_w = (w2 * y2).astype(float)
    neg_w = (w2 * (1 - y2)).astype(float)

    g = (
        pd.DataFrame({"band_value": cat2, "sum_w_pos": pos_w, "sum_w_neg": neg_w})
        .groupby("band_value", sort=False, dropna=False)[["sum_w_pos", "sum_w_neg"]]
        .sum()
        .reset_index()
    )

    if tot_pos <= 0.0 or tot_neg <= 0.0:
        g["woe"] = 0.0
        return g[["band_value", "woe", "sum_w_pos", "sum_w_neg"]]

    # WOE(category) = log( P(cat|pos) / P(cat|neg) ) with additive smoothing.
    num = (g["sum_w_pos"].astype(float) + eps) / float(tot_pos)
    den = (g["sum_w_neg"].astype(float) + eps) / float(tot_neg)
    g["woe"] = np.log(num / den).astype(float)
    return g[["band_value", "woe", "sum_w_pos", "sum_w_neg"]]


def _woe_make_bands(
    df: pd.DataFrame,
    *,
    t0_col: str,
    dsold_thresholds: Tuple[float, float, float, float],
    presentation_cuts: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """
    Build stable, leak-safe bands (no outcome fields) used by the WOE anchor.

    Returns:
      - bands_df: categorical columns
      - presentation_cuts: (c1, c2) used for presentation banding
    """
    out = pd.DataFrame(index=df.index)

    # ---- trust tier (seller quality proxies) ----
    seller_rating = df.get("seller_rating")
    review_count = df.get("review_count")
    member_since_year = df.get("member_since_year")

    # Use edited_date year for the "member for >=3y" rule when available.
    t0 = df.get(t0_col)
    if t0 is not None and pd.api.types.is_datetime64_any_dtype(t0):
        t0_year = t0.dt.year
    else:
        t0_year = pd.Series(index=df.index, dtype="float64")

    trust = pd.Series("LOW", index=df.index, dtype="object")
    try:
        hi = (
            (seller_rating >= 9.7)
            & (review_count >= 50)
            & member_since_year.notna()
            & t0_year.notna()
            & (member_since_year <= (t0_year - 3))
        )
        med = (seller_rating >= 9.0) & (review_count >= 10)
        trust = trust.mask(med, "MED").mask(hi, "HIGH")
    except Exception:
        # If columns missing / wrong dtype, fall back to LOW.
        pass
    out["trust_tier"] = trust

    # ---- shipping / pickup friction ----
    ai_pickup_only = df.get("ai_pickup_only_bin")
    ai_ship_pickup = df.get("ai_ship_pickup")
    ai_ship_can = df.get("ai_ship_can")

    ship = pd.Series("ship_unknown", index=df.index, dtype="object")
    try:
        pickup_heavy = ((ai_pickup_only == 1) | (ai_ship_pickup == 1))
        shipping_ok = (ai_ship_can == 1)
        ship = ship.mask(shipping_ok, "shipping_ok").mask(pickup_heavy, "pickup_heavy")
    except Exception:
        pass
    out["ship_band"] = ship

    # ---- sale mode ----
    firm = df.get("ai_sale_mode_firm")
    bids = df.get("ai_sale_mode_bids")
    obo = df.get("ai_sale_mode_obo")

    sale = pd.Series("sale_unspec", index=df.index, dtype="object")
    try:
        sale = sale.mask(obo == 1, "obo").mask(bids == 1, "bids").mask(firm == 1, "firm")
    except Exception:
        pass
    out["sale_band"] = sale

    # ---- mispricing vs sold median ----
    dsold = df.get("delta_vs_sold_median_30d")
    ds_lo2, ds_lo1, ds_hi1, ds_hi2 = dsold_thresholds

    ds_band = pd.Series("dSold_missing", index=df.index, dtype="object")
    try:
        ds_band = ds_band.mask(dsold.notna(), "dSold_fair")
        ds_band = ds_band.mask(dsold <= ds_lo2, "dSold_cheap")
        ds_band = ds_band.mask((dsold > ds_lo2) & (dsold <= ds_lo1), "dSold_under")
        ds_band = ds_band.mask((dsold > ds_lo1) & (dsold <= ds_hi1), "dSold_fair")
        ds_band = ds_band.mask((dsold > ds_hi1) & (dsold <= ds_hi2), "dSold_over")
        ds_band = ds_band.mask(dsold > ds_hi2, "dSold_overpriced")
    except Exception:
        pass
    out["dsold_band"] = ds_band

    # ---- condition band (0..1 score) ----
    cond = df.get("condition_score")
    cond_band = pd.Series("cond_missing", index=df.index, dtype="object")
    try:
        cond_band = cond_band.mask(cond.notna(), "cond_lo")
        cond_band = cond_band.mask(cond >= 0.70, "cond_mid").mask(cond >= 0.90, "cond_hi")
    except Exception:
        pass
    out["cond_band"] = cond_band

    # ---- damage AI band (sev: 0..3+) ----
    sev = df.get("damage_severity_ai")
    if sev is None:
        sev = df.get("sev")

    dmg_band = pd.Series("dmg_missing", index=df.index, dtype="object")
    try:
        dmg_band = dmg_band.mask(sev.notna(), "dmg_0")
        dmg_band = dmg_band.mask(sev <= 0, "dmg_0")
        dmg_band = dmg_band.mask(sev == 1, "dmg_1")
        dmg_band = dmg_band.mask(sev == 2, "dmg_2")
        dmg_band = dmg_band.mask(sev >= 3, "dmg_3p")
    except Exception:
        pass
    out["dmg_ai_band"] = dmg_band

    # ---- battery band (percentage) ----
    batt = df.get("battery_pct_effective")
    bat_band = pd.Series("bat_missing", index=df.index, dtype="object")
    try:
        bat_band = bat_band.mask(batt.notna(), "bat_lo")
        bat_band = bat_band.mask(batt >= 88, "bat_mid").mask(batt >= 95, "bat_hi")
    except Exception:
        pass
    out["bat_band"] = bat_band

    # ---- vision-only bands (assets present only; avoid missingness shortcut) ----
    img_cnt = df.get("image_count")
    has_assets = pd.Series(False, index=df.index, dtype="bool")
    try:
        has_assets = img_cnt.notna() & (img_cnt.astype(float) > 0.0)
    except Exception:
        pass
    out["has_assets"] = has_assets.astype(int)

    # presentation score
    caption_share = df.get("caption_share")
    stock_photo_share = df.get("stock_photo_share")
    photo_quality_avg = df.get("photo_quality_avg")
    bg_clean_avg = df.get("bg_clean_avg")

    pres_score = pd.Series(np.nan, index=df.index, dtype="float64")
    try:
        pres_score = (
            0.30 * np.log1p(pd.to_numeric(img_cnt, errors="coerce").fillna(0.0))
            + 0.25 * pd.to_numeric(photo_quality_avg, errors="coerce").fillna(0.0)
            + 0.20 * pd.to_numeric(bg_clean_avg, errors="coerce").fillna(0.0)
            + 0.15 * (1.0 - pd.to_numeric(stock_photo_share, errors="coerce").fillna(0.0))
            + 0.10 * pd.to_numeric(caption_share, errors="coerce").fillna(0.0)
        )
        pres_score = pres_score.where(has_assets, np.nan)
    except Exception:
        pass
    out["presentation_score"] = pres_score

    # learn / apply presentation cuts
    if presentation_cuts is None:
        valid = pres_score.dropna()
        if len(valid) >= 50:
            c1 = float(np.nanquantile(valid.values, 1.0 / 3.0))
            c2 = float(np.nanquantile(valid.values, 2.0 / 3.0))
        elif len(valid) > 0:
            c1 = float(np.nanquantile(valid.values, 0.5))
            c2 = c1
        else:
            c1, c2 = 0.0, 0.0
        presentation_cuts = (c1, c2)

    c1, c2 = presentation_cuts
    pres_band = pd.Series(None, index=df.index, dtype="object")
    try:
        pres_band = pres_band.mask(pres_score.notna(), "present_hi")
        pres_band = pres_band.mask(pres_score < c2, "present_mid")
        pres_band = pres_band.mask(pres_score < c1, "present_lo")
    except Exception:
        pass
    out["presentation_band"] = pres_band

    # vision damage band (dmg_band_hq_struct)
    vdmg = df.get("dmg_band_hq_struct")
    vd_band = pd.Series(None, index=df.index, dtype="object")
    try:
        vd_band = vd_band.mask(has_assets, "vdmg_missing")
        vd_band = vd_band.mask(has_assets & vdmg.notna(), "vdmg_0")
        vd_band = vd_band.mask(has_assets & (vdmg <= 0), "vdmg_0")
        vd_band = vd_band.mask(has_assets & (vdmg == 1), "vdmg_1")
        vd_band = vd_band.mask(has_assets & (vdmg == 2), "vdmg_2")
        vd_band = vd_band.mask(has_assets & (vdmg >= 3), "vdmg_3p")
    except Exception:
        pass
    out["vdmg_band"] = vd_band

    # accessories band
    charger_bundle_level = df.get("charger_bundle_level")
    box_present = df.get("box_present")
    receipt_present = df.get("receipt_present")

    acc_score = pd.Series(np.nan, index=df.index, dtype="float64")
    try:
        acc_score = (
            pd.to_numeric(charger_bundle_level, errors="coerce").fillna(0.0)
            + pd.to_numeric(box_present, errors="coerce").fillna(0.0)
            + pd.to_numeric(receipt_present, errors="coerce").fillna(0.0)
        )
        acc_score = acc_score.where(has_assets, np.nan)
    except Exception:
        pass
    out["acc_score"] = acc_score

    acc_band = pd.Series(None, index=df.index, dtype="object")
    try:
        acc_band = acc_band.mask(acc_score.notna(), "acc_hi")
        acc_band = acc_band.mask(acc_score < 2.0, "acc_mid")
        acc_band = acc_band.mask(acc_score <= 0.0, "acc_lo")
    except Exception:
        pass
    out["accessories_band"] = acc_band

    return out, presentation_cuts


def compute_woe_anchor_p_slow21(
    train_sold_df: pd.DataFrame,
    score_df: pd.DataFrame,
    *,
    t0_col: str = "edited_date",
    sold_date_col: str = "sold_date",
    duration_col: str = "duration_h",
    slow_threshold_h: float = 504.0,
    min_duration_h: float = 48.0,
    half_life_days: float = 90.0,
    eps: float = 0.5,
    dsold_thresholds: Optional[Tuple[float, float, float, float]] = None,
    presentation_cuts: Optional[Tuple[float, float]] = None,
    weight_ref_ts: Optional[pd.Timestamp] = None,
    band_schema_version: int = 1,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Learns additive WOE tables on TRAIN-only SOLD rows, then scores arbitrary rows.

    Returns:
      - p_slow21: probability-like score (sigmoid of WOE logit)
      - logit: raw WOE logit
      - ctx: debug context (cuts, base_rate, etc)
    """
    if train_sold_df is None or len(train_sold_df) == 0:
        p = pd.Series(0.5, index=score_df.index, dtype="float64")
        logit = pd.Series(0.0, index=score_df.index, dtype="float64")
        return p, logit, {"note": "empty_train"}

    t = train_sold_df.copy()
    # Basic guard: need duration + sold_date for label_ts logic
    if duration_col not in t.columns:
        p = pd.Series(0.5, index=score_df.index, dtype="float64")
        logit = pd.Series(0.0, index=score_df.index, dtype="float64")
        return p, logit, {"note": "missing_duration_col"}

    t = t[t[duration_col].notna()].copy()
    if len(t) == 0:
        p = pd.Series(0.5, index=score_df.index, dtype="float64")
        logit = pd.Series(0.0, index=score_df.index, dtype="float64")
        return p, logit, {"note": "no_duration_rows"}

    # Filter out ultra-fast (<48h) rows, as in the SQL anchor job.
    try:
        t = t[t[duration_col] >= float(min_duration_h)].copy()
    except Exception:
        pass

    y = (t[duration_col].astype(float) > float(slow_threshold_h)).astype(int)

    # label_ts: positives become "known" at t0+21d, negatives at sold_date (or fallback to t0)
    t0 = t.get(t0_col)
    sold_ts = t.get(sold_date_col)

    if t0 is not None and pd.api.types.is_datetime64_any_dtype(t0):
        t0_ts = t0
    else:
        t0_ts = pd.to_datetime(pd.Series(index=t.index, data=pd.NaT))

    if sold_ts is not None and pd.api.types.is_datetime64_any_dtype(sold_ts):
        sold_ts2 = sold_ts
    else:
        sold_ts2 = pd.to_datetime(pd.Series(index=t.index, data=pd.NaT))

    label_ts = sold_ts2.copy()
    try:
        label_ts = label_ts.where(y == 0, t0_ts + pd.to_timedelta(float(slow_threshold_h), unit="h"))
        label_ts = label_ts.fillna(t0_ts)
    except Exception:
        label_ts = sold_ts2.fillna(t0_ts)

    # Time decay weights (mean-normalized) on label_ts.
    # For OOF cross-fitting we want each fold mapping to use the same reference timestamp
    # (i.e., weights comparable across folds), so we support an explicit weight_ref_ts.
    label_ts_utc = ensure_datetime_utc(label_ts)
    if weight_ref_ts is None:
        ref_ts_used = label_ts_utc.max()
    else:
        ref_ts_used = pd.Timestamp(weight_ref_ts)
        if ref_ts_used.tzinfo is None:
            ref_ts_used = ref_ts_used.tz_localize("UTC")
        else:
            ref_ts_used = ref_ts_used.tz_convert("UTC")

    w = pd.Series(
        time_decay_weights(label_ts_utc, half_life_days=float(half_life_days), ref_ts=ref_ts_used),
        index=t.index,
        dtype="float64",
    )

    # dsold thresholds: allow caller override for strict OOF consistency
    dsold_thresholds_used = dsold_thresholds
    if dsold_thresholds_used is None:
        dsold_vals = pd.to_numeric(t.get("delta_vs_sold_median_30d"), errors="coerce")
        med_abs = float(dsold_vals.abs().median()) if dsold_vals.notna().any() else 0.0
        if med_abs < 5.0:
            # Likely ratio-like
            dsold_thresholds_used = (-0.15, -0.05, 0.05, 0.15)
        else:
            # Likely NOK delta
            dsold_thresholds_used = (-2000.0, -500.0, 500.0, 2000.0)

    # Build bands for train and score (shared presentation cuts)
    train_bands, pres_cuts_learned = _woe_make_bands(
        t,
        t0_col=t0_col,
        dsold_thresholds=dsold_thresholds_used,
        presentation_cuts=presentation_cuts,
    )
    pres_cuts_used = presentation_cuts if presentation_cuts is not None else pres_cuts_learned
    score_bands, _ = _woe_make_bands(
        score_df,
        t0_col=t0_col,
        dsold_thresholds=dsold_thresholds_used,
        presentation_cuts=pres_cuts_used,
    )

    # Global totals (weighted)
    tot_pos = float((w * y).sum())
    tot_neg = float((w * (1 - y)).sum())
    tot = tot_pos + tot_neg
    p_base = (tot_pos / tot) if tot > 0 else 0.5
    base_logit = _safe_logit(p_base)

    # Build WOE tables (global)
    global_cols = [
        "dsold_band",
        "trust_tier",
        "ship_band",
        "sale_band",
        "cond_band",
        "dmg_ai_band",
        "bat_band",
    ]
    band_col_to_name: Dict[str, str] = {
        "dsold_band": "dsold",
        "trust_tier": "trust",
        "ship_band": "ship",
        "sale_band": "sale",
        "cond_band": "cond",
        "dmg_ai_band": "dmg",
        "bat_band": "bat",
        "presentation_band": "present",
        "vdmg_band": "vdmg",
        "accessories_band": "acc",
    }

    woe_tables: Dict[str, List[Dict[str, Any]]] = {}
    woe_maps: Dict[str, Dict[Any, float]] = {}
    for col in global_cols:
        tbl = _woe_table_from_series(
            train_bands[col],
            y,
            w,
            eps=float(eps),
            tot_pos=tot_pos,
            tot_neg=tot_neg,
        )
        mp = {row["band_value"]: float(row["woe"]) for row in tbl.to_dict("records")}
        woe_maps[col] = mp
        woe_tables[band_col_to_name.get(col, col)] = tbl.to_dict("records")

    # Vision-only WOE tables learned on assets-present subset only
    assets_mask = train_bands["has_assets"] == 1
    tot_pos_a = float((w[assets_mask] * y[assets_mask]).sum())
    tot_neg_a = float((w[assets_mask] * (1 - y[assets_mask])).sum())

    vision_cols = ["presentation_band", "vdmg_band", "accessories_band"]
    vision_maps: Dict[str, Dict[Any, float]] = {}
    for col in vision_cols:
        band_name = band_col_to_name.get(col, col)
        # If no assets in train, default to 0
        if assets_mask.sum() == 0 or (tot_pos_a + tot_neg_a) <= 0:
            vision_maps[col] = {}
            woe_tables[band_name] = []
            continue
        tbl = _woe_table_from_series(
            train_bands.loc[assets_mask, col],
            y.loc[assets_mask],
            w.loc[assets_mask],
            eps=float(eps),
            tot_pos=tot_pos_a,
            tot_neg=tot_neg_a,
        )
        mp = {row["band_value"]: float(row["woe"]) for row in tbl.to_dict("records")}
        vision_maps[col] = mp
        woe_tables[band_name] = tbl.to_dict("records")

    # Score rows
    logit = np.full(len(score_df), base_logit, dtype="float64")
    for col in global_cols:
        mp = woe_maps.get(col, {})
        contrib = score_bands[col].astype("object").where(score_bands[col].notna(), "__MISSING__").map(mp).fillna(0.0).astype(float)
        logit += contrib.values

    # Vision contributions only when has_assets==1
    has_assets = score_bands["has_assets"].astype(int).values == 1
    for col in vision_cols:
        mp = vision_maps.get(col, {})
        if not mp:
            continue
        contrib = score_bands[col].astype("object").where(score_bands[col].notna(), "__MISSING__").map(mp).fillna(0.0).astype(float).values
        logit += np.where(has_assets, contrib, 0.0)

    p_slow = _sigmoid(logit)

    p_series = pd.Series(p_slow, index=score_df.index, dtype="float64")
    logit_series = pd.Series(logit, index=score_df.index, dtype="float64")

    ctx = {
        "band_schema_version": int(band_schema_version),
        "base_rate": float(p_base),
        "base_logit": float(base_logit),
        "tot_pos_w": float(tot_pos),
        "tot_neg_w": float(tot_neg),
        "dsold_thresholds": tuple(float(x) for x in dsold_thresholds_used),
        "presentation_cuts": tuple(float(x) for x in pres_cuts_used),
        "half_life_days": float(half_life_days),
        "eps": float(eps),
        "weight_ref_ts": ref_ts_used,
        "woe_tables": woe_tables,
    }
    return p_series, logit_series, ctx


def _woe_time_folds_by_t0_day(t0: pd.Series, n_folds: int) -> np.ndarray:
    """Deterministic blocked folds by t0_day for OOF target encoding.

    We fold by *day* (not by individual row) to avoid splitting near-duplicate listings
    from the same day across folds.
    """
    if n_folds <= 1:
        return np.zeros(len(t0), dtype=int)

    t0_utc = ensure_datetime_utc(t0)
    t0_day = t0_utc.dt.date
    uniq_days = [d for d in pd.unique(t0_day) if pd.notna(d)]
    uniq_days = sorted(uniq_days)
    if len(uniq_days) == 0:
        return np.zeros(len(t0), dtype=int)

    n_days = len(uniq_days)
    day_to_fold: Dict[Any, int] = {}
    for i, d in enumerate(uniq_days):
        fid = int((i * n_folds) / n_days)
        if fid >= n_folds:
            fid = n_folds - 1
        day_to_fold[d] = fid

    return np.array([day_to_fold.get(d, 0) for d in t0_day], dtype=int)


def compute_woe_anchor_oof(
    train_sold_df: pd.DataFrame,
    *,
    n_folds: int,
    t0_col: str = "edited_date",
    sold_date_col: str = "sold_date",
    duration_col: str = "duration_h",
    slow_threshold_h: float = 504.0,
    min_duration_h: float = 48.0,
    half_life_days: float = 90.0,
    eps: float = 0.5,
    dsold_thresholds: Optional[Tuple[float, float, float, float]] = None,
    presentation_cuts: Optional[Tuple[float, float]] = None,
    weight_ref_ts: Optional[pd.Timestamp] = None,
    band_schema_version: int = 1,
) -> Tuple[pd.Series, pd.Series, pd.Series, List[Dict[str, Any]]]:
    """Compute OUT-OF-FOLD WOE anchor scores for TRAIN SOLD rows.

    Returns:
      - oof_p:     probability (same indexing as train_sold_df)
      - oof_logit: logit (same indexing)
      - fold_id:   fold id per row (same indexing)
      - fold_ctxs: list of per-fold ctx dicts (each has fold_id + woe_tables)
    """
    if n_folds <= 1:
        p_all, logit_all, ctx = compute_woe_anchor_p_slow21(
            train_sold_df,
            train_sold_df,
            t0_col=t0_col,
            sold_date_col=sold_date_col,
            duration_col=duration_col,
            slow_threshold_h=slow_threshold_h,
            min_duration_h=min_duration_h,
            half_life_days=half_life_days,
            eps=eps,
            dsold_thresholds=dsold_thresholds,
            presentation_cuts=presentation_cuts,
            weight_ref_ts=weight_ref_ts,
            band_schema_version=band_schema_version,
        )
        fold_id = pd.Series(np.zeros(len(train_sold_df), dtype=int), index=train_sold_df.index)
        ctx = dict(ctx)
        ctx["fold_id"] = 0
        return p_all, logit_all, fold_id, [ctx]

    fold_ids = _woe_time_folds_by_t0_day(train_sold_df[t0_col], n_folds)
    fold_id_series = pd.Series(fold_ids, index=train_sold_df.index, dtype=int)
    oof_p = pd.Series(index=train_sold_df.index, dtype="float64")
    oof_logit = pd.Series(index=train_sold_df.index, dtype="float64")
    fold_ctxs: List[Dict[str, Any]] = []

    for k in range(int(n_folds)):
        m_te = fold_ids == k
        if not bool(m_te.any()):
            continue

        tr_df = train_sold_df.loc[~m_te].copy()
        te_df = train_sold_df.loc[m_te].copy()
        p_te, logit_te, ctx_k = compute_woe_anchor_p_slow21(
            tr_df,
            te_df,
            t0_col=t0_col,
            sold_date_col=sold_date_col,
            duration_col=duration_col,
            slow_threshold_h=slow_threshold_h,
            min_duration_h=min_duration_h,
            half_life_days=half_life_days,
            eps=eps,
            dsold_thresholds=dsold_thresholds,
            presentation_cuts=presentation_cuts,
            weight_ref_ts=weight_ref_ts,
            band_schema_version=band_schema_version,
        )
        oof_p.loc[te_df.index] = p_te
        oof_logit.loc[te_df.index] = logit_te

        ctx_k = dict(ctx_k)
        ctx_k["fold_id"] = int(k)
        fold_ctxs.append(ctx_k)

    # If any rows ended up unscored (should be rare), backfill with final mapping.
    if oof_p.isna().any():
        miss_idx = oof_p.index[oof_p.isna()]
        log(f"[woe] WARNING: OOF produced {len(miss_idx)} NaN scores; backfilling with final mapping.")
        p_fill, logit_fill, _ = compute_woe_anchor_p_slow21(
            train_sold_df,
            train_sold_df.loc[miss_idx],
            t0_col=t0_col,
            sold_date_col=sold_date_col,
            duration_col=duration_col,
            slow_threshold_h=slow_threshold_h,
            min_duration_h=min_duration_h,
            half_life_days=half_life_days,
            eps=eps,
            dsold_thresholds=dsold_thresholds,
            presentation_cuts=presentation_cuts,
            weight_ref_ts=weight_ref_ts,
            band_schema_version=band_schema_version,
        )
        oof_p.loc[miss_idx] = p_fill
        oof_logit.loc[miss_idx] = logit_fill
        fold_id_series.loc[miss_idx] = -1

    return oof_p, oof_logit, fold_id_series, fold_ctxs

def clean_json_numbers(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json_numbers(v) for v in obj]
    return obj

# ---------------- Region price-z ----------------
def region_price_z(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   geo_col: str, region_map_csv: str, shrink_k: float,
                   edited_col: str, half_life_days: float = 150.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not region_map_csv or not pathlib.Path(region_map_csv).exists() or geo_col not in train_df.columns or geo_col not in val_df.columns:
        log("[geo] region map not applied (missing file/column).")
        return train_df, val_df
    m = pd.read_csv(region_map_csv)
    cols = [c.strip().lower() for c in m.columns]
    city_col   = "location_city" if "location_city" in cols else cols[0]
    region_col = "region5"        if "region5"        in cols else cols[1]
    m = m[[city_col, region_col]].dropna()
    m[city_col] = m[city_col].astype(str).str.strip().str.lower()
    m = m.drop_duplicates(subset=[city_col], keep="first").rename(columns={city_col:"__city__", region_col:"region5"})

    def attach(df):
        d = df.copy()
        d["__city__"] = d[geo_col].astype(str).str.strip().str.lower()
        return d.merge(m, on="__city__", how="left")

    tr, va = attach(train_df), attach(val_df)

    w  = time_decay_weights(tr[edited_col], half_life_days=float(half_life_days))
    p  = pd.to_numeric(tr["price"], errors="coerce").astype(float)
    W  = float(np.nansum(w)) or 1.0
    mu = float(np.nansum(w*p)/W)
    var= float(np.nansum(w*(p**2))/W - mu**2)
    var = max(var, 1e-9)
    sd = math.sqrt(var)

    g = tr.assign(_w=w, _p=p).groupby("region5", dropna=False).agg(
        {"_w":"sum","_p":["sum", lambda s: np.nansum(s**2)]})
    g.columns = ["sum_w","sum_wp","sum_wp2"]
    g = g.reset_index()

    g["mu"]  = g["sum_wp"]/g["sum_w"].replace(0,np.nan)
    g["var"] = (g["sum_wp2"]/g["sum_w"].replace(0,np.nan)) - g["mu"]**2
    g["var"] = g["var"].fillna(1e-9).clip(lower=1e-9)
    g["sd"]  = np.sqrt(g["var"])

    alpha = g["sum_w"]/(g["sum_w"] + float(shrink_k))
    g["_mu"] = alpha*g["mu"] + (1-alpha)*mu
    g["_sd"] = alpha*g["sd"] + (1-alpha)*sd

    stats = g[["region5","_mu","_sd"]]

    def apply_z(df):
        d = df.merge(stats, on="region5", how="left")
        d["_mu"] = d["_mu"].fillna(mu)
        d["_sd"] = d["_sd"].replace([0,np.inf,-np.inf], np.nan).fillna(sd)
        price = pd.to_numeric(d["price"], errors="coerce").astype(float)
        d["price_z_in_region"] = (price - d["_mu"])/d["_sd"]
        return d.drop(columns=["_mu","_sd","__city__"], errors="ignore")

    tr = apply_z(tr)
    va = apply_z(va)
    log("[geo] added single feature 'price_z_in_region'")
    return tr, va

# ---------------- DB helpers ----------------
def load_feature_set_hash(conn, feature_set: str) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT features_hash FROM ml.feature_set WHERE name=%s", (feature_set,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"feature_set {feature_set} not found")
        return row[0]
    
def _sbert_table_name(args) -> str:
    if args.sbert_table:
        t = args.sbert_table.strip()
    else:
        t = f"ml.sbert_vec{int(args.sbert_dim)}_v1"
    # Hard validation to avoid SQL injection via --sbert_table
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", t):
        raise ValueError(f"[sbert] invalid table name: {t}")
    return t


# =============================================================================
# Geo mapping helpers (DB-backed super_metro_v4)
# =============================================================================

def _parse_csv_list(x: Optional[str]) -> List[str]:
    if x is None:
        return []
    x = str(x).strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def _validate_relname_str(rel: str) -> str:
    """Very small safety check for user-provided schema-qualified identifiers."""
    rel = (rel or "").strip()
    if not rel:
        raise ValueError("empty relation name")
    # allow schema.rel or just rel
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", rel):
        return rel
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rel):
        return rel
    raise ValueError(f"invalid relation name: {rel!r}")


def ensure_geo_columns(
    conn,
    df: pd.DataFrame,
    *,
    geo_dim_view: str = "ml.geo_dim_super_metro_v4_current_v",
) -> pd.DataFrame:
    """
    Ensure df has the standard geo columns:
      region_geo, pickup_metro_30_200_geo, super_metro_v4_geo, geo_match_method, geo_release_id

    Strategy:
      1) If df already contains region_geo + super_metro_v4_geo, do nothing.
      2) Else, LEFT JOIN (in Python) the geo_dim view by listing_id.
    """
    if df is None or df.empty:
        return df
    if all(c in df.columns for c in ["region_geo", "super_metro_v4_geo", "pickup_metro_30_200_geo"]):
        # already present (e.g. using ml.*_geo_v / *_read_v)
        return df
    if "listing_id" not in df.columns:
        log("[geo] cannot attach geo columns (missing listing_id)")
        return df

    geo_dim_view = _validate_relname_str(geo_dim_view)
    sql = f"""
        SELECT
          listing_id,
          geo_release_id,
          region_geo,
          pickup_metro_30_200_geo,
          super_metro_v4_geo,
          geo_match_method
        FROM {geo_dim_view}
    """
    g = pd.read_sql_query(sql, conn)

    # preserve original row order + index so downstream splits (which rely on df.index) stay aligned
    _orig_index = df.index
    _tmp = df.copy()
    _tmp["_row_id__"] = np.arange(len(_tmp), dtype=np.int64)
    out = _tmp.merge(g, on="listing_id", how="left", sort=False)
    out = out.sort_values("_row_id__").drop(columns=["_row_id__"])
    out.index = _orig_index

    # Fill only if null. Keep DB-generated 'other_*' labels intact.
    out["region_geo"] = out["region_geo"].fillna("unknown")
    out["pickup_metro_30_200_geo"] = out["pickup_metro_30_200_geo"].fillna("unknown")
    out["super_metro_v4_geo"] = out["super_metro_v4_geo"].fillna("unknown")
    out["geo_match_method"] = out["geo_match_method"].fillna("unmapped")
    out["geo_release_id"] = pd.to_numeric(out["geo_release_id"], errors="coerce")

    # Light QA logging
    n = len(out)
    unk = int((out["region_geo"] == "unknown").sum()) if "region_geo" in out.columns else -1
    other = int(out["super_metro_v4_geo"].astype(str).str.startswith("other_").sum()) if "super_metro_v4_geo" in out.columns else -1
    log(f"[geo] attached geo_dim ({geo_dim_view}) -> rows={n} unknown_region={unk} other_super={other}")
    return out


def fit_geo_ohe_template(
    train_events: pd.DataFrame,
    geo_cols: List[str],
    *,
    min_n: int = 50,
) -> Dict[str, List[str]]:
    """Return per-column category lists to use for stable one-hot encoding."""
    template: Dict[str, List[str]] = {}
    if train_events is None or train_events.empty:
        return template

    for col in geo_cols:
        if col not in train_events.columns:
            continue
        s = train_events[col].fillna("missing").astype(str)
        vc = s.value_counts(dropna=False)
        keep = [k for k, n in vc.items() if int(n) >= int(min_n) and k not in ("missing", "__rare__")]
        keep_sorted = sorted(keep)
        template[col] = keep_sorted + ["__rare__", "missing"]
    return template


def apply_geo_ohe(df: pd.DataFrame, template: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply stable one-hot encoding for geo categorical columns.

    Critical invariant: preserve the original index.  If we lose the index (e.g. by converting
    to a bare pd.Categorical array), pandas will union indexes during concat and silently
    create extra rows with NaNs. That corrupts sold_event/sold_date and breaks SOLD-only eval.
    """
    if df is None or df.empty or not template:
        return df

    parts: List[pd.DataFrame] = []
    for col, cats in template.items():
        if col not in df.columns:
            continue

        keep = set(cats) - {"__rare__"}  # includes 'missing'
        s = df[col].fillna("missing").astype(str)
        s = s.where(s.isin(keep), "__rare__")

        # Preserve index and force fixed category set (ensures stable columns)
        s = s.astype(pd.CategoricalDtype(categories=cats))
        oh = pd.get_dummies(s, prefix=f"geo_{col}", prefix_sep="=", dtype=np.uint8)

        # Defensive: ensure exact alignment (prevents accidental row-union)
        oh = oh.reindex(df.index, fill_value=0)
        parts.append(oh)

    if not parts:
        return df

    ohe = pd.concat(parts, axis=1)
    ohe = ohe.reindex(sorted(ohe.columns), axis=1)
    ohe = ohe.reindex(df.index, fill_value=0)
    return pd.concat([df, ohe], axis=1)


def fit_geo_prior_maps(
    train_events: pd.DataFrame,
    geo_cols: List[str],
    *,
    shrink_k: float,
    half_life_days: float,
    duration_col: str = "duration_h",
    edited_col: str = "edited_date",
    slow_threshold_h: float = 21.0 * 24.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Build smoothed, time-decayed priors per geo column using ONLY training sold events.

    Produces, per geo col:
      - geo_prior_slow21_{col}: smoothed probability of being slow21
      - geo_prior_tom_h_{col}: smoothed mean time-on-market in hours
      - geo_n_eff_{col}: effective (time-decayed) sample size
    """
    out: Dict[str, Dict[str, Any]] = {}
    if train_events is None or train_events.empty:
        return out

    # weights: recency by edited_date (same as region_price_z / WOE style)
    w = time_decay_weights(train_events[edited_col], half_life_days=float(half_life_days)).astype(float)
    d = pd.to_numeric(train_events[duration_col], errors="coerce").astype(float)
    y = (d > float(slow_threshold_h)).astype(float)

    w_sum = float(np.nansum(w))
    if w_sum <= 0:
        return out

    base_slow = float(np.nansum(w * y) / w_sum)
    base_tom = float(np.nansum(w * d) / w_sum)

    tmp = train_events.copy()
    tmp["_key_dummy"] = 0  # placeholder
    tmp["_w"] = w
    tmp["_wy"] = w * y
    tmp["_wd"] = w * d

    for col in geo_cols:
        if col not in tmp.columns:
            continue
        tmp["_key_dummy"] = tmp[col].fillna("missing").astype(str)

        stats = (
            tmp.groupby("_key_dummy", dropna=False)[["_w", "_wy", "_wd"]]
            .sum()
            .rename(columns={"_w": "w_sum", "_wy": "wy_sum", "_wd": "wd_sum"})
        )
        stats["mean_slow"] = stats["wy_sum"] / stats["w_sum"].replace(0.0, np.nan)
        stats["mean_tom"] = stats["wd_sum"] / stats["w_sum"].replace(0.0, np.nan)

        k = float(shrink_k)
        stats["prior_slow"] = (stats["w_sum"] * stats["mean_slow"] + k * base_slow) / (stats["w_sum"] + k)
        stats["prior_tom"] = (stats["w_sum"] * stats["mean_tom"] + k * base_tom) / (stats["w_sum"] + k)

        out[col] = {
            "base_slow": base_slow,
            "base_tom": base_tom,
            "table": stats[["prior_slow", "prior_tom", "w_sum"]].copy(),
        }

    return out


def apply_geo_priors(
    df: pd.DataFrame,
    priors: Dict[str, Dict[str, Any]],
    *,
    prefix: str = "",
) -> pd.DataFrame:
    if df is None or df.empty or not priors:
        return df

    out = df
    for col, ctx in priors.items():
        if col not in out.columns:
            continue
        base_slow = float(ctx["base_slow"])
        base_tom = float(ctx["base_tom"])
        table = ctx["table"]

        key = out[col].fillna("missing").astype(str)
        slow = key.map(table["prior_slow"]).astype(float).fillna(base_slow).astype(np.float32)
        tom = key.map(table["prior_tom"]).astype(float).fillna(base_tom).astype(np.float32)
        n_eff = key.map(table["w_sum"]).astype(float).fillna(0.0).astype(np.float32)

        # Avoid blocked substrings ('duration', 'sold', 'label', 'target', ...)
        safe_col = col.replace(".", "_")
        out[f"{prefix}geo_prior_slow21_{safe_col}"] = slow
        out[f"{prefix}geo_prior_tom_h_{safe_col}"] = tom
        out[f"{prefix}geo_n_eff_{safe_col}"] = n_eff
    return out



def resolve_sbert_pca_rev_generic(conn, table_fq: str, source: str, model_rev: str, pca_rev: str) -> str:
    if pca_rev:
        return pca_rev
    schema, table = table_fq.split(".", 1)
    q = psql.SQL("""
        SELECT pca_rev
        FROM {}.{}
        WHERE source=%s AND model_rev=%s
        ORDER BY created_at DESC
        LIMIT 1
    """).format(psql.Identifier(schema), psql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(q, (source, model_rev))
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError(f"[sbert] could not auto-detect pca_rev for table={table_fq} source={source} model_rev={model_rev}")
    return row[0]

def load_sbert_vec_generic(conn, table_fq: str, source: str, model_rev: str, pca_rev: str) -> pd.DataFrame:
    schema, table = table_fq.split(".", 1)
    q = psql.SQL("""
        SELECT listing_id, edited_date, vec::text AS vec_txt
        FROM {}.{}
        WHERE source=%s AND model_rev=%s AND pca_rev=%s
    """).format(psql.Identifier(schema), psql.Identifier(table))
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(q, (source, model_rev, pca_rev))
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def parse_pgvector_text(s: pd.Series, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(s)
    mat = np.zeros((n, dim), dtype=np.float32)
    present = np.zeros(n, dtype=np.int8)
    if n == 0:
        return mat, present

    vals = s.fillna("").astype(str).values
    for i, t in enumerate(vals):
        t = t.strip()
        if not t or t[0] != "[" or t[-1] != "]":
            continue
        v = np.fromstring(t[1:-1], sep=",", dtype=np.float32)
        if v.shape[0] != dim or not np.isfinite(v).all():
            continue
        mat[i, :] = v
        present[i] = 1
    return mat, present

def attach_sbert_vec_features(conn, df: pd.DataFrame, args) -> pd.DataFrame:
    if df.empty:
        return df

    dim = int(args.sbert_dim)
    table_fq = _sbert_table_name(args)
    pca_rev = resolve_sbert_pca_rev_generic(conn, table_fq, args.sbert_source, args.sbert_model_rev, args.sbert_pca_rev)
    pref = (args.sbert_prefix.strip() if args.sbert_prefix else f"sbert{dim}")

    log(f"[sbert] table={table_fq} source={args.sbert_source} model_rev={args.sbert_model_rev} "
        f"pca_rev={pca_rev} dim={dim} prefix={pref}")

    sv = load_sbert_vec_generic(conn, table_fq, args.sbert_source, args.sbert_model_rev, pca_rev)
    if sv.empty:
        log("[sbert] no vectors found for this (table, source, model_rev, pca_rev).")
        return df

    df2 = df.copy()
    df2["edited_date"] = ensure_datetime_utc(df2["edited_date"])
    sv["edited_date"] = ensure_datetime_utc(sv["edited_date"])

    sv = sv.drop_duplicates(subset=["listing_id", "edited_date"], keep="last")

    before = len(df2)
    df2 = df2.merge(sv, on=["listing_id", "edited_date"], how="left")
    assert len(df2) == before

    mat, present = parse_pgvector_text(df2["vec_txt"], dim=dim)

    emb_cols = [f"{pref}_{j:02d}" for j in range(dim)]
    emb_df = pd.DataFrame(mat, columns=emb_cols, index=df2.index)
    present_s = pd.Series(present.astype("int8"), name=f"{pref}_present", index=df2.index)

    df2 = pd.concat(
        [df2.drop(columns=["vec_txt"], errors="ignore"), emb_df, present_s],
        axis=1
    )

    df2 = df2.copy()  # forces defragmentation
    log(f"[sbert] joined vectors present={int(df2[f'{pref}_present'].sum())}/{len(df2)}")
    return df2

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

def _get_emb_cols(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}_{j:02d}" for j in range(dim)]

def fit_sbert_prototypes(
    train_events: pd.DataFrame,
    emb_cols: List[str],
    min_n: int = 50,
    present_col: str = "",
) -> Dict[str, np.ndarray]:
    te = train_events
    if present_col and present_col in te.columns:
        te = te[te[present_col].astype("int8") == 1]

    if te.empty:
        return {}

    X = te[emb_cols].to_numpy(np.float32, copy=False)
    Xn = _l2_normalize_rows(X)

    y = te["duration_h"].to_numpy(np.float32, copy=False)

    masks = {
        "fast24":  (y <= 24.0),
        "fast72":  (y <= 72.0),
        "slow7d":  (y >= 168.0),
        "slow21d": (y >= 21.0 * 24.0),
    }

    protos: Dict[str, np.ndarray] = {}
    for name, m in masks.items():
        if int(m.sum()) < int(min_n):
            continue
        c = Xn[m].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-12)
        protos[name] = c.astype(np.float32)

    return protos

def apply_sbert_prototypes(
    df: pd.DataFrame,
    emb_cols: List[str],
    protos: Dict[str, np.ndarray],
    out_prefix: str,
) -> pd.DataFrame:
    if df.empty or not protos:
        return df

    X = df[emb_cols].to_numpy(np.float32, copy=False)
    Xn = _l2_normalize_rows(X)

    sim_cols = []
    for name, c in protos.items():
        col = f"{out_prefix}_sim_{name}"
        df[col] = (Xn @ c).astype(np.float32)
        sim_cols.append(col)

    if f"{out_prefix}_sim_fast72" in df.columns and f"{out_prefix}_sim_slow21d" in df.columns:
        df[f"{out_prefix}_margin_fast72_slow21d"] = (
            df[f"{out_prefix}_sim_fast72"] - df[f"{out_prefix}_sim_slow21d"]
        ).astype(np.float32)

    df[f"{out_prefix}_novelty"] = (1.0 - df[sim_cols].max(axis=1)).astype(np.float32)

    return df



def load_rows_with_strict_anchor(conn, limit: Optional[int], features_view: str = "ml.tom_features_v1_enriched_ai_clean_mv") -> pd.DataFrame:
    """
    Strict, leak-safe anchors @ t0 (edited_date):
      - Old strict anchor (30/60d blend) kept for comparison → ptv_anchor_strict_t0.
      - NEW advanced anchor v1 at t0:
          cohort = generation × model_norm × storage_bucket × condition_bucket × severity × battery_bucket × trust_tier
          window = [t0-365d, t0-5d] (5d embargo), no post-t0 data
          fallback cascade (require n≥200): keep model/storage; relax trust → battery (merge) → damage (bin) → condition (pool)
          outputs: anchor_price_smart, anchor_tts_median_h, anchor_n_support, anchor_level_k, ptv_anchor_smart

    NOTE: Battery always taken from feature-store's battery_pct_effective (consistent for sold & unsold).
    """
    # Optional socio/affordability columns: only present if features_view includes them
    socio_select = ""
    try:
        _cols = set(get_relation_columns(conn, features_view))
    except Exception:
        _cols = set()

    _want = [
        ("centrality_class", "int4"),
        ("miss_kommune", "int4"),
        ("miss_socio", "int4"),
        ("price_months_income", "float8"),
        ("log_price_to_income_cap", "float8"),
        ("aff_resid_in_seg", "float8"),
        ("aff_decile_in_seg", "int4"),
        ("rel_log_price_to_comp_mean30_super", "float8"),
        ("rel_log_price_to_comp_mean30_kommune", "float8"),
        ("rel_price_best", "float8"),
        ("rel_price_source", "int4"),
        ("miss_rel_price", "int4"),
    ]
    _present = [(c, cast) for (c, cast) in _want if c in _cols]
    if _present:
        _lines = ["      -- Socio + affordability (from features_view; optional)"]
        for (c, cast) in _present:
            _lines.append(f"      eai.{c}::{cast} AS {c},")
        socio_select = "\n".join(_lines)


    sql = f"""
    WITH base AS (
      SELECT
        f.listing_id,
        f.edited_date,
        f.generation,
        f.model,
        f.storage_gb,
        f.price,
        f.location_city,
        f.damage_binary_ai::int AS Binary,
        f.seller_rating::double precision AS seller_rating,
        f.review_count,
        f.member_since_year,
        f.condition_score::double precision AS condition_score,
        -- use the feature-store field; do NOT reference *_fixed_ai here
        f.battery_pct_effective::double precision AS battery_pct_effective,
        COALESCE(f.damage_severity_ai,0)::int AS sev,
        LOWER(REGEXP_REPLACE(f.model,'\\s+',' ','g')) AS model_norm,
        -- storage bucket for strict/speed joins (Apple SKUs only)
        CASE
          WHEN f.storage_gb >= 900 THEN 1024
          WHEN f.storage_gb >= 500 THEN 512
          WHEN f.storage_gb >= 250 THEN 256
          WHEN f.storage_gb >= 120 THEN 128
          ELSE f.storage_gb
        END AS sbucket,
        -- coarse condition buckets for old strict anchor
        CASE
          WHEN f.condition_score >= 0.95 THEN 1.0
          WHEN f.condition_score >= 0.85 THEN 0.9
          WHEN f.condition_score >= 0.65 THEN 0.7
          WHEN f.condition_score IS NULL THEN 0.0
          ELSE 0.0
        END AS cs_bucket
      FROM ml.tom_features_v1_mv f
      WHERE f.edited_date IS NOT NULL
      {f'LIMIT {int(limit)}' if limit else ''}
    ), y AS (
      SELECT listing_id, sold_event, sold_date, duration_hours
      FROM ml.tom_labels_v1_mv
    )
    SELECT
      b.*,

      -- NEW: basic image features per listing (NULL when no assets)
      img.image_count,
      img.caption_count,

      -- NEW: battery health from image screenshots (vision model)
      batt.battery_pct_img,

      -- AI columns (from AI-enriched read view, left-join by listing_id)
      eai.ai_sale_mode_obo, eai.ai_sale_mode_firm, eai.ai_sale_mode_bids, eai.ai_sale_mode_unspecified,
      eai.ai_owner_private, eai.ai_owner_work, eai.ai_owner_unknown,
      eai.ai_ship_can, eai.ai_ship_pickup, eai.ai_ship_unspecified,
      eai.ai_rep_apple, eai.ai_rep_authorized, eai.ai_rep_independent, eai.ai_rep_unknown,
      eai.ai_can_ship_bin, eai.ai_pickup_only_bin, eai.ai_vat_invoice_bin, eai.ai_first_owner_bin, eai.ai_used_with_case_bin,
      eai.ai_negotiability_f, eai.ai_urgency_f, eai.ai_lqs_textonly_f,
      eai.ai_opening_offer_nok_f, eai.ai_opening_offer_ratio,
      eai.ai_storage_gb_fixed,

      -- NEW: value anchors / deltas from enriched view
      eai.ptv_sold_30d,
      eai.ptv_ask_day,
      eai.delta_vs_sold_median_30d,
      eai.delta_vs_ask_median_day,
{socio_select}

      -- Speed anchors by (generation, storage bucket, ptv bucket) per D4 MV
      speed.speed_fast7_anchor,
      speed.speed_fast24_anchor,
      speed.speed_slow21_anchor,
      speed.speed_median_hours_ptv,
      speed.speed_n_eff_ptv,

      y.sold_event, y.sold_date, y.duration_hours,

      -- Old strict anchor (30/60d) — unchanged, for comparison
      a.anchor_30d_t0         AS anchor_strict_30d_t0,
      a.n30_t0                AS anchor_n30_t0,
      a.anchor_60d_t0         AS anchor_strict_60d_t0,
      a.n60_t0                AS anchor_n60_t0,
      a.anchor_blend_t0       AS anchor_blend_t0,
      CASE WHEN a.anchor_blend_t0 > 0 THEN b.price::numeric / a.anchor_blend_t0 ELSE NULL END AS ptv_anchor_strict_t0,

      -- NEW: Advanced anchor v1 @ t0 (price + duration + support + level)
      anc_smart.anchor_price_smart::numeric,
      anc_smart.anchor_tts_median_h::numeric,
      anc_smart.anchor_n_support::int,
      anc_smart.anchor_level_k::int,
      CASE WHEN anc_smart.anchor_price_smart > 0 THEN b.price::numeric / anc_smart.anchor_price_smart ELSE NULL END AS ptv_anchor_smart

    FROM base b
    LEFT JOIN y USING (listing_id)

    -- NEW: image counts per listing (no matview, treat "no assets" as NULL/missing)
    LEFT JOIN LATERAL (
      SELECT
        CASE
          WHEN COUNT(*) = 0 THEN NULL::int
          ELSE COUNT(*)
        END AS image_count,
        CASE
          WHEN COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               ) = 0
          THEN NULL::int
          ELSE COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               )
        END AS caption_count
      FROM "iPhone".iphone_image_assets ia
      WHERE ia.generation = b.generation
        AND ia.listing_id    = b.listing_id
    ) img ON TRUE

    -- NEW: battery % from screenshots (vision model; NULL if no screenshot / no value)
    LEFT JOIN LATERAL (
      SELECT
        MAX(i.battery_health_pct_img) AS battery_pct_img
      FROM ml.iphone_image_features_v1 i
      WHERE i.feature_version = 1
        AND i.battery_screenshot IS TRUE
        AND i.battery_health_pct_img IS NOT NULL
        AND i.generation = b.generation
        AND i.listing_id    = b.listing_id
    ) batt ON TRUE

    -- Old strict anchor logic (model×storage×CS×SEV, 30/60d blend, 5d embargo)
    LEFT JOIN LATERAL (
      WITH t0 AS (SELECT b.edited_date::date AS d),
      sold30 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.sold_price::numeric) AS med,
          COUNT(*)::int AS n
        FROM "iPhone".iphone_listings s, t0
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '35 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.generation = b.generation
          AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) = b.model_norm
          AND (CASE
                 WHEN s.storage_gb >= 900 THEN 1024
                 WHEN s.storage_gb >= 500 THEN 512
                 WHEN s.storage_gb >= 250 THEN 256
                 WHEN s.storage_gb >= 120 THEN 128
                 ELSE s.storage_gb
               END) = b.sbucket
          AND (
                (COALESCE(s.condition_score,0) IN (0.7,0.9,1.0)
                 AND COALESCE(s.damage_severity_ai,0) = b.sev)
                OR (COALESCE(s.condition_score,0)=0.0
                    AND COALESCE(s.damage_severity_ai,0)=0 AND b.sev=0)
              )
      ),
      sold60 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.sold_price::numeric) AS med,
          COUNT(*)::int AS n
        FROM "iPhone".iphone_listings s, t0
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '65 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.generation = b.generation
          AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) = b.model_norm
          AND (CASE
                 WHEN s.storage_gb >= 900 THEN 1024
                 WHEN s.storage_gb >= 500 THEN 512
                 WHEN s.storage_gb >= 250 THEN 256
                 WHEN s.storage_gb >= 120 THEN 128
                 ELSE s.storage_gb
               END) = b.sbucket
          AND (
                (COALESCE(s.condition_score,0) IN (0.7,0.9,1.0)
                 AND COALESCE(s.damage_severity_ai,0) = b.sev)
                OR (COALESCE(s.condition_score,0)=0.0
                    AND COALESCE(s.damage_severity_ai,0)=0 AND b.sev=0)
              )
      )
      SELECT
        (SELECT med FROM sold30) AS anchor_30d_t0,
        (SELECT n   FROM sold30) AS n30_t0,
        (SELECT med FROM sold60) AS anchor_60d_t0,
        (SELECT n   FROM sold60) AS n60_t0,
        CASE
          WHEN COALESCE((SELECT n FROM sold30),0) >= 10 THEN (SELECT med FROM sold30)
          WHEN COALESCE((SELECT n FROM sold30),0) + COALESCE((SELECT n FROM sold60),0) > 0 THEN
            (
              ((COALESCE((SELECT n FROM sold30),0) + 4) * COALESCE((SELECT med FROM sold30),0)) +
              ((COALESCE((SELECT n FROM sold60),0) + 2) * COALESCE((SELECT med FROM sold60),0))
            ) / NULLIF(((COALESCE((SELECT n FROM sold30),0) + 4)
                        + (COALESCE((SELECT n FROM sold60),0) + 2)), 0)
          ELSE NULL
        END AS anchor_blend_t0
    ) a ON TRUE

    -- NEW: Advanced anchor v1 @ t0 (365d lookback, 5d embargo), never mixing model/gen/storage.
    LEFT JOIN LATERAL (
      WITH t0 AS (SELECT b.edited_date::date AS d),

      -- Sold comps in the leak-safe window; battery comes from feature-store view to keep hierarchy consistent.
      sold_s AS (
        SELECT
          s.sold_price::numeric AS sold_price,
          EXTRACT(EPOCH FROM (s.sold_date - s.edited_date))/3600.0 AS duration_h,
          s.generation,
          LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) AS model_norm,
          CASE
            WHEN s.storage_gb >= 1900
                 AND s.generation = 17
                 AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) LIKE '%pro%'
              THEN 2048
            WHEN s.storage_gb >= 900 THEN 1024
            WHEN s.storage_gb >= 500 THEN  512
            WHEN s.storage_gb >= 250 THEN  256
            WHEN s.storage_gb >= 120 THEN  128
            ELSE NULL
          END AS sbucket,
          CASE
            WHEN COALESCE(s.condition_score,0) >= 0.99 THEN 1.0
            WHEN COALESCE(s.condition_score,0) >= 0.90 THEN 0.9
            WHEN COALESCE(s.condition_score,0) >= 0.70 THEN 0.7
            WHEN COALESCE(s.condition_score,0) >= 0.50 THEN 0.5
            WHEN COALESCE(s.condition_score,0) >= 0.20 THEN 0.2
            ELSE 0.0
          END AS cs_bucket,
          COALESCE(s.damage_severity_ai,0)::int AS sev,
          fv.battery_pct_effective AS batt_pct_eff,
          fv.seller_rating::double precision AS seller_rating,
          fv.review_count,
          fv.member_since_year
        FROM "iPhone".iphone_listings s
        JOIN ml.tom_features_v1_mv fv USING (listing_id)
        JOIN t0 ON TRUE
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '365 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.edited_date IS NOT NULL
      ),

      -- Buckets: battery + trust
      s2 AS (
        SELECT
          sold_price, duration_h, generation, model_norm, sbucket, cs_bucket, sev,
          CASE
            WHEN batt_pct_eff IS NULL OR batt_pct_eff = 0 THEN 'B_MISSING'
            WHEN batt_pct_eff <  80 THEN 'B_LOW'
            WHEN batt_pct_eff <  85 THEN 'B80_84'
            WHEN batt_pct_eff <  90 THEN 'B85_89'
            WHEN batt_pct_eff <  95 THEN 'B90_94'
            ELSE 'B95_100'
          END AS battery_bucket,
          CASE
            WHEN seller_rating >= 9.7
             AND review_count >= 50
             AND member_since_year <= EXTRACT(YEAR FROM (SELECT d FROM t0)) - 3 THEN 'HIGH'
            WHEN seller_rating >= 9.0
             AND review_count >= 10 THEN 'MED'
            ELSE 'LOW'
          END AS trust_tier

        FROM sold_s
        WHERE sbucket IS NOT NULL
      ),

      -- Cohort keys for the active listing
      base_keys AS (
        SELECT
          b.generation AS generation,
          b.model_norm AS model_norm,
          CASE
            WHEN b.storage_gb >= 1900
                 AND b.generation = 17
                 AND b.model_norm LIKE '%pro%'
              THEN 2048
            WHEN b.storage_gb >= 900 THEN 1024
            WHEN b.storage_gb >= 500 THEN  512
            WHEN b.storage_gb >= 250 THEN  256
            WHEN b.storage_gb >= 120 THEN  128
            ELSE NULL
          END AS sbucket,
          CASE
            WHEN COALESCE(b.condition_score,0) >= 0.99 THEN 1.0
            WHEN COALESCE(b.condition_score,0) >= 0.90 THEN 0.9
            WHEN COALESCE(b.condition_score,0) >= 0.70 THEN 0.7
            WHEN COALESCE(b.condition_score,0) >= 0.50 THEN 0.5
            WHEN COALESCE(b.condition_score,0) >= 0.20 THEN 0.2
            ELSE 0.0
          END AS cs_bucket,
          COALESCE(b.sev,0)::int AS sev,
          CASE
            WHEN b.battery_pct_effective IS NULL OR b.battery_pct_effective = 0 THEN 'B_MISSING'
            WHEN b.battery_pct_effective <  80 THEN 'B_LOW'
            WHEN b.battery_pct_effective <  85 THEN 'B80_84'
            WHEN b.battery_pct_effective <  90 THEN 'B85_89'
            WHEN b.battery_pct_effective <  95 THEN 'B90_94'
            ELSE 'B95_100'
          END AS battery_bucket,
          CASE
            WHEN b.seller_rating >= 9.7
             AND b.review_count >= 50
             AND b.member_since_year <= EXTRACT(YEAR FROM (SELECT d FROM t0)) - 3 THEN 'HIGH'
            WHEN b.seller_rating >= 9.0
             AND b.review_count >= 10 THEN 'MED'
            ELSE 'LOW'
          END AS trust_tier
      ),

      -- lock model/gen/storage to the active listing exactly
      s3 AS (
        SELECT s2.*
        FROM s2
        JOIN base_keys k
          ON s2.generation=k.generation
         AND s2.model_norm=k.model_norm
         AND s2.sbucket=k.sbucket
      ),

      -- add pooled groups (battery groups, sev_bin, condition groups) for fallbacks
      s3b AS (
        SELECT
          s3.*,
          CASE
            WHEN battery_bucket IN ('B95_100','B90_94') THEN 'B90P'
            WHEN battery_bucket IN ('B85_89','B80_84') THEN 'B80_89'
            ELSE 'B_LOW_OR_MISS'
          END AS bat_grp,
          CASE WHEN sev=0 THEN 0 ELSE 1 END AS sev_bin,
          CASE
            WHEN cs_bucket IN (0.7,0.9,1.0) THEN 'GOODP'
            WHEN cs_bucket = 0.5          THEN 'MID'
            ELSE 'POOR_OR_UNK'
          END AS cs_grp
        FROM s3
      ),

      base_b AS (
        SELECT
          k.*,
          CASE
            WHEN k.battery_bucket IN ('B95_100','B90_94') THEN 'B90P'
            WHEN k.battery_bucket IN ('B85_89','B80_84') THEN 'B80_89'
            ELSE 'B_LOW_OR_MISS'
          END AS bat_grp,
          CASE WHEN k.sev=0 THEN 0 ELSE 1 END AS sev_bin,
          CASE
            WHEN k.cs_bucket IN (0.7,0.9,1.0) THEN 'GOODP'
            WHEN k.cs_bucket = 0.5          THEN 'MID'
            ELSE 'POOR_OR_UNK'
          END AS cs_grp
        FROM base_keys k
      ),

      -- L0: full cohort (trust + battery + sev + condition)
      lvl0 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.battery_bucket = base_b.battery_bucket
          AND s3b.trust_tier     = base_b.trust_tier
      ),

      -- L1: drop trust (pool trust tiers)
      lvl1 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.battery_bucket = base_b.battery_bucket
      ),

      -- L2: merge battery (B95_100+B90_94) and (B85_89+B80_84), keep B_LOW/MISS distinct
      lvl2 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.bat_grp   = base_b.bat_grp
      ),

      -- L3: pool condition (GOOD+ vs MID vs POOR/UNK) and bin damage (0 vs >0)
      lvl3 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_grp = base_b.cs_grp
          AND s3b.sev_bin= base_b.sev_bin
      ),

      -- L4 (last resort): keep damage bin only (0 vs >0), pool all else
      lvl4 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.sev_bin= base_b.sev_bin
      ),

      candidates AS (
        SELECT 0 AS k, 'lvl0' AS lvl, p_med, t_med, n FROM lvl0
        UNION ALL SELECT 1, 'lvl1', p_med, t_med, n FROM lvl1
        UNION ALL SELECT 2, 'lvl2', p_med, t_med, n FROM lvl2
        UNION ALL SELECT 3, 'lvl3', p_med, t_med, n FROM lvl3
        UNION ALL SELECT 4, 'lvl4', p_med, t_med, n FROM lvl4
      )
      SELECT
        p_med AS anchor_price_smart,
        t_med AS anchor_tts_median_h,
        n     AS anchor_n_support,
        k     AS anchor_level_k
      FROM (
        SELECT c.*,
               ROW_NUMBER() OVER (
                 ORDER BY
                   CASE WHEN c.n >= 200 THEN 0 ELSE 1 END,
                   CASE WHEN c.n >= 200 THEN c.k ELSE -c.k END,
                   c.n DESC
               ) AS rnk
        FROM candidates c
      ) z
      WHERE rnk = 1
    ) anc_smart ON TRUE

    LEFT JOIN {features_view} eai
           ON eai.listing_id = b.listing_id

    LEFT JOIN LATERAL (
      -- Map ptv_final to ptv_bucket and join the D4 speed anchor MV
      WITH ptv_bin AS (
        SELECT
          CASE
            WHEN eai.ptv_final IS NULL OR eai.ptv_final < 0.50 OR eai.ptv_final > 2.50 THEN NULL::int
            WHEN eai.ptv_final < 0.80 THEN 1
            WHEN eai.ptv_final < 0.90 THEN 2
            WHEN eai.ptv_final < 1.00 THEN 3
            WHEN eai.ptv_final < 1.10 THEN 4
            WHEN eai.ptv_final < 1.20 THEN 5
            WHEN eai.ptv_final < 1.40 THEN 6
            ELSE 7
          END AS ptv_bucket
      )
      SELECT
        s.speed_fast7_anchor,
        s.speed_fast24_anchor,
        s.speed_slow21_anchor,
        s.speed_median_hours_ptv,
        s.speed_n_eff_ptv
      FROM ptv_bin pb
      LEFT JOIN ml.tom_speed_anchor_v1_mv s
        ON s.generation = b.generation
       AND s.sbucket    = b.sbucket
       AND s.ptv_bucket = pb.ptv_bucket
    ) AS speed ON TRUE;
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return pd.DataFrame(rows)



def load_all_rows(conn, limit: Optional[int], features_view: str = "ml.tom_features_v1_enriched_ai_clean_mv") -> pd.DataFrame:
    # Use the AI-augmented spam-clean read view so ai_* (and speed_* if present) are available.
    sql = f"""
    SELECT
      e.*,

      -- NEW: basic image features per listing (NULL when no assets)
      img.image_count,
      img.caption_count,

      -- NEW: battery health from image screenshots (vision model)
      batt.battery_pct_img,

      y.sold_event,
      y.sold_date,
      y.duration_hours
    FROM {features_view} e
    JOIN ml.tom_labels_v1_mv y USING (listing_id)

    -- Image count + caption count (no assets => NULL)
    LEFT JOIN LATERAL (
      SELECT
        CASE
          WHEN COUNT(*) = 0 THEN NULL::int
          ELSE COUNT(*)
        END AS image_count,
        CASE
          WHEN COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               ) = 0
          THEN NULL::int
          ELSE COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               )
        END AS caption_count
      FROM "iPhone".iphone_image_assets ia
      WHERE ia.generation = e.generation
        AND ia.listing_id    = e.listing_id
    ) img ON TRUE

    -- Battery % from screenshots (vision model; NULL if no screenshot / no value)
    LEFT JOIN LATERAL (
      SELECT
        MAX(i.battery_health_pct_img) AS battery_pct_img
      FROM ml.iphone_image_features_v1 i
      WHERE i.feature_version = 1
        AND i.battery_screenshot IS TRUE
        AND i.battery_health_pct_img IS NOT NULL
        AND i.generation = e.generation
        AND i.listing_id    = e.listing_id
    ) batt ON TRUE

    WHERE e.edited_date IS NOT NULL
    ORDER BY e.edited_date
    {f'LIMIT {int(limit)}' if limit else ''}
    """
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


_COL_OR_CAST_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(::([a-zA-Z_][a-zA-Z0-9_]*))?$")

def _build_select_items(contract_spec: list[tuple[str, str]], alias: str) -> list[psql.Composed]:
    """Build SELECT items from (feature_name, expr_sql) contract pairs.

    Supports:
      - col
      - col::type

    For anything more complex, expr_sql must already be a valid SQL expression
    (and should include the correct table alias).
    """
    items: list[psql.Composed] = []
    for feat_name, expr_sql in contract_spec:
        expr_sql = (expr_sql or "").strip()
        m = _COL_OR_CAST_RE.match(expr_sql)
        if m:
            col = m.group(1)
            typ = m.group(3)
            if typ:
                items.append(
                    psql.SQL(f"{alias}.") + psql.Identifier(col) + psql.SQL(f"::{typ} AS ") + psql.Identifier(feat_name)
                )
            else:
                items.append(
                    psql.SQL(f"{alias}.") + psql.Identifier(col) + psql.SQL(" AS ") + psql.Identifier(feat_name)
                )
        else:
            # Complex expression: trust contract authoring (must be safe SQL).
            items.append(psql.SQL(expr_sql) + psql.SQL(" AS ") + psql.Identifier(feat_name))
    return items


def load_image_agg_features(
    conn,
    listing_ids: list[int],
    generations: list[int],
    *,
    image_features_view: str = "ml.iphone_image_features_unified_v1",
    feature_set_name: str = "iphone_image_unified_v1_model",
    chunk_size: int = 20000,
    fetch_size: int = 5000,
) -> pd.DataFrame:
    """Load contracted vision/image features for a list of (generation, listing_id) pairs."""
    if len(listing_ids) != len(generations):
        raise ValueError("listing_ids and generations must have the same length")
    if not listing_ids:
        # empty input => empty output with contract columns
        spec = load_feature_contract_spec(conn, feature_set_name)
        cols = ['generation','listing_id'] + [c for c, _ in spec]
        return pd.DataFrame(columns=cols)

    assert_feature_contract(conn, feature_set_name)
    spec = load_feature_contract_spec(conn, feature_set_name)
    select_items = _build_select_items(spec, alias="u")

    # Always include join keys for downstream merge.
    select_items = [psql.SQL("u.generation AS generation"), psql.SQL("u.listing_id AS listing_id")] + select_items

    q = psql.SQL(
        """
        SELECT {select_list}
        FROM {view} u
        JOIN (
            SELECT
              UNNEST(%(generations)s::int[])    AS generation,
              UNNEST(%(listing_ids)s::bigint[])    AS listing_id
        ) t
          ON t.generation = u.generation
         AND t.listing_id    = u.listing_id
        """
    ).format(select_list=psql.SQL(", ").join(select_items), view=_rel_ident(image_features_view))

    dfs: list[pd.DataFrame] = []
    for i in range(0, len(listing_ids), chunk_size):
        gen_chunk = generations[i : i + chunk_size]
        id_chunk = listing_ids[i : i + chunk_size]
        with conn.cursor() as cur:
            cur.execute(q, {"generations": gen_chunk, "listing_ids": id_chunk})
            cols = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                try:
                    dfs.append(pd.DataFrame(rows, columns=cols))
                except MemoryError:
                    step = max(200, len(rows) // 4)
                    for k in range(0, len(rows), step):
                        dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                except Exception as e:
                    if type(e).__name__ in ("_ArrayMemoryError", "ArrayMemoryError"):
                        step = max(200, len(rows) // 4)
                        for k in range(0, len(rows), step):
                            dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                    else:
                        raise

    if not dfs:
        cols = ['generation','listing_id'] + [c for c, _ in spec]
        return pd.DataFrame(columns=cols)

    return pd.concat(dfs, ignore_index=True)

def load_damage_fusion_features(
    conn,
    listing_ids: list[int],
    generations: Optional[list[int]] = None,
    *,
    fusion_features_view: str = "ml.iphone_damage_fusion_v2_scored",
    feature_set_name: str = "damage_fusion_v2_scored_model",
    chunk_size: int = 20000,
    fetch_size: int = 2000,
) -> pd.DataFrame:
    """Load contracted damage-fusion features.

    Prefer joining by (generation, listing_id) when `generations` is provided and the
    fusion view exposes a `generation` column. This avoids accidental row
    duplication if a view contains multiple rows per listing_id across generations
    or model revisions.

    Always returns the join keys (listing_id and, when available, generation) plus
    the contracted feature columns.
    """
    if generations is not None and len(listing_ids) != len(generations):
        raise ValueError("listing_ids and generations must have the same length")

    # Inspect relation columns once to decide join keys.
    try:
        rel_cols = set(get_relation_columns(conn, fusion_features_view))
    except Exception:
        rel_cols = set()

    join_on_generation = bool(generations is not None and "generation" in rel_cols)

    if not listing_ids:
        spec = load_feature_contract_spec(conn, feature_set_name)
        feat_cols = [c for c, _ in spec if c not in ("generation", "listing_id")]
        if join_on_generation:
            cols = ["generation", "listing_id"] + feat_cols
        else:
            cols = ["listing_id"] + feat_cols
        return pd.DataFrame(columns=cols)

    assert_feature_contract(conn, feature_set_name)
    spec = load_feature_contract_spec(conn, feature_set_name)

    # Avoid duplicated output column names if the contract also includes join keys.
    spec = [(c, expr) for (c, expr) in spec if c not in ("generation", "listing_id")]
    select_items = _build_select_items(spec, alias="f")

    # Always include join keys for downstream merge.
    if join_on_generation:
        select_items = [psql.SQL("f.generation AS generation"), psql.SQL("f.listing_id AS listing_id")] + select_items
        q = psql.SQL(
            """
            SELECT {select_list}
            FROM {view} f
            JOIN (
                SELECT
                  UNNEST(%(generations)s::int[])   AS generation,
                  UNNEST(%(listing_ids)s::bigint[])   AS listing_id
            ) t
              ON t.generation = f.generation
             AND t.listing_id    = f.listing_id
            """
        ).format(select_list=psql.SQL(", ").join(select_items), view=_rel_ident(fusion_features_view))
    else:
        select_items = [psql.SQL("f.listing_id AS listing_id")] + select_items
        q = psql.SQL(
            """
            SELECT {select_list}
            FROM {view} f
            JOIN (
                SELECT UNNEST(%(listing_ids)s::bigint[]) AS listing_id
            ) t ON t.listing_id = f.listing_id
            """
        ).format(select_list=psql.SQL(", ").join(select_items), view=_rel_ident(fusion_features_view))

    dfs: list[pd.DataFrame] = []
    for i in range(0, len(listing_ids), chunk_size):
        id_chunk = listing_ids[i : i + chunk_size]
        params = {"listing_ids": id_chunk}
        if join_on_generation:
            gen_chunk = generations[i : i + chunk_size]  # type: ignore[index]
            params["generations"] = gen_chunk
        with conn.cursor() as cur:
            cur.execute(q, params)
            cols = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                try:
                    dfs.append(pd.DataFrame(rows, columns=cols))
                except MemoryError:
                    # Defensive fallback vs memory fragmentation / low-RAM situations:
                    # retry by splitting the batch into smaller parts.
                    step = max(200, len(rows) // 4)
                    for k in range(0, len(rows), step):
                        dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                except Exception as e:
                    if type(e).__name__ in ("_ArrayMemoryError", "ArrayMemoryError"):
                        step = max(200, len(rows) // 4)
                        for k in range(0, len(rows), step):
                            dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                    else:
                        raise

    if not dfs:
        feat_cols = [c for c, _ in spec]
        if join_on_generation:
            cols = ["generation", "listing_id"] + feat_cols
        else:
            cols = ["listing_id"] + feat_cols
        return pd.DataFrame(columns=cols)

    out = pd.concat(dfs, ignore_index=True)

    # Defensive: enforce 1 row per join key to avoid merge row explosion.
    if join_on_generation and {"generation", "listing_id"}.issubset(out.columns):
        if out.duplicated(subset=["generation", "listing_id"]).any():
            dup_n = int(out.duplicated(subset=["generation", "listing_id"]).sum())
            log(f"[fusion] WARN: {fusion_features_view} returned {dup_n} duplicate rows by (generation,listing_id); keeping last")
            out = out.drop_duplicates(subset=["generation", "listing_id"], keep="last")
    elif "listing_id" in out.columns:
        if out.duplicated(subset=["listing_id"]).any():
            dup_n = int(out.duplicated(subset=["listing_id"]).sum())
            log(f"[fusion] WARN: {fusion_features_view} returned {dup_n} duplicate rows by listing_id; keeping last")
            out = out.drop_duplicates(subset=["listing_id"], keep="last")

    return out

def load_device_meta_features(
    conn,
    listing_ids: list[int],
    generations: list[int],
    *,
    device_meta_features_view: str = "ml.iphone_device_meta_encoded_v1",
    feature_set_name: str = "iphone_device_meta_encoded_v1_model",
    chunk_size: int = 20000,
    fetch_size: int = 5000,
) -> pd.DataFrame:
    """Load contracted device-meta encoded features for (generation, listing_id) pairs."""
    if len(listing_ids) != len(generations):
        raise ValueError("listing_ids and generations must have the same length")
    if not listing_ids:
        spec = load_feature_contract_spec(conn, feature_set_name)
        cols = [c for c, _ in spec]
        return pd.DataFrame(columns=cols)

    assert_feature_contract(conn, feature_set_name)
    spec = load_feature_contract_spec(conn, feature_set_name)
    select_items = _build_select_items(spec, alias="m")

    q = psql.SQL(
        """
        SELECT {select_list}
        FROM {view} m
        JOIN (
            SELECT
              UNNEST(%(generations)s::int[]) AS generation,
              UNNEST(%(listing_ids)s::bigint[]) AS listing_id
        ) t
          ON t.generation = m.generation
         AND t.listing_id    = m.listing_id
        """
    ).format(select_list=psql.SQL(", ").join(select_items), view=_rel_ident(device_meta_features_view))

    dfs: list[pd.DataFrame] = []
    for i in range(0, len(listing_ids), chunk_size):
        gen_chunk = generations[i : i + chunk_size]
        id_chunk  = listing_ids[i : i + chunk_size]
        with conn.cursor() as cur:
            cur.execute(q, {"generations": gen_chunk, "listing_ids": id_chunk})
            cols = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                try:
                    dfs.append(pd.DataFrame(rows, columns=cols))
                except MemoryError:
                    step = max(200, len(rows) // 4)
                    for k in range(0, len(rows), step):
                        dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                except Exception as e:
                    if type(e).__name__ in ("_ArrayMemoryError", "ArrayMemoryError"):
                        step = max(200, len(rows) // 4)
                        for k in range(0, len(rows), step):
                            dfs.append(pd.DataFrame(rows[k:k+step], columns=cols))
                    else:
                        raise

    if not dfs:
        cols = [c for c, _ in spec]
        return pd.DataFrame(columns=cols)

    return pd.concat(dfs, ignore_index=True)



# (load_damage_fusion_features replaced by contract-governed loader above)
def load_first_inactive_meta_edited_at(
    conn: psycopg.Connection,
    pairs: pd.DataFrame,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Load first inactive meta_edited_at (seller intervention time) for (generation, listing_id) pairs.

    This is intended ONLY for label hygiene / censoring hygiene. Do NOT use these timestamps as features.
    We use PLATFORM's hydration-derived meta_edited_at (not observed_at) by design.
    """
    cols = ["generation", "listing_id", "first_inactive_meta_edited_at"]
    if pairs is None or len(pairs) == 0:
        return pd.DataFrame(columns=cols)

    p = pairs[["generation", "listing_id"]].dropna().drop_duplicates()
    if len(p) == 0:
        return pd.DataFrame(columns=cols)

    q = """
    WITH target AS (
      SELECT * FROM unnest(%(gens)s::int[], %(fids)s::bigint[]) AS t(generation, listing_id)
    )
    SELECT
      e.generation,
      e.listing_id,
      MIN(e.meta_edited_at) AS first_inactive_meta_edited_at
    FROM "iPhone".iphone_inactive_state_events e
    JOIN target t USING (generation, listing_id)
    WHERE e.is_inactive = TRUE
      AND e.meta_edited_at IS NOT NULL
    GROUP BY 1,2
    """

    rows: List[Dict[str, Any]] = []
    for i in range(0, len(p), chunk_size):
        chunk = p.iloc[i : i + chunk_size]
        gens = chunk["generation"].astype(int).tolist()
        fids = chunk["listing_id"].astype("int64").tolist()
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(q, {"gens": gens, "fids": fids})
            rows.extend(cur.fetchall())

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)



def add_damage_fusion_derived_features(df: pd.DataFrame, *, prefix: str = "fusion_") -> pd.DataFrame:
    """
    Convert a small set of string/categorical fusion outputs into numeric ordinals so they can be used
    by encode_numeric() without changing the upstream feature store.

    This is intentionally minimal; the raw string columns are kept (they will be dropped by encode_numeric).
    """
    # damage_fused_tier: tier0..tier3 -> 0..3
    tier_col = f"{prefix}damage_fused_tier"
    if tier_col in df.columns:
        tier_map = {"tier0": 0, "tier1": 1, "tier2": 2, "tier3": 3}
        df[f"{prefix}damage_fused_tier_ord"] = (
            df[tier_col].map(tier_map).astype("float64")
        )

    # batt_bucket: batt_unknown / batt_lo / batt_mid / batt_hi -> 0..3
    batt_col = f"{prefix}batt_bucket"
    if batt_col in df.columns:
        batt_map = {"batt_unknown": 0, "batt_lo": 1, "batt_mid": 2, "batt_hi": 3}
        df[f"{prefix}batt_band_ord"] = (
            df[batt_col].map(batt_map).astype("float64")
        )

    return df



def attach_damage_fusion_features(
    conn,
    df: pd.DataFrame,
    *,
    prefix: str = "fusion_",
    fusion_features_view: str = "ml.iphone_damage_fusion_v2_scored",
    feature_set_name: str = "damage_fusion_v2_scored_model",
    chunk_size: int = 20000,
) -> pd.DataFrame:
    """Left-join contracted damage fusion features onto df.

    Join keys:
      - Prefer (generation, listing_id) when both are available.
      - Fallback to listing_id only if the fusion view lacks generation.
    """
    if df.empty or "listing_id" not in df.columns:
        return df

    listing_ids = df["listing_id"].astype("int64", errors="ignore").tolist()
    generations = df["generation"].astype("int64", errors="ignore").tolist() if "generation" in df.columns else None

    f = load_damage_fusion_features(
        conn,
        listing_ids=listing_ids,
        generations=generations,
        fusion_features_view=fusion_features_view,
        feature_set_name=feature_set_name,
        chunk_size=chunk_size,
    )
    if f.empty:
        return df

    # Decide join keys based on what we actually got back.
    join_keys = ["listing_id"]
    if generations is not None and "generation" in f.columns and "generation" in df.columns:
        join_keys = ["generation", "listing_id"]

    # Prefix non-key columns
    f_cols = [c for c in f.columns if c not in join_keys]
    f = f.rename(columns={c: prefix + c for c in f_cols})

    # Avoid duplicate columns on merge
    overlap = [c for c in f.columns if c in df.columns and c not in join_keys]
    if overlap:
        df = df.drop(columns=overlap)

    out = df.merge(f, how="left", on=join_keys)

    # Fail-closed if the merge accidentally explodes rows (signals duplicate keys upstream).
    if len(out) != len(df):
        raise RuntimeError(
            f"[fusion] merge changed row count {len(df)} -> {len(out)}. "
            f"Check duplicates in {fusion_features_view} on keys {join_keys}."
        )

    return out


def attach_device_meta_features(
    conn,
    df: pd.DataFrame,
    *,
    device_meta_features_view: str = "ml.iphone_device_meta_encoded_v1",
    feature_set_name: str = "iphone_device_meta_encoded_v1_model",
    chunk_size: int = 20000,
) -> pd.DataFrame:
    """Left-join encoded device meta features onto df (join keys: generation,listing_id)."""
    if df.empty or "listing_id" not in df.columns or "generation" not in df.columns:
        return df

    listing_ids = df["listing_id"].astype("int64", errors="ignore").tolist()
    generations = df["generation"].astype("int64", errors="ignore").tolist()

    m = load_device_meta_features(
        conn,
        listing_ids=listing_ids,
        generations=generations,
        device_meta_features_view=device_meta_features_view,
        feature_set_name=feature_set_name,
        chunk_size=chunk_size,
    )
    if m.empty:
        return df

    # Make sure join keys exist in result
    if "generation" not in m.columns or "listing_id" not in m.columns:
        raise RuntimeError(
            "Device meta feature block did not return required join keys (generation,listing_id). "
            "Ensure your feature contract includes them."
        )

    # Avoid collisions (should not happen if you prefix dev_* in the view, but keep safe)
    overlap = [c for c in m.columns if c in df.columns and c not in ("generation", "listing_id")]
    if overlap:
        df = df.drop(columns=overlap)

    return df.merge(m, how="left", on=["generation", "listing_id"])




def load_trainer_derived_features_chunked(
    conn,
    keys: pd.DataFrame,
    *,
    view: str,
    chunk_size: int = 25000,
    fetch_size: int = 5000,
) -> pd.DataFrame:
    """Load trainer-derived (T0-safe) features from a DB relation.

    `keys` must contain: generation, listing_id, edited_date (UTC).
    The relation referenced by `view` is expected to expose:
      generation, listing_id, t0, and the derived feature columns.

    Implementation notes:
      - Uses a psycopg cursor + fetchmany() to avoid pandas.read_sql_query overhead
        (and the pandas/DBAPI warnings) on large key sets.
      - Designed to be fast when `view` is a precomputed MV with a unique index on
        (generation, listing_id, t0).
    """
    if keys is None or keys.empty:
        return pd.DataFrame()

    k = keys[["generation", "listing_id", "edited_date"]].copy()
    k["edited_date"] = ensure_datetime_utc(k["edited_date"])

    gens = k["generation"].astype(int).tolist()
    fids = k["listing_id"].astype("int64").tolist()

    # psycopg wants Python datetime objects for timestamptz[]
    t0s = np.array(k["edited_date"].dt.to_pydatetime()).tolist()

    # Injection-safe relation identifier
    view_ident = _rel_ident(view)

    q = psql.SQL(
        """
        SELECT
          d.generation,
          d.listing_id,
          d.t0 AS edited_date,
          d.dow,
          d.is_weekend,
          d.gen_30d_post_count,
          d.allgen_30d_post_count,
          d.battery_img_conflict,
          d.battery_img_minus_text,
          d.battery_pct_effective_fused,
          d.rocket_clean,
          d.rocket_heavy,
          d.fast_pattern_v2,
          d.is_zombie_pattern
        FROM {view} d
        JOIN (
          SELECT * FROM unnest(
            %(gens)s::int[],
            %(fids)s::bigint[],
            %(t0s)s::timestamptz[]
          ) AS t(generation, listing_id, t0)
        ) k
          ON k.generation = d.generation
         AND k.listing_id   = d.listing_id
         AND k.t0        = d.t0
        """
    ).format(view=view_ident)

    dfs: List[pd.DataFrame] = []
    n = len(k)
    n_chunks = (n + int(chunk_size) - 1) // int(chunk_size)

    for i in range(0, n, int(chunk_size)):
        gens_c = gens[i : i + int(chunk_size)]
        fids_c = fids[i : i + int(chunk_size)]
        t0s_c  = t0s[i : i + int(chunk_size)]

        t_start = dt.datetime.utcnow()
        with conn.cursor() as cur:
            cur.execute(q, {"gens": gens_c, "fids": fids_c, "t0s": t0s_c})
            cols = [d.name for d in cur.description]
            while True:
                rows = cur.fetchmany(int(fetch_size))
                if not rows:
                    break
                dfs.append(pd.DataFrame(rows, columns=cols))

        t_end = dt.datetime.utcnow()
        log(
            f"[trainer_derived_store] fetched chunk {1 + (i // int(chunk_size))}/{n_chunks} "
            f"rows={min(int(chunk_size), n - i)} sec={(t_end - t_start).total_seconds():.2f}"
        )

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not out.empty and "edited_date" in out.columns:
        out["edited_date"] = ensure_datetime_utc(out["edited_date"])
    return out


def attach_trainer_derived_store(
    conn,
    df: pd.DataFrame,
    *,
    trainer_derived_view: str,
    chunk_size: int = 25000,
    min_coverage: float = 0.98,
    cert_entrypoint: str = "ml.trainer_derived_feature_store_t0_v1_v",
    cert_max_age: str = "24 hours",
    skip_cert_check: bool = False,
) -> pd.DataFrame:
    """Attach trainer-derived DB store features to the training frame.

    Join keys:
      - (generation, listing_id, edited_date == store.t0)

    Fail-closed semantics (without per-row overhead):
      1) Require the CERTIFIED entrypoint ONCE up-front (audit.require_certified_strict).
      2) Read the requested `trainer_derived_view` as a plain relation for performance.

    IMPORTANT:
      If you pass a guarded view name (e.g. *_train_v), Postgres may legally execute the
      guard function many times because audit.require_certified_strict is VOLATILE.
      To avoid pathological runtimes, this function will automatically read from the
      precomputed MV (ml.trainer_derived_features_v1_mv) when a *_train_v view is passed,
      after performing the up-front certification check.
    """
    if df is None or df.empty:
        return df
    if not trainer_derived_view:
        return df

    required = {"generation", "listing_id", "edited_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"trainer_derived_store merge requires columns {sorted(required)}; missing {sorted(missing)}"
        )

    # 1) Require certification ONCE (fail closed)
    if not skip_cert_check:
        try:
            audit_require_certified_strict(conn, cert_entrypoint, max_age=cert_max_age)
        except Exception as e:
            raise RuntimeError(
                f"trainer_derived_store entrypoint not certified (or viewdef drifted): {cert_entrypoint} "
                f"(max_age={cert_max_age}) ({e})"
            ) from e
    else:
        log("[trainer_derived_store] skip_cert_check=True; NOT enforcing certification")

    # 2) Choose a fast source relation (avoid guarded view overhead)
    source_view = trainer_derived_view.strip()
    if source_view.endswith("_train_v"):
        source_view = "ml.trainer_derived_features_v1_mv"
    log(f"[trainer_derived_store] source={source_view} (cert_entrypoint={cert_entrypoint})")

    keys = df[["generation", "listing_id", "edited_date"]].copy()
    td = load_trainer_derived_features_chunked(conn, keys, view=source_view, chunk_size=chunk_size)

    if td.empty:
        raise RuntimeError(
            f"trainer_derived_store returned 0 rows for view={source_view}. "
            "Check store population and join key alignment (df.edited_date vs store.t0)."
        )

    td = td.drop_duplicates(subset=["generation", "listing_id", "edited_date"])

    # Prefer DB store values for overlapping columns (avoid pandas merge suffixes)
    overlap = [c for c in td.columns if c in df.columns and c not in ("generation", "listing_id", "edited_date")]
    if overlap:
        df = df.drop(columns=overlap)

    out = df.merge(td, on=["generation", "listing_id", "edited_date"], how="left")

    # Coverage check: 'dow' is deterministic from t0; if missing, join failed
    if "dow" in out.columns:
        cov = float(out["dow"].notna().mean())
        if cov < float(min_coverage):
            raise RuntimeError(
                f"trainer_derived_store coverage too low: {cov:.3%} (min {min_coverage:.1%}). "
                "Likely df.edited_date != store.t0 (timezone/rounding mismatch) or store is incomplete."
            )

    # Normalize fused battery name to preserve downstream expectations
    if "battery_pct_effective_fused" in out.columns:
        if "battery_pct_effective" in out.columns:
            out["battery_pct_effective"] = out["battery_pct_effective_fused"].where(
                out["battery_pct_effective_fused"].notna(),
                out["battery_pct_effective"],
            )
        else:
            out["battery_pct_effective"] = out["battery_pct_effective_fused"]
        out = out.drop(columns=["battery_pct_effective_fused"])

    return out

def load_orderbook_features(
    conn,
    listing_ids: List[int],
    *,
    view: str = OB_FEATURES_VIEW,
    cols: Optional[List[str]] = None,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Load OB features for a set of listing_ids from a precomputed MV."""
    if cols is None:
        cols = OB_FEATURE_COLS

    cols = [c for c in cols if c != "listing_id"]

    if not listing_ids:
        return pd.DataFrame(columns=["listing_id"] + cols)

    ids = sorted(set(int(x) for x in listing_ids if x is not None))

    sel = ", ".join(cols)
    sql = f"""
        SELECT
          listing_id,
          {sel}
        FROM {view}
        WHERE listing_id = ANY(%(fids)s::bigint[])
    """

    rows: List[Dict[str, Any]] = []
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, {"fids": chunk})
            rows.extend(cur.fetchall())

    if not rows:
        return pd.DataFrame(columns=["listing_id"] + cols)

    return pd.DataFrame(rows)


def attach_orderbook_features(conn, df: pd.DataFrame, *, view: str = OB_FEATURES_VIEW) -> pd.DataFrame:
    """Attach OB features if they are missing from df."""
    if df is None or df.empty or "listing_id" not in df.columns:
        return df

    missing = [c for c in OB_FEATURE_COLS if c not in df.columns]
    if not missing:
        log(f"[ob] {view}: ob_* columns already present; skipping attach")
        return df

    listing_ids = df["listing_id"].dropna().astype("int64").unique().tolist()
    if not listing_ids:
        return df

    try:
        ob_df = load_orderbook_features(conn, listing_ids, view=view, cols=missing)
    except Exception as e:
        log(f"[ob] WARN: could not attach OB features from {view}: {e}")
        return df

    if ob_df is None or ob_df.empty:
        log(f"[ob] {view}: no rows returned; continuing without OB features")
        return df

    out = df.merge(ob_df, on="listing_id", how="left")
    log(f"[ob] attached {len(missing)} OB columns from {view}")
    return out



def write_model_registry(conn,
                         model_key: str,
                         feature_set: str,
                         feat_hash: str,
                         algo: str,
                         hyperparams: Dict[str,Any],
                         metrics: Dict[str,Any],
                         train_start: Optional[dt.datetime],
                         train_end: Optional[dt.datetime],
                         artifact_uri: str,
                         created_by: str,
                         notes: str):
    sql = """
    INSERT INTO ml.model_registry
      (model_key, feature_set_name, feature_set_hash, trained_at, train_start, train_end,
       algo, hyperparams, metrics, artifact_uri, is_active, created_by, notes)
    VALUES
      (%s, %s, %s, now(), %s, %s, %s, %s::jsonb, %s::jsonb, %s, FALSE, %s, %s)
    ON CONFLICT (model_key) DO UPDATE
      SET trained_at=EXCLUDED.trained_at,
          train_start=EXCLUDED.train_start, train_end=EXCLUDED.train_end,
          algo=EXCLUDED.algo, hyperparams=EXCLUDED.hyperparams,
          metrics=EXCLUDED.metrics, artifact_uri=EXCLUDED.artifact_uri,
          created_by=EXCLUDED.created_by, notes=EXCLUDED.notes;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            model_key, feature_set, feat_hash, train_start, train_end,
            algo,
            json.dumps(clean_json_numbers(hyperparams)),
            json.dumps(clean_json_numbers(metrics)),
            artifact_uri, created_by, notes
        ))
        conn.commit()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_utc_pydt(ts: Any) -> Optional[dt.datetime]:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def persist_woe_anchor_artifacts_v1(
    conn,
    *,
    model_key: str,
    train_cutoff_ts: Any,
    half_life_days: float,
    eps: float,
    n_folds: int,
    band_schema_version: int,
    code_sha256: str,
    ctx_final: Dict[str, Any],
    fold_ctxs: Optional[List[Dict[str, Any]]] = None,
    oof_scores: Optional[pd.DataFrame] = None,
    set_active: bool = True,
) -> None:
    """Persist a coherent WOE "model package" into Postgres.

    Tables:
      - ml.woe_anchor_model_registry_v1
      - ml.woe_anchor_cuts_v1
      - ml.woe_anchor_map_v1
      - ml.woe_anchor_scores_v1 (optional, OOF scores)
    """
    fold_ctxs = fold_ctxs or []

    # Required parameters from ctx_final
    base_rate = float(ctx_final.get("base_rate", float("nan")))
    base_logit = float(ctx_final.get("base_logit", float("nan")))
    ds1, ds2, ds3, ds4 = tuple(ctx_final.get("dsold_thresholds") or (None, None, None, None))
    c1, c2 = tuple(ctx_final.get("presentation_cuts") or (0.0, 0.0))

    train_cutoff_ts_pydt = _to_utc_pydt(train_cutoff_ts)

    with conn.cursor() as cur:
        # Deactivate prior active models (if requested)
        if set_active:
            cur.execute(
                "UPDATE ml.woe_anchor_model_registry_v1 SET is_active = FALSE WHERE is_active = TRUE AND model_key <> %s",
                (model_key,),
            )

        # Upsert registry row
        cur.execute(
            """
            INSERT INTO ml.woe_anchor_model_registry_v1
              (model_key, train_cutoff_ts, half_life_days, eps, n_folds, band_schema_version, code_sha256,
               is_active, base_rate, base_logit, dsold_t1, dsold_t2, dsold_t3, dsold_t4)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_key) DO UPDATE
              SET train_cutoff_ts      = EXCLUDED.train_cutoff_ts,
                  half_life_days       = EXCLUDED.half_life_days,
                  eps                 = EXCLUDED.eps,
                  n_folds             = EXCLUDED.n_folds,
                  band_schema_version  = EXCLUDED.band_schema_version,
                  code_sha256          = EXCLUDED.code_sha256,
                  is_active            = EXCLUDED.is_active,
                  base_rate            = EXCLUDED.base_rate,
                  base_logit           = EXCLUDED.base_logit,
                  dsold_t1             = EXCLUDED.dsold_t1,
                  dsold_t2             = EXCLUDED.dsold_t2,
                  dsold_t3             = EXCLUDED.dsold_t3,
                  dsold_t4             = EXCLUDED.dsold_t4,
                  created_at           = now();
            """,
            (
                model_key,
                train_cutoff_ts_pydt,
                float(half_life_days),
                float(eps),
                int(n_folds),
                int(band_schema_version),
                str(code_sha256),
                bool(set_active),
                base_rate,
                base_logit,
                ds1,
                ds2,
                ds3,
                ds4,
            ),
        )

        # Replace mapping/cuts (idempotent)
        cur.execute("DELETE FROM ml.woe_anchor_map_v1  WHERE model_key = %s", (model_key,))
        cur.execute("DELETE FROM ml.woe_anchor_cuts_v1 WHERE model_key = %s", (model_key,))

        # Cuts (fold_id NULL = final)
        cur.execute(
            "INSERT INTO ml.woe_anchor_cuts_v1(model_key, fold_id, c1, c2) VALUES (%s, NULL, %s, %s)",
            (model_key, float(c1), float(c2)),
        )
        for k in range(int(n_folds)):
            cur.execute(
                "INSERT INTO ml.woe_anchor_cuts_v1(model_key, fold_id, c1, c2) VALUES (%s, %s, %s, %s)",
                (model_key, int(k), float(c1), float(c2)),
            )

        # Mapping rows
        def _insert_maps(ctx: Dict[str, Any], fold_id: Optional[int]) -> None:
            tables = ctx.get("woe_tables") or {}
            for band_name, rows in tables.items():
                if rows is None:
                    continue
                for r in rows:
                    band_value = r.get("band_value")
                    if band_value is None:
                        continue
                    cur.execute(
                        """
                        INSERT INTO ml.woe_anchor_map_v1
                          (model_key, fold_id, band_name, band_value, woe, sum_w_pos, sum_w_neg)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            model_key,
                            fold_id,
                            str(band_name),
                            str(band_value),
                            float(r.get("woe", 0.0)),
                            None if r.get("sum_w_pos") is None else float(r.get("sum_w_pos")),
                            None if r.get("sum_w_neg") is None else float(r.get("sum_w_neg")),
                        ),
                    )

        _insert_maps(ctx_final, None)
        for ctx_k in fold_ctxs:
            fid = ctx_k.get("fold_id")
            if fid is None:
                continue
            _insert_maps(ctx_k, int(fid))

        # Optional: store OOF scores (1 row per observation)
        if oof_scores is not None and len(oof_scores) > 0:
            cur.execute("DELETE FROM ml.woe_anchor_scores_v1 WHERE model_key = %s", (model_key,))
            rows = []
            for r in oof_scores.itertuples(index=False):
                rows.append(
                    (
                        model_key,
                        int(r.generation),
                        int(r.listing_id),
                        _to_utc_pydt(r.t0),
                        None if pd.isna(r.fold_id) else int(r.fold_id),
                        bool(getattr(r, "is_oof", True)),
                        None if pd.isna(r.woe_logit) else float(r.woe_logit),
                        None if pd.isna(r.woe_p) else float(r.woe_p),
                    )
                )
            cur.executemany(
                """
                INSERT INTO ml.woe_anchor_scores_v1
                  (model_key, generation, listing_id, t0, fold_id, is_oof, woe_logit, woe_p)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (model_key, generation, listing_id, t0) DO UPDATE
                  SET fold_id   = EXCLUDED.fold_id,
                      is_oof    = EXCLUDED.is_oof,
                      woe_logit = EXCLUDED.woe_logit,
                      woe_p     = EXCLUDED.woe_p,
                      created_at= now();
                """,
                rows,
            )

        conn.commit()
    log(f"[db] persisted WOE anchor artifacts v1: model_key={model_key} (active={set_active})")




def write_mae_summary(conn, model_key: str, rows: List[Tuple]):
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany("""
          INSERT INTO ml.mae_summary_v1
            (model_key, horizon_hours, n_eval, mae_hours, med_ae_hours, p90_ae_hours,
             c_index, ibs_24h, ibs_48h, cohort_key, cohort_json)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
          ON CONFLICT (model_key, horizon_hours, cohort_key) DO UPDATE
            SET n_eval      = EXCLUDED.n_eval,
                mae_hours   = EXCLUDED.mae_hours,
                med_ae_hours= EXCLUDED.med_ae_hours,
                p90_ae_hours= EXCLUDED.p90_ae_hours,
                c_index     = EXCLUDED.c_index,
                ibs_24h     = EXCLUDED.ibs_24h,
                cohort_json = EXCLUDED.cohort_json;
        """, rows)
        conn.commit()

# NEW: persist eval-time predictions for SOLD rows (eval_df + pred_eval) into ml.predictions_eval_sold_v1
def write_eval_predictions_for_sold(
    conn: psycopg.Connection,
    model_key: str,
    eval_df: pd.DataFrame,
    pred_eval: np.ndarray,
):
    """
    Persist per-row eval predictions for SOLD rows into ml.predictions_eval_sold_v1.

    We treat asof_ts = edited_date (t0). expected_hours is the predicted total duration.
    """
    if eval_df.empty:
        log("[eval_sold] eval_df is empty; nothing to write.")
        return
    if pred_eval is None or len(pred_eval) != len(eval_df):
        log(f"[eval_sold] mismatch: len(eval_df)={len(eval_df)} vs len(pred_eval)={len(pred_eval)}; skipping.")
        return

    rows: List[Tuple] = []
    for (_, row), pred_h in zip(eval_df.iterrows(), pred_eval):
        try:
            fid = int(row["listing_id"])
        except Exception:
            continue
        edited = pd.to_datetime(row["edited_date"], utc=True)
        sold   = pd.to_datetime(row["sold_date"],   utc=True)
        dur_h  = float(row["duration_hours"])
        asof_ts = edited.to_pydatetime()  # prediction at t0
        rows.append((
            fid,
            model_key,
            asof_ts,
            edited.to_pydatetime(),
            sold.to_pydatetime(),
            dur_h,
            float(pred_h),
            "train_eval",
        ))

    if not rows:
        log("[eval_sold] no rows to insert into ml.predictions_eval_sold_v1.")
        return

    sql = """
      INSERT INTO ml.predictions_eval_sold_v1
        (listing_id, model_key, asof_ts, edited_date, sold_date,
         duration_hours, expected_hours, source)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
      ON CONFLICT (listing_id, model_key, asof_ts) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    log(f"[eval_sold] inserted {len(rows)} eval SOLD predictions into ml.predictions_eval_sold_v1")

# ---------------- AFT with XGBoost ----------------
def build_monotone_constraints(feat_cols: List[str]) -> str:
    cons = []
    for c in feat_cols:
        lc = c.lower()

        # Price / value / anchors: higher -> slower
        if lc in (
            "price",
            "price_z_in_region",
            "ptv_final",
            "ptv_anchor_strict_t0",
            "ptv_anchor_smart",
            "anchor_median_hours_14d",
            "delta_vs_sold_median_30d",
            "ptv_sold_30d",
            "delta_vs_ask_median_day",
            "woe_anchor_p_slow21",
            "woe_anchor_logit_slow21",
        ):
            cons.append(+1)

        # Severity: more damage -> slower or equal (can't speed up by itself)
        elif lc in ("sev", "damage_severity_ai"):
            cons.append(+1)

        # Trust: higher rating / more reviews -> faster
        elif lc in ("seller_rating", "review_count"):
            cons.append(-1)

        else:
            cons.append(0)

    return "(" + ",".join(str(int(v)) for v in cons) + ")"


# ---------------- Feature importance persistence (DB) ----------------
# We persist both gain-based importance and TreeSHAP importance into:
#   ml.model_feature_importance_v1
# so downstream SQL can aggregate by meta->>'feature_block' etc.

def infer_feature_block(feature_name: str) -> str:
    """Heuristic feature block classifier used for model_feature_importance_v1 meta."""
    f = feature_name or ""

    # Stock (live-inventory) — computed in Python from iphone_listings intervals
    if f in {"stock_n_sm4_gen_sbucket", "stock_n_sm4_gen", "stock_share_sbucket"}:
        return "stock_store"


    # Trainer-derived store (DB) — T0-certified derived features
    if f in {
        "dow",
        "is_weekend",
        "gen_30d_post_count",
        "allgen_30d_post_count",
        "battery_img_conflict",
        "battery_img_minus_text",
        "battery_pct_effective",
        "rocket_clean",
        "rocket_heavy",
        "fast_pattern_v2",
        "is_zombie_pattern",
    }:
        return "trainer_derived_store"
    # Missingness flags
    if f.endswith("__missing") or f.startswith("miss_"):
        return "trainer_missing_flags"
    # Supervised prior / target encoding
    if f.startswith("woe_") or f.startswith("woe_anchor"):
        return "anchor_woe_store"
    # Anchors / priors
    if f.startswith(("ptv_", "anchor_", "speed_")):
        return "anchor_priors_store"
    # Geo
    if f.startswith("geo_"):
        return "geo_store"
    # Orderbook
    if f.startswith("ob_"):
        return "orderbook_store"
    # AI enrichment
    if f.startswith("ai_"):
        return "ai_enrichment_store"
    # Fusion
    if f.startswith("fusion_"):
        return "fusion_store"
    # Device meta
    if f.startswith(("dev_", "gmc_")):
        return "device_meta_store"
    # Vision / image-derived
    if f.startswith((
        "battery_", "battery_pct", "bg_", "body_", "caption_", "case_", "charger_", "dmg_",
        "earbuds_", "has_", "image_", "n_battery", "photo_", "receipt_", "stock_", "box_",
        "color_",
    )):
        return "vision_store"
    # Socio / market
    if f in {
        "centrality_class",
        "price_months_income",
        "log_price_to_income_cap",
        "aff_resid_in_seg",
        "aff_decile_in_seg",
        "rel_log_price_to_comp_mean30_super",
        "rel_log_price_to_comp_mean30_kommune",
        "rel_price_best",
        "rel_price_source",
        "miss_socio",
        "miss_kommune",
        "miss_rel_price",
    } or f.startswith(("aff_", "rel_", "log_price", "price_months_income")):
        return "socio_economic_store"
    # Trainer-derived (simple heuristics)
    if f.startswith(("delta_", "fast_")) or f in {"binary", "sev", "rocket_clean"}:
        return "trainer_derived_store"
    # Default
    return "main_feature_store"


def write_model_feature_importance_v1(
    conn,
    model_key: str,
    feat_cols,
    values: dict,
    scope: str,
    importance_type: str,
    n_rows: int,
    meta_common=None,
):
    """Write per-feature importance values to ml.model_feature_importance_v1.

    Expects ml.model_feature_importance_v1 schema:
      (model_key, scope, importance_type, feature_name, importance_value, n_rows, meta jsonb, created_at default now())
    """
    if conn is None:
        return
    meta_common = meta_common or {}
    rows = []
    for fname in feat_cols:
        v = values.get(fname, 0.0)
        if v is None:
            v = 0.0
        # Normalize NaN/inf
        try:
            fv = float(v)
            if not math.isfinite(fv):
                fv = 0.0
        except Exception:
            fv = 0.0
        meta = dict(meta_common)
        meta["feature_block"] = infer_feature_block(fname)
        rows.append((model_key, scope, importance_type, fname, fv, int(n_rows), json.dumps(meta)))

    sql_del = """
        DELETE FROM ml.model_feature_importance_v1
        WHERE model_key = %s AND scope = %s AND importance_type = %s
    """
    sql_ins = """
        INSERT INTO ml.model_feature_importance_v1
            (model_key, scope, importance_type, feature_name, importance_value, n_rows, meta)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
    """
    with conn.cursor() as cur:
        cur.execute(sql_del, (model_key, scope, importance_type))
        cur.executemany(sql_ins, rows)
    conn.commit()
    log(f"[db] persisted feature importance: model_key={model_key} scope={scope} type={importance_type} rows={len(rows)}")


def compute_tree_shap_importance(
    booster,
    X,
    feat_cols,
    max_rows: int = 2000,
    seed: int = 1,
):
    """Compute TreeSHAP mean(|contrib|) and mean(contrib) using XGBoost pred_contribs.

    Returns: (mean_abs_dict, mean_dict, n_used)
    """
    if X is None:
        return {}, {}, 0
    X_arr = np.asarray(X)
    n = int(X_arr.shape[0])
    if n == 0:
        return {}, {}, 0
    if max_rows is not None and n > int(max_rows):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=int(max_rows), replace=False)
        X_use = X_arr[idx]
    else:
        X_use = X_arr

    d = xgb.DMatrix(X_use, feature_names=list(feat_cols))
    contrib = booster.predict(d, pred_contribs=True)
    contrib = np.asarray(contrib)
    # Last column is bias term
    if contrib.ndim != 2 or contrib.shape[1] < 2:
        return {}, {}, int(X_use.shape[0])

    vals = contrib[:, :len(feat_cols)]
    mean_abs = np.mean(np.abs(vals), axis=0)
    mean = np.mean(vals, axis=0)

    mean_abs_dict = {fname: float(v) for fname, v in zip(feat_cols, mean_abs)}
    mean_dict = {fname: float(v) for fname, v in zip(feat_cols, mean)}
    return mean_abs_dict, mean_dict, int(X_use.shape[0])



def xgb_aft_train_predict(
    Xe: np.ndarray, ye: np.ndarray,
    Xc: np.ndarray, yc: np.ndarray,
    X_cal: np.ndarray, y_cal: np.ndarray,
    X_eval: np.ndarray,
    ev_dates_events: pd.Series,
    ev_dates_cens: Optional[pd.Series],
    feat_cols: List[str],
    args
) -> Tuple[np.ndarray, Optional[IsotonicRegression], Dict[str,Any], Dict[str, float], xgb.Booster]:
    
    # Defensive: AFT uses log(time) internally; times must be strictly > 0.
    ye = np.clip(np.asarray(ye, dtype=np.float32), EPS_TIME_H, None)

    if Xc.size > 0:
        yc = np.clip(np.asarray(yc, dtype=np.float32), EPS_TIME_H, None)
    else:
        yc = np.asarray(yc, dtype=np.float32)

    if y_cal is not None and len(y_cal):
        y_cal = np.clip(np.asarray(y_cal, dtype=np.float32), EPS_TIME_H, None)


    # Build training design (events + censored)
    if Xc.size > 0:
        X_train = np.vstack([Xe, Xc])
        lb = np.concatenate([ye, yc]).astype(np.float32)
        ub = np.concatenate([ye, np.full_like(yc, AFT_UB_CENS, dtype=np.float32)]).astype(np.float32)


        w_ev = time_decay_weights(ev_dates_events, args.half_life_days)

        # Loss shaping for the 21d (504h) classifier:
        # - keep recency weighting (time_decay_weights)
        # - upweight SOLD >7d and SOLD >21d (multiplicative tails)
        # - bump samples near the 504h boundary (boundary_focus_* knobs)
        # - mild guard on very fast sellers (<10d) to reduce SAC_LT10 blowups
        slow_w = float(args.slow_tail_weight) if (args.slow_tail_weight and args.slow_tail_weight > 1.0) else 1.0
        vslow_w = float(getattr(args, "very_slow_tail_weight", 1.0))
        vslow_w = vslow_w if (vslow_w and vslow_w > 1.0) else 1.0

        tail_mult = np.ones_like(ye, dtype=np.float32)
        tail_mult *= np.where(ye > 168.0, slow_w, 1.0).astype(np.float32)
        tail_mult *= np.where(ye > 504.0, vslow_w, 1.0).astype(np.float32)

        boundary_mult = boundary_focus_weights(ye, args.boundary_focus_k, args.boundary_focus_sigma)

        # Very fast sellers are the most sensitive to being incorrectly flagged as SLOW (>504h).
        fast_guard_k = 0.25
        bf_k = float(args.boundary_focus_k) if getattr(args, "boundary_focus_k", None) is not None else 0.0
        fast_guard_mult = np.where(ye < 240.0, 1.0 + fast_guard_k * max(bf_k, 0.0), 1.0).astype(np.float32)

        w_ev = (w_ev * tail_mult * boundary_mult * fast_guard_mult).astype(np.float32)
        m = np.nanmean(w_ev)
        if m and m > 0:
            w_ev = (w_ev / m).astype(np.float32)

        # Apply the same 21d-focused shaping to censored rows using their censor-time lower bound.
        w_ce = time_decay_weights(ev_dates_cens, args.half_life_days)
        yc_arr = yc.astype(np.float32)

        ce_tail = np.ones_like(yc_arr, dtype=np.float32)
        ce_tail *= np.where(yc_arr > 168.0, slow_w, 1.0).astype(np.float32)
        ce_tail *= np.where(yc_arr > 504.0, vslow_w, 1.0).astype(np.float32)

        ce_boundary = boundary_focus_weights(yc_arr, args.boundary_focus_k, args.boundary_focus_sigma)
        w_ce = (w_ce * ce_tail * ce_boundary).astype(np.float32)
        m = np.nanmean(w_ce)
        if m and m > 0:
            w_ce = (w_ce / m).astype(np.float32)

        weights = np.concatenate([w_ev, w_ce]).astype(np.float32)


    else:
        X_train = Xe
        lb = ye.astype(np.float32)
        ub = ye.astype(np.float32)

        w_ev = time_decay_weights(ev_dates_events, args.half_life_days)

        # Loss shaping for the 21d (504h) classifier:
        # - keep recency weighting (time_decay_weights)
        # - upweight SOLD >7d and SOLD >21d (multiplicative tails)
        # - bump samples near the 504h boundary (boundary_focus_* knobs)
        # - mild guard on very fast sellers (<10d) to reduce SAC_LT10 blowups
        slow_w = float(args.slow_tail_weight) if (args.slow_tail_weight and args.slow_tail_weight > 1.0) else 1.0
        vslow_w = float(getattr(args, "very_slow_tail_weight", 1.0))
        vslow_w = vslow_w if (vslow_w and vslow_w > 1.0) else 1.0

        tail_mult = np.ones_like(ye, dtype=np.float32)
        tail_mult *= np.where(ye > 168.0, slow_w, 1.0).astype(np.float32)
        tail_mult *= np.where(ye > 504.0, vslow_w, 1.0).astype(np.float32)

        boundary_mult = boundary_focus_weights(ye, args.boundary_focus_k, args.boundary_focus_sigma)

        fast_guard_k = 0.25
        bf_k = float(args.boundary_focus_k) if getattr(args, "boundary_focus_k", None) is not None else 0.0
        fast_guard_mult = np.where(ye < 240.0, 1.0 + fast_guard_k * max(bf_k, 0.0), 1.0).astype(np.float32)

        w_ev = (w_ev * tail_mult * boundary_mult * fast_guard_mult).astype(np.float32)
        m = np.nanmean(w_ev)
        if m and m > 0:
            w_ev = (w_ev / m).astype(np.float32)

        weights = w_ev.astype(np.float32)


    # use feature names so get_score() returns real column names, not f0,f1,...
    dtrain = xgb.DMatrix(
        X_train.astype(np.float32),
        feature_names=list(feat_cols)
    )
    dtrain.set_float_info('label_lower_bound', lb)
    dtrain.set_float_info('label_upper_bound', ub)
    dtrain.set_weight(np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0))

    dcal = None
    if len(y_cal):
        dcal = xgb.DMatrix(
            X_cal.astype(np.float32),
            feature_names=list(feat_cols)
        )
        y_cal_f = y_cal.astype(np.float32)
        dcal.set_float_info('label_lower_bound', y_cal_f)
        dcal.set_float_info('label_upper_bound', y_cal_f)

    deval = xgb.DMatrix(
        X_eval.astype(np.float32),
        feature_names=list(feat_cols)
    )

    xgb_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": args.aft_dist,
        "aft_loss_distribution_scale": float(args.aft_scale),
        "tree_method": "gpu_hist",          # GPU training
        "predictor": "gpu_predictor",       # GPU prediction

        # Make max_leaves behave as intended (leaf-wise growth like LGBM)
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": int(max(8, args.num_leaves)),

        "learning_rate": float(args.learning_rate),
        "subsample": float(args.bagging_fraction),
        "colsample_bytree": float(args.feature_fraction),
        "reg_lambda": float(args.lambda_l2),
        "min_child_weight": max(1.0, float(args.min_data_in_leaf)),
        "max_bin": int(args.max_bin),
        "verbosity": 0,
        "seed": int(args.seed),
        "monotone_constraints": build_monotone_constraints(feat_cols),
    }


    evals = [(dtrain, "train")]
    if dcal is not None and len(y_cal) >= 50:
        evals.append((dcal, "cal"))

    # Custom eval metric aligned to the *21-day (504h) classifier* target:
    # maximize SLOW-class F1 at threshold 504h. Used for early stopping/model selection only;
    # the training objective remains survival:aft.
    def _slow21_custom_metric(preds: np.ndarray, dmat: xgb.DMatrix):
        try:
            y_true = dmat.get_float_info("label_lower_bound")
        except Exception:
            y_true = None
        if y_true is None or len(y_true) == 0:
            return [("slow21_slow_f1", 0.0)]
        p = np.asarray(preds, dtype=float)
        p = np.clip(p, 0.0, None)
        rep = classify_at_threshold(y_true, p, SLOW21_H)
        val = rep.get("slow", {}).get("f1", 0.0)
        if val is None or not np.isfinite(val):
            val = 0.0
        return [("slow21_slow_f1", float(val))]

    callbacks = []
    if len(evals) > 1 and getattr(args, "early_stopping_rounds", 0) and int(args.early_stopping_rounds) > 0:
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=int(args.early_stopping_rounds),
                metric_name="slow21_slow_f1",
                data_name="cal",
                maximize=True,
                save_best=True,
            )
        )

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=evals,
        custom_metric=_slow21_custom_metric,
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Safety: avoid pathological early-stopping collapse (e.g. best_iteration=1–2).
    es_best_iter = getattr(booster, "best_iteration", None)
    if es_best_iter is not None and len(evals) > 1:
        min_required = min(50, max(10, int(0.02 * int(args.n_estimators))))
        if int(es_best_iter) < min_required:
            log(f"early stopping picked best_iteration={es_best_iter} < {min_required}; retraining without early stopping")
            booster = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=int(args.n_estimators),
                evals=evals,
                custom_metric=_slow21_custom_metric,
                verbose_eval=False,
            )


    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        pred_eval = booster.predict(deval)
        pred_cal  = booster.predict(dcal) if dcal is not None else None
    else:
        try:
            pred_eval = booster.predict(deval, iteration_range=(0, int(best_iter)+1))
            pred_cal  = booster.predict(dcal,  iteration_range=(0, int(best_iter)+1)) if dcal is not None else None
        except TypeError:
            pred_eval = booster.predict(deval, ntree_limit=int(best_iter)+1)
            pred_cal  = booster.predict(dcal,  ntree_limit=int(best_iter)+1) if dcal is not None else None

    # Treat outputs as HOURS, clamp to sane range, remove NaN/Inf
    def _clean(v):
        if v is None:
            return None
        v = np.asarray(v, dtype=float)
        if np.isfinite(v).any():
            vmax = np.nanmax(v[np.isfinite(v)])
        else:
            vmax = 1e5
        v = np.clip(v, 0.0, vmax * 3.0)
        return np.nan_to_num(v, nan=0.0, posinf=vmax * 3.0, neginf=0.0)

    pred_eval = _clean(pred_eval)
    pred_cal  = _clean(pred_cal) if pred_cal is not None else None

    # Boundary-aware isotonic on CAL
    iso = None
    if pred_cal is not None and len(y_cal) >= 100:
        cal_w = boundary_focus_weights(y_cal, args.boundary_focus_k, args.boundary_focus_sigma)
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pred_cal, y_cal, sample_weight=cal_w)
            pred_eval = iso.predict(pred_eval)
            log("applied boundary-aware isotonic calibration on recent SOLD slice")
        except Exception as e:
            log(f"[warn] isotonic failed: {e}")

    # Gain-based FI
    gain_map = booster.get_score(importance_type="gain")
    total_gain = float(sum(gain_map.values())) if gain_map else 0.0
    share = {k: (100.0 * float(v) / total_gain) if total_gain > 0 else 0.0
             for k, v in gain_map.items()}

    return pred_eval, iso, xgb_params, share, booster

# ---------------- Optuna objective (multi-metric) ----------------
def optuna_objective(trial: optuna.Trial, args: argparse.Namespace, data_bundle: Dict[str, Any]) -> float:
    """
    Optuna objective: optimize ONLY the 21-day (504h) classifier quality.

    Target to maximize:
      slow_f1 = metrics["fast504_classification"]["slow"]["f1"]

    With constraints/penalties to prevent degenerate "predict everything slow" solutions:
      - Hard cap: SAC_LT10 <= FP_LT10_CAP_DEFAULT
      - Soft penalty: score -= mid_penalty * SAC_MID
    """
    # Make a per-trial copy so Optuna does not mutate global args across trials
    args = copy.deepcopy(args)

    # Hyperparameter search space (keep same family/CLI semantics; only objective changes).
    args.learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    args.num_leaves = trial.suggest_int("num_leaves", 16, 128)
    args.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 120)
    args.feature_fraction = trial.suggest_float("feature_fraction", 0.4, 1.0)
    args.bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    args.lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 10.0)
    args.aft_scale = trial.suggest_float("aft_scale", 0.5, 2.0)

    # Loss-shaping knobs already exposed by CLI.
    args.slow_tail_weight = trial.suggest_float("slow_tail_weight", 1.0, 10.0)
    args.very_slow_tail_weight = trial.suggest_float("very_slow_tail_weight", 1.0, 10.0)
    args.boundary_focus_k = trial.suggest_float("boundary_focus_k", 0.0, 2.0)
    args.boundary_focus_sigma = trial.suggest_float("boundary_focus_sigma", 3.0, 36.0)

    X_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    event_dates_train = data_bundle["event_dates_train"]
    X_cens = data_bundle["X_cens"]
    y_cens = data_bundle["y_cens"]
    event_dates_cens = data_bundle["event_dates_cens"]
    X_cal = data_bundle["X_cal"]
    y_cal = data_bundle["y_cal"]
    X_eval = data_bundle["X_eval"]
    y_eval = data_bundle["y_eval"]
    feat_cols = data_bundle["feat_cols"]


    pred_eval, _, _, _, booster = xgb_aft_train_predict(
        Xe=X_train,
        ye=y_train,
        Xc=X_cens,
        yc=y_cens,
        X_cal=X_cal,
        y_cal=y_cal,
        X_eval=X_eval,
        ev_dates_events=event_dates_train,
        ev_dates_cens=event_dates_cens,
        feat_cols=feat_cols,
        args=args,
    )

    rep = slow21_focus_report(y_eval, pred_eval, slow_thresh_hours=SLOW21_H, lt10_thresh_hours=LT10_H)
    slow_stats = rep.get("slow", {})
    sac = rep.get("sacrifice", {})
    conf_slow = rep.get("confusion_slow", {})

    slow_f1 = float(slow_stats.get("f1", 0.0) or 0.0)
    slow_prec = float(slow_stats.get("precision", 0.0) or 0.0)
    slow_rec = float(slow_stats.get("recall", 0.0) or 0.0)

    sac_lt10 = float(sac.get("sac_lt10", 0.0) or 0.0)
    sac_mid = float(sac.get("sac_mid", 0.0) or 0.0)

    # Pure F1 objective (slow21): maximize slow21 F1 only.
    # NOTE: We intentionally do NOT hard-reject trials based on SAC_LT10 / SAC_MID.
    score = float(slow_f1)


    # Pure-F1 objective: keep these for audit only (no penalties applied)
    fp_cap = float(FP_LT10_CAP_DEFAULT)
    mid_penalty = 0.0
    # Trial attribution (auditable).
    trial.set_user_attr("threshold_hours", float(SLOW21_H))
    trial.set_user_attr("slow_f1", slow_f1)
    trial.set_user_attr("slow_precision", slow_prec)
    trial.set_user_attr("slow_recall", slow_rec)
    trial.set_user_attr("sac_lt10", sac_lt10)
    trial.set_user_attr("sac_mid", sac_mid)
    trial.set_user_attr("fp_lt10_cap", fp_cap)
    trial.set_user_attr("mid_penalty", mid_penalty)
    trial.set_user_attr("tp_slow", int(conf_slow.get("tp_slow", 0)))
    trial.set_user_attr("fp_slow", int(conf_slow.get("fp_slow", 0)))
    trial.set_user_attr("fn_slow", int(conf_slow.get("fn_slow", 0)))
    trial.set_user_attr("tn_slow", int(conf_slow.get("tn_slow", 0)))
    trial.set_user_attr("best_iteration", int(getattr(booster, "best_iteration", -1)) if hasattr(booster, "best_iteration") else -1)

    return float(score)

# ---------------- Main ----------------
def main():
    args = parse_args()
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        log("ERROR: PG_DSN not set")
        raise SystemExit(2)
    outdir = pathlib.Path(args.outdir)
    ensure_dir(outdir)

    # Stable model_key for the entire run (also used for WOE artifact persistence)
    if args.model_key:
        model_key = args.model_key
    else:
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_key = f"tom_aft_v1_{ts}"
    log(f"model_key={model_key}")

    log("connecting to Postgres…")
    conn = psycopg.connect(dsn, application_name=getattr(args, "pg_application_name", "train_aft"))
    # Apply session-level safety/perf settings (timeouts, jit, app_name)
    apply_pg_session_settings(conn, args)

    # Fail fast on uncertified feature stores (cheap checks, avoids long hangs later)
    preflight_cert_checks(conn, args)


    # Feature contract hash
    try:
        feat_hash = load_feature_set_hash(conn, args.feature_set)
        log(f"feature_set={args.feature_set} hash={feat_hash}")
    except Exception as e:
        log(f"[warn] feature_set hash lookup failed: {e}")
        feat_hash = "unknown"

    # External feature-block contracts (fail-fast + log; guarantees no silent drift)
    def _log_and_assert_block(tag: str, feature_set: str, view: str) -> None:
        try:
            h = load_feature_set_hash(conn, feature_set)
        except Exception as e:
            h = "unknown"
            log(f"[warn] {tag} feature_set hash lookup failed: {e}")
        log(f"[contract] {tag} feature_set={feature_set} hash={h} view={view}")
        assert_feature_contract(conn, feature_set)

    # Contracts for side feature blocks (only if enabled)
    _log_and_assert_block("image", args.image_feature_set, args.image_features_view)
    if not args.disable_fusion:
        _log_and_assert_block("fusion", args.fusion_feature_set, args.fusion_features_view)
    if not args.disable_device_meta:
        _log_and_assert_block("device_meta", args.device_meta_feature_set, args.device_meta_features_view)

    # Load data
    log("loading dataset…")
    if args.use_strict_anchor:
        df = load_rows_with_strict_anchor(conn, args.train_limit, features_view=args.features_view)
        log("strict anchors @ t0 enabled (model×storage×CS×SEV; 30/60d blend; gen preserved) + advanced_anchor_v1 (price+tts+support with fallback cascade, no storage mixing)")
    else:
        df = load_all_rows(conn, args.train_limit, features_view=args.features_view)

    if df.empty:
        log("no rows found; abort")
        raise SystemExit(1)

    # --- socio/affordability (from features_view, if present) ---
    _socio_cols = [
        "centrality_class",
        "miss_kommune",
        "miss_socio",
        "price_months_income",
        "log_price_to_income_cap",
        "aff_resid_in_seg",
        "aff_decile_in_seg",
        "rel_log_price_to_comp_mean30_super",
        "rel_log_price_to_comp_mean30_kommune",
        "rel_price_best",
        "rel_price_source",
        "miss_rel_price",
    ]
    _present_socio = [c for c in _socio_cols if c in df.columns]
    if _present_socio:
        log(f"[socio] attached {len(_present_socio)} socio/market cols from features_view: {', '.join(_present_socio)}")
    else:
        log("[socio] no socio cols present on base df (features_view likely not *_socio_mv)")


    # Geo mapping (super_metro_v4) attach (optional)
    geo_mode = (args.geo_mode or "off").lower()
    if geo_mode != "off":
        try:
            df = ensure_geo_columns(conn, df, geo_dim_view=args.geo_dim_view)

            # Quick QA logging (proves geo wiring at training time)
            if "geo_match_method" in df.columns:
                mm = df["geo_match_method"].value_counts(dropna=False).to_dict()
                log(f"[geo] geo_match_method_counts={mm}")
            if "region_geo" in df.columns and "super_metro_v4_geo" in df.columns:
                unk = int((df["region_geo"] == "unknown").sum())
                other = int(df["super_metro_v4_geo"].astype(str).str.startswith("other_").sum())
                log(f"[geo] coverage unknown_region={unk}/{len(df)} other_super={other}/{len(df)}")
        except Exception as e:
            log(f"[geo] failed to attach geo columns; continuing with geo_mode=off ({e})")
            args.geo_mode = "off"

    
    # Optional generation filter: train only on a single generation
    if args.gen_filter is not None:
        before = len(df)
        df = df[df["generation"] == args.gen_filter].copy()
        after = len(df)
        log(f"[seg] gen_filter={args.gen_filter} applied: {before} -> {after} rows")
        if df.empty:
            log(f"[seg] no rows left after gen_filter={args.gen_filter}; abort.")
            raise SystemExit(1)
        
    # Drop legacy light-vision columns loaded in SQL so the unified merge doesn't create *_x/*_y duplicates.
    drop_before_vision_merge = ["image_count", "caption_count", "battery_pct_img"]
    df.drop(columns=[c for c in drop_before_vision_merge if c in df.columns], inplace=True, errors="ignore")

    # ---- Optional: live-inventory ('stock') features (computed in Python; no DB objects) ----
    if args.use_stock_features:
        df = attach_stock_features_python(
            conn,
            df,
            t0_col="edited_date",
            geo_dim_view=args.geo_dim_view,
            meta_view=args.features_view,
        )


        
    # ---- Attach aggregated image features (damage / quality / accessories / color) ----
    img_feats = load_image_agg_features(
        conn,
        listing_ids=df["listing_id"].tolist(),
        generations=df["generation"].tolist(),
        image_features_view=args.image_features_view,
        feature_set_name=args.image_feature_set,
        chunk_size=args.image_chunk_size,
    )
    if not img_feats.empty:
        before_cols = set(df.columns)
        df = df.merge(
            img_feats,
            on=["generation", "listing_id"],
            how="left",
        )
        added_img_cols = sorted(set(df.columns) - before_cols)
        log(f"[img] merged image features (+{len(added_img_cols)} cols): {', '.join(added_img_cols)}")

        if not args.disable_device_meta:
            df = attach_device_meta_features(
                conn,
                df,
                device_meta_features_view=args.device_meta_features_view,
                feature_set_name=args.device_meta_feature_set,
                chunk_size=args.device_meta_chunk_size,
            )
            added_dm_cols = sorted(set(df.columns) - before_cols - set(added_img_cols))
            log(f"[device_meta] merged device meta features (+{len(added_dm_cols)} cols): {', '.join(added_dm_cols)}")

        # ---- Battery: prefer screenshot-derived if present ----
        if args.disable_trainer_derived_store:
            # Rules:
            # - text battery_pct_effective often 0/NULL; treat <=0 as missing
            # - prefer image battery_pct_img when available
            batt_img = pd.to_numeric(df.get("battery_pct_img"), errors="coerce")
            batt_txt = pd.to_numeric(df.get("battery_pct_effective"), errors="coerce")
            batt_txt = batt_txt.where(batt_txt > 0)
            df["battery_img_conflict"] = (
                batt_img.notna() & batt_txt.notna() & (np.abs(batt_img - batt_txt) >= 4.0)
            ).astype("int8")
            df["battery_img_minus_text"] = (batt_img - batt_txt)
            df["battery_pct_effective"] = np.where(batt_img.notna(), batt_img, batt_txt)
            log("[img] derived battery features: battery_img_conflict, battery_img_minus_text, battery_pct_effective")
        else:
            log("[trainer_derived_store] enabled: skipping Python battery fusion (will use DB trainer-derived store)")
    else:
        log("[img] no image agg features found for these rows")


    # Damage fusion feature store (image+text) — optional join
    if not args.disable_fusion:
        before_fusion_cols = set(df.columns)
        df = attach_damage_fusion_features(
            conn,
            df,
            fusion_features_view=args.fusion_features_view,
            feature_set_name=args.fusion_feature_set,
            chunk_size=args.fusion_chunk_size,
        )
        added_fusion_cols = sorted(set(df.columns) - before_fusion_cols)
        log(f"[fusion] merged fusion features (+{len(added_fusion_cols)} cols): {', '.join(added_fusion_cols)}")
    df = attach_orderbook_features(conn, df)

    # Body color flags (numeric) so encode_numeric picks them up
    if "body_color_key_main" in df.columns:
        color_map = [
            "graphite",
            "midnight",
            "sierra_blue",
            "blue",
            "silver",
            "starlight",
            "gold",
            "alpine_green",
            "pink",
            "green",
            "productred",
        ]

        # Use object dtype so comparison gives plain bools, NaN -> False
        bck = df["body_color_key_main"].astype(object)

        for ck in color_map:
            colname = f"color_{ck}"
            # (bck == ck) is a normal bool Series here, no <NA>
            df[colname] = (bck == ck).astype("int8")

        # Explicit flag for unknown / missing color
        df["color_unknown"] = df["body_color_key_main"].isna().astype("int8")

        log("[img] added body color flags for keys: " + ", ".join(color_map) + " + color_unknown")




    # Prep durations
    df["edited_date"] = ensure_datetime_utc(df["edited_date"])
    df["sold_date"]   = ensure_datetime_utc(df.get("sold_date"))
    df["duration_h"]  = pd.to_numeric(df["duration_hours"], errors="coerce").astype(float)
    df = df[df["duration_h"].notna()].copy()
    df["duration_h"]  = np.clip(df["duration_h"].values, EPS_TIME_H, None)


    if args.use_sbert_vec:
        df = attach_sbert_vec_features(conn, df, args)
        pref = (args.sbert_prefix.strip() if args.sbert_prefix else f"sbert{int(args.sbert_dim)}")
        log("[sbert] example cols: " + ", ".join([c for c in df.columns if c.startswith(pref + "_")][:8]))


    # Winsorize extreme durations to stabilize tails
    if args.winsor_q and 0 < args.winsor_q < 1:
        old_max = float(np.nanmax(df["duration_h"]))
        df["duration_h"] = winsorize_upper(df["duration_h"].values, args.winsor_q)
        new_max = float(np.nanmax(df["duration_h"]))
        log(f"winsorized durations at q={args.winsor_q}: max {old_max:.2f} -> {new_max:.2f}")

    # ---- Trainer-derived feature store (DB; certified) ----
    if (not args.disable_trainer_derived_store) and args.trainer_derived_features_view:
        before_cols = set(df.columns)
        df = attach_trainer_derived_store(
            conn,
            df,
            trainer_derived_view=args.trainer_derived_features_view,
            chunk_size=args.trainer_derived_chunk_size,
            cert_entrypoint="ml.trainer_derived_feature_store_t0_v1_v",
            cert_max_age=f"{getattr(args, 'cert_max_age_hours', 24)} hours",
            skip_cert_check=getattr(args, 'skip_cert_checks', False),
        )
        added = [c for c in df.columns if c not in before_cols]
        log(f"[trainer_derived_store] merged (+{len(added)} cols) from {args.trainer_derived_features_view}")
    else:
        # ---- New calendar & inventory features (no-leak) ----
        # DOW (0=Mon, 6=Sun) and weekend flag, based on edited_date (t₀)
        df["dow"] = df["edited_date"].dt.weekday.astype("int8")
        df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")

        # Inventory proxy 1: last-30-days flow per generation
        # For each row, count how many *same-generation* listings were edited in the
        # last 30 days up to and including this row's edited_date.
        # This uses only past edited_date -> no leakage.
        df_idx = df.index.copy()
        df_sorted = df.sort_values(["generation", "edited_date"]).copy()

        gen_30d_counts = (
            df_sorted
            .set_index("edited_date")
            .groupby("generation")["listing_id"]
            .rolling("30D")
            .count()
            .reset_index(level=0, drop=True)
        )

        df_sorted["gen_30d_post_count"] = gen_30d_counts.astype("float32").values

        # Inventory proxy 2: last-30-days flow across *all* gens (13–17)
        all_sorted = df_sorted.sort_values("edited_date").copy()
        all_30d_counts = (
            all_sorted
            .set_index("edited_date")["listing_id"]
            .rolling("30D")
            .count()
        )
        all_sorted["allgen_30d_post_count"] = all_30d_counts.astype("float32").values

        # Restore original row order
        df = all_sorted.loc[df_idx].copy()


        # Optional region price-z
        if args.region_map_csv:
            df, df = region_price_z(df, df, args.geo_col, args.region_map_csv,
                                    args.region_shrink_k, edited_col="edited_date", half_life_days=float(args.half_life_days))


        # ---- Rocket & zombie pattern features (price × quality interaction) ----
        # These are built from t0-safe feature-store columns (no outcome leakage).

        # Severity: prefer 'sev' if present from strict-anchor SQL, otherwise fallback to damage_severity_ai.
        sev_raw = None
        if "sev" in df.columns:
            sev_raw = df["sev"]
        elif "damage_severity_ai" in df.columns:
            sev_raw = df["damage_severity_ai"]
        else:
            sev_raw = pd.Series(0, index=df.index)

        sev = pd.to_numeric(sev_raw, errors="coerce").fillna(0).astype(int)

        # Condition, battery, PTV, discount vs sold-median.
        cs    = pd.to_numeric(df.get("condition_score"), errors="coerce")
        batt  = pd.to_numeric(df.get("battery_pct_effective"), errors="coerce")

        # PTV: prefer ptv_final if available from the enriched MV; otherwise fall back to ptv_anchor_smart.
        if "ptv_final" in df.columns:
            ptv_raw = df["ptv_final"]
        elif "ptv_anchor_smart" in df.columns:
            ptv_raw = df["ptv_anchor_smart"]
        else:
            ptv_raw = pd.Series(np.nan, index=df.index)

        ptv = pd.to_numeric(ptv_raw, errors="coerce")
        delta_sold = pd.to_numeric(df.get("delta_vs_sold_median_30d"), errors="coerce")

        # CLEAN ROCKET: what we actually observed works in your 180d analysis.
        rocket_clean = (
            sev.isin([0, 1])
            & (cs >= 0.70)
            & ((batt.isna()) | (batt >= 84))
            & (ptv <= 0.95)
            & (delta_sold <= -500)
        )

        # HEAVY ROCKET: cracked but stupid cheap (sev=3, ptv<=0.95, delta<=-2000).
        rocket_heavy = (
            (sev == 3)
            & (ptv <= 0.95)
            & (delta_sold <= -2000)
        )

        df["rocket_clean"] = rocket_clean.astype("int8")
        df["rocket_heavy"] = rocket_heavy.astype("int8")
        df["fast_pattern_v2"] = (rocket_clean | rocket_heavy).astype("int8")

        # ZOMBIE PATTERN: systematically slow, overpriced crap (high ptv and positive delta vs sold).
        zombie = (ptv > 1.20) & (delta_sold > 500)
        df["is_zombie_pattern"] = zombie.astype("int8")
    # ---- Missingness flags (important with median-imputation) ----
    df = add_missing_flags(
        df,
        cols=[
            # core
            "price",
            "condition_score",
            "battery_pct_effective",
            "seller_rating",
            "review_count",
            "member_since_year",

            # anchors / speed
            "ptv_anchor_strict_t0",
            "ptv_anchor_smart",
            "anchor_price_smart",
            "anchor_tts_median_h",
            "anchor_n_support",
            "anchor_level_k",
            "speed_fast7_anchor",
            "speed_fast24_anchor",
            "speed_slow21_anchor",
            "speed_median_hours_ptv",
            "speed_n_eff_ptv",

            # region feature
            "price_z_in_region",

            # image/unified vision features (present in your merged unified table)
            "image_count",
            "caption_count",
            "battery_pct_img",
            "battery_img_minus_text",
            # damage fusion v2 (optional)
            "fusion_img_damage_score",
            "fusion_img_damage_source_level",
            "fusion_batt_fused",
            "fusion_text_sev_corr",
            "fusion_damage_fused_tier_ord",
            "fusion_batt_band_ord",
        ],
    )




    # Build eval/train slices (sold-only eval)
    sold_df  = df[(df["sold_event"] == True) & df["sold_date"].notna()].copy()
    max_sold = sold_df["sold_date"].max()
    if pd.isna(max_sold):
        log("no sold_date found; abort")
        raise SystemExit(1)

    eval_cutoff = max_sold - pd.Timedelta(days=int(args.eval_days))
    eval_df = sold_df[sold_df["sold_date"] >= eval_cutoff].copy()
    train_events = sold_df[sold_df["sold_date"] < eval_cutoff].copy()

    # Freeze SOLD-only frames so nothing later can contaminate them
    train_sold_df = train_events.copy()
    eval_sold_df  = eval_df.copy()

    cens_min_h = float(int(args.censored_min_days) * 24)
    cens_df = df[(df["sold_event"] == False) & (df["duration_h"] >= cens_min_h)].copy()

    # --- split sanity (SOLD-only eval) ---
    log(
        f"[split_sold_only] base_rows={len(df)} sold_total={len(sold_df)} "
        f"train_sold={len(train_events)} eval_sold_last_{int(args.eval_days)}d={len(eval_df)} "
        f"censored_ge_{int(args.censored_min_days)}d={len(cens_df)} max_sold={max_sold} eval_cutoff={eval_cutoff}"
    )
    if len(eval_df) > 0:
        assert eval_df["sold_date"].notna().all(), "eval_df contains NULL sold_date"
        assert (eval_df["sold_event"] == True).all(), "eval_df contains non-sold rows"
    if len(train_events) > 0:
        assert train_events["sold_date"].notna().all(), "train_events contains NULL sold_date"
        assert (train_events["sold_event"] == True).all(), "train_events contains non-sold rows"

    # --- inactive-based censoring hygiene (seller intervention) ---
    censored_before = len(cens_df)
    dropped_inactive = 0
    mode = getattr(args, "censored_inactive_filter_mode", "drop_early_inactive")
    min_active_days = float(getattr(args, "censored_inactive_min_active_days", 1.0))
    if censored_before > 0 and mode != "off" and min_active_days > 0:
        try:
            inactive_meta = load_first_inactive_meta_edited_at(conn, cens_df[["generation", "listing_id"]])
            if not inactive_meta.empty:
                inactive_meta["first_inactive_meta_edited_at"] = ensure_datetime_utc(
                    inactive_meta["first_inactive_meta_edited_at"]
                )
                tmp = cens_df.merge(inactive_meta, on=["generation", "listing_id"], how="left")
                t0 = ensure_datetime_utc(tmp["edited_date"])
                t_inact = ensure_datetime_utc(tmp["first_inactive_meta_edited_at"])
                age_days = (t_inact - t0).dt.total_seconds() / 86400.0
                exclude = (
                    t_inact.notna()
                    & age_days.notna()
                    & (age_days >= 0)
                    & (age_days < min_active_days)
                )
                dropped_inactive = int(exclude.sum())
                cens_df = (
                    tmp.loc[~exclude]
                    .drop(columns=["first_inactive_meta_edited_at"], errors="ignore")
                    .copy()
                )
        except Exception as e:
            # Fail-safe: do not crash production training if the inactive events query fails.
            log(
                f"[inactive_censor] WARN: failed to apply inactive censor hygiene (continuing without filtering): {e}"
            )
            dropped_inactive = 0
            cens_df = cens_df.copy()

    log(
        f"[inactive_censor] censored_before={censored_before} censored_after={len(cens_df)} dropped={dropped_inactive} "
        f"min_active_days={min_active_days} mode={mode}"
    )



    sbert_proto_ctx = None

    if args.use_sbert_vec and getattr(args, "sbert_use_prototypes", False):
        dim  = int(args.sbert_dim)
        pref = (args.sbert_prefix.strip() if args.sbert_prefix else f"sbert{dim}")
        emb_cols = _get_emb_cols(pref, dim)

        protos = fit_sbert_prototypes(
            train_events,
            emb_cols,
            min_n=int(getattr(args, "sbert_proto_min_n", 50)),
            present_col=f"{pref}_present",
        )

        log(f"[sbert] protos fitted: {', '.join(sorted(protos.keys())) if protos else '(none)'}")

        outp = f"{pref}p"
        train_events = apply_sbert_prototypes(train_events, emb_cols, protos, out_prefix=outp)
        eval_df      = apply_sbert_prototypes(eval_df,      emb_cols, protos, out_prefix=outp)
        cens_df      = apply_sbert_prototypes(cens_df,      emb_cols, protos, out_prefix=outp)

        if getattr(args, "sbert_drop_raw", False):
            for dfx in (train_events, eval_df, cens_df):
                dfx.drop(columns=[c for c in emb_cols if c in dfx.columns], inplace=True, errors="ignore")
            log("[sbert] dropped raw embedding dims after deriving prototype features (kept *_present)")

        sbert_proto_ctx = {"protos": protos, "emb_cols": emb_cols, "outp": outp, "pref": pref}


    # WOE / log-odds additive anchor for SLOW21 (train-only; scored for all rows)
    try:
        woe_fold_ctxs: List[Dict[str, Any]] = []
        woe_oof_scores_df: Optional[pd.DataFrame] = None

        # (1) Final mapping learned on TRAIN SOLD; scores for all rows
        p_woe_all, logit_woe_all, woe_ctx = compute_woe_anchor_p_slow21(
            train_events,
            df,
            half_life_days=float(args.half_life_days),
            eps=float(args.woe_eps),
            band_schema_version=int(args.woe_band_schema_version),
        )
        df["woe_anchor_p_slow21"] = p_woe_all.astype("float32")
        df["woe_anchor_logit_slow21"] = logit_woe_all.astype("float32")

        # (2) Strict leak-control: replace TRAIN SOLD rows with OUT-OF-FOLD scores
        if (not args.disable_woe_oof) and int(args.woe_folds) > 1 and len(train_events) > 0:
            p_oof, logit_oof, fold_ids, woe_fold_ctxs = compute_woe_anchor_oof(
                train_events,
                n_folds=int(args.woe_folds),
                half_life_days=float(args.half_life_days),
                eps=float(args.woe_eps),
                dsold_thresholds=woe_ctx.get("dsold_thresholds"),
                presentation_cuts=woe_ctx.get("presentation_cuts"),
                weight_ref_ts=woe_ctx.get("weight_ref_ts"),
                band_schema_version=int(args.woe_band_schema_version),
            )
            df.loc[train_events.index, "woe_anchor_p_slow21"] = p_oof.astype("float32")
            df.loc[train_events.index, "woe_anchor_logit_slow21"] = logit_oof.astype("float32")

            woe_oof_scores_df = pd.DataFrame(
                {
                    "generation": df.loc[train_events.index, "generation"].astype(int).values,
                    "listing_id": df.loc[train_events.index, "listing_id"].astype(np.int64).values,
                    "t0": df.loc[train_events.index, "edited_date"].values,
                    "fold_id": fold_ids.astype(int).values,
                    "is_oof": True,
                    "woe_logit": df.loc[train_events.index, "woe_anchor_logit_slow21"].astype(float).values,
                    "woe_p": df.loc[train_events.index, "woe_anchor_p_slow21"].astype(float).values,
                }
            )
            log(f"[woe] OOF scoring applied to TRAIN SOLD: folds={int(args.woe_folds)} rows={len(train_events)}")
        else:
            log(f"[woe] OOF scoring skipped: disable_woe_oof={args.disable_woe_oof} woe_folds={int(args.woe_folds)} train_rows={len(train_events)}")

        # (3) Propagate into the split copies (indices are preserved from df)
        for _d in (sold_df, train_events, eval_df, cens_df):
            _d["woe_anchor_p_slow21"] = df.loc[_d.index, "woe_anchor_p_slow21"].values
            _d["woe_anchor_logit_slow21"] = df.loc[_d.index, "woe_anchor_logit_slow21"].values

        _br = float(woe_ctx.get("base_rate", float("nan")))
        _thr = woe_ctx.get("dsold_thresholds")
        _cuts = woe_ctx.get("presentation_cuts")
        log(f"[woe] trained slow21 WOE anchor on TRAIN SOLD n={len(train_events)} base_rate={_br:.4f} dsold_thr={_thr} pres_cuts={_cuts}")

        # (4) Persist artifacts for SQL-native scoring
        if (not args.no_db_writes) and (not args.disable_woe_persist):
            code_sha256 = sha256_file(str(pathlib.Path(__file__).resolve()))
            persist_woe_anchor_artifacts_v1(
                conn,
                model_key=model_key,
                train_cutoff_ts=eval_cutoff,
                half_life_days=float(args.half_life_days),
                eps=float(args.woe_eps),
                n_folds=int(args.woe_folds) if (not args.disable_woe_oof) else 1,
                band_schema_version=int(args.woe_band_schema_version),
                code_sha256=code_sha256,
                ctx_final=woe_ctx,
                fold_ctxs=woe_fold_ctxs,
                oof_scores=woe_oof_scores_df,
            )
            log(f"[woe] persisted WOE anchor artifacts to Postgres (model_key={model_key})")
        else:
            log(f"[woe] DB persistence skipped: no_db_writes={args.no_db_writes} disable_woe_persist={args.disable_woe_persist}")
    except Exception as e:
        # If the WOE block (especially DB persistence) errors, psycopg leaves the
        # connection in an aborted transaction state. Roll back so later writes
        # (eval predictions, model registry) don't fail with InFailedSqlTransaction.
        try:
            if conn is not None:
                conn.rollback()
        except Exception:
            pass
        log(f"[woe] WARN: WOE anchor computation failed; continuing without it: {e}")
        for _d in (df, sold_df, train_events, eval_df, cens_df):
            _d["woe_anchor_p_slow21"] = np.nan
            _d["woe_anchor_logit_slow21"] = np.nan
        if (not getattr(args, "disable_woe_persist", False)) and (not getattr(args, "no_db_writes", False)):
            raise RuntimeError("WOE anchor failed and persistence is enabled; aborting run.") from e

    # ---------------------------------------------------------------------
    # GEO FEATURES (super_metro_v4) — categorical + priors
    # ---------------------------------------------------------------------
    geo_mode = (args.geo_mode or "off").lower()
    if geo_mode != "off":
        geo_cols = _parse_csv_list(args.geo_cols)
        geo_prior_cols = _parse_csv_list(args.geo_prior_cols)
        geo_prior_k = float(args.geo_prior_k) if args.geo_prior_k is not None else float(args.region_shrink_k)

        # One-hot geo categoricals (stable template learned from TRAIN sold events only)
        if geo_mode in ("ohe", "both") and geo_cols:
            try:
                ohe_template = fit_geo_ohe_template(train_events, geo_cols, min_n=int(args.geo_ohe_min_n))
                if ohe_template:
                    sold_df = apply_geo_ohe(sold_df, ohe_template)
                    train_events = apply_geo_ohe(train_events, ohe_template)
                    eval_df = apply_geo_ohe(eval_df, ohe_template)
                    cens_df = apply_geo_ohe(cens_df, ohe_template)

                    # Persist template for reproducibility
                    with open(os.path.join(args.outdir, "geo_ohe_template.json"), "w", encoding="utf-8") as f:
                        json.dump(ohe_template, f, indent=2, sort_keys=True)

                    n_ohe_cols = sum(len(v) for v in ohe_template.values())
                    log(f"[geo] one-hot enabled cols={list(ohe_template.keys())} total_levels={n_ohe_cols}")
                else:
                    log("[geo] one-hot requested but no geo columns found in training frame")
            except Exception as e:
                log(f"[geo] one-hot failed (continuing without ohe): {e}")

        # Smoothed numeric priors (trained on TRAIN sold events only; applied to all splits)
        if geo_mode in ("priors", "both") and geo_prior_cols:
            try:
                priors = fit_geo_prior_maps(
                    train_events,
                    geo_prior_cols,
                    shrink_k=geo_prior_k,
                    half_life_days=float(args.half_life_days),
                )
                if priors:
                    sold_df = apply_geo_priors(sold_df, priors)
                    train_events = apply_geo_priors(train_events, priors)
                    eval_df = apply_geo_priors(eval_df, priors)
                    cens_df = apply_geo_priors(cens_df, priors)

                    # Persist tables for reproducibility / debugging
                    for _col, _ctx in priors.items():
                        tbl = _ctx["table"].copy()
                        tbl.to_csv(os.path.join(args.outdir, f"geo_priors__{_col}.csv"))

                    log(f"[geo] priors enabled cols={list(priors.keys())} shrink_k={geo_prior_k}")
                else:
                    log("[geo] priors requested but no geo columns found in training frame")
            except Exception as e:
                log(f"[geo] priors failed (continuing without priors): {e}")


    # Defensive: enforce SOLD-only invariants again AFTER feature engineering.
    # Some downstream feature-attach steps may introduce NA/duplicate index alignment issues; we fail-closed to SOLD-only.
    def _enforce_sold_only(_d: pd.DataFrame, _name: str) -> pd.DataFrame:
        if _d is None or _d.empty:
            return _d
        if ('sold_event' not in _d.columns) or ('sold_date' not in _d.columns):
            return _d
        _m = (_d['sold_event'] == True) & _d['sold_date'].notna()
        if not bool(_m.all()):
            _before = len(_d)
            _after = int(_m.sum())
            log(f"[split_sold_only_guard] {_name}: dropping {_before - _after} non-sold/NULL-sold_date rows to enforce SOLD-only")
            _d = _d.loc[_m].copy()
        return _d
    sold_df = _enforce_sold_only(sold_df, 'sold_df')
    train_events = _enforce_sold_only(train_events, 'train_events')
    eval_df = _enforce_sold_only(eval_df, 'eval_df')

    # Re-freeze SOLD-only frames AFTER WOE/GEO feature engineering so they include new columns
    # (The earlier freeze happens before geo_ohe/geo_priors/woe columns are added.)
    train_sold_df = train_events.copy()
    eval_sold_df  = eval_df.copy()

    # Encode numeric features
    med_src_df = train_events if not train_events.empty else df
    X_ev, keep_ev = encode_numeric(train_events, median_from=med_src_df)
    if not cens_df.empty:
        X_ce, keep_ce = encode_numeric(cens_df, median_from=med_src_df)
    else:
        X_ce, keep_ce = pd.DataFrame(index=cens_df.index), []
    feat_cols = sorted(set(keep_ev).intersection(keep_ce)) if not cens_df.empty else sorted(keep_ev)

    # Drop degenerate
    deg = [c for c in feat_cols
           if (c in X_ev.columns and np.nanstd(pd.to_numeric(X_ev[c], errors="coerce").values.astype(float)) == 0.0)]
    for c in deg:
        feat_cols.remove(c)

    # Final matrices
    med_vals = med_src_df[feat_cols].median(numeric_only=True)

    # Save feature column order and training medians for inference
    feat_cols_path = outdir / "feature_cols.json"
    medians_path   = outdir / "feature_medians.json"

    with open(feat_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(feat_cols), f)

    with open(medians_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: (None if pd.isna(v) else float(v)) for k, v in med_vals.items()},
            f,
        )

    log(f"saved feature_cols to {feat_cols_path}")
    log(f"saved feature medians to {medians_path}")

    Xe = train_sold_df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(med_vals).values.astype(np.float32)
    ye = train_sold_df["duration_h"].values.astype(np.float32)


    if not cens_df.empty:
        Xc = cens_df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(med_vals).values.astype(np.float32)
        yc = cens_df["duration_h"].values.astype(np.float32)
        ev_dates_cens = cens_df["edited_date"]
    else:
        Xc = np.empty((0, len(feat_cols)), dtype=np.float32)
        yc = np.empty((0,), dtype=np.float32)
        ev_dates_cens = None

    X_eval = eval_sold_df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(med_vals).values.astype(np.float32)
    y_eval = eval_sold_df["duration_h"].values.astype(np.float32)


    # Calibration slice (recent SOLD before eval window) — aka "sval"
    #
    # Default (backwards compatible): cal_from_days = max(eval_days + 14, 21)
    # Override (CLI): --sval_days / --cal_days sets the *length* of the slice immediately before eval_cutoff.
    if getattr(args, "sval_days", None) is not None and int(args.sval_days) > 0:
        sval_days = int(args.sval_days)
        cal_lower = eval_cutoff - pd.Timedelta(days=sval_days)
        log(f"[sval] using CLI sval_days={sval_days} (cal_lower={cal_lower}, eval_cutoff={eval_cutoff})")
    else:
        cal_from_days = max(int(args.eval_days) + 14, 21)
        cal_lower = max_sold - pd.Timedelta(days=int(cal_from_days))
        sval_days = int((eval_cutoff - cal_lower) / pd.Timedelta(days=1))
        log(f"[sval] default sval_days={sval_days} via cal_from_days={cal_from_days} (cal_lower={cal_lower}, eval_cutoff={eval_cutoff})")

    cal_df = sold_df[(sold_df["sold_date"] >= cal_lower) & (sold_df["sold_date"] < eval_cutoff)].copy()

    if sbert_proto_ctx and sbert_proto_ctx["protos"]:
        cal_df = apply_sbert_prototypes(
            cal_df,
            sbert_proto_ctx["emb_cols"],
            sbert_proto_ctx["protos"],
            out_prefix=sbert_proto_ctx["outp"],
        )
        if getattr(args, "sbert_drop_raw", False):
            cal_df.drop(
                columns=[c for c in sbert_proto_ctx["emb_cols"] if c in cal_df.columns],
                inplace=True,
                errors="ignore",
            )

    X_cal = cal_df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(med_vals).values.astype(np.float32)
    y_cal = cal_df["duration_h"].values.astype(np.float32)

    # --- sanity: enforce SOLD-only eval for AFT objective ---
    # At this point, train_events MUST be SOLD-only rows (sold_date < eval_cutoff)
    # and eval_df MUST be SOLD-only rows (sold_date >= eval_cutoff).
    # SOLD-only invariant: eval must be sold-only (training set may include censored)
    if "sold_event" in eval_df.columns:
        assert bool((eval_df["sold_event"] == True).all()), "EVAL contains non-sold rows; eval must be SOLD-only"

    assert (eval_df.get('sold_event') is None) or bool((eval_df['sold_event'] == True).all())
    assert train_events['sold_date'].notna().all() and eval_df['sold_date'].notna().all()
    assert len(train_events) == len(sold_df[sold_df['sold_date'] < eval_cutoff])
    assert len(eval_df) == len(sold_df[sold_df['sold_date'] >= eval_cutoff])
    # censored rows are controlled ONLY by --censored_min_days (+ optional inactive filter)
    if not cens_df.empty:
        assert bool((cens_df['sold_event'] == False).all())
        assert bool((cens_df['duration_h'] >= cens_min_h).all())
    log(f"[aft_data_sold_eval] train_sold={len(train_events)} censored_ge_{int(args.censored_min_days)}d={len(cens_df)} eval_sold_last_{args.eval_days}d={len(eval_df)}")

    if len(y_eval) == 0 or len(ye) == 0:
        log("insufficient rows for training/eval; abort")
        raise SystemExit(1)

    # ---------------- Optuna tuning (optional) ----------------
    if args.optuna_trials and args.optuna_trials > 0:
        log(f"starting Optuna hyperparameter tuning with {args.optuna_trials} trials…")
        study = optuna.create_study(direction="maximize")
        data_bundle = {
            "X_train": Xe,
            "y_train": ye,
            "event_dates_train": train_sold_df["edited_date"],
            "X_cens": Xc,
            "y_cens": yc,
            "event_dates_cens": ev_dates_cens,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_eval": X_eval,
            "y_eval": y_eval,
            "feat_cols": feat_cols,
        }
        import xgboost
        study.optimize(
            lambda trial: optuna_objective(trial, args=args, data_bundle=data_bundle),
            n_trials=args.optuna_trials,
            catch=(xgboost.core.XGBoostError, MemoryError),
        )
        log(f"Optuna best slow21 score (SLOW F1 @504h with constraints) = {study.best_value:.4f}")
        log(f"Optuna best params = {json.dumps(study.best_params, indent=2)}")


        # Update args with best params for final training run
        for k, v in study.best_params.items():
            setattr(args, k, v)

        # Save Optuna best params
        with open(outdir / "optuna_best_params.json", "w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)

    # Train AFT (XGB) + predict eval with final args (possibly tuned) — still GPU
    pred_eval, iso, xgb_params, fi_share, booster = xgb_aft_train_predict(
        Xe, ye, Xc, yc, X_cal, y_cal, X_eval,
        ev_dates_events=train_events["edited_date"],
        ev_dates_cens=ev_dates_cens,
        feat_cols=feat_cols,
        args=args
    )

    # Save booster model for inference
    model_path = outdir / "model_xgb.json"
    booster.save_model(model_path.as_posix())
    log(f"saved booster to {model_path}")

    # Persist calibration artifact for train/serve parity
    if iso is not None:
        cal_path = outdir / "calibrator_isotonic_time.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(iso, f)
        log(f"saved calibrator to {cal_path}")

    # ---------------- 21d (504h) classifier focus diagnostics ----------------
    # These logs are non-breaking and are used both for Optuna auditing and for
    # day-to-day visibility into the 21d classifier behavior.
    slow21_eval_focus = slow21_focus_report(y_eval, pred_eval, slow_thresh_hours=SLOW21_H, lt10_thresh_hours=LT10_H)
    slow_stats_eval = slow21_eval_focus.get("slow", {})
    sac_eval = slow21_eval_focus.get("sacrifice", {})
    conf_eval = slow21_eval_focus.get("confusion_slow", {})
    log(
        "EVAL (SOLD last N days) — 21d classifier @504h — "
        f"SLOW prec={float(slow_stats_eval.get('precision', 0.0) or 0.0):.4f} "
        f"rec={float(slow_stats_eval.get('recall', 0.0) or 0.0):.4f} "
        f"f1={float(slow_stats_eval.get('f1', 0.0) or 0.0):.4f} | "
        f"SAC_LT10={float(sac_eval.get('sac_lt10', 0.0) or 0.0):.4f} "
        f"SAC_MID={float(sac_eval.get('sac_mid', 0.0) or 0.0):.4f} | "
        f"conf_slow(tp={int(conf_eval.get('tp_slow', 0))}, "
        f"fp={int(conf_eval.get('fp_slow', 0))}, "
        f"fn={int(conf_eval.get('fn_slow', 0))}, "
        f"tn={int(conf_eval.get('tn_slow', 0))})"
    )

    slow21_cal_focus = None
    if X_cal is not None and len(y_cal) > 0:
        try:
            dcal_pred = xgb.DMatrix(X_cal.astype(np.float32), feature_names=list(feat_cols))
            best_iter = getattr(booster, "best_iteration", None)
            if best_iter is None:
                pred_cal_raw = booster.predict(dcal_pred)
            else:
                pred_cal_raw = booster.predict(dcal_pred, iteration_range=(0, int(best_iter) + 1))

            # Match the same post-processing used elsewhere (finite + clip).
            pred_cal_raw = np.asarray(pred_cal_raw, dtype=float)
            pred_cal_raw = np.where(np.isfinite(pred_cal_raw), pred_cal_raw, 0.0)
            pred_cal_raw = np.clip(pred_cal_raw, 0.0, 24.0 * 365.0 * 5.0)

            pred_cal = iso.predict(pred_cal_raw) if iso is not None else pred_cal_raw

            slow21_cal_focus = slow21_focus_report(y_cal, pred_cal, slow_thresh_hours=SLOW21_H, lt10_thresh_hours=LT10_H)
            slow_stats_cal = slow21_cal_focus.get("slow", {})
            sac_cal = slow21_cal_focus.get("sacrifice", {})
            conf_cal = slow21_cal_focus.get("confusion_slow", {})
            log(
                "CAL (recent SOLD) — 21d classifier @504h — "
                f"SLOW prec={float(slow_stats_cal.get('precision', 0.0) or 0.0):.4f} "
                f"rec={float(slow_stats_cal.get('recall', 0.0) or 0.0):.4f} "
                f"f1={float(slow_stats_cal.get('f1', 0.0) or 0.0):.4f} | "
                f"SAC_LT10={float(sac_cal.get('sac_lt10', 0.0) or 0.0):.4f} "
                f"SAC_MID={float(sac_cal.get('sac_mid', 0.0) or 0.0):.4f} | "
                f"conf_slow(tp={int(conf_cal.get('tp_slow', 0))}, "
                f"fp={int(conf_cal.get('fp_slow', 0))}, "
                f"fn={int(conf_cal.get('fn_slow', 0))}, "
                f"tn={int(conf_cal.get('tn_slow', 0))})"
            )
        except Exception as e:
            log(f"[warn] slow21 CAL diagnostic failed: {e}")


    # ---------------- Persist eval predictions for SOLD rows (backfill) ----------------
    # model_key was computed once at the start of the run (stable across all artifacts)
    if not args.no_db_writes:
        try:
            write_eval_predictions_for_sold(conn, model_key, eval_df, pred_eval)
        except Exception as e:
            log(f"[warn] write_eval_predictions_for_sold failed: {e}")

    # Metrics (overall + buckets)
    mm_overall = mae_med_p90(y_eval, pred_eval)
    buckets = {
        "fast_<=24h":      (y_eval <= 24),
        "mid_24_72h":      ((y_eval > 24) & (y_eval <= 72)),
        "slow_72h_21d":    ((y_eval > 72) & (y_eval <= 21*24)),
        "very_slow_>21d":  (y_eval > 21*24),
    }
    bucket_rows, bucket_report = [], {}
    for name, mask in buckets.items():
        idx = np.where(mask)[0]
        if len(idx):
            m = mae_med_p90(y_eval[idx], pred_eval[idx])
            rm = {"mae": float(m["mae"]), "med": float(m["med"]), "p90": float(m["p90"])}
        else:
            rm = {"mae": float("nan"), "med": float("nan"), "p90": float("nan")}
        bucket_report[name] = {"rows": int(len(idx)), **rm}
        horizon = {"fast_<=24h":24, "mid_24_72h":72,
                   "slow_72h_21d":504, "very_slow_>21d":9999}[name]
        bucket_rows.append((horizon, int(len(idx)),
                            rm["mae"], rm["med"], rm["p90"], name))

    # Classification metrics at multiple thresholds
    raw_fast72 = fast72_metrics(y_eval, pred_eval)         # 7d (168h) raw (fast72/slow72)
    fast7d     = canonicalize_fast72_for_json(raw_fast72)  # canonical schema with 'fast'/'slow'

    fast24  = classify_at_threshold(y_eval, pred_eval, 24.0)             # 24h
    fast72h = classify_at_threshold(y_eval, pred_eval, 72.0)             # 72h
    fast240 = classify_at_threshold(y_eval, pred_eval, 240.0)            # 10d
    fast504 = classify_at_threshold(y_eval, pred_eval, 21.0 * 24.0)      # 21d (504h)


    metrics = {
        "rows_eval_sold_last_days": int(len(y_eval)),
        "overall": {
            "mae_hours": float(mm_overall["mae"]),
            "med_ae_hours": float(mm_overall["med"]),
            "p90_ae_hours": float(mm_overall["p90"]),
        },
        "buckets": bucket_report,

        # 7-day metric, stored under BOTH keys with SAME schema:
        "fast7d_classification": fast7d,
        "fast72_classification": fast7d,

        "fast24_classification":  fast24,
        "fast72h_classification": fast72h,
        "fast240_classification": fast240,
        "fast504_classification": fast504,     # 21d (504h)
        "fast21d_classification": fast504,     # alias for readability


        # Explicitly surface the 21d-focused target + collateral metrics.
        "slow21_focus": {
            "threshold_hours": float(SLOW21_H),
            "fp_lt10_cap": float(FP_LT10_CAP_DEFAULT),
            "cal": slow21_cal_focus,
            "eval": slow21_eval_focus,
        },
        "eval_days": int(args.eval_days),
        "sval_days": int(sval_days) if 'sval_days' in locals() else None,
        "censored_min_days": int(args.censored_min_days),
        "used_aft": True,
        "aft_impl": "xgboost",
    }



    # ---------------- Feature importance (top 50) ----------------
    if fi_share:
        top_fi = sorted(fi_share.items(), key=lambda kv: kv[1], reverse=True)[:50]
        log("Top 30 features by gain importance (% of total):")
        for fname, pct in top_fi:
            log(f"  {fname}: {pct:.3f}%")
    else:
        log("No feature importance data available from booster.")


    

    # ---------------- Persist feature importance (gain + TreeSHAP) ----------------
    # Console logging above is not sufficient for downstream SQL; we persist:
    #   - gain_pct (from fi_share)
    #   - shap_mean_abs (TreeSHAP on eval SOLD slice, capped for speed)
    if (conn is not None) and (not args.no_db_writes):
        scope_eval = f"eval_sold_last_{int(args.eval_days)}d"
        try:
            # Gain-based share (already computed during training)
            write_model_feature_importance_v1(
                conn,
                model_key=model_key,
                feat_cols=feat_cols,
                values=fi_share or {},
                scope=scope_eval,
                importance_type="gain_pct",
                n_rows=int(X_eval.shape[0]) if X_eval is not None else 0,
                meta_common={"source": "xgb_gain_share"},
            )
        except Exception as e:
            log(f"[warn] gain importance persistence failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

        try:
            shap_abs, shap_mean, shap_n = compute_tree_shap_importance(
                booster,
                X_eval,
                feat_cols,
                max_rows=min(int(X_eval.shape[0]), 2000) if X_eval is not None else 0,
                seed=int(args.seed),
            )
            if shap_n > 0:
                write_model_feature_importance_v1(
                    conn,
                    model_key=model_key,
                    feat_cols=feat_cols,
                    values=shap_abs,
                    scope=scope_eval,
                    importance_type="shap_mean_abs",
                    n_rows=shap_n,
                    meta_common={"source": "xgb_pred_contribs", "max_rows": int(shap_n)},
                )
                # Optional: signed mean contribution (can be useful for sanity checks)
                write_model_feature_importance_v1(
                    conn,
                    model_key=model_key,
                    feat_cols=feat_cols,
                    values=shap_mean,
                    scope=scope_eval,
                    importance_type="shap_mean",
                    n_rows=shap_n,
                    meta_common={"source": "xgb_pred_contribs", "max_rows": int(shap_n)},
                )
        except Exception as e:
            log(f"[warn] SHAP persistence (eval) failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

# ---------------- Per-generation metrics ----------------
    per_gen_metrics: Dict[str, Any] = {}
    per_gen_rows_for_db: List[Tuple[int, int, float, float, float, str, str]] = []

    if "generation" in eval_df.columns:
        gens_series = pd.to_numeric(eval_df["generation"], errors="coerce")
        unique_gens = [g for g in sorted(gens_series.dropna().unique())]
        for g in unique_gens:
            mask = (gens_series.values == g)
            y_g = y_eval[mask]
            p_g = pred_eval[mask]
            mm_g = mae_med_p90(y_g, p_g)
            buckets_g = {
                "fast_<=24h":      (y_g <= 24),
                "mid_24_72h":      ((y_g > 24) & (y_g <= 72)),
                "slow_72h_21d":    ((y_g > 72) & (y_g <= 21*24)),
                "very_slow_>21d":  (y_g > 21*24),
            }
            bucket_report_g = {}
            gen_bucket_rows = []
            for name, msk in buckets_g.items():
                idx = np.where(msk)[0]
                if len(idx):
                    m = mae_med_p90(y_g[idx], p_g[idx])
                    rm = {"mae": float(m["mae"]), "med": float(m["med"]), "p90": float(m["p90"])}
                else:
                    rm = {"mae": float("nan"), "med": float("nan"), "p90": float("nan")}
                bucket_report_g[name] = {"rows": int(len(idx)), **rm}
                horizon = {"fast_<=24h":24, "mid_24_72h":72,
                           "slow_72h_21d":504, "very_slow_>21d":9999}[name]
                gen_bucket_rows.append((horizon, int(len(idx)),
                                        rm["mae"], rm["med"], rm["p90"], name))

            raw_fast72_g = fast72_metrics(y_g, p_g)
            fast7d_g     = canonicalize_fast72_for_json(raw_fast72_g)

            g_key = str(int(g)) if float(g).is_integer() else str(g)
            per_gen_metrics[g_key] = {
                "rows_eval_sold_last_days": int(len(y_g)),
                "overall": {
                    "mae_hours": float(mm_g["mae"]),
                    "med_ae_hours": float(mm_g["med"]),
                    "p90_ae_hours": float(mm_g["p90"]),
                },
                "buckets": bucket_report_g,
                "fast72_classification": fast7d_g,
            }


            cohort_key = f"generation={g_key}"
            cohort_json = json.dumps({"generation": g_key})
            per_gen_rows_for_db.append(
                (0, int(len(y_g)), float(mm_g["mae"]), float(mm_g["med"]),
                 float(mm_g["p90"]), cohort_key, cohort_json)
            )
            for (horizon, n_g, mae_g, med_g, p90_g, bname) in gen_bucket_rows:
                per_gen_rows_for_db.append(
                    (horizon, int(n_g), float(mae_g), float(med_g), float(p90_g),
                     f"generation={g_key};bucket={bname}",
                     json.dumps({"generation": g_key, "bucket": bname}))
                )

    if per_gen_metrics:
        metrics["per_generation"] = per_gen_metrics

    # Registry + MAE summary
    if not args.no_db_writes:
        train_start = pd.to_datetime(df["edited_date"]).min().to_pydatetime() if "edited_date" in df.columns else None
        train_end   = pd.to_datetime(df["edited_date"]).max().to_pydatetime() if "edited_date" in df.columns else None

        mae_rows: List[Tuple] = [
            (model_key, 0, metrics["rows_eval_sold_last_days"],
             metrics["overall"]["mae_hours"],
             metrics["overall"]["med_ae_hours"],
             metrics["overall"]["p90_ae_hours"],
             None, None, None, "GLOBAL", json.dumps({"scope":"GLOBAL"})),
        ]
        for horizon, n, mae, med, p90, bucket_name in bucket_rows:
            mae_rows.append(
                (model_key, horizon, n, mae, med, p90,
                 None, None, None,
                 f"bucket={bucket_name}", json.dumps({"bucket": bucket_name}))
            )
        for (horizon, n_g, mae_g, med_g, p90_g, cohort_key, cohort_json) in per_gen_rows_for_db:
            mae_rows.append(
                (model_key, horizon, n_g, mae_g, med_g, p90_g,
                 None, None, None, cohort_key, cohort_json)
            )

        write_model_registry(
            conn, model_key, args.feature_set, feat_hash,
            algo="xgboost-survival-aft",
            hyperparams={
                "aft_dist":args.aft_dist, "aft_scale":args.aft_scale,
                "n_estimators":args.n_estimators, "max_leaves":int(max(8, args.num_leaves)),
                "min_child_weight": max(1.0, float(args.min_data_in_leaf)),
                "colsample_bytree": float(args.feature_fraction),
                "subsample": float(args.bagging_fraction),
                "reg_lambda": float(args.lambda_l2),
                "learning_rate":float(args.learning_rate),
                "max_bin": int(args.max_bin),
                "seed": int(args.seed),
                "monotone_constraints": build_monotone_constraints(feat_cols),
            },
            metrics=metrics,
            train_start=train_start, train_end=train_end,
            artifact_uri=str(pathlib.Path(outdir).resolve()),
            created_by="pipeline",
            notes="aft_xgb_strict_anchor_t0 + advanced_anchor_v1_in_trainer + eval_sold_logging + optuna_multi_metric",
        )

        write_mae_summary(conn, model_key, mae_rows)

    with open(outdir/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_json_numbers(metrics), f, indent=2)
    with open(outdir/"params.json",  "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_features": len(feat_cols),
                "used_aft": True,
                "aft_impl": "xgboost",
            },
            f,
            indent=2,
        )

    log(f"registered model_key={model_key}")
    log("done.")

if __name__ == "__main__":
    main() 
