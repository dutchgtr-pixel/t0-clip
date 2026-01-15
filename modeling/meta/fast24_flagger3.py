#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
fast24_flagger3_public.py — end-to-end pipeline with JOINT Stage-1 ⨉ Stage-2 tuning

Goal:
- For every Stage-1 Optuna trial, run a LEAK-FREE Stage-2 (mask-23) inline (silent), compute flips+regression
  metrics (baseline/tuned MAE, RMSE, R², improvements), PRINT a concise summary, and LOG per-trial rows to
  CSV + JSONL. Select the best trial by a JOINT score combining (TP−FP), flips quality, and Δmetrics.
- Optionally run a larger outer Stage-2 pass at the end (when --tune_mask23 &gt; 0) to write files &amp; a JSON summary.
"""

import argparse, json, os, warnings, csv
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
# add near other imports
try:
    import lightgbm as lgb
except Exception:
    lgb = None  # Stage-2 ranker gracefully falls back to prob-only if LightGBM is unavailable

from bisect import bisect_left, insort

# silence warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
except Exception:
    pass

# Optional Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner
except Exception:
    optuna = None

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ───────────────────────── defaults (wired to your env)
# NOTE: For public release, defaults are portable and do not include developer-specific absolute paths.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", os.path.join(SCRIPT_DIR, "data"))
DEFAULT_TRAIN   = os.getenv("TRAIN_CSV",   os.path.join(DEFAULT_DATA_DIR, "train.csv"))
DEFAULT_SLOW    = os.getenv("SLOW_CSV",    os.path.join(DEFAULT_DATA_DIR, "val_slow.csv"))
DEFAULT_VALTEXT = os.getenv("VALTEXT_CSV", os.path.join(DEFAULT_DATA_DIR, "val_text.csv"))
DEFAULT_OOF     = os.getenv("OOF_CSV",     os.path.join(DEFAULT_DATA_DIR, "oof_predictions.csv"))
DEFAULT_OUT_DIR = os.getenv("OUT_DIR",     os.path.join(DEFAULT_DATA_DIR, "out"))

# ───────────────────────── model knobs
N_SPLITS     = 7
CALIBRATION  = "isotonic"     # isotonic | platt | none
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"

# ───────────────────────── threshold knobs
PREC_MIN        = 0.75
REC_MIN         = 0.45
MAX_SHARE       = 0.50
PER_MODEL_MIN_N = 30
TOPK_FALLBACK   = 25

# ───────────────────────── DAMAGE FILTER
DAMAGE_KEEP_LEVELS = None  # e.g., {0,1}


# Battery corridor (train+val)
BATT_GATE_MIN = None     # e.g., 84.0
BATT_GATE_MAX = None     # e.g., 101.0   (exclusive upper bound)
BATT_KEEP_NA  = False    # if True, keep rows with missing battery_pct




# Optional global fixed prob cutoff (used when --fixed_prob_threshold is set)
FIXED_PROB_THRESHOLD = None
# ───────────────────────── soft gate knobs (probability penalty; NOT hard drop)
G_SOFT_ENABLE        = True
G_MIN_COND           = 0.7
G_MAX_DAMAGE         = 1.0
G_MIN_BATT           = 0.0
G_PTV_OVERALL_MAX    = 1.07
G_DELTA_OVERALL_MAX  = -350.0
G_PTV_ANCHOR_MAX     = 0.98
G_MIN_STORAGE_GB     = 128
G_MAX_PPG_RATIO      = 50  # NOK/GB

# text/seller knobs
G_WARRANTY_BOOST     = 1.20
G_RECEIPT_BOOST      = -1.00
G_OPLOCK_PENALTY     = 1.00
G_NEGOTIATION_BOOST  = -2.00
G_FASTPRIS_PENALTY   = -1.0
G_ACC_MAX_BOOST      = -1.00
G_MIN_SELLER_RATING  = 10.00
G_MIN_REVIEWS        = 200
G_MIN_ACCOUNT_AGE_Y  = 10


# Seller corridor (hard gates; None = disabled)
SELLER_GATE_MIN_RATING   = None    # e.g., 12  (use your rating scale; you used 10..30 style in logs)
SELLER_GATE_MIN_REVIEWS  = None    # e.g., 50
SELLER_GATE_MIN_AGE_Y    = None    # e.g., 3 (years on platform)
SELLER_GATE_MIN_QUALITY  = None    # combined metric gate; see compute_seller_quality()


# per-variant penalties (1.0 = neutral; &lt;= 0.0 = hard block)
G_MODEL_PENALTIES = {
    "iPhone 13 Mini": 0.00,
    "iPhone 13": 1.00,
    "iPhone 13 Pro": 1.00,
    "iPhone 13 Pro Max": 1.00,
}

G_PENALTY_MIN        = 0.90
G_TOTAL_PENALTY_MIN  = 0.15
G_TOTAL_PENALTY_MAX  = 1.00

# ───────────────────────── Stage-1 HARD FILTER exports (reporting only)
HARD_FILTER_ENABLE          = True
HF_MIN_PROB_FAST24          = 0.134
HF_DROP_DAMAGE_GTE          = 2.0
HF_DROP_COND_MIN            = 0.2
HF_DROP_COND_EQ0_TOO        = True
HF_OUT_FULL_FILENAME        = "slow_bucket_pred_fast24_flagged_full_hardfiltered.csv"
HF_OUT_MIN_FILENAME         = "slow_bucket_pred_fast24_flagged_min_hardfiltered.csv"
STAGE1_HF_EXPORT            = 0  # 0=off (no log/export), 1=on

# ───────────────────────── embeddings
EMB_CFGS = {
    "32":        [("embed32_",  32)],
    "64":        [("embed64_",  64)],
    "64w":       [("embed64w_", 64)],
    "32+64":     [("embed32_",  32), ("embed64_",  64)],
    "32+64w":    [("embed32_",  32), ("embed64w_", 64)],
    "32+64+64w": [("embed32_",  32), ("embed64_",  64), ("embed64w_", 64)],
    "all":       [("embed32_",  32), ("embed64_",  64), ("embed64w_", 64)],
}
EMB_DIMS_DEFAULT = 32

# ───────────────────────── text SVD knobs
SVD_ENABLE = True
SVD_DIMS = 400
SVD_TFIDF_MAX_FEATURES = 120000


# AFTER
EXPECTED_SLOW_COUNT    = None

# ───────────────────────── JOINT objective weights (Stage-1 ⨉ Stage-2)
JOINT_W_TPFP = 1.0
JOINT_W_FAST = 1.5
JOINT_W_SLOW = 4.0
JOINT_W_MID  = 2.0
JOINT_W_MAE  = 0.10
JOINT_W_RMSE = 0.02
JOINT_W_R2   = 80.0


# How many Stage-2 trials to run **inside each Stage-1 Optuna trial** (set via CLI)
TUNE_MASK23_PER_TRIAL = 0


def build_stage2_features(df, model_pred_col: str, include_emb: bool = False):
    """Build compact Stage-2 features out of val_out (already merged)."""
    use_cols = [
        # risk / anchors
        "prob_fast24","overall_median_price","delta_from_overall_price",
        "ptv_overall","ptv_anchor","delta_anchor_price",
        "anchor_time14_model_row","delta_anchor_time14_minus_pred","delta_pred_model_med14",
        # condition triad
        "condition_score","battery_pct","damage_severity",
        # precomputed utilities
        "_delta_overall_norm","_underprice_quality","_stor_fac",
        # seller/docs/text flags
        "seller_rating","review_count","ratings_count","member_since_year","member_year",
        "warranty_flag_text","receipt_flag_text","operator_lock_flag_text",
        "negotiation_flag_text","fast_pris_flag_text","accessory_count_text",
        # pricing per GB
        "price_per_gb","ppg_ratio","price_pe_rgb",
    ]
    if include_emb:
        use_cols += [c for c in df.columns if c.lower().startswith(("embed32_","embed64_","embed64w_"))]
    X = __import__('pandas').DataFrame(index=df.index)
    def safe_num_local(s):
        import pandas as pd
        return pd.to_numeric(s, errors="coerce")
    for c in use_cols:
        if c in df.columns:
            X[c] = safe_num_local(df[c])
        else:
            X[c] = 0.0
    # distance-to-23 target (monotone penalty)
    if model_pred_col in df.columns:
        delta23 = (safe_num_local(df[model_pred_col]) - 23.0).clip(lower=0.0)
    else:
        pcol = preferred_model_pred_col(df) or "pred_hours"
        delta23 = (safe_num_local(df.get(pcol, 23.0)) - 23.0).clip(lower=0.0)
    X["_delta23"] = delta23
    # synthesize _underprice_quality if absent
    if "_underprice_quality" not in X.columns:
    # (removed inner 'import numpy as np' – using global np)
        cond = X.get("condition_score").clip(0,1).fillna(0.0) if "condition_score" in X else 0.0
        dmg  = X.get("damage_severity").clip(0,3).fillna(3.0) if "damage_severity" in X else 3.0
        batt = X.get("battery_pct").clip(0,100).fillna(0.0) if "battery_pct" in X else 0.0
        dmg_fac  = 1.0 - (dmg / 3.0)
        batt_fac = batt / 100.0
        overn = X.get("_delta_overall_norm", 0.0)
        stor  = X.get("_stor_fac", 0.0)
        X["_underprice_quality"] = (-overn.clip(-1,1)) * cond * dmg_fac * batt_fac * (1.0 + stor)
    return coerce_numeric_df(X), list(X.columns)

# ───────────────────────── Stage-2 (mask-23) default knobs (overridable via CLI)
MASK23_TRAIN_FRAC      = 0.40
MASK23_CV_BLOCKS       = 1
MASK23_MIN_FLIPS       = 10
MASK23_MIN_SHARE       = 0.20
MASK23_MAX_SHARE       = 0.75
MASK23_SLOW_CAP_ABS    = 2
MASK23_SLOW_CAP_RATIO  = 0.10

# Stage-2 objective weights

# ───────────────────────── Stage-2 (ranker) lightweight LightGBM controls
S2_USE_LGBM = True  # can be toggled by CLI --s2_use_lgbm
S2_USE_EMB = False  # optional: include embeddings in Stage-2 features (risk of overfit)
S2_LGBM_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.08,
    num_leaves=15,
    max_depth=4,
    min_data_in_leaf=28,
    feature_fraction=0.8,
    bagging_fraction=0.9,
    bagging_freq=1,
    lambda_l2=10.0,
    n_estimators=250,
    verbosity=-1,
    seed=RANDOM_STATE,
)
S2_VFUNC_DEFAULT = {
    "w_fast": 0.40,
    "w_slow_risk": 0.35,
    "w_util": 0.20,
    "w_delta23": 0.05,
    "delta23_max": 48.0,
    "cvar_alpha": 0.90,
    "cvar_ceiling": 0.30,
}

MASK23_W_FAST        = 1.0
MASK23_W_SLOW        = 10.0
MASK23_W_MID         = 2.0
MASK23_W_MAE         = 3.0
MASK23_W_RMSE        = 1.0
MASK23_W_R2          = 3.0
MASK23_W_SHARE_SAT   = 0.25
MASK23_BARRIER_POWER = 2.0

# Bonus for trials that flip zero slow rows (>72h)
SLOW_ZERO_BONUS = 0.0



# Apply-time max allowed predicted slow probability for Stage-2 gating
S2_P_SLOW_MAX = 0.20

# Baseline overlay mode: 'hard' (strict), 'soft' (lenient), or 'off'
OVERLAY_MODE = 'hard'
# ───────────── union-of-zero-slow trials (CLI-controlled; default OFF)
UNION_ZERO_SLOW = False
UNION_ZERO_SLOW_MIN_FAST = 1            # ignore zero-slow trials that flipped < this many fast rows
UNION_ZERO_SLOW_MIN_CONSENSUS = 1       # require row to appear in >= this many zero-slow trials
UNION_ZERO_SLOW_OUTNAME = "joint_zero_slow_union.csv"


# ───────────────────────── misc helpers
VARIANTS = ["iPhone 13 Mini","iPhone 13","iPhone 13 Pro","iPhone 13 Pro Max"]
LEAK_SUBS = ("dur", "duration", "sold", "label", "target", "actual", "ground", "is_24", "fast24", "seen", "fetch", "postal")
BAN_COLS  = ("id", "listing", "url", "description", "title")  # public: exclude identifier/text columns

from typing import Optional, List

def pick(cols, candidates: List[str]) -> Optional[str]:
    s = set(cols)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in s:
            return c
        if c.lower() in low:
            return low[c.lower()]
    for c in cols:
        lc = c.lower()
        for k in candidates:
            if k.lower() in lc:
                return c
    return None


def safe_num(s): return pd.to_numeric(s, errors="coerce")

def compute_seller_quality(df: pd.DataFrame) -> pd.Series:
    """
    seller_quality = seller_rating * log1p(review_count) * account_age_years
    - seller_rating: numeric
    - review_count : numeric; we use log1p() to dampen very large counts
    - account_age  : max(0, 2025 - member_since_year)  (kept consistent with soft_gate_penalty style)
    Returns a float Series aligned to df.index; NaNs -> 0.0
    """
    sr  = safe_num(df.get("seller_rating"))
    rc1 = np.log1p(safe_num(df.get("review_count", df.get("ratings_count"))))
    msy = safe_num(df.get("member_since_year", df.get("member_year")))
    age = 2025.0 - msy
    age = np.where(np.isfinite(age), np.maximum(0.0, age), 0.0)
    q = (sr * rc1 * age)
    q = np.where(np.isfinite(q), q, 0.0)
    return pd.Series(q, index=df.index, dtype=float)


def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df

def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    X = dedup_columns(X.copy())
    X = X.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    for c in X.columns.tolist():
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)

def canon_model(name: str) -> str:
    s = str(name).strip().lower()
    if "pro max" in s: return "iPhone 13 Pro Max"
    if "pro" in s:     return "iPhone 13 Pro"
    if "mini" in s:    return "iPhone 13 Mini"
    if "13" in s:      return "iPhone 13"
    return str(name) if name is not None else ""

def find_emb_cols_any(df: pd.DataFrame, emb_cfg: List[Tuple[str,int]]) -> List[str]:
    cols = []
    for pref, _dims in emb_cfg:
        pref_low = pref.lower()
        pref_cols = [c for c in df.columns if c.lower().startswith(pref_low)]
        def sufnum(c: str) -> int:
            s = c[len(pref):]
            try: return int(''.join(ch for ch in s if ch.isdigit()))
            except: return 10**9
        pref_cols = sorted(pref_cols, key=sufnum)
        cols.extend(pref_cols)
    return cols

# ───────────────────────── threshold sweep
def threshold_sweep(y_true: np.ndarray, p: np.ndarray, steps: int = 1001) -> pd.DataFrame:
    ths = np.linspace(0.0, 1.0, steps); n = len(y_true); rows = []
    for t in ths:
        y_pred = (p >= t).astype(int)
        TP = int(((y_true == 1) & (y_pred == 1)).sum())
        FP = int(((y_true == 0) & (y_pred == 1)).sum())
        TN = int(((y_true == 0) & (y_pred == 0)).sum())
        FN = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc  = (TP + TN) / n if n > 0 else 0.0
        bal  = 0.5 * (rec + (TN / (TN + FP) if (TN + FP) > 0 else 0.0))
        rows.append({"threshold": float(t), "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                     "precision": prec, "recall_TPR": rec, "F1": f1, "accuracy": acc,
                     "balanced_accuracy": bal, "predicted_fast_share": (TP+FP)/n if n>0 else 0.0})
    return pd.DataFrame(rows)

def choose_threshold_with_constraints(sweep: pd.DataFrame,
                                     prec_min: float = PREC_MIN,
                                     rec_min: float = REC_MIN,
                                     max_share: Optional[float] = MAX_SHARE) -> float:
    df = sweep.copy()
    cand = df[(df["precision"] >= prec_min) & (df["recall_TPR"] >= rec_min)]
    if len(cand):
        return float(cand.sort_values(["F1","threshold"], ascending=[False, True]).iloc[0]["threshold"])
    cand = df[df["precision"] >= prec_min]
    if len(cand):
        return float(cand.sort_values(["F1","threshold"], ascending=[False, True]).iloc[0]["threshold"])
    if max_share is not None:
        cand = df[df["predicted_fast_share"] <= max_share]
        if len(cand):
            return float(cand.sort_values(["F1","threshold"], ascending=[False, True]).iloc[0]["threshold"])
    return float(df.sort_values(["balanced_accuracy","threshold"], ascending=[False, True]).iloc[0]["threshold"])

# ───────────────────────── pools + rolling medians (leak-safe)
def build_clean_pool(df: pd.DataFrame,
                     model_col: Optional[str],
                     edited_col: Optional[str],
                     sold_col: Optional[str],
                     sold_price_col: Optional[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    out["model"] = df[model_col].map(canon_model) if model_col else ""
    out["_edited"] = pd.to_datetime(df[edited_col], errors="coerce", utc=True) if edited_col else pd.NaT
    out["_sold"]   = pd.to_datetime(df[sold_col],   errors="coerce", utc=True) if sold_col   else pd.NaT
    out["_sold_price"] = safe_num(df[sold_price_col]) if sold_price_col else np.nan
    out = out[out["model"].isin(VARIANTS)]
    out = out.dropna(subset=["model","_edited","_sold","_sold_price"])
    out["duration_h"] = (out["_sold"] - out["_edited"]).dt.total_seconds() / 3600.0
    out = out[(out["duration_h"] >= 0.0) & (out["duration_h"] <= 504.0)]
    out = out.sort_values("_sold").reset_index(drop=True)
    return out

def rolling_anchor14_for_targets(target: pd.DataFrame,
                                 target_model_col: str,
                                 target_sold_col: str,
                                 pool: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=target.index, dtype=float)
    if pool.empty or target.empty or target_sold_col not in target.columns:
        return out
    DAY14_NS = np.int64(14*24*3600) * np.int64(1_000_000_000)
    for v in VARIANTS:
        tgt_v = target[target[target_model_col] == v].copy()
        if tgt_v.empty: continue
        p = pool[pool["model"] == v].copy()
        if p.empty: continue
        p_times = p["_sold"].astype("int64").to_numpy()
        p_durs  = p["duration_h"].to_numpy()
        n = len(p)
        tgt_v = tgt_v.sort_values(target_sold_col)
        t_times = pd.to_datetime(tgt_v[target_sold_col], errors="coerce", utc=True).astype("int64").to_numpy()
        idxs    = tgt_v.index.to_numpy()
        left = right = 0
        window, prior = [], []
        def med(lst: List[float]) -> float:
            m = len(lst)
            if m == 0: return np.nan
            k = m // 2
            return float(lst[k]) if (m % 2) else float((lst[k-1] + lst[k]) / 2.0)
        for t, row_idx in zip(t_times, idxs):
            if np.isnan(t): out.loc[row_idx] = np.nan; continue
            while right < n and p_times[right] < t:
                d = float(p_durs[right]); insort(prior, d); insort(window, d); right += 1
            cutoff = t - DAY14_NS
            while left < right and p_times[left] < cutoff:
                d = float(p_durs[left])
                pos = bisect_left(window, d)
                if pos < len(window) and window[pos] == d: window.pop(pos)
                left += 1
            m14 = med(window);  m14 = med(prior) if np.isnan(m14) else m14
            out.loc[row_idx] = m14
    return out

def rolling_price30_for_targets(target: pd.DataFrame,
                                target_model_col: str,
                                target_sold_col: str,
                                pool: pd.DataFrame,
                                embargo_days: int = 0) -> pd.Series:
    EMBARGO_NS = np.int64(embargo_days*24*3600) * np.int64(1_000_000_000)

    out = pd.Series(np.nan, index=target.index, dtype=float)
    if pool.empty or target.empty or target_sold_col not in target.columns:
        return out

    DAY30_NS = np.int64(30*24*3600) * np.int64(1_000_000_000)

    for v in VARIANTS:
        tgt_v = target[target[target_model_col] == v].copy()
        if tgt_v.empty:
            continue

        p = pool[pool["model"] == v].copy()
        if p.empty:
            continue

        # Ensure pool is time-sorted and clean
        p = p.dropna(subset=["_sold", "_sold_price"]).copy()
        p_times = p["_sold"].astype("int64").to_numpy()
        p_prices = p["_sold_price"].to_numpy()
        order = np.argsort(p_times)
        p_times = p_times[order]
        p_prices = p_prices[order]
        n = len(p_times)

        # Sort target rows by their timestamp
        tgt_v = tgt_v.sort_values(target_sold_col)
        t_times = pd.to_datetime(tgt_v[target_sold_col], errors="coerce", utc=True).astype("int64").to_numpy()
        idxs    = tgt_v.index.to_numpy()

        left = right = 0
        window, prior = [], []

        def med(lst: List[float]) -> float:
            m = len(lst)
            if m == 0:
                return np.nan
            k = m // 2
            return float(lst[k]) if (m % 2) else float((lst[k-1] + lst[k]) / 2.0)

        for t, row_idx in zip(t_times, idxs):
            # Skip NaT rows
            # (t is int64; NaT -> very negative; if needed, guard explicitly)
            if not np.isfinite(float(t)):
                out.loc[row_idx] = np.nan
                continue

            # Enforce embargo: use only pool rows strictly before (t - EMBARGO_NS)
            t_eff = t - EMBARGO_NS

            while right < n and p_times[right] < t_eff:
                pr = float(p_prices[right])
                insort(prior, pr)
                insort(window, pr)
                right += 1

            # Sliding 30-day window ending at t_eff
            cutoff = t_eff - DAY30_NS
            while left < right and p_times[left] < cutoff:
                pr = float(p_prices[left])
                pos = bisect_left(window, pr)
                if pos < len(window) and window[pos] == pr:
                    window.pop(pos)
                left += 1

            m30 = med(window)
            if np.isnan(m30):
                m30 = med(prior)
            out.loc[row_idx] = m30

    return out

def rolling_anchor60_for_targets(target: pd.DataFrame,
                                 target_model_col: str,
                                 target_storage_col: str,
                                 target_sold_col: str,
                                 pool: pd.DataFrame,
                                 embargo_days: int = 0) -> pd.Series:
    out = pd.Series(np.nan, index=target.index, dtype=float)
    if pool.empty or target.empty or (target_sold_col not in target.columns):
        return out
    DAY60_NS   = np.int64(60*24*3600) * np.int64(1_000_000_000)
    EMBARGO_NS = np.int64(embargo_days*24*3600) * np.int64(1_000_000_000)

    # For each (model, storage) group, compute rolling 60D median of SOLD prices as-of t-embargo
    for (m, s) in pool[["model","storage"]].drop_duplicates().itertuples(index=False, name=None):
        tgt = target[(target[target_model_col] == m) & (target[target_storage_col] == s)].copy()
        if tgt.empty: 
            continue
        p = pool[(pool["model"] == m) & (pool["storage"] == s)].copy()
        if p.empty:
            continue

        p_times   = p["_sold"].astype("int64").to_numpy()
        p_prices  = p["_sold_price"].to_numpy()
        n         = len(p)
        order     = np.argsort(p_times)
        p_times   = p_times[order]
        p_prices  = p_prices[order]

        tgt = tgt.sort_values(target_sold_col)
        t_times = pd.to_datetime(tgt[target_sold_col], errors="coerce", utc=True).astype("int64").to_numpy()
        idxs    = tgt.index.to_numpy()

        left = right = 0
        window = []
        def insert_sorted(lst, x):
            pos = bisect_left(lst, x)
            lst.insert(pos, x)
        def median(lst):
            m = len(lst)
            if m == 0: 
                return np.nan
            k = m // 2
            if m % 2:
                return float(lst[k])
            return float((lst[k-1] + lst[k]) / 2.0)

        for t, row_idx in zip(t_times, idxs):
            if np.isnan(t):
                out.loc[row_idx] = np.nan
                continue

            # Exclude events on/after (t - EMBARGO)
            t_eff = t - EMBARGO_NS

            while right < n and p_times[right] < t_eff:
                insert_sorted(window, float(p_prices[right]))
                right += 1

            cutoff = t_eff - DAY60_NS
            while left < right and p_times[left] < cutoff:
                # remove p_prices[left] from sorted window
                x = float(p_prices[left])
                pos = bisect_left(window, x)
                if pos < len(window) and window[pos] == x:
                    window.pop(pos)
                left += 1

            out.loc[row_idx] = median(window)

    return out


# ───────────────────────── feature harvest
def non_leaky(col: str) -> bool:
    lc = col.lower()
    return not any(sub in lc for sub in LEAK_SUBS)

def allowed_base(col: str) -> bool:
    lc = col.lower()
    if lc.startswith("embed32_") or lc.startswith("embed64_") or lc.startswith("embed64w_"):
        return True
    return not any(b in lc for b in BAN_COLS)

EXTRA_NUMERIC_COLS = [
    "warranty_flag_text","receipt_flag_text","operator_lock_flag_text",
    "negotiation_flag_text","fast_pris_flag_text","accessory_count_text","age_months_text",
    "seller_rating","review_count","ratings_count","member_since_year","member_year",
    "price_per_gb","ppg_ratio","price_pe_rgb"
]

def storage_factor(s: pd.Series) -> pd.Series:
    STOR_RANK = {64:0.0, 128:0.1, 256:0.2, 512:0.3, 1024:0.4}
    v = pd.to_numeric(s, errors="coerce")
    return v.map(STOR_RANK).fillna(0.0)

def _combine_title_desc(df: pd.DataFrame, prefer_title: Optional[str]=None, prefer_desc: Optional[str]=None) -> pd.Series:
    tcol = prefer_title or pick(df.columns, ["title","ad_title","listing_title","headline","title_text"])
    dcol = prefer_desc  or pick(df.columns, ["description","desc","ad_description","listing_description","body","text","description_text","details"])
    title = df[tcol].astype(str) if tcol else pd.Series([""]*len(df), index=df.index)
    desc  = df[dcol].astype(str) if dcol else pd.Series([""]*len(df), index=df.index)
    return (title.fillna("") + " " + desc.fillna("")).str.strip().fillna("")

def compute_svd_arrays_from_text(train_lab: pd.DataFrame,
                                 vtxt: pd.DataFrame,
                                 val_like_df: pd.DataFrame,
                                 id_vtxt: str, id_val: str,
                                 svd_dims: int = SVD_DIMS,
                                 max_features: int = SVD_TFIDF_MAX_FEATURES) -> Tuple[np.ndarray, np.ndarray, int]:
    try:
        tr_text = _combine_title_desc(train_lab)
        vt_text_all = _combine_title_desc(vtxt)
        id_map = dict(zip(vtxt[id_vtxt].astype(str), vt_text_all.astype(str)))
        va_ids = val_like_df[id_val].astype(str)
        va_text = va_ids.map(id_map).fillna("")
        if tr_text.fillna("").str.len().sum() == 0:
            return np.zeros((len(train_lab), 0)), np.zeros((len(val_like_df), 0)), 0
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
        Xtr_tfidf = vectorizer.fit_transform(tr_text.values)
        Xva_tfidf = vectorizer.transform(va_text.values)
        max_k = max(1, min(svd_dims,
                           (Xtr_tfidf.shape[1]-1) if Xtr_tfidf.shape[1] > 1 else 1,
                           (Xtr_tfidf.shape[0]-1) if Xtr_tfidf.shape[0] > 1 else 1))
        svd = TruncatedSVD(n_components=max_k, random_state=RANDOM_STATE)
        Ztr = svd.fit_transform(Xtr_tfidf)
        Zva = svd.transform(Xva_tfidf)
        return Ztr.astype(float), Zva.astype(float), int(max_k)
    except Exception:
        return np.zeros((len(train_lab), 0)), np.zeros((len(val_like_df), 0)), 0

def harvest_features(train: pd.DataFrame, val: pd.DataFrame, extra_emb_cols: Optional[List[str]] = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common = [c for c in train.columns if c in val.columns and non_leaky(c) and allowed_base(c)]
    num_cols = []
    for c in common:
        if c.startswith("embed32_") or c.startswith("embed64_") or c.startswith("embed64w_"):
            continue
        if train[c].dtype != "O" and val[c].dtype != "O":
            num_cols.append(c)

    base_tr = train[num_cols].copy() if num_cols else pd.DataFrame(index=train.index)
    base_va = val[num_cols].copy()   if num_cols else pd.DataFrame(index=val.index)

    # embeddings
    if extra_emb_cols:
        for c in extra_emb_cols:
            base_tr[c] = safe_num(train.get(c)); base_va[c] = safe_num(val.get(c))
        used_embs = list(extra_emb_cols)
    else:
        emb_dim = EMB_DIMS_DEFAULT
        emb_cols = [f"embed32_{i:02d}" for i in range(emb_dim) if f"embed32_{i:02d}" in train.columns]
        for c in emb_cols:
            if c in val.columns:
                base_tr[c] = safe_num(train[c]); base_va[c] = safe_num(val[c])
        used_embs = emb_cols

    # engineered/known numeric columns
    for c in [
        "overall_median_price","ptv_overall","delta_from_overall_price",
        "ptv_anchor","delta_anchor_price",
        "anchor_time14_model_row","delta_anchor_time14_minus_pred","delta_pred_model_med14",
        "condition_score","battery_pct","damage_severity"
    ]:
        if c in train.columns and c in val.columns:
            base_tr[c] = safe_num(train[c]); base_va[c] = safe_num(val[c])

    # extra text/seller numerics
    for c in EXTRA_NUMERIC_COLS:
        ct = pick(train.columns, [c]); cv = pick(val.columns, [c])
        if ct and cv:
            base_tr[ct] = safe_num(train[ct]); base_va[cv] = safe_num(val[cv])

    # storage utilities
    stor_tr = train.get("storage_gb", train.get("storage_norm", train.get("storage_num", train.get("storage", np.nan))))
    stor_va = val.get("storage_gb",   val.get("storage_norm",   val.get("storage_num",   val.get("storage",   np.nan))))
    base_tr["_stor_fac"] = storage_factor(stor_tr)
    base_va["_stor_fac"] = storage_factor(stor_va)

    # normalized deltas
    if "overall_median_price" in base_tr.columns:
        denom_tr = safe_num(train["overall_median_price"]).replace(0, np.nan)
        base_tr["_delta_overall_norm"] = safe_num(base_tr["delta_from_overall_price"]) / denom_tr
    else:
        base_tr["_delta_overall_norm"] = np.nan
    if "overall_median_price" in base_va.columns:
        denom_va = safe_num(val["overall_median_price"]).replace(0, np.nan)
        base_va["_delta_overall_norm"] = safe_num(base_va["delta_from_overall_price"]) / denom_va
    else:
        base_va["_delta_overall_norm"] = np.nan

    # composite underprice×quality×battery×storage
    def _comp(df):
        cond = df.get("condition_score", 0).clip(0,1).fillna(0.0)
        dmg  = df.get("damage_severity", 3).clip(0,3).fillna(3.0)
        batt = df.get("battery_pct", 0).clip(0,100).fillna(0.0)
        dmg_fac  = 1.0 - (dmg / 3.0)
        batt_fac = batt / 100.0
        return (-df["_delta_overall_norm"].clip(-1,1).fillna(0.0)) * cond * dmg_fac * batt_fac * (1.0 + df["_stor_fac"])
    base_tr["_underprice_quality"] = _comp(base_tr)
    base_va["_underprice_quality"] = _comp(base_va)

    # simple hinges
    def add_hinges(df_tr, df_va, col, knots, prefix):
        if col in df_tr.columns and col in df_va.columns:
            vtr, vva = safe_num(df_tr[col]), safe_num(df_va[col])
            for k in knots:
                df_tr[f"{prefix}_lt_{k}"] = (vtr <= k).astype(float)
                df_va[f"{prefix}_lt_{k}"] = (vva <= k).astype(float)
                df_tr[f"{prefix}_hinge_{k}"] = (k - vtr).clip(lower=0)
                df_va[f"{prefix}_hinge_{k}"] = (k - vva).clip(lower=0)
    add_hinges(base_tr, base_va, "delta_from_overall_price", [-300,-250,-200,-150,-100,-50], "_hovr")
    add_hinges(base_tr, base_va, "delta_anchor_price",       [-250,-150,-100,-50],          "_hanc")

    # interactions
    def mul(a,b): return pd.to_numeric(a, errors="coerce") * pd.to_numeric(b, errors="coerce")
    if "ptv_anchor" in base_tr.columns and "condition_score" in base_tr.columns:
        base_tr["ptv_anchor_x_cond"] = mul(base_tr["ptv_anchor"], base_tr["condition_score"]).fillna(0.0)
        base_va["ptv_anchor_x_cond"] = mul(base_va["ptv_anchor"], base_va["condition_score"]).fillna(0.0)
    if "ptv_overall" in base_tr.columns and "battery_pct" in base_tr.columns:
        base_tr["ptv_overall_x_batt"] = mul(base_tr["ptv_overall"], base_tr["battery_pct"]).fillna(0.0)
        base_va["ptv_overall_x_batt"] = mul(base_va["ptv_overall"], base_va["battery_pct"]).fillna(0.0)
    if "delta_anchor_time14_minus_pred" in base_tr.columns and "condition_score" in base_tr.columns:
        base_tr["d_anchor_time_x_cond"] = mul(base_tr["delta_anchor_time14_minus_pred"], base_tr["condition_score"]).fillna(0.0)
        base_va["d_anchor_time_x_cond"] = mul(base_va["delta_anchor_time14_minus_pred"], base_va["condition_score"]).fillna(0.0)

    base_tr = coerce_numeric_df(base_tr)
    base_va = coerce_numeric_df(base_va)
    base_va = base_va.reindex(columns=base_tr.columns, fill_value=0.0)
    return base_tr, base_va, used_embs

# ───────────────────────── soft anti-FP penalty
def _get_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    return pick(df.columns, names)

def _to_storage_gb(df: pd.DataFrame) -> pd.Series:
    col = _get_col(df, ["storage_gb","storage_norm","storage_num","capacity","storage"])
    if not col: return pd.Series(np.nan, index=df.index)
    s = df[col].astype(str).str.lower()
    gb = pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    tb_mask = s.str.contains("tb", na=False)
    gb = np.where(tb_mask, gb*1024.0, gb)
    return pd.Series(gb, index=df.index)

def _price_per_gb(df: pd.DataFrame) -> pd.Series:
    price_col = _get_col(df, ["price","ask","ask_price"])
    if not price_col: return pd.Series(np.nan, index=df.index)
    price = safe_num(df[price_col])
    gb = _to_storage_gb(df)
    ppg = price / gb.replace(0, np.nan)
    return pd.Series(ppg, index=df.index)

def _flag(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    c = _get_col(df, names)
    if not c: return None
    return safe_num(df[c])

def soft_gate_penalty(df: pd.DataFrame) -> pd.Series:
    if not G_SOFT_ENABLE: return pd.Series(1.0, index=df.index)
    def clamp(v, lo, hi): return np.maximum(lo, np.minimum(hi, v))
    pen = pd.Series(1.0, index=df.index, dtype=float)

    if "condition_score" in df:
        cond = pd.to_numeric(df["condition_score"], errors="coerce")
        f = np.where(cond >= G_MIN_COND, 1.0,
                     np.where(np.isnan(cond), 1.0, clamp(cond / max(G_MIN_COND, 1e-6), G_PENALTY_MIN, 1.0)))
        pen *= f
    if "damage_severity" in df:
        dmg = pd.to_numeric(df["damage_severity"], errors="coerce")
        f = np.where(dmg <= G_MAX_DAMAGE, 1.0,
                     np.where(np.isnan(dmg), 1.0, clamp(1.0 - 0.3*(dmg - G_MAX_DAMAGE), G_PENALTY_MIN, 1.0)))
        pen *= f
    if "battery_pct" in df:
        batt = pd.to_numeric(df["battery_pct"], errors="coerce")
        f = np.where(batt >= G_MIN_BATT, 1.0,
                     np.where(np.isnan(batt), 1.0, clamp(batt / max(G_MIN_BATT,1e-6), G_PENALTY_MIN, 1.0)))
        pen *= f
    if "ptv_overall" in df:
        po = pd.to_numeric(df["ptv_overall"], errors="coerce")
        f = np.where(po <= G_PTV_OVERALL_MAX, 1.0,
                     np.where(np.isnan(po), 1.0, clamp(G_PTV_OVERALL_MAX / np.maximum(po,1e-6), G_PENALTY_MIN, 1.0)))
        pen *= f
    if "delta_from_overall_price" in df:
        delt = pd.to_numeric(df["delta_from_overall_price"], errors="coerce")
        f = np.where(delt <= G_DELTA_OVERALL_MAX, 1.0,
                     np.where(np.isnan(delt), 1.0, clamp(np.exp(-(delt - G_DELTA_OVERALL_MAX)/120.0), G_PENALTY_MIN, 1.0)))
        pen *= f
    if "ptv_anchor" in df:
        pa = pd.to_numeric(df["ptv_anchor"], errors="coerce")
        f = np.where(pa <= G_PTV_ANCHOR_MAX, 1.0,
                     np.where(np.isnan(pa), 1.0, clamp(G_PTV_ANCHOR_MAX / np.maximum(pa,1e-6), G_PENALTY_MIN, 1.0)))
        pen *= f

    stor_gb = _to_storage_gb(df)
    if stor_gb.notna().any() and G_MIN_STORAGE_GB is not None:
        pen *= np.where(stor_gb >= G_MIN_STORAGE_GB, 1.0, 0.9)

    ppg_col = _get_col(df, ["price_per_gb","ppg_ratio","price_pe_rgb","price_per_storage"])
    ppg = pd.to_numeric(df[ppg_col], errors="coerce") if ppg_col else _price_per_gb(df)
    if ppg.notna().any() and G_MAX_PPG_RATIO is not None:
        pen *= np.where(ppg <= G_MAX_PPG_RATIO, 1.0,
                        np.where(np.isnan(ppg), 1.0, np.maximum(0.7, G_MAX_PPG_RATIO / np.maximum(ppg,1e-6))))

    for nm, boost in [
        (["warranty_flag_text","warranty_flag"], G_WARRANTY_BOOST),
        (["receipt_flag_text","receipt_flag"],   G_RECEIPT_BOOST),
        (["operator_lock_flag_text","simlock_flag","oplock_flag"], G_OPLOCK_PENALTY),
        (["negotiation_flag_text","negotiation_flag"], G_NEGOTIATION_BOOST),
        (["fast_pris_flag_text","fast_pris_flag"], G_FASTPRIS_PENALTY),
    ]:
        s = _flag(df, nm)
        if s is not None and boost != 1.0:
            pen *= np.where(s.fillna(0) >= 1, boost, 1.0)

    acc = _flag(df, ["accessory_count_text","accessory_count"])
    if acc is not None and G_ACC_MAX_BOOST != 1.0:
        pen *= 1.0 + (np.clip(acc.fillna(0), 0, 5)/5.0)*(G_ACC_MAX_BOOST-1.0)

    srat = _flag(df, ["seller_rating","rating","user_rating"])
    if srat is not None and G_MIN_SELLER_RATING is not None:
        pen *= np.where(srat.fillna(G_MIN_SELLER_RATING) >= G_MIN_SELLER_RATING, 1.0, 0.9)

    rcnt = _flag(df, ["review_count","reviews_count","ratings_count"])
    if rcnt is not None and G_MIN_REVIEWS is not None:
        pen *= np.where(rcnt.fillna(G_MIN_REVIEWS) >= G_MIN_REVIEWS, 1.0, 0.92)

    msy = _flag(df, ["member_since_year","member_year"])
    if msy is not None and G_MIN_ACCOUNT_AGE_Y is not None:
        age = 2025 - msy
        pen *= np.where(age.fillna(G_MIN_ACCOUNT_AGE_Y) >= G_MIN_ACCOUNT_AGE_Y, 1.0, 0.9)

    model_col = "__canon_model" if "__canon_model" in df.columns else _get_col(df, ["model","variant","device_model"])
    if model_col:
        model_factors = df[model_col].map(G_MODEL_PENALTIES).fillna(1.0).astype(float)
        pen *= np.where(model_factors > 0, model_factors, 1.0)

    pen = np.clip(pen, G_TOTAL_PENALTY_MIN, G_TOTAL_PENALTY_MAX)
    return pd.Series(pen, index=df.index)

def clamp(v, lo, hi): return np.maximum(lo, np.minimum(hi, v))

# ───────────────────────── training + calibration
def train_blend_and_calibrate(X: np.ndarray, y: np.ndarray):
    if y.min() == y.max():
        raise SystemExit("Training labels are single-class; cannot train classifier.")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]; y_tr = y[tr_idx]
        scaler = StandardScaler(); Z_tr = scaler.fit_transform(X_tr); Z_va = scaler.transform(X_va)
        lr = LogisticRegression(max_iter=5000, class_weight=CLASS_WEIGHT, solver="lbfgs", random_state=RANDOM_STATE)
        lr.fit(Z_tr, y_tr)
        oof[va_idx] = lr.predict_proba(Z_va)[:,1]
    use_cal = CALIBRATION
    spread = float(np.nanmax(oof) - np.nanmin(oof))
    if use_cal == "isotonic" and (spread < 1e-3 or np.unique(np.round(oof, 6)).size < 10):
        use_cal = "platt"
    if use_cal == "isotonic":
        calib = IsotonicRegression(out_of_bounds="clip"); calib.fit(oof, y)
    elif use_cal == "platt":
        pl = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE); pl.fit(oof.reshape(-1,1), y); calib = ("platt", pl)
    elif use_cal == "none":
        calib = None
    else:
        raise SystemExit(f"Unknown calibration: {CALIBRATION}")
    scaler_f = StandardScaler(); Z_full = scaler_f.fit_transform(X)
    lr_f = LogisticRegression(max_iter=5000, class_weight=CLASS_WEIGHT, solver="lbfgs", random_state=RANDOM_STATE)
    lr_f.fit(Z_full, y)
    return dict(scaler=scaler_f, lr=lr_f, calib=calib, cal_kind=use_cal)

def predict_proba(X: np.ndarray, comps: dict) -> np.ndarray:
    Z = comps["scaler"].transform(X)
    p = comps["lr"].predict_proba(Z)[:,1]
    calib = comps["calib"]
    if calib is None: return p
    if isinstance(calib, IsotonicRegression): return calib.predict(p)
    if isinstance(calib, tuple) and calib[0] == "platt": return calib[1].predict_proba(p.reshape(-1,1))[:,1]
    return p

# ───────────────────────── eval helpers
def metrics_union(y_true_bin: np.ndarray, y_hat_bin: np.ndarray) -> Dict[str, float]:
    TP = int(((y_true_bin==1)&(y_hat_bin==1)).sum())
    FP = int(((y_true_bin==0)&(y_hat_bin==1)).sum())
    TN = int(((y_true_bin==0)&(y_hat_bin==0)).sum())
    FN = int(((y_true_bin==1)&(y_hat_bin==0)).sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    acc  = (TP+TN)/len(y_true_bin) if len(y_true_bin)>0 else 0.0
    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                precision=prec, recall=rec, F1=f1, accuracy=acc,
                flags=int(y_hat_bin.sum()), n=int(len(y_true_bin)))

def eval_40_72(val_out: pd.DataFrame, cond_col: Optional[str], dmg_col: Optional[str], dur_col: Optional[str]) -> Dict[str, float]:
    out = {"F1_40_72": 0.0, "prec_40": 0.0, "rec_40": 0.0, "TP_40": 0, "FP_slow72": 0, "MID": 0, "kept": 0, "fast40_total": 0}
    if dur_col is None or dur_col not in val_out.columns: return out
    dur = pd.to_numeric(val_out[dur_col], errors="coerce")
    fast40 = dur <= 40
    slow72 = dur > 72
    mid4072 = (dur > 40) & (dur <= 72)
    kept = val_out["is_fast24_pred"].astype(bool).copy()
    if HARD_FILTER_ENABLE:
        prob_ok = val_out["prob_fast24"] >= float(HF_MIN_PROB_FAST24)
        cond_series = safe_num(val_out.get(cond_col, np.nan)) if cond_col else pd.Series(np.nan, index=val_out.index)
        dmg_series  = safe_num(val_out.get(dmg_col,  np.nan)) if dmg_col  else pd.Series(np.nan, index=val_out.index)
        mask_damage_drop = (dmg_series >= float(HF_DROP_DAMAGE_GTE)) & (
            (cond_series >= float(HF_DROP_COND_MIN)) | ((cond_series == 0.0) if HF_DROP_COND_EQ0_TOO else False)
        )
        kept = kept & prob_ok & (~mask_damage_drop)
    TP = int((kept & fast40).sum()); FP = int((kept & slow72).sum()); MID = int((kept & mid4072).sum())
    kept_total = int(kept.sum()); fast40_total = int(fast40.sum())
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0; rec  = TP/fast40_total if fast40_total>0 else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    out.update(dict(F1_40_72=f1, prec_40=prec, rec_40=rec, TP_40=TP, FP_slow72=FP, MID=MID, kept=kept_total, fast40_total=fast40_total))
    return out

def preferred_model_pred_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["pred_hours_or_best","pred_final_best","pred_hours_enriched","pred_hours_base","pred_hours"]:
        if c in df.columns: return c
    return None

# ───────────────────────── anchors by model×storage (ask-price medians)
def attach_anchor_price_by_model_storage(train: pd.DataFrame, val: pd.DataFrame,
                                         price_tr: Optional[str], price_va: Optional[str],
                                         model_tr: Optional[str], model_va: Optional[str],
                                         storage_tr: Optional[str], storage_va: Optional[str]):
    if not (price_tr and model_tr and storage_tr): return
    df = train[[model_tr, storage_tr, price_tr]].dropna(subset=[price_tr]).copy()
    med_ms = df.groupby([model_tr, storage_tr])[price_tr].median()
    train["_ms_key"] = list(zip(train.get(model_tr), train.get(storage_tr)))
    train["_anchor_price_ms"] = pd.Series(train["_ms_key"]).map(med_ms).values
    train["ptv_anchor"] = safe_num(train.get(price_tr)) / safe_num(train.get("_anchor_price_ms"))
    train["delta_anchor_price"] = safe_num(train.get(price_tr)) - safe_num(train.get("_anchor_price_ms"))
    train.drop(columns=["_ms_key","_anchor_price_ms"], inplace=True, errors="ignore")
    if model_va and storage_va and price_va and model_va in val.columns and storage_va in val.columns:
        val["_ms_key"] = list(zip(val.get(model_va), val.get(storage_va)))
        val["_anchor_price_ms"] = pd.Series(val["_ms_key"]).map(med_ms).values
        val["ptv_anchor"] = safe_num(val.get(price_va)) / safe_num(val.get("_anchor_price_ms"))
        val["delta_anchor_price"] = safe_num(val.get(price_va)) - safe_num(val.get("_anchor_price_ms"))
        val.drop(columns=["_ms_key","_anchor_price_ms"], inplace=True, errors="ignore")

# ───────────────────────── PIPELINE (Stage 1)
def pipeline_run(train_path: str, slow_path: str, val_text_path: str, oof_path: Optional[str],
                 out_dir: str, emb_choice: str, threshold_steps: int, write_outputs: bool = True
                ) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    train = pd.read_csv(train_path)
    slow  = pd.read_csv(slow_path)
    vtxt  = pd.read_csv(val_text_path) if os.path.exists(val_text_path) else slow
    

    
    # ─── Condition corridor + predicted slow filter (order can be swapped by flag) ───
    def _apply_cond_gate_local(df, within_pred=False):
        def _ensure_ptv_inline(_df: pd.DataFrame) -> pd.DataFrame:
            # Pick a price column
            price_col = None
            for cand in ["price", "price_sv_eff", "price_b", "ask_price", "ask_price_nok", "price_nok"]:
                if cand in _df.columns:
                    price_col = cand
                    break
            if price_col is None:
                print("[PTVGate] Could not build PTVs: no price column found.")
                return _df

            # ---- ptv_overall
            if "ptv_overall" not in _df.columns:
                src = None
                if "overall_median_price" in _df.columns:
                    overall = pd.to_numeric(_df["overall_median_price"], errors="coerce")
                    src = "overall_median_price"
                elif "delta_from_overall_price" in _df.columns:
                    overall = pd.to_numeric(_df[price_col], errors="coerce") - pd.to_numeric(_df["delta_from_overall_price"], errors="coerce")
                    src = "price - delta_from_overall_price"
                elif "variant" in _df.columns:
                    overall = _df.groupby("variant")[price_col].transform("median")
                    src = "variant median (inline)"
                elif "model" in _df.columns:
                    overall = _df.groupby("model")[price_col].transform("median")
                    src = "model median (inline)"
                else:
                    med = pd.to_numeric(_df[price_col], errors="coerce").median()
                    overall = pd.Series(med, index=_df.index)
                    src = "global median (inline)"
                _df["ptv_overall"] = pd.to_numeric(_df[price_col], errors="coerce") / pd.to_numeric(overall, errors="coerce")
                print(f"[PTVGate] Built ptv_overall via {src}.")

            # ---- ptv_anchor
            if "ptv_anchor" not in _df.columns:
                if "delta_anchor_price" in _df.columns:
                    anchor_price = pd.to_numeric(_df[price_col], errors="coerce") - pd.to_numeric(_df["delta_anchor_price"], errors="coerce")
                    _df["ptv_anchor"] = pd.to_numeric(_df[price_col], errors="coerce") / anchor_price.replace({0: np.nan})
                    print("[PTVGate] Built ptv_anchor via delta_anchor_price.")
                elif ("model" in _df.columns) and ("storage" in _df.columns):
                    ms_med = _df.groupby(["model","storage"])[price_col].transform("median")
                    _df["ptv_anchor"] = pd.to_numeric(_df[price_col], errors="coerce") / pd.to_numeric(ms_med, errors="coerce")
                    print("[PTVGate] Built ptv_anchor via (model, storage) medians (inline).")
                elif "variant" in _df.columns:
                    var_med = _df.groupby("variant")[price_col].transform("median")
                    _df["ptv_anchor"] = pd.to_numeric(_df[price_col], errors="coerce") / pd.to_numeric(var_med, errors="coerce")
                    print("[PTVGate] Built ptv_anchor via variant median (fallback).")
                else:
                    print("[PTVGate] Could not build ptv_anchor — missing model/storage/variant.")

            return _df

        try:
            # Ensure PTV columns exist before any gating
            df = _ensure_ptv_inline(df)

            if "condition_score" not in df.columns:
                # Damage filter even when cond gate is skipped
                levels = globals().get("DAMAGE_KEEP_LEVELS", None)
                if levels is not None and "damage_severity" in df.columns:
                    n0 = len(df)
                    ds = pd.to_numeric(df["damage_severity"], errors="coerce").astype("Int64")
                    df = df[ds.isin(levels)].copy().reset_index(drop=True)
                    print(f"[DamageGate] damage_severity in {sorted(levels)}: {n0} -> {len(df)} (cond_gate_absent)")

                # BattGate even when cond gate is skipped
                bmin = globals().get("BATT_GATE_MIN", None)
                bmax = globals().get("BATT_GATE_MAX", None)
                bkeep = bool(globals().get("BATT_KEEP_NA", False))
                if (bmin is not None) or (bmax is not None):
                    if "battery_pct" in df.columns:
                        n_b0 = len(df)
                        bp = pd.to_numeric(df["battery_pct"], errors="coerce")
                        mask_b = np.ones(len(df), dtype=bool)
                        if bmin is not None: mask_b &= (bp >= float(bmin))
                        if bmax is not None: mask_b &= (bp <  float(bmax))
                        if not bkeep:       mask_b &= bp.notna()
                        df = df[mask_b].copy().reset_index(drop=True)
                        print(f"[BattGate] battery_pct in [{bmin if bmin is not None else '-inf'}, {bmax if bmax is not None else '+inf'})"
                              f"{' +keep_na' if bkeep else ''}: {n_b0} -> {len(df)} (cond_gate_absent)")
                    else:
                        print("[BattGate] Skipped: 'battery_pct' not found (cond_gate_absent).")

                # PTV gates even when cond gate is skipped
                p_over = globals().get("PTV_OVERALL_MAX", None)
                p_anch = globals().get("PTV_ANCHOR_MAX",  None)
                if (p_over is not None):
                    if "ptv_overall" in df.columns:
                        n0 = len(df)
                        vo = pd.to_numeric(df["ptv_overall"], errors="coerce")
                        df = df[vo <= float(p_over)].copy().reset_index(drop=True)
                        print(f"[PTVGate] ptv_overall <= {p_over}: {n0} -> {len(df)} (cond_gate_absent)")
                    else:
                        print("[PTVGate] Skipped: 'ptv_overall' not found (cond_gate_absent).")
                if (p_anch is not None):
                    if "ptv_anchor" in df.columns:
                        n0 = len(df)
                        va = pd.to_numeric(df["ptv_anchor"], errors="coerce")
                        df = df[va <= float(p_anch)].copy().reset_index(drop=True)
                        print(f"[PTVGate] ptv_anchor  <= {p_anch}: {n0} -> {len(df)} (cond_gate_absent)")
                    else:
                        print("[PTVGate] Skipped: 'ptv_anchor' not found (cond_gate_absent).")


                # <<< SellerGate (rating/reviews/age & combined quality)
                sr_min  = globals().get("SELLER_GATE_MIN_RATING",  None)
                rc_min  = globals().get("SELLER_GATE_MIN_REVIEWS", None)
                age_min = globals().get("SELLER_GATE_MIN_AGE_Y",   None)
                q_min   = globals().get("SELLER_GATE_MIN_QUALITY", None)

                _sg_df = df
                n_s0 = len(_sg_df)
                mask_s = np.ones(n_s0, dtype=bool)

                if sr_min is not None and "seller_rating" in _sg_df.columns:
                    mask_s &= (safe_num(_sg_df["seller_rating"]).to_numpy(dtype=float) >= float(sr_min))

                if rc_min is not None:
                    if "review_count" in _sg_df.columns:
                        rc_here = safe_num(_sg_df["review_count"]).to_numpy(dtype=float)
                    elif "ratings_count" in _sg_df.columns:
                        rc_here = safe_num(_sg_df["ratings_count"]).to_numpy(dtype=float)
                    else:
                        rc_here = None
                    if rc_here is not None:
                        mask_s &= (rc_here >= float(rc_min))

                if age_min is not None:
                    msy = None
                    if "member_since_year" in _sg_df.columns:
                        msy = safe_num(_sg_df["member_since_year"]).to_numpy(dtype=float)
                    elif "member_year" in _sg_df.columns:
                        msy = safe_num(_sg_df["member_year"]).to_numpy(dtype=float)
                    if msy is not None:
                        age_y = 2025.0 - msy
                        age_y = np.where(np.isfinite(age_y), np.maximum(0.0, age_y), 0.0)
                        mask_s &= (age_y >= float(age_min))

                if q_min is not None:
                    sq = compute_seller_quality(_sg_df).to_numpy(dtype=float)
                    mask_s &= (sq >= float(q_min))

                df = _sg_df[mask_s].copy().reset_index(drop=True)
                print(f"[SellerGate] rating>={sr_min if sr_min is not None else '-'} "
                      f"reviews>={rc_min if rc_min is not None else '-'} "
                      f"age_y>={age_min if age_min is not None else '-'} "
                      f"quality>={q_min if q_min is not None else '-'} : "
                      f"{n_s0} -> {len(df)} (within_pred={within_pred})")

                return df

            if (globals().get("COND_GATE_MIN") is None) and (globals().get("COND_GATE_MAX") is None):
                # Damage filter with no cond bounds
                levels = globals().get("DAMAGE_KEEP_LEVELS", None)
                if levels is not None and "damage_severity" in df.columns:
                    n0 = len(df)
                    ds = pd.to_numeric(df["damage_severity"], errors="coerce").astype("Int64")
                    df = df[ds.isin(levels)].copy().reset_index(drop=True)
                    print(f"[DamageGate] damage_severity in {sorted(levels)}: {n0} -> {len(df)} (no_cond_bounds)")

                # BattGate with no cond bounds
                bmin = globals().get("BATT_GATE_MIN", None)
                bmax = globals().get("BATT_GATE_MAX", None)
                bkeep = bool(globals().get("BATT_KEEP_NA", False))
                if (bmin is not None) or (bmax is not None):
                    if "battery_pct" in df.columns:
                        n_b0 = len(df)
                        bp = pd.to_numeric(df["battery_pct"], errors="coerce")
                        mask_b = np.ones(len(df), dtype=bool)
                        if bmin is not None: mask_b &= (bp >= float(bmin))
                        if bmax is not None: mask_b &= (bp <  float(bmax))
                        if not bkeep:       mask_b &= bp.notna()
                        df = df[mask_b].copy().reset_index(drop=True)
                        print(f"[BattGate] battery_pct in [{bmin if bmin is not None else '-inf'}, {bmax if bmax is not None else '+inf'})"
                              f"{' +keep_na' if bkeep else ''}: {n_b0} -> {len(df)} (no_cond_bounds)")
                    else:
                        print("[BattGate] Skipped: 'battery_pct' not found (no_cond_bounds).")

                # PTV gates with no cond bounds
                p_over = globals().get("PTV_OVERALL_MAX", None)
                p_anch = globals().get("PTV_ANCHOR_MAX",  None)
                if (p_over is not None):
                    if "ptv_overall" in df.columns:
                        n0 = len(df)
                        vo = pd.to_numeric(df["ptv_overall"], errors="coerce")
                        df = df[vo <= float(p_over)].copy().reset_index(drop=True)
                        print(f"[PTVGate] ptv_overall <= {p_over}: {n0} -> {len(df)} (no_cond_bounds)")
                    else:
                        print("[PTVGate] Skipped: 'ptv_overall' not found (no_cond_bounds).")
                if (p_anch is not None):
                    if "ptv_anchor" in df.columns:
                        n0 = len(df)
                        va = pd.to_numeric(df["ptv_anchor"], errors="coerce")
                        df = df[va <= float(p_anch)].copy().reset_index(drop=True)
                        print(f"[PTVGate] ptv_anchor  <= {p_anch}: {n0} -> {len(df)} (no_cond_bounds)")
                    else:
                        print("[PTVGate] Skipped: 'ptv_anchor' not found (no_cond_bounds).")
                        
                # <<< SellerGate (no_cond_bounds)
                sr_min  = globals().get("SELLER_GATE_MIN_RATING",  None)
                rc_min  = globals().get("SELLER_GATE_MIN_REVIEWS", None)
                age_min = globals().get("SELLER_GATE_MIN_AGE_Y",   None)
                q_min   = globals().get("SELLER_GATE_MIN_QUALITY", None)

                _sg_df = df
                n_s0 = len(_sg_df)
                mask_s = np.ones(n_s0, dtype=bool)

                if sr_min is not None and "seller_rating" in _sg_df.columns:
                    mask_s &= (safe_num(_sg_df["seller_rating"]).to_numpy(dtype=float) >= float(sr_min))

                if rc_min is not None:
                    if "review_count" in _sg_df.columns:
                        rc_here = safe_num(_sg_df["review_count"]).to_numpy(dtype=float)
                    elif "ratings_count" in _sg_df.columns:
                        rc_here = safe_num(_sg_df["ratings_count"]).to_numpy(dtype=float)
                    else:
                        rc_here = None
                    if rc_here is not None:
                        mask_s &= (rc_here >= float(rc_min))

                if age_min is not None:
                    msy = None
                    if "member_since_year" in _sg_df.columns:
                        msy = safe_num(_sg_df["member_since_year"]).to_numpy(dtype=float)
                    elif "member_year" in _sg_df.columns:
                        msy = safe_num(_sg_df["member_year"]).to_numpy(dtype=float)
                    if msy is not None:
                        age_y = 2025.0 - msy
                        age_y = np.where(np.isfinite(age_y), np.maximum(0.0, age_y), 0.0)
                        mask_s &= (age_y >= float(age_min))

                if q_min is not None:
                    sq = compute_seller_quality(_sg_df).to_numpy(dtype=float)
                    mask_s &= (sq >= float(q_min))

                df = _sg_df[mask_s].copy().reset_index(drop=True)
                print(f"[SellerGate] rating>={sr_min if sr_min is not None else '-'} "
                      f"reviews>={rc_min if rc_min is not None else '-'} "
                      f"age_y>={age_min if age_min is not None else '-'} "
                      f"quality>={q_min if q_min is not None else '-'} : "
                      f"{n_s0} -> {len(df)} (within_pred={within_pred})")

                return df

            # Main case: condition-gated slice
            cs = pd.to_numeric(df["condition_score"], errors="coerce")
            lo = float(globals().get("COND_GATE_MIN")) if globals().get("COND_GATE_MIN") is not None else -1e9
            hi = float(globals().get("COND_GATE_MAX")) if globals().get("COND_GATE_MAX") is not None else  1e9
            n0 = len(df)
            mask = (cs >= lo) & (cs < hi)
            if bool(globals().get("COND_KEEP_ZERO", False)):
                mask = mask | cs.eq(0.0)
            out = df[mask].copy().reset_index(drop=True)

            # Damage filter inside the cond-gated slice
            levels = globals().get("DAMAGE_KEEP_LEVELS", None)
            if levels is not None:
                if "damage_severity" in out.columns:
                    n1 = len(out)
                    ds = pd.to_numeric(out["damage_severity"], errors="coerce").astype("Int64")
                    out = out[ds.isin(levels)].copy().reset_index(drop=True)
                    print(f"[DamageGate] damage_severity in {sorted(levels)}: {n1} -> {len(out)} (within_pred={within_pred})")
                else:
                    print("[DamageGate] Skipped: 'damage_severity' not found.")

            # BattGate inside the cond-gated slice
            bmin = globals().get("BATT_GATE_MIN", None)
            bmax = globals().get("BATT_GATE_MAX", None)
            bkeep = bool(globals().get("BATT_KEEP_NA", False))
            if (bmin is not None) or (bmax is not None):
                if "battery_pct" in out.columns:
                    n_b0 = len(out)
                    bp = pd.to_numeric(out["battery_pct"], errors="coerce")
                    mask_b = np.ones(len(out), dtype=bool)
                    if bmin is not None: mask_b &= (bp >= float(bmin))
                    if bmax is not None: mask_b &= (bp <  float(bmax))
                    if not bkeep:       mask_b &= bp.notna()
                    out = out[mask_b].copy().reset_index(drop=True)
                    print(f"[BattGate] battery_pct in [{bmin if bmin is not None else '-inf'}, {bmax if bmax is not None else '+inf'})"
                          f"{' +keep_na' if bkeep else ''}: {n_b0} -> {len(out)} (within_pred={within_pred})")
                else:
                    print("[BattGate] Skipped: 'battery_pct' not found.")

            # Ensure PTVs exist in 'out' and apply PTV gates inside the cond-gated slice
            out = _ensure_ptv_inline(out)
            p_over = globals().get("PTV_OVERALL_MAX", None)
            p_anch = globals().get("PTV_ANCHOR_MAX",  None)
            if (p_over is not None):
                if "ptv_overall" in out.columns:
                    n1 = len(out)
                    vo = pd.to_numeric(out["ptv_overall"], errors="coerce")
                    out = out[vo <= float(p_over)].copy().reset_index(drop=True)
                    print(f"[PTVGate] ptv_overall <= {p_over}: {n1} -> {len(out)} (within_pred={within_pred})")
                else:
                    print("[PTVGate] Skipped: 'ptv_overall' not found.")
            if (p_anch is not None):
                if "ptv_anchor" in out.columns:
                    n1 = len(out)
                    va = pd.to_numeric(out["ptv_anchor"], errors="coerce")
                    out = out[va <= float(p_anch)].copy().reset_index(drop=True)
                    print(f"[PTVGate] ptv_anchor  <= {p_anch}: {n1} -> {len(out)} (within_pred={within_pred})")
                else:
                    print("[PTVGate] Skipped: 'ptv_anchor' not found.")

            # <<< SellerGate
            sr_min  = globals().get("SELLER_GATE_MIN_RATING",  None)
            rc_min  = globals().get("SELLER_GATE_MIN_REVIEWS", None)
            age_min = globals().get("SELLER_GATE_MIN_AGE_Y",   None)
            q_min   = globals().get("SELLER_GATE_MIN_QUALITY", None)

            _sg_df = out if 'out' in locals() else df
            n_s0 = len(_sg_df)
            mask_s = np.ones(n_s0, dtype=bool)

            # seller_rating (NaN -> pass)
            if sr_min is not None and "seller_rating" in _sg_df.columns:
                sr = safe_num(_sg_df["seller_rating"]).to_numpy(dtype=float)
                mask_s &= np.where(np.isfinite(sr), sr >= float(sr_min), True)

            # review_count OR ratings_count (NaN -> pass)
            if rc_min is not None:
                rc_here = None
                if "review_count" in _sg_df.columns:
                    rc_here = safe_num(_sg_df["review_count"]).to_numpy(dtype=float)
                elif "ratings_count" in _sg_df.columns:
                    rc_here = safe_num(_sg_df["ratings_count"]).to_numpy(dtype=float)
                if rc_here is not None:
                    mask_s &= np.where(np.isfinite(rc_here), rc_here >= float(rc_min), True)

            # member_since_year -> age years (NaN -> pass)
            if age_min is not None:
                msy = None
                if "member_since_year" in _sg_df.columns:
                    msy = safe_num(_sg_df["member_since_year"]).to_numpy(dtype=float)
                elif "member_year" in _sg_df.columns:
                    msy = safe_num(_sg_df["member_year"]).to_numpy(dtype=float)
                if msy is not None:
                    age_y = 2025.0 - msy
                    age_y = np.where(np.isfinite(age_y), np.maximum(0.0, age_y), np.nan)
                    mask_s &= np.where(np.isfinite(age_y), age_y >= float(age_min), True)

            # combined seller_quality (NaN -> pass)
            if q_min is not None:
                sq = compute_seller_quality(_sg_df).to_numpy(dtype=float)
                mask_s &= np.where(np.isfinite(sq), sq >= float(q_min), True)


            _sg_df = _sg_df[mask_s].copy().reset_index(drop=True)
            if 'out' in locals():
                out = _sg_df
            else:
                df = _sg_df

            print(f"[SellerGate] rating>={sr_min if sr_min is not None else '-'} "
                  f"reviews>={rc_min if rc_min is not None else '-'} "
                  f"age_y>={age_min if age_min is not None else '-'} "
                  f"quality>={q_min if q_min is not None else '-'} : "
                  f"{n_s0} -> {len(_sg_df)} (within_pred={within_pred})")



            if within_pred:
                print(f"[CondGate] within_pred_slow: {n0} -> {len(out)} (bounds=[{lo},{hi}), keep_zero={bool(globals().get('COND_KEEP_ZERO', False))})")
            else:
                print(f"[CondGate] condition_score in [{lo}, {hi}) (+keep_zero={bool(globals().get('COND_KEEP_ZERO', False))}) : {n0} -> {len(out)}")
            return out


        except Exception as _e:
            # Ensure filters still apply on exceptions
            levels = globals().get("DAMAGE_KEEP_LEVELS", None)
            if levels is not None and "damage_severity" in df.columns:
                n0 = len(df)
                ds = pd.to_numeric(df["damage_severity"], errors="coerce").astype("Int64")
                df = df[ds.isin(levels)].copy().reset_index(drop=True)
                print(f"[DamageGate] damage_severity in {sorted(levels)}: {n0} -> {len(df)} (cond_gate_exception)")

            # <<< BattGate on exception path
            bmin = globals().get("BATT_GATE_MIN", None)
            bmax = globals().get("BATT_GATE_MAX", None)
            bkeep = bool(globals().get("BATT_KEEP_NA", False))
            if (bmin is not None) or (bmax is not None):
                if "battery_pct" in df.columns:
                    n_b0 = len(df)
                    bp = pd.to_numeric(df["battery_pct"], errors="coerce")
                    mask_b = np.ones(len(df), dtype=bool)
                    if bmin is not None: mask_b &= (bp >= float(bmin))
                    if bmax is not None: mask_b &= (bp <  float(bmax))
                    if not bkeep:       mask_b &= bp.notna()
                    df = df[mask_b].copy().reset_index(drop=True)
                    print(f"[BattGate] battery_pct in [{bmin if bmin is not None else '-inf'}, {bmax if bmax is not None else '+inf'})"
                          f"{' +keep_na' if bkeep else ''}: {n_b0} -> {len(df)} (cond_gate_exception)")
                else:
                    print("[BattGate] Skipped: 'battery_pct' not found (cond_gate_exception).")

            print("[CondGate] Skipped (no condition_score or bad bounds).")
            return df


        except Exception as _e:
            # <<< Make sure damage filter still applies on errors
            levels = globals().get("DAMAGE_KEEP_LEVELS", None)
            if levels is not None and "damage_severity" in df.columns:
                n0 = len(df)
                ds = pd.to_numeric(df["damage_severity"], errors="coerce").astype("Int64")
                df = df[ds.isin(levels)].copy().reset_index(drop=True)
                print(f"[DamageGate] damage_severity in {sorted(levels)}: {n0} -> {len(df)} (cond_gate_exception)")
            print("[CondGate] Skipped (no condition_score or bad bounds).")
            return df

    # >>> Leak-free PTVs for the 'slow' slice (TRAIN-only, per-row, embargo 1 day) BEFORE corridor gating
    try:
        # Identify core columns
        id_tr   = pick(train.columns, ["listing_id","listingid","id"])
        id_slow = pick(slow.columns,  ["listing_id","listingid","id"])
        id_vtxt = pick(vtxt.columns,  ["listing_id","listingid","id"])

        model_tr = pick(train.columns, ["model","variant","device_model"])
        model_sv = pick(slow.columns,  ["model","variant","device_model"]) or pick(vtxt.columns, ["model","variant","device_model"])
        storage_tr = pick(train.columns, ["storage","capacity","storage_gb","storage_norm","storage_num"])
        storage_sv = pick(slow.columns,  ["storage","capacity","storage_gb","storage_norm","storage_num"]) or pick(vtxt.columns, ["storage","capacity","storage_gb","storage_norm","storage_num"])
        price_tr  = pick(train.columns, ["sold_price","final_sold_price","sold_nok","sold_price_nok","soldprice"])
        price_sv  = pick(slow.columns,  ["price","ask","ask_price","ask_price_nok","price_nok"]) or pick(vtxt.columns, ["price","ask","ask_price","ask_price_nok","price_nok"])
        sold_tr   = pick(train.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
        sold_vt   = pick(vtxt.columns,  ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])

        # Canonicalize models
        if model_tr:
            train["__canon_model"] = train[model_tr].map(canon_model)
        if model_sv:
            slow["__canon_model"] = (slow[model_sv] if model_sv in slow.columns else vtxt[model_sv]).map(canon_model)

        # Merge row timestamps from val-text into slow (for per-row time-aware medians)
        if sold_vt and id_slow and id_vtxt:
            slow = slow.merge(
                vtxt[[id_vtxt, sold_vt]].drop_duplicates(id_vtxt),
                left_on=id_slow, right_on=id_vtxt, how="left", suffixes=("", "__join")
            )
            slow["__sold"] = pd.to_datetime(slow[sold_vt], errors="coerce", utc=True)

        # Build TRAIN pool (leak-free universe for overall 30D medians)
        pool_train = build_clean_pool(
            train,
            model_col="__canon_model" if "__canon_model" in train.columns else model_tr,
            edited_col=pick(train.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"]),
            sold_col=sold_tr,
            sold_price_col=price_tr
        )


        # ===== TRAIN-first merge of seller fields for the 20% holdout (slow) =====
        # These fields are static listing attributes; we prefer TRAIN to guarantee the exact row slice.
        seller_cols_train = []
        # Presence checks on TRAIN
        if "seller_rating" in train.columns:
            seller_cols_train.append("seller_rating")
        if "review_count" in train.columns:
            seller_cols_train.append("review_count")
        elif "ratings_count" in train.columns:
            seller_cols_train.append("ratings_count")
        if "member_since_year" in train.columns:
            seller_cols_train.append("member_since_year")
        elif "member_year" in train.columns:
            seller_cols_train.append("member_year")

        if seller_cols_train and id_tr and id_slow:
            # Left-join into 'slow' by ID; these columns will then flow into 'val' below
            try:
                slow = slow.merge(
                    train[[id_tr] + seller_cols_train].drop_duplicates(id_tr),
                    left_on=id_slow, right_on=id_tr,
                    how="left", suffixes=("", "__from_train")
                )
                print(f"[SellerMerge/TRAIN] merged {seller_cols_train} by {id_tr} into slow.")

                # Fallback: also pull seller fields from VTXΤ into slow and fill remaining NaNs
                seller_cols_vtxt = []
                if "seller_rating" in vtxt.columns:         seller_cols_vtxt.append("seller_rating")
                if "review_count" in vtxt.columns:          seller_cols_vtxt.append("review_count")
                elif "ratings_count" in vtxt.columns:       seller_cols_vtxt.append("ratings_count")
                if "member_since_year" in vtxt.columns:     seller_cols_vtxt.append("member_since_year")
                elif "member_year" in vtxt.columns:         seller_cols_vtxt.append("member_year")

                if seller_cols_vtxt and id_slow and id_vtxt:
                    try:
                        slow = slow.merge(
                            vtxt[[id_vtxt] + seller_cols_vtxt].drop_duplicates(id_vtxt),
                            left_on=id_slow, right_on=id_vtxt, how="left", suffixes=("", "__from_vtxt")
                        )
                        # fill NaNs in primary columns from the __from_vtxt columns
                        for c in ["seller_rating","review_count","ratings_count","member_since_year","member_year"]:
                            cv = f"{c}__from_vtxt"
                            if cv in slow.columns:
                                if c in slow.columns:
                                    slow[c] = slow[c].fillna(slow[cv])
                                else:
                                    slow[c] = slow[cv]
                                slow.drop(columns=[cv], inplace=True)
                        print("[SellerMerge/VTXT] fallback filled seller fields into slow.")
                    except Exception as e:
                        print("[SellerMerge/VTXT] WARN:", e)

            except Exception as e:
                print("[SellerMerge/TRAIN] WARN: TRAIN-first seller merge failed:", e)
        else:
            print("[SellerMerge/TRAIN] skipped: no seller cols in TRAIN or missing IDs.")
        # ========================================================================


        

        # ptv_overall(row): 30D median by model, as-of row_date - 1 day (embargo=1)
        if price_sv and model_sv and "__sold" in slow.columns and not pool_train.empty:
            slow["overall_median_price"] = rolling_price30_for_targets(
                target=slow, target_model_col="__canon_model", target_sold_col="__sold",
                pool=pool_train, embargo_days=1
            ).values
            slow["delta_from_overall_price"] = safe_num(slow.get(price_sv)) - safe_num(slow.get("overall_median_price"))
            slow["ptv_overall"]              = safe_num(slow.get(price_sv)) / safe_num(slow.get("overall_median_price"))
            print("[PTVGate] ptv_overall computed: TRAIN-only, 30D, embargo=1.")

        # ptv_anchor(row): 60D median by (model×storage), as-of row_date - 1 day (embargo=1)
        if price_tr and price_sv and model_tr and model_sv and storage_tr and storage_sv and "__sold" in slow.columns:
            pool_anchor = pd.DataFrame({
                "model":   train["__canon_model"] if "__canon_model" in train.columns else train[model_tr].map(canon_model),
                "storage": train[storage_tr],
                "_sold":   pd.to_datetime(train[sold_tr], errors="coerce", utc=True),
                "_sold_price": safe_num(train[price_tr]),
            }).dropna(subset=["model","storage","_sold","_sold_price"])
            pool_anchor = pool_anchor[pool_anchor["model"].isin(VARIANTS)]

            slow["anchor_price_60d"] = rolling_anchor60_for_targets(
                target=slow,
                target_model_col="__canon_model",
                target_storage_col=storage_sv if storage_sv in slow.columns else storage_sv,
                target_sold_col="__sold",
                pool=pool_anchor,
                embargo_days=1
            ).values
            slow["ptv_anchor"] = safe_num(slow.get(price_sv)) / safe_num(slow.get("anchor_price_60d"))
            print("[PTVGate] ptv_anchor computed: TRAIN-only, 60D, embargo=1.")
    except Exception as e:
        print("[PTVGate] WARN: TRAIN-only PTV precompute failed; corridor PTV gates may be skipped.", e)




    # Decide ordering
    if bool(globals().get("FILTER_PRED_SLOW_FIRST", False)) and ("pred_hours" in slow.columns):
        # 1) Filter predicted SLOW first
        n0 = len(slow)
        ph = pd.to_numeric(slow["pred_hours"], errors="coerce")
        dur_col = pick(slow.columns, ["duration_h","dur_hours","duration_hours"])
        if dur_col is None:
            ed = pick(slow.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])
            sd = pick(slow.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
            if ed and sd:
                dur = (pd.to_datetime(slow[sd], errors="coerce", utc=True) -
                       pd.to_datetime(slow[ed], errors="coerce", utc=True)).dt.total_seconds()/3600.0
                slow["duration_h"] = dur
                dur_col = "duration_h"
        if dur_col:
            slow = slow[(ph > 72.0) & (pd.to_numeric(slow[dur_col], errors="coerce").notna())].copy().reset_index(drop=True)
        else:
            slow = slow[(ph > 72.0)].copy().reset_index(drop=True)
        print(f"[FilterFirst] predicted_slow_first: {n0} -> {len(slow)}")

        # Optional guard: only assert if caller explicitly set EXPECTED_SLOW_COUNT
        exp = globals().get("EXPECTED_SLOW_COUNT", None)
        if globals().get("RUN_MODE", "apply") == "apply" and exp is not None:
            assert len(slow) == int(exp), (
                f"Unexpected slow count: {len(slow)} (expected {int(exp)}). Wrong input slice?"
            )


        # 2) Apply condition corridor within the predicted-slow slice (if requested)
        slow = _apply_cond_gate_local(slow, within_pred=True)

        # (Skip the later slow filter block)
        _skip_slow_filter_later = True
    else:
        # Original order: condition corridor first (if any), then predicted slow filter
        _skip_slow_filter_later = False
        slow = _apply_cond_gate_local(slow, within_pred=False)

    # filter to SOLD & predicted SLOW (pred_hours>72) — skipped if we filtered first
    if not bool(globals().get("FILTER_PRED_SLOW_FIRST", False)) or not locals().get("_skip_slow_filter_later", False):
        if "pred_hours" in slow.columns:
            n0 = len(slow)
            ph = pd.to_numeric(slow["pred_hours"], errors="coerce")
            dur_col = pick(slow.columns, ["duration_h","dur_hours","duration_hours"])
            if dur_col is None:
                ed = pick(slow.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])
                sd = pick(slow.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
                if ed and sd:
                    dur = (pd.to_datetime(slow[sd], errors="coerce", utc=True) -
                           pd.to_datetime(slow[ed], errors="coerce", utc=True)).dt.total_seconds()/3600.0
                    slow["duration_h"] = dur
                    dur_col = "duration_h"
            if dur_col:
                slow = slow[(ph > 72.0) & (pd.to_numeric(slow[dur_col], errors="coerce").notna())].copy().reset_index(drop=True)
            else:
                slow = slow[(ph > 72.0)].copy().reset_index(drop=True)
            print(f"[Filter] SOLD & predicted SLOW rows: {n0} → {len(slow)}")
        else:
            print("[Warn] 'pred_hours' absent in slow_val; NOT filtering by predicted SLOW.")
            dur_col = pick(slow.columns, ["duration_h","dur_hours","duration_hours"])
# ids & cols
    id_tr   = pick(train.columns, ["listing_id","listingid","id"])
    id_slow = pick(slow.columns,  ["listing_id","listingid","id"])
    id_vtxt = pick(vtxt.columns,  ["listing_id","listingid","id"])
    if not id_tr or not id_slow or not id_vtxt: raise SystemExit("Need an ID column (listing_id/id).")

    model_tr_orig = pick(train.columns, ["model","variant","device_model"])
    model_sv_orig = pick(slow.columns,  ["model","variant","device_model"]) or pick(vtxt.columns, ["model","variant","device_model"])
    train["__canon_model"] = train[model_tr_orig].map(canon_model) if model_tr_orig else ""
    slow["__canon_model"]  = slow[model_sv_orig].map(canon_model)  if model_sv_orig else (slow.get("model") if "model" in slow else "")
    vtxt["__canon_model"]  = vtxt[model_sv_orig].map(canon_model)  if model_sv_orig else (vtxt.get("model") if "model" in vtxt else "")
    model_tr = "__canon_model"; model_sv = "__canon_model"

    # dates/price/storage
    edited_tr = pick(train.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])
    sold_tr   = pick(train.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
    price_tr  = pick(train.columns, ["price","ask","ask_price"])
    price_sv  = pick(slow.columns,  ["price","ask","ask_price"]) or pick(vtxt.columns, ["price","ask","ask_price"])
    storage_tr = pick(train.columns, ["storage","capacity","storage_gb","storage_norm","storage_num"])
    storage_sv = pick(slow.columns,  ["storage","capacity","storage_gb","storage_norm","storage_num"]) or pick(vtxt.columns, ["storage","capacity","storage_gb","storage_norm","storage_num"])
    sold_price_tr = pick(train.columns, ["sold_price","final_sold_price","sold_nok","sold_price_nok","soldprice"])
    sold_price_va = pick(vtxt.columns,  ["sold_price","final_sold_price","sold_nok","sold_price_nok","soldprice"])

    # duration cols
    dur_tr = pick(train.columns, ["duration_h","dur_hours","duration_hours"])
    if dur_tr is None:
        edited_c = pick(train.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])
        sold_c   = pick(train.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
        if edited_c and sold_c:
            dt_edit = pd.to_datetime(train[edited_c], errors="coerce", utc=True)
            dt_sold = pd.to_datetime(train[sold_c],   errors="coerce", utc=True)
            train["__dur_h_calc"] = (dt_sold - dt_edit).dt.total_seconds() / 3600.0
            dur_tr = "__dur_h_calc"
        else:
            raise SystemExit("No duration column and cannot compute from dates.")
    dur_sv = pick(slow.columns,  ["duration_h","dur_hours","duration_hours"])

    # OOF → TRAIN pred_hours (optional)
    if oof_path and os.path.exists(oof_path):
        oof = pd.read_csv(oof_path)
        id_oof = pick(oof.columns, ["listing_id","listingid","id"])
        ph_oof = pick(oof.columns, ["pred_hours","pred_hours_oof","oof_pred_hours","pred_hours_enriched"])
        if id_oof and ph_oof:
            train = train.merge(oof[[id_oof, ph_oof]].drop_duplicates(id_oof),
                                left_on=id_tr, right_on=id_oof, how="left")
            train.rename(columns={ph_oof:"pred_hours"}, inplace=True)

    train_lab = train.dropna(subset=[dur_tr]).copy()


    # BatteryGate on TRAIN (same corridor as val)
    bmin = globals().get("BATT_GATE_MIN", None)
    bmax = globals().get("BATT_GATE_MAX", None)
    bkeep = bool(globals().get("BATT_KEEP_NA", False))
    if (bmin is not None) or (bmax is not None):
        if "battery_pct" in train_lab.columns:
            n_t0 = len(train_lab)
            bp_t = pd.to_numeric(train_lab["battery_pct"], errors="coerce")
            mask_bt = np.ones(len(train_lab), dtype=bool)
            if bmin is not None: mask_bt &= (bp_t >= float(bmin))
            if bmax is not None: mask_bt &= (bp_t <  float(bmax))
            if not bkeep:
                mask_bt &= bp_t.notna()
            train_lab = train_lab[mask_bt].copy().reset_index(drop=True)
            print(f"[BattGate][TRAIN] battery_pct in [{bmin if bmin is not None else '-inf'}, {bmax if bmax is not None else '+inf'})"
                  f"{' +keep_na' if bkeep else ''}: {n_t0} -> {len(train_lab)}")
        else:
            print("[BattGate][TRAIN] Skipped: 'battery_pct' not found.")


    # embeddings selection
    emb_cfg = EMB_CFGS.get(emb_choice, EMB_CFGS["32"])
    emb_cols_train = find_emb_cols_any(train, emb_cfg)
    emb_cols_vtxt  = find_emb_cols_any(vtxt,  emb_cfg)
    emb_cols_use = [c for c in emb_cols_train if c in emb_cols_vtxt]

    # merge extras+embs to slow slice
    EXTRA_PULL = [
        "__canon_model",
        "warranty_flag_text","receipt_flag_text","operator_lock_flag_text",
        "negotiation_flag_text","fast_pris_flag_text","accessory_count_text","age_months_text",
        "seller_rating","review_count","ratings_count","member_since_year","member_year",
        "price_per_gb","ppg_ratio","price_pe_rgb"
    ]
    present_extras = []
    for nm in EXTRA_PULL:
        c = pick(vtxt.columns, [nm])
        if c and c not in present_extras: present_extras.append(c)

    merge_cols = [id_vtxt] + present_extras + emb_cols_use
    merge_cols = [c for c in merge_cols if c in vtxt.columns]
    val = slow.merge(
        vtxt[merge_cols].drop_duplicates(id_vtxt),
        left_on=id_slow, right_on=id_vtxt, how="left", suffixes=("", "__emb")
    )

    # pools + rolling medians
    pool_train = build_clean_pool(train, model_tr_orig, edited_tr, sold_tr, sold_price_tr)
    sold_val_col   = pick(vtxt.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
    edited_val_col = pick(vtxt.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])
    pool_val_add = build_clean_pool(vtxt, model_sv_orig, edited_val_col, sold_val_col, sold_price_va) if sold_price_va else pd.DataFrame(columns=["model","_edited","_sold","_sold_price","duration_h"])
    pool_val = pd.concat([pool_train, pool_val_add], ignore_index=True) if not pool_val_add.empty else pool_train

    sold_train_for_target = pick(train_lab.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"]) or sold_tr
    train_lab["anchor_time14_model_row"] = rolling_anchor14_for_targets(
        target=train_lab.assign(__sold=pd.to_datetime(train_lab[sold_train_for_target], errors="coerce", utc=True)),
        target_model_col="__canon_model", target_sold_col="__sold", pool=pool_train
    ).values

    sold_val_for_target = pick(val.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc"])
    if sold_val_for_target is None and sold_val_col:
        val["__sold"] = pd.to_datetime(vtxt[sold_val_col], errors="coerce", utc=True); sold_val_for_target = "__sold"
    elif sold_val_for_target is not None:
        val["__sold"] = pd.to_datetime(val[sold_val_for_target], errors="coerce", utc=True); sold_val_for_target = "__sold"

    val["anchor_time14_model_row"] = rolling_anchor14_for_targets(
        target=val, target_model_col="__canon_model", target_sold_col=sold_val_for_target if sold_val_for_target else id_slow, pool=pool_val
    ).values

# fast24_flagger3_public.py — Part 2/2  (CONTINUATION)

    # 30d price medians (SOLD)
    train_lab["overall_median_price"] = rolling_price30_for_targets(
        target=train_lab.assign(__sold_target=pd.to_datetime(
            train_lab[sold_train_for_target], errors="coerce", utc=True
        )),
        target_model_col="__canon_model",
        target_sold_col="__sold_target",
        pool=pool_train
    ).values
    if sold_val_for_target:
        val["overall_median_price"] = rolling_price30_for_targets(
            target=val,
            target_model_col="__canon_model",
            target_sold_col=sold_val_for_target,
            pool=pool_train   # STRICT TRAIN-ONLY for the 20% hold-out
        ).values
    else:
        val["overall_median_price"] = np.nan

    # price deltas & anchors by (model×storage)
    price_sv_eff = price_sv if price_sv in val.columns else pick(val.columns, ["price","ask","ask_price"])
    val["delta_from_overall_price"] = safe_num(val.get(price_sv_eff)) - safe_num(val.get("overall_median_price"))
    val["ptv_overall"]              = safe_num(val.get(price_sv_eff)) / safe_num(val.get("overall_median_price"))
    train_lab["delta_from_overall_price"] = safe_num(train_lab.get(price_tr)) - safe_num(train_lab.get("overall_median_price"))
    train_lab["ptv_overall"]              = safe_num(train_lab.get(price_tr)) / safe_num(train_lab.get("overall_median_price"))
    attach_anchor_price_by_model_storage(train_lab, val, price_tr, price_sv_eff, model_tr_orig, model_sv_orig, storage_tr, storage_sv)

    # delta anchor time vs pred
    if "anchor_time14_model_row" in train_lab.columns and "pred_hours" in train_lab.columns:
        train_lab["delta_anchor_time14_minus_pred"] = safe_num(train_lab["anchor_time14_model_row"]) - safe_num(train_lab["pred_hours"])
        train_lab["delta_pred_model_med14"]         = safe_num(train_lab["pred_hours"]) - safe_num(train_lab["anchor_time14_model_row"])
    else:
        train_lab["delta_anchor_time14_minus_pred"] = np.nan
        train_lab["delta_pred_model_med14"]         = np.nan

    pred_best_sv = pick(val.columns, ["pred_final_best","pred_hours","pred_hours_base","pred_hours_enriched"])
    if "anchor_time14_model_row" in val.columns and pred_best_sv:
        val["delta_anchor_time14_minus_pred"] = safe_num(val["anchor_time14_model_row"]) - safe_num(val[pred_best_sv])
        val["delta_pred_model_med14"]         = safe_num(val[pred_best_sv]) - safe_num(val["anchor_time14_model_row"])
    else:
        val["delta_anchor_time14_minus_pred"] = np.nan
        val["delta_pred_model_med14"]         = np.nan

    # features
    X_tr_df, X_va_df, emb_cols_used = harvest_features(train_lab, val, extra_emb_cols=emb_cols_use)
    y_tr = (safe_num(train_lab[dur_tr]) <= 24.0).astype(int).to_numpy()
    X_tr = X_tr_df.values.astype(float)
    X_va = X_va_df.values.astype(float)

    # text SVD
    svd_used = 0
    if SVD_ENABLE:
        Ztr, Zva, svd_used = compute_svd_arrays_from_text(
            train_lab=train_lab, vtxt=vtxt, val_like_df=val,
            id_vtxt=id_vtxt, id_val=id_slow, svd_dims=SVD_DIMS, max_features=SVD_TFIDF_MAX_FEATURES
        )
        if Ztr.shape[0] == len(X_tr) and Zva.shape[0] == len(X_va) and Ztr.shape[1] > 0:
            X_tr = np.hstack([X_tr, Ztr]); X_va = np.hstack([X_va, Zva])

    # train + calibrate
    comps = train_blend_and_calibrate(X_tr, y_tr)

    # predict + soft penalty
    p_va = predict_proba(X_va, comps)
    val_for_penalty = dedup_columns(val.copy())
    pen = soft_gate_penalty(val_for_penalty).values
    p_final = np.where(np.isfinite(p_va * pen), p_va * pen, 0.0)

    # hard-block models
    model_factor = val_for_penalty["__canon_model"].map(G_MODEL_PENALTIES).fillna(1.0).astype(float)
    hard_block_mask = model_factor.values <= 0.0
    if hard_block_mask.any(): p_final[hard_block_mask] = 0.0

    val_out = val.copy()
    val_out["prob_fast24"] = p_final
    val_out["fast24_rank_desc"] = val_out["prob_fast24"].rank(method="first", ascending=False, na_option="bottom").fillna(0).astype(int)

    # thresholding (global; label-free when requested; fallback top-K)
    flags_model = np.zeros(len(val_out), dtype=bool)
    out_sweep   = os.path.join(DEFAULT_OUT_DIR, "fast24_threshold_sweep_calibrated.csv")
    p = val_out["prob_fast24"].values.astype(float)

    if globals().get("NO_CURRENT_THRESHOLD", False):
        # LABEL-FREE THRESHOLDING (no current-slice durations used)
        fixed = globals().get("FIXED_PROB_THRESHOLD", None)
        if fixed is not None:
            t_global = float(fixed)
        else:
            share = globals().get("MAX_PRED_SHARE", None)
            if share is not None:
                share = float(share)
                ps = np.sort(p)
                # choose cutoff so that predicted share <= MAX_PRED_SHARE
                k = int(np.clip(np.floor((1.0 - share) * max(len(ps) - 1, 0)), 0, max(len(ps) - 1, 0)))
                t_global = float(ps[k]) if len(ps) else 1.0
            else:
                # safe default cutoff if none provided
                t_global = 0.70

        thr_vec = np.full(len(val_out), t_global, dtype=float)
        flags_model = (p >= thr_vec)

    else:
        # ORIGINAL LABEL-BASED SWEEP (uses current-slice durations)
        if dur_sv and dur_sv in val_out.columns:
            y_true_all = (safe_num(val_out[dur_sv]) <= 24.0).astype(int).to_numpy()
            mask_lab = ~np.isnan(safe_num(val_out[dur_sv]))
            sweep = threshold_sweep(y_true_all[mask_lab], p[mask_lab], steps=threshold_steps)
            if write_outputs:
                sweep.to_csv(out_sweep, index=False)
            t_global = choose_threshold_with_constraints(sweep, PREC_MIN, REC_MIN, MAX_SHARE)

            thr_vec = np.full(len(val_out), t_global, dtype=float)
            if "__canon_model" in val_out.columns:
                for m in val_out["__canon_model"].dropna().unique():
                    mask_m = (val_out["__canon_model"] == m)
                    if (mask_m & mask_lab).sum() >= PER_MODEL_MIN_N:
                        y_true_m = (safe_num(val_out.loc[mask_m, dur_sv]) <= 24).to_numpy(dtype=int)
                        p_m = val_out.loc[mask_m, "prob_fast24"].to_numpy()
                        sw_m = threshold_sweep(y_true_m, p_m, steps=threshold_steps)
                        t_m = choose_threshold_with_constraints(sw_m, PREC_MIN, REC_MIN, MAX_SHARE)
                        thr_vec[mask_m.values] = t_m
            flags_model = (p >= thr_vec)
        else:
            # No duration column available: fallback to fixed cutoff
            thr_vec = np.full(len(val_out), 0.70, dtype=float)
            flags_model = (p >= thr_vec)

    # Fallback top-K underpriced if nothing passes (target-free)
    if flags_model.sum() == 0 and TOPK_FALLBACK and TOPK_FALLBACK > 0:
        d_over = safe_num(val_out.get("delta_from_overall_price", np.nan)).fillna(0.0)
        mask_under = (d_over <= -50).values
        order = np.argsort(-p)  # highest probs first
        pick_idx = [i for i in order if mask_under[i]][:TOPK_FALLBACK]
        if pick_idx:
            flags_model[pick_idx] = True


    # Rule-1 fast-lane
    cond_col = pick(val_out.columns, ["condition_score","cond_score"])
    dmg_col  = pick(val_out.columns, ["damage_severity","damage_level"])
    rule1 = (
        (safe_num(val_out.get("delta_pred_model_med14")) < 33) &
        (safe_num(val_out.get(cond_col))   >= 0.7 if cond_col else True) &
        (safe_num(val_out.get(dmg_col))    <= 1   if dmg_col  else True) &
        (safe_num(val_out.get("delta_from_overall_price")) <= 750)
    )
    if "__canon_model" in val_out.columns:
        block_mask = val_out["__canon_model"].map(G_MODEL_PENALTIES).fillna(1.0).astype(float).values <= 0.0
        if block_mask.any(): rule1 = rule1 & (~block_mask)

    val_out["flag_fastlane"] = rule1.astype(int)
    val_out["is_fast24_pred"] = (val_out["flag_fastlane"].astype(bool) | flags_model.astype(bool)).astype(int)

    # write outputs
    out_full_all  = os.path.join(DEFAULT_OUT_DIR, "slow_bucket_pred_fast24_with_fastflag.csv")
    out_flag_full = os.path.join(DEFAULT_OUT_DIR, "slow_bucket_pred_fast24_flagged_full.csv")
    out_flag_min  = os.path.join(DEFAULT_OUT_DIR, "slow_bucket_pred_fast24_flagged_min.csv")
    if write_outputs:
        val_out.to_csv(out_full_all, index=False)
        val_out[val_out["is_fast24_pred"] == 1].to_csv(out_flag_full, index=False)

    batt_col = pick(val_out.columns, ["battery_pct","battery_percent","battery"])
    stor_txt = pick(val_out.columns, ["storage","capacity"])
    stor_num = pick(val_out.columns, ["storage_gb","storage_norm","storage_num"])
    pred_best_sv = pick(val_out.columns, ["pred_final_best","pred_hours","pred_hours_base","pred_hours_enriched"])

    # minimal export set for inspection
    want_cols = []
    for c in [
        id_slow, model_sv, model_sv_orig,
        cond_col, dmg_col, price_sv_eff,
        "overall_median_price","delta_from_overall_price","ptv_overall",
        "ptv_anchor","delta_anchor_price",
        "anchor_time14_model_row","delta_anchor_time14_minus_pred","delta_pred_model_med14",
        pred_best_sv, batt_col, (stor_txt or stor_num),
        dur_sv, "flag_fastlane","is_fast24_pred","prob_fast24","fast24_rank_desc"
    ]:
        if c and (c in val_out.columns) and (c not in want_cols):
            want_cols.append(c)

    rename_map = {
        id_slow: "listing_id",
        model_sv: "model_canonical",
        model_sv_orig: "model_original",
        cond_col: "condition_score",
        dmg_col: "damage_severity",
        batt_col: "battery_pct",
        price_sv_eff: "price",
        "overall_median_price": "overall_median_price",
        "delta_from_overall_price": "delta_from_overall_price",
        "ptv_overall": "ptv_overall",
        "ptv_anchor": "ptv_anchor",
        "delta_anchor_price": "delta_anchor_price",
        "anchor_time14_model_row": "median_dur_14d_before_h",
        "delta_anchor_time14_minus_pred": "delta_anchor_time14_minus_pred",
        "delta_pred_model_med14": "delta_time_pred_minus_median14h",
        pred_best_sv: "pred_hours_or_best",
        (stor_txt or stor_num): "storage",
        dur_sv: "duration_h",
        "flag_fastlane": "flag_fastlane",
        "is_fast24_pred": "is_fast24_pred",
        "prob_fast24": "prob_fast24",
        "fast24_rank_desc": "fast24_rank_desc",
    }
    flagged_min_df = (
        val_out[val_out["is_fast24_pred"] == 1][want_cols]
        .rename(columns={k: v for k, v in rename_map.items() if k in want_cols})
    )
    if write_outputs:
        flagged_min_df.to_csv(out_flag_min, index=False)

    # per-model probability report
    if "__canon_model" in val_out.columns and write_outputs:
        per_model = (
            val_out.groupby("__canon_model")["prob_fast24"]
            .mean()
            .reset_index()
            .sort_values("prob_fast24", ascending=False)
        )
        per_model.to_csv(os.path.join(DEFAULT_OUT_DIR, "per_model_probabilities_fast24.csv"), index=False)

    # HARD FILTER exports (optional reporting only)
    if HARD_FILTER_ENABLE and STAGE1_HF_EXPORT and write_outputs:
        df_flagged_full = val_out[val_out["is_fast24_pred"] == 1].copy()
        prob_ok = df_flagged_full["prob_fast24"] >= float(HF_MIN_PROB_FAST24)
        cond_series = safe_num(df_flagged_full.get(cond_col, np.nan)) if cond_col else pd.Series(np.nan, index=df_flagged_full.index)
        dmg_series  = safe_num(df_flagged_full.get(dmg_col,  np.nan)) if dmg_col  else pd.Series(np.nan, index=df_flagged_full.index)
        mask_damage_drop = (dmg_series >= float(HF_DROP_DAMAGE_GTE)) & (
            (cond_series >= float(HF_DROP_COND_MIN)) | ((cond_series == 0.0) if HF_DROP_COND_EQ0_TOO else False)
        )
        mask_keep = prob_ok & (~mask_damage_drop)
        hard_full = df_flagged_full[mask_keep].copy()
        out_hf_full = os.path.join(DEFAULT_OUT_DIR, HF_OUT_FULL_FILENAME)
        hard_full.to_csv(out_hf_full, index=False)
        hard_min = hard_full[want_cols].rename(columns={k: v for k, v in rename_map.items() if k in want_cols})
        out_hf_min = os.path.join(DEFAULT_OUT_DIR, HF_OUT_MIN_FILENAME)
        hard_min.to_csv(out_hf_min, index=False)
        print(f"[HARD FILTER][Stage-1 export] Kept {len(hard_full)} / {len(df_flagged_full)} flagged rows.")
        print("Wrote HARD-FILTERED FULL:", out_hf_full)
        print("Wrote HARD-FILTERED MIN :", out_hf_min)

    # [UNION] metrics ≤24h (info & tuning target)
    metrics = {}
    if dur_sv and dur_sv in val_out.columns:
        y_true = (safe_num(val_out[dur_sv]) <= 24.0).astype(int).to_numpy()
        y_hat  = val_out["is_fast24_pred"].astype(int).to_numpy()
        metrics = metrics_union(y_true, y_hat)
        print(f"[UNION] Flags={metrics['flags']} / {metrics['n']} | "
              f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} F1={metrics['F1']:.4f} Acc={metrics['accuracy']:.4f} "
              f"TP={metrics['TP']} FP={metrics['FP']} TN={metrics['TN']} FN={metrics['FN']} | "
              f"(fastlane={int(val_out.get('flag_fastlane', pd.Series(0)).sum())})")

    # 40/72 post-selection eval
    cond_col_here = pick(val_out.columns, ["condition_score","cond_score"])
    dmg_col_here  = pick(val_out.columns, ["damage_severity","damage_level"])
    eval_4072 = eval_40_72(val_out, cond_col_here, dmg_col_here, dur_sv)

    # baseline regression (informational only)
    model_pred_col = preferred_model_pred_col(val_out)
    baseline_reg = {}

    # summary
    if write_outputs:
        summary = {
            "train_rows": int(len(train_lab)),
            "val_rows": int(len(val_out)),
            "flagged_rows": int((val_out["is_fast24_pred"] == 1).sum()),
            "features_train": int(X_tr.shape[1]),
            "features_val": int(X_va.shape[1]),
            "svd_dims_used": int(svd_used),
            "emb_choice": emb_choice,
            "duration_train_col": dur_tr,
            "duration_val_col": dur_sv,
            "rule1_fastlane_kept": int(val_out.get("flag_fastlane", pd.Series(0)).sum()),
            "calibration": CALIBRATION,
            "metrics_union_24h": metrics,
            "metrics_40vs72_postHF": eval_4072,
            "baseline_regression": baseline_reg,
            "outputs": {
                "full_all_with_fastflag": out_full_all,
                "flagged_full": out_flag_full,
                "flagged_min": out_flag_min,
                "sweep": out_sweep,
                **({
                    "hardfilter_full": os.path.join(DEFAULT_OUT_DIR, HF_OUT_FULL_FILENAME),
                    "hardfilter_min": os.path.join(DEFAULT_OUT_DIR, HF_OUT_MIN_FILENAME),
                } if HARD_FILTER_ENABLE and STAGE1_HF_EXPORT else {})
            }
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    return {
        "val_out": val_out,
        "flagged_min_df": flagged_min_df,
        "dur_col": dur_sv,
        "cond_col": cond_col,
        "dmg_col": dmg_col,
        "model_pred_col": model_pred_col,
        "union": metrics,
        "eval_40_72": eval_4072,
        "baseline_regression": baseline_reg,
        "outputs": dict(full_all=out_full_all, flagged_full=out_flag_full, flagged_min=out_flag_min)
    }

# ───────────────────────── Stage-2 (LEAK-FREE): mask-23 tuner (INLINE-capable, with Optuna hush in silent mode)
def run_mask23_tuning(val_out: pd.DataFrame,

                      dur_col: Optional[str],
                      model_pred_col: Optional[str],
                      out_dir: str,
                      n_trials: int = 200,
                      *,
                      silent: bool = False,
                      write_outputs: bool = True) -> Dict[str, Any]:
    """
    Leak-free Stage-2 tuning: tune on early flagged rows (using their durations),
    apply learned policy to later flagged rows WITHOUT reading their durations.
    When `silent=True` and `write_outputs=False`, this is safe to call inside Stage-1 trials.
    """
    # ensure _delta23 present for eligibility/d23 gating
    try:
        if '_delta23' not in val_out.columns:
            pcol = model_pred_col if (model_pred_col in val_out.columns) else preferred_model_pred_col(val_out)
            if pcol:
                val_out['_delta23'] = (safe_num(val_out[pcol]) - 23.0).clip(lower=0.0)
    except Exception:
        pass

    if dur_col is None or model_pred_col is None or dur_col not in val_out.columns or model_pred_col not in val_out.columns:
        if not silent:
            print("[WARN] mask-23 tuner: missing duration or model prediction column; skip.")
        return {}

    # ---------- helpers ----------
    def _pick_time_col(df: pd.DataFrame) -> Optional[str]:
        sold = pick(df.columns, ["sold_date","sold_utc","sold_date_utc","sold_time","sold_dt","sold_ts_utc","__sold"])
        if sold: return sold
        return pick(df.columns, ["edited_date_utc","edited_ts_utc","edited_date","edited_utc"])

    def _metrics_regression_local(y_true, y_pred):
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(m):
            return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
        yt = yt[m]; yp = yp[m]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt))**2))
        r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan
        return {"n": int(len(yt)), "mae": mae, "rmse": rmse, "r2": r2}

    def _flagged_indices(df: pd.DataFrame) -> np.ndarray:
        return np.where(df.get("is_fast24_pred", 0).astype(int).to_numpy(dtype=bool))[0]

    def _eligible_mask_nodur(df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        n = len(df)
        p_all = df.get("prob_fast24", pd.Series(np.ones(n))).to_numpy(dtype=float)
        use_prob = p_all >= float(params.get("p_threshold", 0.20))

        d_over_ok = np.ones(n, dtype=bool)
        if "delta_from_overall_price" in df.columns:
            d_over_ok = safe_num(df["delta_from_overall_price"]).to_numpy(dtype=float) <= float(params.get("delta_overall_max", 50.0))

        ptv_ok = np.ones(n, dtype=bool)
        if "ptv_overall" in df.columns:
            ptv_ok = safe_num(df["ptv_overall"]).to_numpy(dtype=float) <= float(params.get("ptv_overall_max", 1.05))
        

        anchor_ok = np.ones(n, dtype=bool)
        if "ptv_anchor" in df.columns:
            anchor_ok = safe_num(df["ptv_anchor"]).to_numpy(dtype=float) <= float(params.get("ptv_anchor_max", 1.05))


        cond_ok = np.ones(n, dtype=bool)
        if "condition_score" in df.columns:
            cond_ok = safe_num(df["condition_score"]).to_numpy(dtype=float) >= float(params.get("cond_min", 0.0))

        dmg_ok = np.ones(n, dtype=bool)
        if "damage_severity" in df.columns:
            dmg_ok = safe_num(df["damage_severity"]).to_numpy(dtype=float) <= float(params.get("damage_max", 1.8))

        batt_ok = np.ones(n, dtype=bool)
        if "battery_pct" in df.columns:
            batt_ok = safe_num(df["battery_pct"]).to_numpy(dtype=float) >= float(params.get("batt_min", 0.0))

        dpm_ok = np.ones(n, dtype=bool)
        if "delta_pred_model_med14" in df.columns:
            dpm_ok = safe_num(df["delta_pred_model_med14"]).to_numpy(dtype=float) <= float(params.get("dpm14_max", 40.0))

        seller_ok = np.ones(n, dtype=bool)

        # seller rating (NaN -> pass)
        if "seller_rating" in df.columns:
            sr_min = float(params.get("seller_min_rating", -1e9))
            sr = safe_num(df["seller_rating"]).to_numpy(dtype=float)
            seller_ok &= np.where(np.isfinite(sr), sr >= sr_min, True)

        # review_count OR ratings_count (NaN -> pass)
        rc_min = float(params.get("seller_min_reviews", -1e9))
        if rc_min > -1e8:
            if "review_count" in df.columns:
                rc = safe_num(df["review_count"]).to_numpy(dtype=float)
                seller_ok &= np.where(np.isfinite(rc), rc >= rc_min, True)
            elif "ratings_count" in df.columns:
                rc = safe_num(df["ratings_count"]).to_numpy(dtype=float)
                seller_ok &= np.where(np.isfinite(rc), rc >= rc_min, True)

        # age years (NaN -> pass)
        age_min = float(params.get("seller_min_age_y", -1e9))
        if age_min > -1e8:
            msy = safe_num(df.get("member_since_year", df.get("member_year"))).to_numpy(dtype=float)
            age = 2025.0 - msy
            age = np.where(np.isfinite(age), np.maximum(0.0, age), np.nan)
            seller_ok &= np.where(np.isfinite(age), age >= age_min, True)

        # combined quality (NaN -> pass)
        q_min = float(params.get("seller_quality_min", -1e9))
        if q_min > -1e8:
            sq = compute_seller_quality(df).to_numpy(dtype=float)
            seller_ok &= np.where(np.isfinite(sq), sq >= q_min, True)



        return (
            use_prob & d_over_ok & ptv_ok & anchor_ok & cond_ok & dmg_ok & batt_ok & dpm_ok
            & (df['__hf_keep'].astype(bool).to_numpy() if ('__hf_keep' in df.columns) else np.ones(n, dtype=bool))
            & (safe_num(df['_delta23']).to_numpy(dtype=float) <= float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max'])))
        ) if ('_delta23' in df.columns) else (
            use_prob & d_over_ok & ptv_ok & anchor_ok & cond_ok & dmg_ok & batt_ok & dpm_ok & seller_ok
        )



    def _select_train_with_caps(df: pd.DataFrame, cand_idx: np.ndarray, params: Dict[str, Any],
                                rank_scores, dur_all: np.ndarray,
                                slow_cap_abs: int, slow_cap_ratio: float) -> np.ndarray:
        """Greedy selection by score that ENFORCES slow caps on TRAIN (uses true durations)."""
        n = len(df)
        out = np.zeros(n, dtype=bool)
        if cand_idx.size == 0:
            return out
        # Scores
        if rank_scores is None:
            scores = df.get("prob_fast24", pd.Series(np.ones(n))).to_numpy(dtype=float)
        else:
            scores = rank_scores
        # Target K from flip_share (but we may pick fewer to respect caps)
        flip_share = float(params.get("flip_share", 0.40))
        flip_share = max(float(globals().get("MASK23_MIN_SHARE", 0.02)), min(float(globals().get("MASK23_MAX_SHARE", 0.75)), flip_share))
        k = int(np.ceil(flip_share * cand_idx.size))
        # Order candidates by score desc
        order = cand_idx[np.argsort(-scores[cand_idx])]
        sel = []
        slow_cnt = 0
        for idx in order:
            dur = float(dur_all[idx])
            will_be_slow = (dur > 72.0)
            # If taking this would violate absolute or ratio cap, skip
            if will_be_slow and (slow_cnt + 1) > int(slow_cap_abs):
                continue
            new_total = len(sel) + 1
            new_slow_ratio = (slow_cnt + (1 if will_be_slow else 0)) / float(new_total)
            if new_slow_ratio > float(slow_cap_ratio):
                continue
            sel.append(idx)
            if will_be_slow:
                slow_cnt += 1
            if len(sel) >= k:
                break
        if len(sel) == 0:
            return out  # zero allowed; objective will handle
        out[np.array(sel, dtype=int)] = True
        return out

    def _select_flips_over_candidates(df: pd.DataFrame, cand_idx: np.ndarray, params: Dict[str, Any], rank_scores=None) -> np.ndarray:
        n = len(df)
        out = np.zeros(n, dtype=bool)
        if cand_idx.size == 0:
            return out
        eligible = _eligible_mask_nodur(df, params)

        # ranking source: value-function scores if provided, else calibrated prob
        if rank_scores is None:
            scores = df.get("prob_fast24", pd.Series(np.ones(n))).to_numpy(dtype=float)
        else:
            scores = rank_scores

        # size with floors
        min_flips = max(int(np.ceil(MASK23_MIN_SHARE * cand_idx.size)), int(MASK23_MIN_FLIPS))
        flip_share = float(params.get("flip_share", 0.40))
        flip_share = max(MASK23_MIN_SHARE, min(MASK23_MAX_SHARE, flip_share))
        k = int(np.ceil(flip_share * cand_idx.size))
        k = max(min_flips, min(k, cand_idx.size))

        elig_idx = cand_idx[eligible[cand_idx]]
        if elig_idx.size >= k:
            keep = elig_idx[np.argsort(-scores[elig_idx])][:k]
            out[keep] = True
            return out

        # backfill by score from remaining candidates
        keep_list = []
        if elig_idx.size > 0:
            keep_list.append(elig_idx)
        need = k - (elig_idx.size if elig_idx.size > 0 else 0)
        if need > 0:
            remain = np.setdiff1d(cand_idx, elig_idx, assume_unique=False)
            if remain.size > 0:
                add = remain[np.argsort(-scores[remain])][:need]
                if add.size > 0:
                    keep_list.append(add)
        if keep_list:
            out[np.concatenate(keep_list)] = True
            return out

        # last resort: top-k by score
        out[cand_idx[np.argsort(-scores[cand_idx])][:k]] = True
        return out
        keep_list = []
        if elig_idx.size > 0: keep_list.append(elig_idx)
        need = k - (elig_idx.size if elig_idx.size > 0 else 0)
        if need > 0:
            remain = np.setdiff1d(cand_idx, elig_idx, assume_unique=False)
            if remain.size > 0:
                add = remain[np.argsort(-scores[remain])][:need]
                if add.size > 0: keep_list.append(add)
        if keep_list:
            out[np.concatenate(keep_list)] = True
            return out

        # last resort: top-k by prob
        out[cand_idx[np.argsort(-scores[cand_idx])][:k]] = True
        return out

    # ---------- split by time (earliest = train) ----------
    flagged_idx = _flagged_indices(val_out)
    if flagged_idx.size == 0:
        if not silent:
            print("[WARN] mask-23 tuner: no flagged rows; nothing to tune/apply.")
        return {}

    tcol = _pick_time_col(val_out)
    if tcol:
        tser = pd.to_datetime(val_out[tcol], errors="coerce", utc=True)
        ts_key = tser.fillna(pd.Timestamp(0, tz="UTC")).view("int64").to_numpy()
    else:
        ts_key = np.arange(len(val_out), dtype=np.int64)
    flagged_sorted = flagged_idx[np.argsort(ts_key[flagged_idx])]

    split_k = int(np.floor(MASK23_TRAIN_FRAC * flagged_sorted.size))
    train_idx = flagged_sorted[:split_k]
    apply_idx = flagged_sorted[split_k:]

    # fallback if tiny
    if train_idx.size == 0 or apply_idx.size == 0:
        train_idx = flagged_sorted[:max(1, min(10, flagged_sorted.size // 3))]
        apply_idx = flagged_sorted[max(1, min(10, flagged_sorted.size // 3)):]
        if apply_idx.size == 0:
            apply_idx = flagged_sorted

    # ---------- per-trial telemetry store ----------
    trial_rows = []
    trials_csv = os.path.join(out_dir, "mask23_trials_log.csv")

    # baseline vectors
    dur_all = safe_num(val_out[dur_col]).to_numpy(dtype=float)
    pred_col = model_pred_col if (model_pred_col in val_out.columns) else preferred_model_pred_col(val_out)
    if pred_col is None or pred_col not in val_out.columns:
        pred_col = "pred_hours"
    model_pred_all = safe_num(val_out[pred_col]).to_numpy(dtype=float)

    def _objective_on_train(params: Dict[str, Any], trial_number: Optional[int]) -> float:
        # Prune trivial/degenerate trials early (tiny corridor or too few flagged rows)
        try:
            import optuna as _optuna_for_prune  # optional
            _min_rows = int(globals().get("PRUNE_MIN_ROWS", 40))
            _min_flag = int(globals().get("PRUNE_MIN_FLAGGED", 20))
            if (_optuna_for_prune is not None):
                if (len(val_out) < _min_rows) or (flagged_idx.size < _min_flag):
                    raise _optuna_for_prune.TrialPruned(f"Too few rows/flags for Stage-2: rows={len(val_out)} flagged={flagged_idx.size}")
        except Exception:
            pass

        # ---------- S2 ranker: train on EARLY flagged rows (leak-free) ----------
        rank_scores_all = None
        s2_weight_vec = None
        slow_model = None
        try:
            if S2_USE_LGBM and (lgb is not None):
                X_all_df, s2_feat_cols = build_stage2_features(val_out, pred_col, include_emb=S2_USE_EMB)
                X_all = X_all_df.values.astype(float)
                y_slow_train = (dur_all[train_idx] > 72.0).astype(int)
                X_train = X_all[train_idx]
                if np.unique(y_slow_train).size >= 2 and len(train_idx) >= 30:
                    hp = dict(S2_LGBM_PARAMS)
                    # per-trial overrides
                    for ksrc, ktgt in [
                        ('s2_num_leaves','num_leaves'),
                        ('s2_max_depth','max_depth'),
                        ('s2_min_data_in_leaf','min_data_in_leaf'),
                        ('s2_feature_fraction','feature_fraction'),
                        ('s2_bagging_fraction','bagging_fraction'),
                        ('s2_lambda_l2','lambda_l2'),
                        ('s2_learning_rate','learning_rate'),
                        ('s2_n_estimators','n_estimators'),
                    ]:
                        if ksrc in params:
                            hp[ktgt] = params[ksrc]
                    slow_model = lgb.LGBMClassifier(**hp)
                    slow_model.fit(X_train, y_slow_train)
                    p_slow_all = slow_model.predict_proba(X_all)[:, 1]
                    # value-function terms
                    q  = np.clip(safe_num(val_out['prob_fast24']).to_numpy(dtype=float), 0.0, 1.0)
                    U  = X_all_df['_underprice_quality'].to_numpy(dtype=float)
                    d23 = X_all_df['_delta23'].to_numpy(dtype=float)
                    eps = 1e-9
                    Uz = (U - np.nanpercentile(U, 5)) / (np.nanpercentile(U, 95) - np.nanpercentile(U, 5) + eps)
                    Uz = np.clip(Uz, 0.0, 1.0)
                    d23z = np.clip(d23 / float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max'])), 0.0, 1.0)
                    wf = float(params.get('s2_w_fast_raw', S2_VFUNC_DEFAULT['w_fast']))
                    ws = float(params.get('s2_w_slow_raw', S2_VFUNC_DEFAULT['w_slow_risk']))
                    wu = float(params.get('s2_w_util_raw', S2_VFUNC_DEFAULT['w_util']))
                    wd = float(params.get('s2_w_d23_raw', S2_VFUNC_DEFAULT['w_delta23']))
                    wsum = wf + ws + wu + wd + eps
                    w_fast, w_slow, w_util, w_d23 = wf/wsum, ws/wsum, wu/wsum, wd/wsum
                    s2_weight_vec = (w_fast, w_slow, w_util, w_d23)
                    ps = np.clip(p_slow_all, 0.0, 1.0)
                    rank_scores_all = (w_fast * q) - (w_slow * ps) + (w_util * Uz) - (w_d23 * d23z)
                    if not silent:
                        print(f"[Stage2][ranker] LGBM trained on EARLY: X={X_train.shape}, feats={len(s2_feat_cols)}, emb={S2_USE_EMB}")
                        print(f"[Stage2][ranker] weights: w_fast={w_fast:.3f}, w_slow={w_slow:.3f}, w_util={w_util:.3f}, w_d23={w_d23:.3f}, delta23_max={float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max'])):.1f}")
        except Exception as e:
            if not silent:
                print('[Stage2][ranker][WARN]', e)
            slow_model = None

        # Select flips on TRAIN using rank scores if available
        flips_mask = _select_flips_over_candidates
        # (patched) OOF slow-risk gating on TRAIN to mirror APPLY
        try:
            ps_oof_all = np.full(len(val_out), np.nan, dtype=float)
            if (S2_USE_LGBM and (lgb is not None)) and (len(train_idx) >= 30):
                X_all_df_oof, _ = build_stage2_features(val_out, pred_col, include_emb=S2_USE_EMB)
                X_all_oof = X_all_df_oof.values.astype(float)
                y_slow_tr = (dur_all[train_idx] > 72.0).astype(int)
                # time-ordered contiguous blocks
                B = max(2, int(min(max(2, int(globals().get("MASK23_CV_BLOCKS", 3))), 5)))
                splits = np.array_split(train_idx, B)
                for b, va_block in enumerate(splits):
                    tr_blocks = [blk for i, blk in enumerate(splits) if i != b]
                    tr_idx_fold = np.concatenate(tr_blocks) if tr_blocks else np.array([], dtype=int)
                    if tr_idx_fold.size == 0 or np.unique((dur_all[tr_idx_fold] > 72.0).astype(int)).size < 2:
                        continue
                    hp = dict(S2_LGBM_PARAMS)
                    for ksrc, ktgt in [('s2_num_leaves','num_leaves'),('s2_max_depth','max_depth'),('s2_min_data_in_leaf','min_data_in_leaf'),('s2_feature_fraction','feature_fraction'),('s2_bagging_fraction','bagging_fraction'),('s2_lambda_l2','lambda_l2'),('s2_learning_rate','learning_rate'),('s2_n_estimators','n_estimators')]:
                        if ksrc in params: hp[ktgt] = params[ksrc]
                    mdl = lgb.LGBMClassifier(**hp)
                    mdl.fit(X_all_oof[tr_idx_fold], (dur_all[tr_idx_fold] > 72.0).astype(int))
                    ps_oof_all[va_block] = mdl.predict_proba(X_all_oof[va_block])[:,1]
            # fallback: in-sample ps if OOF couldn't be built
            if np.isnan(ps_oof_all[train_idx]).all() and ('p_slow_all' in locals()):
                ps_oof_all[train_idx] = np.clip(p_slow_all[train_idx], 0.0, 1.0)
            # gate cand set by OOF risk
            gate = np.isfinite(ps_oof_all[train_idx]) & (ps_oof_all[train_idx] <= float(globals().get('S2_P_SLOW_MAX', 0.20)))
            cand_idx_train = train_idx[gate]
            if cand_idx_train.size == 0:
                # if nothing survives, keep original train_idx to avoid degenerate objective
                cand_idx_train = train_idx
        except Exception:
            cand_idx_train = train_idx

        flips_mask = _select_flips_over_candidates(val_out, cand_idx_train, params, rank_scores=rank_scores_all)
        picked = flips_mask.astype(bool)

        # enforce size floors on TRAIN
        min_need = max(int(np.ceil(MASK23_MIN_SHARE * train_idx.size)), MASK23_MIN_FLIPS)
        if picked.sum() < min_need:
            obj = 5e5 + 1e5 * (min_need - int(picked.sum()))
            if not silent:
                print(f"[Stage2][trial {trial_number}] undersized: flips={int(picked.sum())} need≥{min_need} | obj={obj:.1f}")
            return obj
        # ---------- S2 ranker: train on EARLY flagged rows (leak-free) ----------
        rank_scores_all = None
        s2_weight_vec = None
        slow_model = None
        try:
            if S2_USE_LGBM and (lgb is not None):
                X_all_df, s2_feat_cols = build_stage2_features(val_out, pred_col, include_emb=S2_USE_EMB)
                X_all = X_all_df.values.astype(float)
                y_slow_train = (dur_all[train_idx] > 72.0).astype(int)
                X_train = X_all[train_idx]
                if __import__('numpy').unique(y_slow_train).size >= 2 and len(train_idx) >= 30:
                    slow_model = lgb.LGBMClassifier(**S2_LGBM_PARAMS)
                    slow_model.fit(X_train, y_slow_train)
                    p_slow_all = slow_model.predict_proba(X_all)[:, 1]
    # (removed inner 'import numpy as np' – using global np)
                    eps = 1e-9
                    q  = np.clip(safe_num(val_out["prob_fast24"]).to_numpy(dtype=float), 0.0, 1.0)
                    ps = np.clip(p_slow_all, 0.0, 1.0)
                    U  = X_all_df["_underprice_quality"].to_numpy(dtype=float)
                    d23 = X_all_df["_delta23"].to_numpy(dtype=float)
                    Uz = (U - np.nanpercentile(U, 5)) / (np.nanpercentile(U, 95) - np.nanpercentile(U, 5) + eps)
                    Uz = np.clip(Uz, 0.0, 1.0)
                    d23z = np.clip(d23 / float(params.get("delta23_max", S2_VFUNC_DEFAULT["delta23_max"])), 0.0, 1.0)
                    wf = S2_VFUNC_DEFAULT["w_fast"]; ws = S2_VFUNC_DEFAULT["w_slow_risk"]; wu = S2_VFUNC_DEFAULT["w_util"]; wd = S2_VFUNC_DEFAULT["w_delta23"]
                    wsum = wf + ws + wu + wd + eps
                    w_fast, w_slow, w_util, w_d23 = wf/wsum, ws/wsum, wu/wsum, wd/wsum
                    s2_weight_vec = (w_fast, w_slow, w_util, w_d23)
                    V = (w_fast * q) - (w_slow * ps) + (w_util * Uz) - (w_d23 * d23z)
                    rank_scores_all = V
        except Exception:
            slow_model = None

        # Use rank scores if available for flip selection on TRAIN
        flips_mask = _select_flips_over_candidates(val_out, train_idx, params, rank_scores=rank_scores_all)


        # counts on TRAIN
        dur_tr = dur_all[train_idx]
        sel_tr = picked[train_idx]
        flips = int(sel_tr.sum())
        slow = int(((dur_tr > 72) & sel_tr).sum())
        fast = int(((dur_tr <= 40) & sel_tr).sum())
        mid  = int((((dur_tr > 40) & (dur_tr <= 72)) & sel_tr).sum())
        slow_ratio = slow / max(1, flips)
        share = flips / max(1, train_idx.size)

        # hard caps (absolute + ratio)
        if slow > MASK23_SLOW_CAP_ABS:
            obj = 1e6 + 1e5 * (slow - MASK23_SLOW_CAP_ABS)
            if not silent:
                print(f"[Stage2][trial {trial_number}] slow_abs cap hit: flips={flips} slow={slow} | obj={obj:.1f}")
            return obj
        if slow_ratio > MASK23_SLOW_CAP_RATIO:
            obj = 1e6 + 1e5 * (slow_ratio - MASK23_SLOW_CAP_RATIO)
            if not silent:
                print(f"[Stage2][trial {trial_number}] slow_ratio cap hit: flips={flips} slow%={slow_ratio:.2%} | obj={obj:.1f}")
            return obj

        # regression gains on TRAIN (baseline vs. 23h-on-selected)
        y_pred_tr_base = model_pred_all[train_idx].copy()
        y_pred_tr_tuned = y_pred_tr_base.copy()
        y_pred_tr_tuned[sel_tr] = 23.0

        base_m = _metrics_regression_local(dur_tr, y_pred_tr_base)
        tuned_m = _metrics_regression_local(dur_tr, y_pred_tr_tuned)

        d_mae  = (base_m["mae"]  - tuned_m["mae"])
        d_rmse = (base_m["rmse"] - tuned_m["rmse"])
        d_r2   = (tuned_m["r2"]  - base_m["r2"])


        # CVaR tail penalty on selected flips (train-side)
        cvar_pen = 0.0
        try:
            if (slow_model is not None) and (rank_scores_all is not None):
                X_tmp, _ = build_stage2_features(val_out, pred_col, include_emb=S2_USE_EMB)
                ps_all = slow_model.predict_proba(X_tmp.values.astype(float))[:, 1]
                sel_ps = ps_all[train_idx][sel_tr]
                if sel_ps.size > 0:
    # (removed inner 'import numpy as np' – using global np)
                    alpha = float(params.get("cvar_alpha", S2_VFUNC_DEFAULT["cvar_alpha"]))
                    ceiling = float(params.get("cvar_ceiling", S2_VFUNC_DEFAULT["cvar_ceiling"]))
                    qalpha = np.quantile(sel_ps, alpha)
                    tail = sel_ps[sel_ps >= qalpha]
                    cvar = float(np.mean(tail)) if tail.size > 0 else 0.0
                    if cvar > ceiling:
                        cvar_pen = 1000.0 * (cvar - ceiling) ** 2
        except Exception:
            cvar_pen = 0.0
        # barrier on slow, soft share saturation, weighted objective (minimize)
        slow_barrier = (slow ** MASK23_BARRIER_POWER)
        share_sat = max(0.0, share - (MASK23_MIN_SHARE + MASK23_W_SHARE_SAT))
        obj = (
            + MASK23_W_SLOW * slow_barrier
            + MASK23_W_MID  * mid
            - MASK23_W_FAST * fast
            + 100.0 * (share_sat ** 2)
            - MASK23_W_MAE  * d_mae
            - MASK23_W_RMSE * d_rmse
            - MASK23_W_R2   * d_r2
            + cvar_pen
        )
        # zero-slow bonus
        try:
            if slow == 0 and float(SLOW_ZERO_BONUS) > 0.0:
                obj -= float(SLOW_ZERO_BONUS)
        except Exception:
            pass

        if not silent:
            print(
                f"[Stage2][trial {trial_number}] flips={flips} fast={fast} slow={slow} mid={mid} "
                f"slow%={slow_ratio:.2%} share={share:.2%} | ΔMAE={d_mae:.3f} ΔRMSE={d_rmse:.3f} ΔR2={d_r2:.4f} | obj={obj:.1f}"
            )
        trial_rows.append({
            "trial": trial_number,
            "flips": flips, "fast": fast, "slow": slow, "mid": mid,
            "slow_ratio": slow_ratio, "share": share,
            "d_mae": d_mae, "d_rmse": d_rmse, "d_r2": d_r2,
            "p_threshold": params.get("p_threshold"),
            "delta_overall_max": params.get("delta_overall_max"),
            "ptv_overall_max": params.get("ptv_overall_max"),
            "ptv_anchor_max":  params.get("ptv_anchor_max"),
            "cond_min": params.get("cond_min"),
            "damage_max": params.get("damage_max"),
            "batt_min": params.get("batt_min"),
            "dpm14_max": params.get("dpm14_max"),
            "flip_share": params.get("flip_share"),
            "obj": obj,
        })
        return float(obj)

    # ----- Optuna hush if silent -----
    prev_level = None
    if silent and (optuna is not None):
        prev_level = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---------- Optuna search (minimize) ----------
    if optuna is None:
        params = {
            "p_threshold": 0.20, "delta_overall_max": -150.0, "ptv_overall_max": 0.98,
            "cond_min": 0.55 if "condition_score" in val_out.columns else 0.0,
            "damage_max": 1.5 if "damage_severity" in val_out.columns else 3.0,
            "batt_min": 25.0 if "battery_pct" in val_out.columns else 0.0,
            "dpm14_max": 20.0 if "delta_pred_model_med14" in val_out.columns else 1e9,
            "flip_share": 0.45,
        }
        best_val = _objective_on_train(params, trial_number=None)
        if not silent:
            print("[TUNING:Stage2] heuristic objective:", float(best_val))
    else:
        def _obj(trial):
            params = {
                "p_threshold":       trial.suggest_float("p_threshold", 0.05, 0.70),
                "delta_overall_max": trial.suggest_float("delta_overall_max", -800.0, 150.0) if "delta_from_overall_price" in val_out.columns else 1e9,
                "ptv_overall_max":   trial.suggest_float("ptv_overall_max", 0.80, 1.15)     if "ptv_overall" in val_out.columns else 10.0,
                "ptv_anchor_max":  trial.suggest_float("ptv_anchor_max", 0.95, 1.00) if "ptv_anchor" in val_out.columns else 10.0,
                "cond_min":          trial.suggest_float("cond_min", 0.0, 0.95)             if "condition_score" in val_out.columns else 0.0,
                "damage_max":        trial.suggest_float("damage_max", 0.5, 2.0)            if "damage_severity" in val_out.columns else 3.0,
                "batt_min":          trial.suggest_float("batt_min", 0.0, 90.0)             if "battery_pct" in val_out.columns else 0.0,
                "dpm14_max":         trial.suggest_float("dpm14_max", -10.0, 60.0)          if "delta_pred_model_med14" in val_out.columns else 1e9,
                "flip_share":        trial.suggest_float("flip_share", MASK23_MIN_SHARE, MASK23_MAX_SHARE),
                "seller_min_rating":  trial.suggest_float("seller_min_rating",  0.0, 30.0)   if "seller_rating" in val_out.columns else -1e9,
                "seller_min_reviews": trial.suggest_float("seller_min_reviews", 0.0, 400.0)  if (("review_count" in val_out.columns) or ("ratings_count" in val_out.columns)) else -1e9,
                "seller_min_age_y":   trial.suggest_float("seller_min_age_y",   0.0, 15.0)   if (("member_since_year" in val_out.columns) or ("member_year" in val_out.columns)) else -1e9,
                "seller_quality_min": trial.suggest_float("seller_quality_min", 0.0, 400.0),
                
            
                # Stage-2 ranker value-function weights & risk knobs
                's2_w_fast_raw': trial.suggest_float('s2_w_fast_raw', 0.10, 1.50),
                's2_w_slow_raw': trial.suggest_float('s2_w_slow_raw', 0.10, 1.50),
                's2_w_util_raw': trial.suggest_float('s2_w_util_raw', 0.10, 1.50),
                's2_w_d23_raw' : trial.suggest_float('s2_w_d23_raw', 0.05, 1.00),
                'delta23_max'  : trial.suggest_float('delta23_max', 18.0, 72.0),
                'cvar_alpha'   : trial.suggest_float('cvar_alpha', 0.80, 0.98),
                'cvar_ceiling' : trial.suggest_float('cvar_ceiling', 0.15, 0.40),
                # Stage-2 LightGBM hyperparams (tuned lightly each trial)
                's2_num_leaves': trial.suggest_int('s2_num_leaves', 7, 31),
                's2_max_depth' : trial.suggest_int('s2_max_depth', 3, 4),
                's2_min_data_in_leaf': trial.suggest_int('s2_min_data_in_leaf', 18, 80),
                's2_feature_fraction': trial.suggest_float('s2_feature_fraction', 0.50, 0.70),
                's2_bagging_fraction': trial.suggest_float('s2_bagging_fraction', 0.60, 0.85),
                's2_lambda_l2' : trial.suggest_float('s2_lambda_l2', 5.0, 40.0, log=True),
                's2_learning_rate': trial.suggest_float('s2_learning_rate', 0.03, 0.06, log=True),
                's2_n_estimators': trial.suggest_int('s2_n_estimators', 80, 400),
            }
            return _objective_on_train(params, trial.number)

        study = optuna.create_study(direction="minimize", study_name="mask23_stage2_leakfree")
        study.optimize(_obj, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
        params = study.best_trial.params
        if not silent:
            print("[TUNING:Stage2] Best (leak-free) objective:", float(study.best_value))

    # restore Optuna verbosity
    if prev_level is not None:
        optuna.logging.set_verbosity(prev_level)

    # write trials CSV for inspection (only if requested)
    try:
        if write_outputs and trial_rows:
            pd.DataFrame(trial_rows).to_csv(trials_csv, index=False)
            if not silent:
                print(f"[Stage2] Wrote trial telemetry: {trials_csv}")
    except Exception as e:
        if not silent:
            print("[WARN] Failed writing trial log:", e)

    # ---------- apply learned policy to LATE block (no durations used) ----------
    flips_all = np.zeros(len(val_out), dtype=bool)
    rank_apply = None

    # Train final slow-risk model on EARLY (train_idx) using best params for APPLY gating
    slow_model = None
    s2_weight_vec = None
    try:
        if S2_USE_LGBM and (lgb is not None) and (len(train_idx) >= 30):
            X_all_df, s2_feat_cols = build_stage2_features(val_out, model_pred_col, include_emb=S2_USE_EMB)
            X_all = X_all_df.values.astype(float)
            y_slow_train = (dur_all[train_idx] > 72.0).astype(int)
            hp = dict(S2_LGBM_PARAMS)
            for ksrc, ktgt in [('s2_num_leaves','num_leaves'),('s2_max_depth','max_depth'),('s2_min_data_in_leaf','min_data_in_leaf'),('s2_feature_fraction','feature_fraction'),('s2_bagging_fraction','bagging_fraction'),('s2_lambda_l2','lambda_l2'),('s2_learning_rate','learning_rate'),('s2_n_estimators','n_estimators')]:
                if ksrc in params: hp[ktgt] = params[ksrc]
            slow_model = lgb.LGBMClassifier(**hp)
            if np.unique(y_slow_train).size >= 2:
                slow_model.fit(X_all[train_idx], y_slow_train)
                ps_all = slow_model.predict_proba(X_all)[:, 1]
                # compute value-function ranking pieces
                q  = np.clip(safe_num(val_out['prob_fast24']).to_numpy(dtype=float), 0.0, 1.0)
                U  = X_all_df['_underprice_quality'].to_numpy(dtype=float)
                d23 = X_all_df['_delta23'].to_numpy(dtype=float)
                eps = 1e-9
                Uz = (U - np.nanpercentile(U, 5)) / (np.nanpercentile(U, 95) - np.nanpercentile(U, 5) + eps)
                Uz = np.clip(Uz, 0.0, 1.0)
                wf = float(params.get('s2_w_fast_raw', S2_VFUNC_DEFAULT['w_fast']))
                ws = float(params.get('s2_w_slow_raw', S2_VFUNC_DEFAULT['w_slow_risk']))
                wu = float(params.get('s2_w_util_raw', S2_VFUNC_DEFAULT['w_util']))
                wd = float(params.get('s2_w_d23_raw', S2_VFUNC_DEFAULT['w_delta23']))
                wsum = wf + ws + wu + wd + eps
                w_fast, w_slow, w_util, w_d23 = wf/wsum, ws/wsum, wu/wsum, wd/wsum
                s2_weight_vec = (w_fast, w_slow, w_util, w_d23)
                d23z = np.clip(d23 / float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max'])), 0.0, 1.0)
                rank_apply = (w_fast * q) - (w_slow * ps_all) + (w_util * Uz) - (w_d23 * d23z)
                val_out['_s2_rank'] = rank_apply

                # Apply-time gating by p_slow/prob/delta23
            try:
                # prefer fixed prob threshold if provided
                fixed_q = globals().get('FIXED_PROB_THRESHOLD', None)
                q_thr = float(fixed_q) if (fixed_q is not None) else float(params.get('p_threshold', 0.5))
                d23max = float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max']))
                keep_mask = (ps_all[apply_idx] <= float(S2_P_SLOW_MAX)) & (q[apply_idx] >= q_thr) & (X_all_df['_delta23'].to_numpy(dtype=float)[apply_idx] <= d23max)
                apply_idx = apply_idx[keep_mask]
                # gentle fallback if gating empties the set
                if apply_idx.size == 0:
                    # relax prob first
                    q_thr_relax = min(q_thr, 0.70)
                    keep_mask = (ps_all[flagged_idx] <= float(S2_P_SLOW_MAX)) & (q[flagged_idx] >= q_thr_relax) & (X_all_df['_delta23'].to_numpy(dtype=float)[flagged_idx] <= d23max)
                    apply_idx = flagged_idx[keep_mask]
                if apply_idx.size == 0:
                    # relax slow cap a little
                    ps_cap = max(float(S2_P_SLOW_MAX), 0.20)
                    keep_mask = (ps_all[flagged_idx] <= ps_cap) & (q[flagged_idx] >= q_thr) & (X_all_df['_delta23'].to_numpy(dtype=float)[flagged_idx] <= d23max)
                    apply_idx = flagged_idx[keep_mask]
            except Exception:
                pass

                # Save artifacts for apply-only mode
                if (not silent) and write_outputs:
                    try:
                        slow_model.booster_.save_model(os.path.join(out_dir, 'mask23_lgbm_slow.txt'))
                        with open(os.path.join(out_dir, 'mask23_stage2_features.json'), 'w', encoding='utf-8') as ffeat:
                            json.dump({'features': list(X_all_df.columns)}, ffeat, indent=2)
                        with open(os.path.join(out_dir, 'mask23_vfunc_weights.json'), 'w', encoding='utf-8') as fw:
                            json.dump({'w_fast': w_fast, 'w_slow_risk': w_slow, 'w_util': w_util, 'w_delta23': w_d23,
                                       'delta23_max': float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max']))}, fw, indent=2)
                    except Exception:
                        pass
    except Exception:
        slow_model = None
        rank_apply = None

    
    # Guard: if gating emptied APPLY, try flagged_idx once (already attempted earlier)
    if apply_idx.size == 0:
        apply_idx = flagged_idx.copy()
    flips_all |= _select_flips_over_candidates(val_out, apply_idx, params, rank_scores=rank_apply)


    # ---------- final evaluation (safe) ----------
    pred_23_or_model = model_pred_all.copy()
    pred_23_or_model[flips_all] = 23.0

    slow_flipped = int(((dur_all > 72) & flips_all).sum())
    fast_flipped = int(((dur_all <= 40) & flips_all).sum())
    mid_flipped  = int((((dur_all > 40) & (dur_all <= 72)) & flips_all).sum())

    base_mets  = _metrics_regression_local(dur_all, model_pred_all)
    tuned_mets = _metrics_regression_local(dur_all, pred_23_or_model)


    # [INLINE EXPORT HOOK] — stash per-trial flipped rows (no files) so the winner can be exported later
    if silent and not write_outputs:
        try:
            trial_no = globals().get("_CURRENT_TRIAL_NO")
            mask_df = val_out.copy()
            mask_df["flip_to_23"] = flips_all.astype(int)
            mask_df["y_pred_23_or_model"] = pred_23_or_model
            mask_rows = mask_df[mask_df["flip_to_23"] == 1].copy()
            store = globals().setdefault("_mask23_inline_store", {})
            # key by trial number; fall back to -1 if not available
            store[int(trial_no) if trial_no is not None else -1] = {"mask_rows": mask_rows}
        except Exception:
            pass


    # [INLINE STASH] Keep per-trial full summary for JOINT best-trial reporting
    if silent and not write_outputs:
        try:
            trial_no = globals().get("_CURRENT_TRIAL_NO")
            # compose a compact summary dict
            summary_inline = {
                "flips_total": int(flips_all.sum()),
                "fast_flipped": fast_flipped,
                "slow_flipped": slow_flipped,
                "mid_flipped":  mid_flipped,
                "baseline": dict(n=base_mets.get("n"), mae=base_mets.get("mae"),
                                 rmse=base_mets.get("rmse"), r2=base_mets.get("r2")),
                "tuned":    dict(n=tuned_mets.get("n"), mae=tuned_mets.get("mae"),
                                 rmse=tuned_mets.get("rmse"), r2=tuned_mets.get("r2")),
                "improvement": {
                    "mae":  base_mets.get("mae")  - tuned_mets.get("mae"),
                    "rmse": base_mets.get("rmse") - tuned_mets.get("rmse"),
                    "r2":   tuned_mets.get("r2")  - base_mets.get("r2"),
                },
            }
            # also stash the flipped rows for that trial
            mask_df = val_out.copy()
            mask_df["flip_to_23"] = flips_all.astype(int)
            mask_df["y_pred_23_or_model"] = pred_23_or_model
            mask_rows = mask_df[mask_df["flip_to_23"] == 1].copy()

            store = globals().setdefault("_mask23_inline_store", {})
            store[int(trial_no) if trial_no is not None else -1] = {
                "full_summary": summary_inline,
                "mask_rows": mask_rows,
            }
        except Exception:
            pass
    



    # Expose inline summaries for JOINT objective (full + compact)
    globals()["_mask23_inline_last_full"] = {
        "leak_free": True,
        "train_frac": MASK23_TRAIN_FRAC,
        "cv_blocks": MASK23_CV_BLOCKS,
        "flips_total": int(flips_all.sum()),
        "slow_flipped": slow_flipped,
        "fast_flipped": fast_flipped,
        "mid_flipped":  mid_flipped,
        "baseline": base_mets,
        "tuned": tuned_mets,
        "improvement": {
            "mae": base_mets.get("mae") - tuned_mets.get("mae"),
            "rmse": base_mets.get("rmse") - tuned_mets.get("rmse"),
            "r2":  tuned_mets.get("r2")  - base_mets.get("r2"),
        }
    }
    globals()["_mask23_inline_last"] = {
        "fast_flipped": fast_flipped,
        "slow_flipped": slow_flipped,
        "mid_flipped":  mid_flipped,
        "improvement": {
            "mae": base_mets.get("mae") - tuned_mets.get("mae"),
            "rmse": base_mets.get("rmse") - tuned_mets.get("rmse"),
            "r2":  tuned_mets.get("r2")  - base_mets.get("r2"),
        }
    }

    # outputs (only if requested)
    try:
        if write_outputs:
            out_full = val_out.copy()
            out_full["flip_to_23"] = flips_all.astype(int)
            out_full["y_pred_23_or_model"] = pred_23_or_model
            out_path_full = os.path.join(out_dir, "preds_with_mask23.csv")
            out_full.to_csv(out_path_full, index=False)

            best_rows = out_full[out_full["flip_to_23"] == 1].copy()
            out_path_mask = os.path.join(out_dir, "best_mask.csv")
            best_rows.to_csv(out_path_mask, index=False)

            summary = {
                "leak_free": True,
                "train_frac": MASK23_TRAIN_FRAC,
                "cv_blocks": 1,
                "flips_total": int(flips_all.sum()),
                "slow_flipped": slow_flipped,
                "fast_flipped": fast_flipped,
                "mid_flipped": mid_flipped,
                "baseline": base_mets,
                "tuned": tuned_mets,
                "improvement": {
                    "mae": base_mets.get("mae") - tuned_mets.get("mae"),
                    "rmse": base_mets.get("rmse") - tuned_mets.get("rmse"),
                    "r2":  tuned_mets.get("r2")  - base_mets.get("r2"),
                },
                "best_params": params,
                "weights": {
                    "W_FAST": MASK23_W_FAST, "W_SLOW": MASK23_W_SLOW, "W_MID": MASK23_W_MID,
                    "W_MAE": MASK23_W_MAE, "W_RMSE": MASK23_W_RMSE, "W_R2": MASK23_W_R2,
                    "W_SHARE_SAT": MASK23_W_SHARE_SAT, "BARRIER_POWER": MASK23_BARRIER_POWER,
                }
            }
            with open(os.path.join(out_dir, "mask23_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if not silent:
                print(json.dumps(summary, indent=2))
                

        # Also dump raw Stage-2 params for frozen re-use
        mask23_path = os.path.join(out_dir, "mask23_best_params.json")
        try:
            with open(mask23_path, "w", encoding="utf-8") as f2:
                json.dump(params, f2, indent=2)
            if not silent:
                print(f"[Stage2] Saved best params to {mask23_path}")
        except Exception as e2:
            if not silent:
                print(f"[Stage2] Failed to save best params: {e2}")

    except Exception as e:
        if not silent:
            print("[WARN] Failed writing mask-23 outputs:", e)



    return {
        "params": params,
        "preds_path": os.path.join(out_dir, "preds_with_mask23.csv") if write_outputs else None,
        "mask_path": os.path.join(out_dir, "best_mask.csv") if write_outputs else None,
        "summary_path": os.path.join(out_dir, "mask23_summary.json") if write_outputs else None,
    }


def apply_mask23_once(val_out: pd.DataFrame,
                      dur_col: Optional[str],
                      model_pred_col: Optional[str],
                      out_dir: str) -> None:
    """
    Apply the already-tuned mask-23 policy ONE time using globals set from --load_mask23_params.
    Writes: preds_with_mask23.csv, best_mask.csv, mask23_summary.json
    """
    if dur_col is None or model_pred_col is None or dur_col not in val_out.columns or model_pred_col not in val_out.columns:
        print("[Stage-2] apply-only: missing duration or model prediction column; skip.")
        return

    # Pull params from globals (set in main() from the JSON)
    params = dict(
        p_threshold       = float(globals().get("MASK23_P_THRESHOLD",       0.20)),
        delta_overall_max = float(globals().get("MASK23_DELTA_OVERALL_MAX", 50.0)),
        ptv_overall_max   = float(globals().get("MASK23_PTV_OVERALL_MAX",   1.05)),
        ptv_anchor_max     = float(globals().get("MASK23_PTV_ANCHOR_MAX",   1.05)),
        cond_min          = float(globals().get("MASK23_COND_MIN",          0.0)),
        damage_max        = float(globals().get("MASK23_DAMAGE_MAX",        1.8)),
        batt_min          = float(globals().get("MASK23_BATT_MIN",          0.0)),
        dpm14_max         = float(globals().get("MASK23_DPM14_MAX",         40.0)),
        flip_share        = float(globals().get("MASK23_FLIP_SHARE",        0.40)),
        seller_min_rating  = float(globals().get("MASK23_SELLER_MIN_RATING",   -1e9)),
        seller_min_reviews = float(globals().get("MASK23_SELLER_MIN_REVIEWS",  -1e9)),
        seller_min_age_y   = float(globals().get("MASK23_SELLER_MIN_AGE_Y",    -1e9)),
        seller_quality_min = float(globals().get("MASK23_SELLER_QUALITY_MIN",  -1e9)),
    )
    


    # Merge full Stage-2 JSON keys (weights + LGBM HPs + delta23/cvar) for apply-only
    _mfull = globals().get("_MASK23_PARAMS_FULL")
    if isinstance(_mfull, dict):
        for k in [
            "s2_w_fast_raw","s2_w_slow_raw","s2_w_util_raw","s2_w_d23_raw",
            "delta23_max","cvar_alpha","cvar_ceiling",
            "s2_num_leaves","s2_max_depth","s2_min_data_in_leaf","s2_feature_fraction",
            "s2_bagging_fraction","s2_lambda_l2","s2_learning_rate","s2_n_estimators"
        ]:
            if k in _mfull:
                params[k] = _mfull[k]


    # Candidates = currently flagged rows (is_fast24_pred==1)
    flagged_idx = np.where(val_out.get("is_fast24_pred", 0).astype(int).to_numpy(dtype=bool))[0]
    if flagged_idx.size == 0:
        print("[Stage-2] apply-only: no flagged rows; nothing to apply.")
        return

    # --- eligibility (same gates as tuner)
    def _eligible(df, ps):
        n = len(df); out = np.ones(n, dtype=bool)
        p_all = df.get("prob_fast24", pd.Series(np.ones(n))).to_numpy(dtype=float)
        out &= (p_all >= ps["p_threshold"])
        if "delta_from_overall_price" in df: out &= (safe_num(df["delta_from_overall_price"]).to_numpy() <= ps["delta_overall_max"])
        if "ptv_overall" in df:              out &= (safe_num(df["ptv_overall"]).to_numpy()            <= ps["ptv_overall_max"])
        if "ptv_anchor"  in df:              out &= (safe_num(df["ptv_anchor"]).to_numpy()             <= ps.get("ptv_anchor_max", 10.0))
        if "condition_score" in df:          out &= (safe_num(df["condition_score"]).to_numpy()        >= ps["cond_min"])
        if "damage_severity" in df:          out &= (safe_num(df["damage_severity"]).to_numpy()        <= ps["damage_max"])
        if "battery_pct" in df:              out &= (safe_num(df["battery_pct"]).to_numpy()            >= ps["batt_min"])
        if "delta_pred_model_med14" in df:   out &= (safe_num(df["delta_pred_model_med14"]).to_numpy() <= ps["dpm14_max"])

        # seller corridor (apply-only, NaN -> pass)
        if "seller_rating" in df and ("seller_min_rating" in ps):
            sr = safe_num(df["seller_rating"]).to_numpy()
            out &= np.where(np.isfinite(sr), sr >= float(ps["seller_min_rating"]), True)

        if ("seller_min_reviews" in ps):
            if "review_count" in df:
                rc = safe_num(df["review_count"]).to_numpy()
                out &= np.where(np.isfinite(rc), rc >= float(ps["seller_min_reviews"]), True)
            elif "ratings_count" in df:
                rc = safe_num(df["ratings_count"]).to_numpy()
                out &= np.where(np.isfinite(rc), rc >= float(ps["seller_min_reviews"]), True)

        if ("seller_min_age_y" in ps):
            msy = safe_num(df.get("member_since_year", df.get("member_year"))).to_numpy(dtype=float)
            age = 2025.0 - msy
            age = np.where(np.isfinite(age), np.maximum(0.0, age), np.nan)
            out &= np.where(np.isfinite(age), age >= float(ps["seller_min_age_y"]), True)

        if ("seller_quality_min" in ps):
            sq = compute_seller_quality(df).to_numpy(dtype=float)
            out &= np.where(np.isfinite(sq), sq >= float(ps["seller_quality_min"]), True)


        return out


    # Try to use saved Stage-2 ranker if available in out_dir
    rank_apply = None
    # Apply-only: if we can score p_slow, hard-gate by S2_P_SLOW_MAX and prob threshold and delta23
    apply_gate_mask = None
    try:
        # We'll compute gating after we get ps_all below; we store params for thresholds now
        pass
    except Exception:
        pass
    try:
        if S2_USE_LGBM and (lgb is not None):
            param_dir = globals().get("_MASK23_PARAMS_DIR", out_dir)
            slow_path = os.path.join(param_dir, 'mask23_lgbm_slow.txt') if os.path.exists(os.path.join(param_dir, 'mask23_lgbm_slow.txt')) else os.path.join(out_dir, 'mask23_lgbm_slow.txt')
            feat_path = os.path.join(param_dir, 'mask23_stage2_features.json') if os.path.exists(os.path.join(param_dir, 'mask23_stage2_features.json')) else os.path.join(out_dir, 'mask23_stage2_features.json')
            vwt_path  = os.path.join(param_dir, 'mask23_vfunc_weights.json') if os.path.exists(os.path.join(param_dir, 'mask23_vfunc_weights.json')) else os.path.join(out_dir, 'mask23_vfunc_weights.json')

            if os.path.exists(slow_path) and os.path.exists(feat_path) and os.path.exists(vwt_path):
                booster = lgb.Booster(model_file=slow_path)
                with open(feat_path, 'r', encoding='utf-8') as ffeat:
                    feats = __import__('json').load(ffeat).get('features', [])
                with open(vwt_path, 'r', encoding='utf-8') as fw:
                    v = __import__('json').load(fw)
                X_df, _ = build_stage2_features(val_out, model_pred_col, include_emb=S2_USE_EMB)
                if feats and all(c in X_df.columns for c in feats):
                    X = X_df[feats].values.astype(float)
                else:
                    X = X_df.values.astype(float)
                ps_all = booster.predict(X)
    # (removed inner 'import numpy as np' – using global np)
                q  = np.clip(safe_num(val_out['prob_fast24']).to_numpy(dtype=float), 0.0, 1.0)
                U  = X_df['_underprice_quality'].to_numpy(dtype=float)
                d23 = X_df['_delta23'].to_numpy(dtype=float)
                eps = 1e-9
                Uz = (U - np.nanpercentile(U, 5)) / (np.nanpercentile(U, 95) - np.nanpercentile(U, 5) + eps)
                Uz = np.clip(Uz, 0.0, 1.0)
                wf, ws, wu, wd = (v.get('w_fast', 0.4), v.get('w_slow_risk', 0.35), v.get('w_util', 0.2), v.get('w_delta23', 0.05))
                d23z = np.clip(d23 / float(v.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max'])), 0.0, 1.0)
                rank_apply = (wf * q) - (ws * ps_all) + (wu * Uz) - (wd * d23z)
    
                # Apply-only gating by p_slow/prob/delta23 if available
                try:
                    q_thr = float(params.get('p_threshold', 0.5))
                    d23max = float(params.get('dpm14_max', 1e9)) if '_delta23' not in X_df.columns else float(params.get('delta23_max', S2_VFUNC_DEFAULT['delta23_max']))
                    mask_gate = (ps_all <= float(S2_P_SLOW_MAX)) & (q >= q_thr) & (d23 <= d23max)
                except Exception:
                    mask_gate = None
    except Exception:
        rank_apply = None

        
    # ===== BEGIN REPLACEMENT: robust Stage-2 selection (no helper needed) =====

    # Build scoring vector (use rank scores if available; else calibrated prob)
    p_all    = val_out.get("prob_fast24", pd.Series(np.ones(len(val_out)))).to_numpy(dtype=float)
    rank_vec = rank_apply if ('rank_apply' in locals() and (rank_apply is not None)) else p_all

    # Flip count: flip_share with floors/caps
    k_floor = max(
        int(np.ceil(float(globals().get("MASK23_MIN_SHARE", 0.20)) * flagged_idx.size)),
        int(globals().get("MASK23_MIN_FLIPS", 10))
    )
    k = int(np.ceil(float(params.get("flip_share", 0.40)) * flagged_idx.size))
    k = max(k_floor, min(k, flagged_idx.size))

    # Candidate set starts as flagged; intersect with p_slow/prob/d23 mask if present
    try:
        if ('mask_gate' in locals()) and (mask_gate is not None):
            cand_idx = flagged_idx[mask_gate[flagged_idx]]
        else:
            cand_idx = flagged_idx
    except Exception:
        cand_idx = flagged_idx

    # Corridor eligibility for all rows
    elig_mask = _eligible(val_out, params)  # existing helper in your file
    # Eligible among candidates
    try:
        elig_idx = np.intersect1d(cand_idx, flagged_idx[elig_mask[flagged_idx]], assume_unique=False)
    except Exception:
        # very defensive fallback
        elig_idx = cand_idx.copy()

    # Primary pick: eligible by corridor
    if elig_idx.size > 0:
        k_use = min(k, int(elig_idx.size))
        pick  = elig_idx[np.argsort(-rank_vec[elig_idx])][:k_use]
    else:
        # Corridor too strict -> fallback: top-K by score from candidate set (still honors floors/caps)
        if cand_idx.size > 0:
            k_use = min(max(k_floor, 1), int(cand_idx.size))
            pick  = cand_idx[np.argsort(-rank_vec[cand_idx])][:k_use]
        else:
            pick = np.array([], dtype=int)

    # ===== END REPLACEMENT =====




    flips_all = np.zeros(len(val_out), dtype=bool)
    flips_all[pick] = True

    # Build 23h-or-model vector and write the same artifacts as tuner
    dur_all = safe_num(val_out[dur_col]).to_numpy(dtype=float)
    model_pred_all = safe_num(val_out[model_pred_col]).to_numpy(dtype=float)
    pred_23_or_model = model_pred_all.copy()
    pred_23_or_model[flips_all] = 23.0

    out_full = val_out.copy()
    out_full["flip_to_23"] = flips_all.astype(int)
    out_full["y_pred_23_or_model"] = pred_23_or_model
    out_full.to_csv(os.path.join(out_dir, "preds_with_mask23.csv"), index=False)
    out_full[out_full["flip_to_23"] == 1].to_csv(os.path.join(out_dir, "best_mask.csv"), index=False)

    # Minimal summary (mirrors tuner)
    def _metr(y, yhat):
        m = np.isfinite(y) & np.isfinite(yhat)
        if not np.any(m): return dict(n=0, mae=np.nan, rmse=np.nan, r2=np.nan)
        yt, yp = y[m], yhat[m]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = float(np.sum((yt - yp) ** 2)); ss_tot = float(np.sum((yt - np.mean(yt))**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan
        return dict(n=int(m.sum()), mae=mae, rmse=rmse, r2=r2)

    base_m = _metr(dur_all, model_pred_all)
    tuned_m = _metr(dur_all, pred_23_or_model)
    summary = {
        "leak_free": True,
        "train_frac": float(globals().get("MASK23_TRAIN_FRAC", 0.40)),
        "cv_blocks": int(globals().get("MASK23_CV_BLOCKS", 1)),
        "flips_total": int(flips_all.sum()),
        "slow_flipped": int(((dur_all > 72) & flips_all).sum()),
        "fast_flipped": int(((dur_all <= 40) & flips_all).sum()),
        "mid_flipped":  int((((dur_all > 40) & (dur_all <= 72)) & flips_all).sum()),
        "baseline": base_m, "tuned": tuned_m,
        "improvement": {"mae": base_m["mae"]-tuned_m["mae"], "rmse": base_m["rmse"]-tuned_m["rmse"], "r2": tuned_m["r2"]-base_m["r2"]},
        "best_params": params
    }
    with open(os.path.join(out_dir, "mask23_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


# ───────────────────────── Stage-1 tuning (Optuna) — JOINT objective with inline Stage-2 and per-trial metrics print+CSV
def _suggest_knobs(trial) -> Dict[str, Any]:
    cal = trial.suggest_categorical("CALIBRATION", ["isotonic", "platt"])
    cls_w = trial.suggest_categorical("CLASS_WEIGHT", ["balanced", None])
    knobs = {
        "N_SPLITS": trial.suggest_int("N_SPLITS", 5, 9),
        "CALIBRATION": cal,
        "RANDOM_STATE": trial.suggest_int("RANDOM_STATE", 1, 9999),
        "CLASS_WEIGHT": cls_w,
        "PREC_MIN": trial.suggest_float("PREC_MIN", 0.60, 0.90),
        "REC_MIN": trial.suggest_float("REC_MIN", 0.30, 0.65),
        "MAX_SHARE": trial.suggest_float("MAX_SHARE", 0.20, 0.70),
        "PER_MODEL_MIN_N": trial.suggest_int("PER_MODEL_MIN_N", 15, 60),
        "TOPK_FALLBACK": trial.suggest_int("TOPK_FALLBACK", 0, 50),
        "G_SOFT_ENABLE": trial.suggest_categorical("G_SOFT_ENABLE", [True, False]),
        "G_MIN_COND": trial.suggest_float("G_MIN_COND", 0.5, 0.9),
        "G_MAX_DAMAGE": trial.suggest_float("G_MAX_DAMAGE", 0.5, 2.0),
        "G_MIN_BATT": trial.suggest_float("G_MIN_BATT", 0.0, 80.0),
        "G_PTV_OVERALL_MAX": trial.suggest_float("G_PTV_OVERALL_MAX", 0.90, 1.20),
        "G_DELTA_OVERALL_MAX": trial.suggest_float("G_DELTA_OVERALL_MAX", -800.0, 100.0),
        "G_PTV_ANCHOR_MAX": trial.suggest_float("G_PTV_ANCHOR_MAX", 0.90, 1.05),
        "G_MIN_STORAGE_GB": trial.suggest_categorical("G_MIN_STORAGE_GB", [64, 128, 256]),
        "G_MAX_PPG_RATIO": trial.suggest_float("G_MAX_PPG_RATIO", 30.0, 90.0),
        "G_WARRANTY_BOOST": trial.suggest_float("G_WARRANTY_BOOST", -2.0, 2.0),
        "G_RECEIPT_BOOST": trial.suggest_float("G_RECEIPT_BOOST", -2.0, 2.0),
        "G_OPLOCK_PENALTY": trial.suggest_float("G_OPLOCK_PENALTY", -2.0, 2.0),
        "G_NEGOTIATION_BOOST": trial.suggest_float("G_NEGOTIATION_BOOST", -2.0, 2.0),
        "G_FASTPRIS_PENALTY": trial.suggest_float("G_FASTPRIS_PENALTY", -2.0, 2.0),
        "G_ACC_MAX_BOOST": trial.suggest_float("G_ACC_MAX_BOOST", -2.0, 2.0),
        "G_MIN_SELLER_RATING": trial.suggest_float("G_MIN_SELLER_RATING", 0.0, 30.0),
        "G_MIN_REVIEWS": trial.suggest_int("G_MIN_REVIEWS", 0, 300),
        "G_MIN_ACCOUNT_AGE_Y": trial.suggest_int("G_MIN_ACCOUNT_AGE_Y", 0, 15),
        "G_PENALTY_MIN": trial.suggest_float("G_PENALTY_MIN", 0.50, 0.95),
        "G_TOTAL_PENALTY_MIN": trial.suggest_float("G_TOTAL_PENALTY_MIN", 0.05, 0.50),
        "G_TOTAL_PENALTY_MAX": trial.suggest_float("G_TOTAL_PENALTY_MAX", 0.85, 1.00),

        
        # ── Per-variant penalties (0.0 = hard block; 1.0 = neutral; >1.0 = boost)
        "MP_13_MINI":      trial.suggest_float("MP_13_MINI", 0.0, 1.50),
        "MP_13":           trial.suggest_float("MP_13", 0.0, 1.50),
        "MP_13_PRO":       trial.suggest_float("MP_13_PRO", 0.0, 1.50),
        "MP_13_PRO_MAX":   trial.suggest_float("MP_13_PRO_MAX", 0.0, 1.50),

    }
    if knobs["G_TOTAL_PENALTY_MIN"] > knobs["G_TOTAL_PENALTY_MAX"]:
        knobs["G_TOTAL_PENALTY_MIN"], knobs["G_TOTAL_PENALTY_MAX"] = knobs["G_TOTAL_PENALTY_MAX"], knobs["G_TOTAL_PENALTY_MIN"]
    if knobs["G_PENALTY_MIN"] > knobs["G_TOTAL_PENALTY_MAX"]:
        knobs["G_PENALTY_MIN"] = knobs["G_TOTAL_PENALTY_MAX"]
    return knobs

def _apply_knobs(k: Dict[str, Any]) -> None:
    g = globals()
    for key, val in k.items():
        g[key] = val


    # If CLI overrides were provided, re-apply them after trial knobs so they always win
    if g.get("_CLI_MP_OVERRIDE"):
        mp = dict(g["G_MODEL_PENALTIES"])
        mp.update(g["_CLI_MP_OVERRIDE"])
        g["G_MODEL_PENALTIES"] = mp


    
    # ── Rebuild per-variant penalties from knobs (if provided this trial)
    if any(name in k for name in ("MP_13_MINI", "MP_13", "MP_13_PRO", "MP_13_PRO_MAX")):
        curr = g.get("G_MODEL_PENALTIES", {
            "iPhone 13 Mini": 1.0,
            "iPhone 13": 1.0,
            "iPhone 13 Pro": 1.0,
            "iPhone 13 Pro Max": 1.0,
        })
        g["G_MODEL_PENALTIES"] = {
            "iPhone 13 Mini":   float(k.get("MP_13_MINI",    curr.get("iPhone 13 Mini",   1.0))),
            "iPhone 13":        float(k.get("MP_13",         curr.get("iPhone 13",        1.0))),
            "iPhone 13 Pro":    float(k.get("MP_13_PRO",     curr.get("iPhone 13 Pro",    1.0))),
            "iPhone 13 Pro Max":float(k.get("MP_13_PRO_MAX", curr.get("iPhone 13 Pro Max",1.0))),
        }


    # If CLI overrides were provided, re-apply them after trial knobs so they always win
    if g.get("_CLI_MP_OVERRIDE"):
        mp = dict(g["G_MODEL_PENALTIES"])
        mp.update(g["_CLI_MP_OVERRIDE"])
        g["G_MODEL_PENALTIES"] = mp



def _stage1_objective_factory(train_path, slow_path, val_text_path, oof_path, out_dir, emb_choice, threshold_steps):
    def _obj(trial):
        knobs = _suggest_knobs(trial)
        _apply_knobs(knobs)
        try:
            # Stage-1 run (no writes during tuning)
            res = pipeline_run(
                train_path=train_path, slow_path=slow_path, val_text_path=val_text_path,
                oof_path=oof_path, out_dir=out_dir, emb_choice=emb_choice,
                threshold_steps=threshold_steps, write_outputs=False
            )
            # Prune trials with too few rows after gating/corridor (prevents NaN explosions)
            try:
                import optuna as _optuna_for_prune  # may be None if Optuna unavailable
            except Exception:
                _optuna_for_prune = None
            if (_optuna_for_prune is not None) and (len(res.get("val_out", [])) < int(globals().get("PRUNE_MIN_ROWS", 40))):
                raise _optuna_for_prune.TrialPruned(f"Too few rows after corridor gating: {len(res.get('val_out', []))} < {int(globals().get('PRUNE_MIN_ROWS', 40))}")

            u = res.get("union", {}) or {}
            tp    = int(u.get("TP", 0))
            fp    = int(u.get("FP", 0))
            f1    = float(u.get("F1", 0.0))
            flags = int(u.get("flags", 0))

            # barrier flag (we still run Stage-2 to collect metrics; score is penalized later)
            bad_trial = (tp <= fp)



            if globals().get("NO_STAGE1_LABEL_SCORING", False):
                tp = fp = flags = 0
                f1 = 0.0
                bad_trial = False



            # Inline Stage-2 (silent)
            n2 = int(globals().get("TUNE_MASK23_PER_TRIAL", 0))
            fast_flipped = slow_flipped = mid_flipped = flips_total = 0
            d_mae = d_rmse = d_r2 = 0.0
            base_mae = base_rmse = base_r2 = ""
            tuned_mae = tuned_rmse = tuned_r2 = ""

            if n2 > 0:

                # tag current trial number so Stage-2 can stash per-trial rows
                globals()["_CURRENT_TRIAL_NO"] = trial.number

                _ = run_mask23_tuning(
                    val_out=res["val_out"],
                    dur_col=res.get("dur_col"),
                    model_pred_col=res.get("model_pred_col"),
                    out_dir=out_dir,
                    n_trials=n2,
                    silent=True,
                    write_outputs=False
                ) or {}
                inline_full = globals().get("_mask23_inline_last_full", None)
                if inline_full:
                    flips_total  = int(inline_full.get("flips_total", 0))
                    fast_flipped = int(inline_full.get("fast_flipped", 0))
                    slow_flipped = int(inline_full.get("slow_flipped", 0))
                    mid_flipped  = int(inline_full.get("mid_flipped", 0))
                    base = inline_full.get("baseline", {}) or {}
                    tuned = inline_full.get("tuned", {}) or {}
                    imp = inline_full.get("improvement", {}) or {}

                    base_mae  = float(base.get("mae"))  if base.get("mae")  is not None else ""
                    base_rmse = float(base.get("rmse")) if base.get("rmse") is not None else ""
                    base_r2   = float(base.get("r2"))   if base.get("r2")   is not None else ""

                    tuned_mae  = float(tuned.get("mae"))  if tuned.get("mae")  is not None else ""
                    tuned_rmse = float(tuned.get("rmse")) if tuned.get("rmse") is not None else ""
                    tuned_r2   = float(tuned.get("r2"))   if tuned.get("r2")   is not None else ""

                    d_mae  = float(imp.get("mae", 0.0))
                    d_rmse = float(imp.get("rmse", 0.0))
                    d_r2   = float(imp.get("r2", 0.0))

            # JOINT score
            w_tpfp = float(globals().get("JOINT_W_TPFP", 1.0))
            w_fast = float(globals().get("JOINT_W_FAST", 1.0))
            w_slow = float(globals().get("JOINT_W_SLOW", 8.0))
            w_mae  = float(globals().get("JOINT_W_MAE", 0.05))
            w_rmse = float(globals().get("JOINT_W_RMSE", 0.02))
            w_r2   = float(globals().get("JOINT_W_R2", 50.0))

            score = (
                w_tpfp * (tp - fp)
                + w_fast * fast_flipped
                - w_slow * slow_flipped
                + w_mae  * d_mae
                + w_rmse * d_rmse
                + w_r2   * d_r2
            )

            if globals().get("NO_STAGE1_LABEL_SCORING", False):
                # Stage-2-only selection: keep the score we already computed from flips/Δmetrics
                score = float(score)


            # Hard barrier applied here (AFTER Stage-2 runs so we still log metrics)
            if bad_trial:
                score -= 1_000_000.0  # barrier penalty (was hard-kill)


            # Per-trial print
            print(
                f"[JOINT][trial {trial.number}] TP={tp} FP={fp} Flags={flags} | "
                f"flips={flips_total} fast={fast_flipped} slow={slow_flipped} mid={mid_flipped} | "
                f"MAE {base_mae}->{tuned_mae} (Δ{d_mae:+.3f})  "
                f"RMSE {base_rmse}->{tuned_rmse} (Δ{d_rmse:+.3f})  "
                f"R2 {base_r2}->{tuned_r2} (Δ{d_r2:+.4f}) | "
                f"score={score:.3f}"
            )

            # CSV log
            try:
                log_path = os.path.join(out_dir, "joint_stage1_trials_log.csv")
                header = not os.path.exists(log_path)
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if header:
                        w.writerow([
                            "trial","tp","fp","flags","f1",
                            "flips_total","fast_flipped","slow_flipped","mid_flipped",
                            "baseline_mae","tuned_mae","delta_mae",
                            "baseline_rmse","tuned_rmse","delta_rmse",
                            "baseline_r2","tuned_r2","delta_r2",
                            "joint_score"
                        ])
                    w.writerow([
                        trial.number, tp, fp, flags, f1,
                        flips_total, fast_flipped, slow_flipped, mid_flipped,
                        base_mae, tuned_mae, d_mae,
                        base_rmse, tuned_rmse, d_rmse,
                        base_r2, tuned_r2, d_r2,
                        score
                    ])
            except Exception:
                pass

            # attrs for best-trial introspection
            try:
                trial.set_user_attr("TP", tp)
                trial.set_user_attr("FP", fp)
                trial.set_user_attr("TP_minus_FP", tp - fp)
                trial.set_user_attr("F1", f1)
                trial.set_user_attr("Flags", flags)
                trial.set_user_attr("Stage2_used", (n2 > 0))
                trial.set_user_attr("S2_fast_flipped", fast_flipped)
                trial.set_user_attr("S2_slow_flipped", slow_flipped)
                trial.set_user_attr("S2_mid_flipped",  mid_flipped)
                trial.set_user_attr("S2_dMAE",  d_mae)
                trial.set_user_attr("S2_dRMSE", d_rmse)
                trial.set_user_attr("S2_dR2",   d_r2)
                trial.set_user_attr("JOINT_score", score)
            except Exception:
                pass

            return float(score)

        except Exception:
            return -1e12
    return _obj

def run_stage1_tuning(n_trials: int,
                      train_path: str, slow_path: str, val_text_path: str, oof_path: Optional[str],
                      out_dir: str, emb_choice: str, threshold_steps: int) -> Dict[str, Any]:
    if optuna is None:
        print("[WARN] Optuna not available; skipping Stage-1 tuning.")
        return {}
    pruner = MedianPruner(n_startup_trials=max(10, min(20, n_trials // 5)))
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="fast24_stage1")
    study.optimize(
        _stage1_objective_factory(train_path, slow_path, val_text_path, oof_path, out_dir, emb_choice, threshold_steps),
        n_trials=n_trials, n_jobs=1, gc_after_trial=True
    )
    best = study.best_trial.params
    _apply_knobs(best)
    try:
        with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
    except Exception:
        pass

    # print best trial JOINT summary
    try:
        best_no = study.best_trial.number
        ua = study.best_trial.user_attrs

        # pull full baseline/tuned metrics for the winning trial
        store = globals().get("_mask23_inline_store", {})
        full = (store.get(best_no) or {}).get("full_summary") or globals().get("_mask23_inline_last_full") or {}
        base = full.get("baseline", {}) or {}
        tuned = full.get("tuned", {}) or {}
        imp = full.get("improvement", {}) or {}

        # safe numbers
        def _f(x, nd=3):
            try: return f"{float(x):.{nd}f}"
            except: return "nan"

        print(
            "[JOINT:best] "
            f"TP={ua.get('TP')} FP={ua.get('FP')} Flags={ua.get('Flags')} "
            f"S2_fast={ua.get('S2_fast_flipped')} S2_slow={ua.get('S2_slow_flipped')} S2_mid={ua.get('S2_mid_flipped')} "
            f"| baseline MAE={_f(base.get('mae'))} RMSE={_f(base.get('rmse'))} R2={_f(base.get('r2'),4)} "
            f"| tuned MAE={_f(tuned.get('mae'))} RMSE={_f(tuned.get('rmse'))} R2={_f(tuned.get('r2'),4)} "
            f"| ΔMAE={_f(ua.get('S2_dMAE',0))} ΔRMSE={_f(ua.get('S2_dRMSE',0))} ΔR2={_f(ua.get('S2_dR2',0),4)} "
            f"| score={study.best_value:.3f}"
        )
    except Exception:
        pass


    print("[TUNING:Stage1] Best value:", study.best_value)


    # Write CSV of flipped rows for the winning trial (from inline Stage-2)
    try:
        best_no = study.best_trial.number
        store = globals().get("_mask23_inline_store", {})
        if best_no in store and "mask_rows" in store[best_no]:
            out_csv = os.path.join(out_dir, "joint_best_mask_rows.csv")
            store[best_no]["mask_rows"].to_csv(out_csv, index=False)
            print(f"[JOINT] Wrote best-trial flipped rows: {out_csv}")
        else:
            print("[JOINT] No inline flipped rows found to write for the best trial.")
    except Exception as e:
        print("[JOINT] Failed to write best-trial flipped rows CSV:", e)


    # --- Union across trials (zero-slow OR any-trials) ---
    if globals().get("UNION_ZERO_SLOW", False) or globals().get("UNION_ANY_TRIALS", False):
        try:
            store = globals().get("_mask23_inline_store", {}) or {}
            id_candidates = ["listing_id", "listingid", "id"]
            dfs = []
            kept_trials = 0

            any_mode = bool(globals().get("UNION_ANY_TRIALS", False))
            min_fast = int(globals().get("UNION_ZERO_SLOW_MIN_FAST", 1))

            for tr in study.trials:
                ua = getattr(tr, "user_attrs", None) or {}
                s2_slow = ua.get("S2_slow_flipped", None)
                s2_fast = int(ua.get("S2_fast_flipped", 0) or 0)

                # include: ANY mode ignores slow; ZERO-SLOW mode requires s2_slow==0; always require min fast flips
                if ((any_mode or s2_slow == 0) and s2_fast >= min_fast):
                    rec = store.get(tr.number)
                    if rec and "mask_rows" in rec:
                        df = rec["mask_rows"].copy()
                        if "flip_to_23" in df.columns:
                            df = df[df["flip_to_23"] == 1]
                        df["_source_trial"] = tr.number
                        df["_source_score"] = ua.get("JOINT_score", None)
                        dfs.append(df)
                        kept_trials += 1

            if not dfs:
                print("[JOINT][UNION] No eligible trials to union.")
            else:
                union_df = pd.concat(dfs, ignore_index=True)

                # choose an ID column to dedup on
                id_col = pick(union_df.columns, id_candidates) or "listing_id"

                # consensus count per ID (how many trials selected this ID)
                cons = (
                    union_df[[id_col, "_source_trial"]]
                    .drop_duplicates()
                    .groupby(id_col)["_source_trial"]
                    .size()
                    .rename("consensus_count")
                )
                union_df = union_df.merge(cons, on=id_col, how="left")

                # filter by required consensus (e.g., ≥5)
                min_cons = int(globals().get("UNION_ZERO_SLOW_MIN_CONSENSUS", 1))
                if min_cons > 1:
                    union_df = union_df[union_df["consensus_count"] >= min_cons].copy()

                # dedup by ID (keep one row per ID), prefer higher consensus then higher prob_fast24
                if "prob_fast24" in union_df.columns:
                    union_df = (
                        union_df.sort_values(["consensus_count", "prob_fast24"], ascending=[False, False])
                                 .drop_duplicates(subset=[id_col])
                    )
                else:
                    union_df = (
                        union_df.sort_values(["consensus_count"], ascending=False)
                                 .drop_duplicates(subset=[id_col])
                    )

                # write output
                out_name = globals().get("UNION_ZERO_SLOW_OUTNAME") or "joint_union_consensus.csv"
                out_union = os.path.join(out_dir, out_name)
                union_df.to_csv(out_union, index=False)

                mode = "ANY-TRIALS" if any_mode else "ZERO-SLOW"
                print(f"[JOINT][UNION] Wrote {mode} union: {out_union} | rows={len(union_df)} "
                      f"| trials_used={kept_trials} | min_cons={min_cons} | min_fast={min_fast}")
        except Exception as e:
            print("[JOINT][UNION] Union failed:", e)

    

    return best

# ========================== CLI (BUILD / OVERRIDES) ==========================
def _build_argparser():
    ap = argparse.ArgumentParser(
        description="fast24_flagger — Stage1 (classifier+calibration) + Stage2 (mask-23 tuner) with JOINT tuning"
    )
    # paths
    ap.add_argument("--train",   default=DEFAULT_TRAIN,   type=str)
    ap.add_argument("--slow",    default=DEFAULT_SLOW,    type=str)
    ap.add_argument("--valtext", default=DEFAULT_VALTEXT, type=str)
    ap.add_argument("--oof",     default=DEFAULT_OOF,     type=str)
    ap.add_argument("--outdir",  default=DEFAULT_OUT_DIR, type=str)

    # Stage-2 ranker toggles
    ap.add_argument("--s2_use_lgbm", action="store_true", default=S2_USE_LGBM,
                    help="Enable lightweight LightGBM slow-risk ranker in Stage-2 (default ON).")
    ap.add_argument("--s2_use_emb", action="store_true", default=S2_USE_EMB,
                    help="Also feed embeddings to the Stage-2 ranker (default OFF; enables if set).")

    ap.add_argument("--overlay_mode", choices=["off","soft","hard"], default=OVERLAY_MODE,
                    help="Baseline overlay gating: off (none), soft (lenient), hard (strict; default).")

    ap.add_argument('--s2_pslow_max', type=float, default=S2_P_SLOW_MAX,
                    help='Apply-time p(slow) cap for Stage-2 APPLY; rows with p_slow above this are blocked (default 0.20).')


    ap.add_argument("--no_current_threshold", action="store_true",
                    help="Do NOT use current-slice durations to pick thresholds; use a fixed prob cutoff or target-free share cap.")
    ap.add_argument("--fixed_prob_threshold", type=float, default=None,
                    help="If set, use this probability cutoff (0–1) label-free.")
    ap.add_argument("--max_pred_share", type=float, default=None,
                    help="If set (0–1) and no fixed threshold, choose the largest prob cutoff with predicted share <= this (label-free).")
    ap.add_argument("--no_stage1_label_scoring", action="store_true",
                    help="Disable Stage-1 barrier/JOINT scoring based on current-slice labels (no TP/FP barrier).")
    
    # ========================== DAMAGE FILTER ==========================

    ap.add_argument("--damage_keep_levels", type=str, default=None,
                help='Comma-separated damage levels to keep (e.g., "0,1"). If not set, keep all.')


    ap.add_argument("--expected_slow_count", type=int, default=None,
                help="If set, assert the predicted-slow-first count equals this (fail-fast).")


    ap.add_argument("--union_any_trials", action="store_true",
                    help="Union flips from ALL trials (ignore S2_slow_flipped). Use with --union_zero_slow_min_consensus.")



    ap.add_argument("--cond_gate_min", type=float, default=None,
                    help="If set, filter slow slice to rows with condition_score >= this.")
    ap.add_argument("--cond_gate_max", type=float, default=None,
                    help="If set, filter slow slice to rows with condition_score < this.")

    ap.add_argument("--filter_pred_slow_first", action="store_true",
                    help="Filter to predicted slow (pred_hours>72) before applying the condition corridor.")

    # PTV FILTERS (overall & anchor)
    ap.add_argument("--ptv_overall_max", type=float, default=None,
                    help="If set, keep rows with ptv_overall <= this (train+val corridor).")
    ap.add_argument("--ptv_anchor_max", type=float, default=None,
                    help="If set, keep rows with ptv_anchor  <= this (train+val corridor).")

    

    # Seller corridor (hard gates)
    ap.add_argument("--seller_min_rating",  type=float, default=None,
                    help="If set, keep rows with seller_rating >= this.")
    ap.add_argument("--seller_min_reviews", type=float, default=None,
                    help="If set, keep rows with review_count (or ratings_count) >= this.")
    ap.add_argument("--seller_min_age_y",   type=float, default=None,
                    help="If set, keep rows with (2025 - member_since_year) >= this.")
    ap.add_argument("--seller_quality_min", type=float, default=None,
                    help="If set, keep rows with seller_quality >= this (rating*log1p(reviews)*age_years).")





    # general
    ap.add_argument("--emb",              default="all", choices=list(EMB_CFGS.keys()))
    ap.add_argument("--threshold_steps",  default=1001, type=int)

    # Stage-1 tuning
    ap.add_argument("--tune", type=int, default=0,
                    help="Optuna trials for Stage-1 tuning (0=off). If >0, JOINT score can include Stage-2 per trial (see --tune_mask23_per_trial).")

    # Stage-2 tuning (outer, after final Stage-1 run)
    ap.add_argument("--tune_mask23", type=int, default=0,
                    help="Optuna trials for leak-free Stage-2 tuning after Stage-1 finalization (0=off)")

    # Inline Stage-2 per Stage-1 trial
    ap.add_argument("--tune_mask23_per_trial", default=0, type=int,
                    help="Run this many Stage-2 trials INSIDE EACH Stage-1 Optuna trial to compute JOINT score (0=off)")

    # JOINT objective weights (Stage-1 ⨉ Stage-2)
    ap.add_argument("--joint_w_tp_fp", type=float, default=JOINT_W_TPFP)
    ap.add_argument("--joint_w_fast",  type=float, default=JOINT_W_FAST)
    ap.add_argument("--joint_w_slow",  type=float, default=JOINT_W_SLOW)
    
    ap.add_argument("--joint_w_mid",  type=float, default=JOINT_W_MID)
    ap.add_argument("--joint_w_mae",   type=float, default=JOINT_W_MAE)
    ap.add_argument("--joint_w_rmse",  type=float, default=JOINT_W_RMSE)
    ap.add_argument("--joint_w_r2",    type=float, default=JOINT_W_R2)

    # Stage-1 HF export toggle (reporting only)
    ap.add_argument("--stage1_hf_export", type=int, default=STAGE1_HF_EXPORT, choices=[0, 1],
                    help="If 1, Stage-1 writes/logs *_hardfiltered.csv for reporting. Default 0 (off).")
    # Optional Stage-1 HF knobs
    ap.add_argument("--hf_min_prob",          type=float, default=None, help="Override HF_MIN_PROB_FAST24")
    ap.add_argument("--hf_drop_damage_gte",   type=float, default=None)
    ap.add_argument("--hf_drop_cond_min",     type=float, default=None)
    ap.add_argument("--hf_drop_cond_eq0_too", action="store_true", default=None)

    # ── Stage-2 leak-free split controls
    ap.add_argument("--mask23_train_frac", type=float, default=MASK23_TRAIN_FRAC,
                    help="Fraction of flagged rows (earliest by time) used to tune Stage-2. Default 0.40")
    ap.add_argument("--mask23_cv_blocks",  type=int,   default=MASK23_CV_BLOCKS,
                    help="Rolling blocks for leak-free tuning (1 = single early/late split). Default 1")

    # ── Stage-2: floors/caps/share
    ap.add_argument("--mask23_min_flips",      type=int,   default=MASK23_MIN_FLIPS,
                    help="Minimum flips Stage-2 will select if candidates exist (default 10)")
    ap.add_argument("--mask23_min_share",      type=float, default=MASK23_MIN_SHARE,
                    help="Minimum share of flagged to flip (default 0.20)")
    ap.add_argument("--mask23_max_share",      type=float, default=MASK23_MAX_SHARE,
                    help="Maximum share of flagged to flip (default 0.75)")
    ap.add_argument("--mask23_slow_cap_abs",   type=int,   default=MASK23_SLOW_CAP_ABS,
                    help="Absolute cap on slow(>72h) among flips on TRAIN (default 2)")
    ap.add_argument("--mask23_slow_cap_ratio", type=float, default=MASK23_SLOW_CAP_RATIO,
                    help="Max slow/flips ratio on TRAIN (default 0.10)")
    
    # Per-variant penalty overrides (0.0 = hard block; 1.0 = neutral; >1.0 = boost)
    ap.add_argument("--mp_13_mini",     type=float, default=None)
    ap.add_argument("--mp_13",          type=float, default=None)
    ap.add_argument("--mp_13_pro",      type=float, default=None)
    ap.add_argument("--mp_13_pro_max",  type=float, default=None)


    ap.add_argument("--joint_barrier", choices=["hard","soft"], default="hard")
    ap.add_argument("--barrier_gain_min", type=float, default=0.0)


    # condition score 0 kept )
    ap.add_argument("--cond_keep_zero", action="store_true",
                help="Keep rows with condition_score == 0.0 regardless of cond gate bounds.")

     # STAGE 2 BEST PARAMAS )
    ap.add_argument("--load_mask23_params", type=str, default=None,
                help="Path to a JSON file of Stage-2 (mask-23) best params to APPLY (no tuning).")
    


    # BATTERY PCT FILTER  )
    ap.add_argument("--batt_gate_min", type=float, default=None,
                help="If set, keep rows with battery_pct >= this (train+val corridor).")
    ap.add_argument("--batt_gate_max", type=float, default=None,
                help="If set, keep rows with battery_pct < this (train+val corridor).")
    ap.add_argument("--batt_keep_na", action="store_true",
                help="Keep rows with missing battery_pct when applying the battery corridor.")






    # Union of all zero-slow inline Stage-2 trials (optional; OFF unless enabled)
    ap.add_argument("--union_zero_slow", action="store_true",
                    help="If set, after Stage-1 tuning, union all inline Stage-2 trials with S2_slow_flipped==0 and write a deduped CSV.")
    ap.add_argument("--union_zero_slow_min_fast", type=int, default=None,
                    help="Min S2_fast_flipped required for a trial to be included in the zero-slow union (default 1).")
    ap.add_argument("--union_zero_slow_min_consensus", type=int, default=None,
                    help="Keep only rows that appear in at least this many zero-slow trials (default 1 = any).")
    ap.add_argument("--union_zero_slow_out", type=str, default=None,
                    help="Output filename for the union CSV (default 'joint_zero_slow_union.csv').")



    # ----------------Parmas enabled -----------------------------------------------)

    ap.add_argument("--load_params", type=str, default=None,
                help="Path to a JSON file of best params to load & apply (e.g. best_params.json).")

    


    # ── Stage-2 objective weights (advanced)
    ap.add_argument("--mask23_w_fast",        type=float, default=MASK23_W_FAST)
    ap.add_argument("--mask23_w_slow",        type=float, default=MASK23_W_SLOW)
    ap.add_argument("--mask23_w_mid",         type=float, default=MASK23_W_MID)
    ap.add_argument("--mask23_w_mae",         type=float, default=MASK23_W_MAE)
    ap.add_argument("--mask23_w_rmse",        type=float, default=MASK23_W_RMSE)
    ap.add_argument("--mask23_w_r2",          type=float, default=MASK23_W_R2)
    ap.add_argument("--mask23_w_share_sat",   type=float, default=MASK23_W_SHARE_SAT)
    ap.add_argument("--mask23_barrier_power", type=float, default=MASK23_BARRIER_POWER)

    ap.add_argument("--mask23_zero_slow_bonus", type=float, default=SLOW_ZERO_BONUS,
                    help="Extra reward subtracted from objective when slow_flipped==0 (default 0.0)")

    return ap

def _apply_cli_hf_overrides(args):
    global DEFAULT_OUT_DIR
    # Filter predicted-slow first ordering
    global FILTER_PRED_SLOW_FIRST
    FILTER_PRED_SLOW_FIRST = bool(getattr(args, "filter_pred_slow_first", False))

    global SLOW_ZERO_BONUS
    global S2_P_SLOW_MAX
    
    
    global FIXED_PROB_THRESHOLD
    global OVERLAY_MODE
    global S2_USE_LGBM, S2_USE_EMB
    global HF_MIN_PROB_FAST24, HF_DROP_DAMAGE_GTE, HF_DROP_COND_MIN, HF_DROP_COND_EQ0_TOO, STAGE1_HF_EXPORT
    global MASK23_MIN_FLIPS, MASK23_MIN_SHARE, MASK23_MAX_SHARE, MASK23_SLOW_CAP_ABS, MASK23_SLOW_CAP_RATIO
    global MASK23_TRAIN_FRAC, MASK23_CV_BLOCKS
    global MASK23_W_FAST, MASK23_W_SLOW, MASK23_W_MID, MASK23_W_MAE, MASK23_W_RMSE, MASK23_W_R2, MASK23_W_SHARE_SAT, MASK23_BARRIER_POWER
    global TUNE_MASK23_PER_TRIAL, JOINT_W_TPFP, JOINT_W_FAST, JOINT_W_SLOW, JOINT_W_MAE, JOINT_W_RMSE, JOINT_W_R2

    # Output dir
    if getattr(args, "outdir", None):
        DEFAULT_OUT_DIR = args.outdir

    # Stage-2 ranker toggles
    S2_USE_LGBM = bool(getattr(args, "s2_use_lgbm", S2_USE_LGBM))
    S2_USE_EMB  = bool(getattr(args, "s2_use_emb", S2_USE_EMB))

    # Stage-2 apply-time slow-probability cap
    if getattr(args, 's2_pslow_max', None) is not None:
        S2_P_SLOW_MAX = float(args.s2_pslow_max)
    # Fixed probability threshold for Stage-1 label-free & Stage-2 APPLY preference
    if getattr(args, 'fixed_prob_threshold', None) is not None:
        FIXED_PROB_THRESHOLD = float(args.fixed_prob_threshold)

    # overlay mode from CLI
    if getattr(args, 'overlay_mode', None) is not None:
        OVERLAY_MODE = str(args.overlay_mode).lower()

    global EXPECTED_SLOW_COUNT
    EXPECTED_SLOW_COUNT = getattr(args, "expected_slow_count", None)



    # Stage-1 HF overrides
    if getattr(args, "hf_min_prob", None) is not None:
        HF_MIN_PROB_FAST24 = float(args.hf_min_prob)
    if getattr(args, "hf_drop_damage_gte", None) is not None:
        HF_DROP_DAMAGE_GTE = float(args.hf_drop_damage_gte)
    if getattr(args, "hf_drop_cond_min", None) is not None:
        HF_DROP_COND_MIN = float(args.hf_drop_cond_min)
    if getattr(args, "hf_drop_cond_eq0_too", None) is not None:
        HF_DROP_COND_EQ0_TOO = bool(args.hf_drop_cond_eq0_too)
    if getattr(args, "stage1_hf_export", None) in (0, 1):
        STAGE1_HF_EXPORT = int(args.stage1_hf_export)

    # Stage-2 leak-free split
    if getattr(args, "mask23_train_frac", None) is not None:
        MASK23_TRAIN_FRAC = float(args.mask23_train_frac)
    if getattr(args, "mask23_cv_blocks", None) is not None:
        MASK23_CV_BLOCKS = int(args.mask23_cv_blocks)

    # DAMAGE FILTER 
    global DAMAGE_KEEP_LEVELS
    dlevels = getattr(args, "damage_keep_levels", None)
    if dlevels:
        DAMAGE_KEEP_LEVELS = set(int(s.strip()) for s in str(dlevels).split(",") if s.strip() != "")
    else:
        DAMAGE_KEEP_LEVELS = None

    # Optional PTV corridors via CLI
    global PTV_OVERALL_MAX, PTV_ANCHOR_MAX
    PTV_OVERALL_MAX = getattr(args, "ptv_overall_max", None)
    PTV_ANCHOR_MAX  = getattr(args, "ptv_anchor_max",  None)



    # Stage-2 floors/caps/share
    if getattr(args, "mask23_min_flips", None) is not None:
        MASK23_MIN_FLIPS = int(args.mask23_min_flips)
    if getattr(args, "mask23_min_share", None) is not None:
        MASK23_MIN_SHARE = float(args.mask23_min_share)
    if getattr(args, "mask23_max_share", None) is not None:
        MASK23_MAX_SHARE = float(args.mask23_max_share)
    if getattr(args, "mask23_slow_cap_abs", None) is not None:
        MASK23_SLOW_CAP_ABS = int(args.mask23_slow_cap_abs)
    if getattr(args, "mask23_slow_cap_ratio", None) is not None:
        MASK23_SLOW_CAP_RATIO = float(args.mask23_slow_cap_ratio)

    # Union-zero-slow controls
    global UNION_ZERO_SLOW, UNION_ZERO_SLOW_MIN_FAST, UNION_ZERO_SLOW_MIN_CONSENSUS, UNION_ZERO_SLOW_OUTNAME
    UNION_ZERO_SLOW = bool(getattr(args, "union_zero_slow", False))
    if getattr(args, "union_zero_slow_min_fast", None) is not None:
        UNION_ZERO_SLOW_MIN_FAST = int(args.union_zero_slow_min_fast)
    if getattr(args, "union_zero_slow_min_consensus", None) is not None:
        UNION_ZERO_SLOW_MIN_CONSENSUS = int(args.union_zero_slow_min_consensus)
    if getattr(args, "union_zero_slow_out", None):
        UNION_ZERO_SLOW_OUTNAME = str(args.union_zero_slow_out)

    global UNION_ANY_TRIALS
    UNION_ANY_TRIALS = bool(getattr(args, "union_any_trials", False))

   

    # Stage-2 objective weights
    if getattr(args, "mask23_w_fast", None) is not None:
        MASK23_W_FAST = float(args.mask23_w_fast)
    if getattr(args, "mask23_w_slow", None) is not None:
        MASK23_W_SLOW = float(args.mask23_w_slow)
    if getattr(args, "mask23_w_mid", None) is not None:
        MASK23_W_MID = float(args.mask23_w_mid)
    if getattr(args, "mask23_w_mae", None) is not None:
        MASK23_W_MAE = float(args.mask23_w_mae)
    if getattr(args, "mask23_w_rmse", None) is not None:
        MASK23_W_RMSE = float(args.mask23_w_rmse)
    if getattr(args, "mask23_w_r2", None) is not None:
        MASK23_W_R2 = float(args.mask23_w_r2)
    if getattr(args, "mask23_w_share_sat", None) is not None:
        MASK23_W_SHARE_SAT = float(args.mask23_w_share_sat)
    if getattr(args, "mask23_barrier_power", None) is not None:
        MASK23_BARRIER_POWER = float(args.mask23_barrier_power)

    
    # condition 0 kept )
    global COND_KEEP_ZERO
    COND_KEEP_ZERO = bool(getattr(args, "cond_keep_zero", False))


    # BATTERY PCT FILTER  )
    global BATT_GATE_MIN, BATT_GATE_MAX, BATT_KEEP_NA
    BATT_GATE_MIN = getattr(args, "batt_gate_min", None)
    BATT_GATE_MAX = getattr(args, "batt_gate_max", None)
    BATT_KEEP_NA  = bool(getattr(args, "batt_keep_na", False))

    # Seller corridor (hard gates)
    global SELLER_GATE_MIN_RATING, SELLER_GATE_MIN_REVIEWS, SELLER_GATE_MIN_AGE_Y, SELLER_GATE_MIN_QUALITY
    SELLER_GATE_MIN_RATING  = getattr(args, "seller_min_rating",  None)
    SELLER_GATE_MIN_REVIEWS = getattr(args, "seller_min_reviews", None)
    SELLER_GATE_MIN_AGE_Y   = getattr(args, "seller_min_age_y",   None)
    SELLER_GATE_MIN_QUALITY = getattr(args, "seller_quality_min", None)




    global JOINT_BARRIER, BARRIER_GAIN_MIN
    JOINT_BARRIER    = getattr(args, "joint_barrier", "hard")
    BARRIER_GAIN_MIN = float(getattr(args, "barrier_gain_min", 0.0))




    # Per-variant penalties from CLI (override for this run)
    global G_MODEL_PENALTIES, _CLI_MP_OVERRIDE
    _CLI_MP_OVERRIDE = {}
    if getattr(args, "mp_13_mini", None)     is not None: _CLI_MP_OVERRIDE["iPhone 13 Mini"]    = float(args.mp_13_mini)
    if getattr(args, "mp_13", None)          is not None: _CLI_MP_OVERRIDE["iPhone 13"]         = float(args.mp_13)
    if getattr(args, "mp_13_pro", None)      is not None: _CLI_MP_OVERRIDE["iPhone 13 Pro"]     = float(args.mp_13_pro)
    if getattr(args, "mp_13_pro_max", None)  is not None: _CLI_MP_OVERRIDE["iPhone 13 Pro Max"] = float(args.mp_13_pro_max)
    if _CLI_MP_OVERRIDE:
        G_MODEL_PENALTIES.update(_CLI_MP_OVERRIDE)


    # JOINT tuning controls
    if getattr(args, "tune_mask23_per_trial", None) is not None:
        TUNE_MASK23_PER_TRIAL = int(args.tune_mask23_per_trial)

    global JOINT_W_TPFP, JOINT_W_FAST, JOINT_W_SLOW, JOINT_W_MID, JOINT_W_MAE, JOINT_W_RMSE, JOINT_W_R2
    JOINT_W_TPFP = float(getattr(args, "joint_w_tp_fp", JOINT_W_TPFP))
    JOINT_W_FAST = float(getattr(args, "joint_w_fast",  JOINT_W_FAST))
    JOINT_W_SLOW = float(getattr(args, "joint_w_slow",  JOINT_W_SLOW))
    JOINT_W_MID  = float(getattr(args, "joint_w_mid",   JOINT_W_MID))
    JOINT_W_MAE  = float(getattr(args, "joint_w_mae",   JOINT_W_MAE))
    JOINT_W_RMSE = float(getattr(args, "joint_w_rmse",  JOINT_W_RMSE))
    JOINT_W_R2   = float(getattr(args, "joint_w_r2",    JOINT_W_R2))



    global NO_STAGE1_LABEL_SCORING
    NO_STAGE1_LABEL_SCORING = bool(getattr(args, "no_stage1_label_scoring", False))



    # Optional condition gating (per-condition corridors via CLI)
    global COND_GATE_MIN, COND_GATE_MAX
    COND_GATE_MIN = getattr(args, "cond_gate_min", None)
    COND_GATE_MAX = getattr(args, "cond_gate_max", None)


def main():
    ap = _build_argparser()
    args = ap.parse_args()
    # Mark run mode for downstream invariants/asserts
    global RUN_MODE
    RUN_MODE = "tune" if (args.tune and args.tune > 0) else "apply"

    # Make sure outdir is set and exists
    global DEFAULT_OUT_DIR
    DEFAULT_OUT_DIR = args.outdir
    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

    # Load best-params JSON (optional)
    if getattr(args, "load_params", None):
        try:
            with open(args.load_params, "r", encoding="utf-8") as f:
                best = json.load(f)
            _apply_knobs(best)  # pushes keys like CALIBRATION, PREC_MIN, MP_* into globals
            print(f"[PARAMS] Loaded {len(best)} params from {args.load_params}")
        except Exception as e:
            print(f"[PARAMS] Failed to load params from {args.load_params}: {e}")

    # Apply CLI overrides to globals
    _apply_cli_hf_overrides(args)

    # Stage-1: optionally tune (JOINT objective if --tune_mask23_per_trial > 0)
    if args.tune and args.tune > 0:
        print(f"[Stage-1] Tuning for {args.tune} trials…")
        run_stage1_tuning(
            n_trials=int(args.tune),
            train_path=args.train, slow_path=args.slow, val_text_path=args.valtext, oof_path=args.oof,
            out_dir=args.outdir, emb_choice=args.emb, threshold_steps=int(args.threshold_steps)
        )

    print("[Stage-1] Final pipeline run…")
    # Keep RUN_MODE='tune' when tuning; only force 'apply' in apply-only runs
    if not (args.tune and args.tune > 0):
        RUN_MODE = "apply"

    res = pipeline_run(
        train_path=args.train, slow_path=args.slow, val_text_path=args.valtext, oof_path=args.oof,
        out_dir=args.outdir, emb_choice=args.emb, threshold_steps=int(args.threshold_steps), write_outputs=True
    )

    # Stage-2: outer mask-23 tuner OR apply-only path
    if args.tune_mask23 and args.tune_mask23 > 0:
        print(f"[Stage-2] mask-23 tuning for {args.tune_mask23} trials…")
        run_mask23_tuning(
            val_out=res["val_out"], dur_col=res.get("dur_col"), model_pred_col=res.get("model_pred_col"),
            out_dir=args.outdir, n_trials=int(args.tune_mask23), silent=False, write_outputs=True
        )
    else:
        # Stage-2 apply-only: load & apply params if provided
        if getattr(args, "load_mask23_params", None):
            try:
                with open(args.load_mask23_params, "r", encoding="utf-8") as f:
                    m23 = json.load(f)

                # Stash full Stage-2 JSON + where it lives (for apply-only)
                globals()["_MASK23_PARAMS_FULL"] = dict(m23)
                globals()["_MASK23_PARAMS_DIR"]  = os.path.dirname(args.load_mask23_params)

                # Push into globals used by the policy
                globals()["MASK23_P_THRESHOLD"]       = float(m23.get("p_threshold",       globals().get("MASK23_P_THRESHOLD", 0.0)))
                globals()["MASK23_DELTA_OVERALL_MAX"] = float(m23.get("delta_overall_max", globals().get("MASK23_DELTA_OVERALL_MAX", 0.0)))
                globals()["MASK23_PTV_OVERALL_MAX"]   = float(m23.get("ptv_overall_max",   globals().get("MASK23_PTV_OVERALL_MAX", 1.0)))
                globals()["MASK23_PTV_ANCHOR_MAX"]    = float(m23.get("ptv_anchor_max",    globals().get("MASK23_PTV_ANCHOR_MAX", 1.0)))
                globals()["MASK23_COND_MIN"]          = float(m23.get("cond_min",          globals().get("MASK23_COND_MIN", 0.0)))
                globals()["MASK23_DAMAGE_MAX"]        = float(m23.get("damage_max",        globals().get("MASK23_DAMAGE_MAX", 99.0)))
                globals()["MASK23_BATT_MIN"]          = float(m23.get("batt_min",          globals().get("MASK23_BATT_MIN", 0.0)))
                globals()["MASK23_DPM14_MAX"]         = float(m23.get("dpm14_max",         globals().get("MASK23_DPM14_MAX", 1e9)))
                globals()["MASK23_FLIP_SHARE"]        = float(m23.get("flip_share",        globals().get("MASK23_FLIP_SHARE", 0.0)))
                globals()["MASK23_SELLER_MIN_RATING"]   = float(m23.get("seller_min_rating",   globals().get("MASK23_SELLER_MIN_RATING",  -1e9)))
                globals()["MASK23_SELLER_MIN_REVIEWS"]  = float(m23.get("seller_min_reviews",  globals().get("MASK23_SELLER_MIN_REVIEWS", -1e9)))
                globals()["MASK23_SELLER_MIN_AGE_Y"]    = float(m23.get("seller_min_age_y",    globals().get("MASK23_SELLER_MIN_AGE_Y",   -1e9)))
                globals()["MASK23_SELLER_QUALITY_MIN"]  = float(m23.get("seller_quality_min",  globals().get("MASK23_SELLER_QUALITY_MIN", -1e9)))
                print(f"[Stage-2] Loaded mask-23 params from {args.load_mask23_params}")

                # Apply once over the current flagged set (no tuning)
                apply_mask23_once(
                    res["val_out"],
                    res.get("dur_col"),
                    res.get("model_pred_col"),
                    args.outdir
                )
            except Exception as e:
                print(f"[Stage-2] Failed to apply mask-23 params: {e}")
        else:
            print("[Stage-2] Skipped (no tuning and no --load_mask23_params).")

if __name__ == "__main__":
    main()
