"""Archived numerical definitions; see provenance.json for exact source hashes.
Only the module preamble is new. Bodies and decorators remain unchanged.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

CONTAM_1021_MULT = 1.0
CONTAM_GT21_MULT = 1.0
CONTAM_710_CUT_H = 240.0
CONTAM_1021_CUT_H = 504.0

@dataclass(frozen=True)
class Policy:
    precision_min: float
    sac_max: float
    min_bucket_total: int
    bucket_frac: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def safe_logit(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compatibility wrapper for logit with clipping.

    Some meta-ensemble paths call `safe_logit`; define it as an alias of
    `_logit` so the script does not crash.
    """
    return _logit(p, eps=eps)


def method_base(method: str) -> str:
    """Normalize method names (e.g., 'phase2_wlogit' -> 'wlogit')."""
    if method is None:
        return ""
    m = str(method).strip().lower()
    # Strip common phase/tag prefixes without breaking method names like 'logit_lcb'
    m = re.sub(r"^phase\d+_", "", m)
    m = re.sub(r"^(combined|best|ref)_", "", m)
    return m


def _nll_binary(y01: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    y = y01.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature_scaler(p: np.ndarray, y01: np.ndarray, t_min: float = 0.5, t_max: float = 5.0, grid: int = 50) -> float:
    """
    Fit temperature T>0 for: p_cal = sigmoid(logit(p) / T)
    using a simple log-spaced grid-search to avoid SciPy dependency.
    """
    p = np.asarray(p, dtype=float)
    y01 = np.asarray(y01, dtype=int)

    # If calibration set is degenerate, return identity.
    if p.size == 0 or np.unique(y01).size < 2:
        return 1.0

    l = _logit(p, eps=1e-9)

    # If logits are (almost) constant, scaling cannot help.
    if float(np.nanstd(l)) < 1e-9:
        return 1.0

    t_min = max(1e-3, float(t_min))
    t_max = max(t_min * 1.0001, float(t_max))
    grid = int(max(10, grid))

    Ts = np.logspace(np.log10(t_min), np.log10(t_max), grid)
    best_T = 1.0
    best_nll = float("inf")
    for T in Ts:
        pT = _sigmoid(l / T)
        nll = _nll_binary(y01, pT)
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    T = max(1e-6, float(T))
    return _sigmoid(_logit(p, eps=1e-9) / T)


def fit_isotonic_calibrator(p: np.ndarray, y01: np.ndarray) -> Optional[IsotonicRegression]:
    p = np.asarray(p, dtype=float)
    y01 = np.asarray(y01, dtype=int)
    if p.size == 0 or np.unique(y01).size < 2:
        return None
    # If predictions are constant, isotonic won't help.
    if float(np.nanstd(p)) < 1e-12:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y01.astype(float))
    return iso


def calibrate_prediction_matrices(
    P_sval: np.ndarray,
    P_hold: np.ndarray,
    y_sval_full: np.ndarray,
    calib_mask_full: np.ndarray,
    kind: str = "temp",
    temp_min: float = 0.5,
    temp_max: float = 5.0,
    temp_grid: int = 50,
    clip_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Calibrate each base trial's p_fast using SVAL calibration set (mask over FULL SVAL rows),
    then apply the same mapping to both SVAL and HOLDOUT.

    Returns (P_sval_cal, P_hold_cal, summary_dict).
    """
    kind = str(kind).lower()
    if kind in ("none", "off", "false", "0"):
        return P_sval, P_hold, {"kind": "none"}

    P_sval = np.asarray(P_sval, dtype=float)
    P_hold = np.asarray(P_hold, dtype=float)

    # Clip once up-front to avoid inf logits.
    P_sval = np.clip(P_sval, clip_eps, 1.0 - clip_eps)
    P_hold = np.clip(P_hold, clip_eps, 1.0 - clip_eps)

    calib_mask_full = np.asarray(calib_mask_full, dtype=bool)
    y_cal = np.asarray(y_sval_full, dtype=int)[calib_mask_full]

    n_trials = P_sval.shape[0]
    temps: List[float] = []
    iso_used = 0

    P_s_cal = P_sval.copy()
    P_h_cal = P_hold.copy()

    for ti in range(n_trials):
        p_calib = P_sval[ti, calib_mask_full]

        if kind.startswith("temp"):
            T = fit_temperature_scaler(p_calib, y_cal, t_min=temp_min, t_max=temp_max, grid=temp_grid)
            temps.append(float(T))
            P_s_cal[ti, :] = apply_temperature(P_sval[ti, :], T)
            P_h_cal[ti, :] = apply_temperature(P_hold[ti, :], T)

        elif kind.startswith("iso"):
            iso = fit_isotonic_calibrator(p_calib, y_cal)
            if iso is None:
                # identity
                P_s_cal[ti, :] = P_sval[ti, :]
                P_h_cal[ti, :] = P_hold[ti, :]
            else:
                iso_used += 1
                P_s_cal[ti, :] = iso.predict(P_sval[ti, :])
                P_h_cal[ti, :] = iso.predict(P_hold[ti, :])

        else:
            raise ValueError(f"Unknown calibration kind: {kind} (expected none/temp/isotonic)")

    summary: Dict[str, Any] = {"kind": kind, "n_trials": int(n_trials)}
    if temps:
        temps_arr = np.asarray(temps, dtype=float)
        summary.update(
            {
                "temp_min": float(np.min(temps_arr)),
                "temp_median": float(np.median(temps_arr)),
                "temp_mean": float(np.mean(temps_arr)),
                "temp_max": float(np.max(temps_arr)),
                "temp_grid": int(temp_grid),
                "temp_range": [float(temp_min), float(temp_max)],
            }
        )
    if kind.startswith("iso"):
        summary["iso_used"] = int(iso_used)

    return P_s_cal, P_h_cal, summary


def prcs_score(P: float, R: float, C: float, S: float, wP: float, wR: float, wC: float, wS: float) -> float:
    # logit(P) is the "precision on log-odds" term you liked (strong sensitivity near high P)
    P_eff = float(P)
    if P_LOGIT_CAP is not None and float(P_LOGIT_CAP) < 1.0:
        P_eff = min(P_eff, float(P_LOGIT_CAP))
    lp = float(_logit(np.array([P_eff], dtype=float))[0])
    return float(wP) * lp + float(wR) * float(R) - float(wC) * float(C) - float(wS) * float(S)


def _scaled_outer_min_bucket(min_bucket_inner: int, inner_n: int, outer_n: int) -> int:
    """Scale an INNER absolute bucket minimum to an OUTER subset size."""
    if outer_n <= 0:
        return 0
    scaled = int(math.ceil(float(min_bucket_inner) * (float(outer_n) / max(1.0, float(inner_n)))))
    return max(5, min(scaled, outer_n))


def apply_outer_soft_penalty_prcs(obj: float, cm_outer: Dict[str, float], policy: Policy, inner_n: int, outer_n: int) -> float:
    """Soft penalties if OUTER violates policy; avoids -1e18 trial waste while discouraging overfit."""
    if outer_n <= 0:
        return float(obj)

    p = float(cm_outer.get("precision", 0.0))
    s = float(cm_outer.get("sacrifice", 1.0))
    b = int(cm_outer.get("bucket", 0))

    pen = 0.0
    # Precision shortfall is the most serious.
    if p < float(policy.precision_min):
        pen += 30.0 * (float(policy.precision_min) - p)

    # Sacrifice above cap.
    if s > float(policy.sac_max):
        pen += 15.0 * (s - float(policy.sac_max))

    # Bucket too small (scaled).
    b_min = _scaled_outer_min_bucket(int(policy.min_bucket_total), inner_n, outer_n)
    if b_min > 0 and b < b_min:
        pen += 5.0 * (float(b_min - b) / float(b_min))

    return float(obj) - pen


def outer_objective_recall_under_contam_cap(
    cm_outer: Dict[str, float],
    policy: Policy,
    contam_cap: float,
    inner_n: int,
    outer_n: int,
    pen_contam: float = 120.0,
    pen_precision: float = 30.0,
    pen_bucket: float = 5.0,
) -> float:
    """Phase-3 OUTER objective: maximize recall while respecting contam cap.

    If OUTER violates cap/precision/bucket, we return recall minus penalties (NOT -1e18),
    so Optuna can still learn without burning most trials.
    """
    # No OUTER split -> just return recall.
    if outer_n <= 0:
        return float(cm_outer.get("recall", 0.0))

    p = float(cm_outer.get("precision", 0.0))
    r = float(cm_outer.get("recall", 0.0))
    c = float(cm_outer.get("contamination", 1.0))
    b = int(cm_outer.get("bucket", 0))

    obj = r

    if contam_cap is not None and float(contam_cap) > 0.0 and c > float(contam_cap):
        obj -= float(pen_contam) * (c - float(contam_cap))

    if p < float(policy.precision_min):
        obj -= float(pen_precision) * (float(policy.precision_min) - p)

    b_min = _scaled_outer_min_bucket(int(policy.min_bucket_total), inner_n, outer_n)
    if b_min > 0 and b < b_min:
        obj -= float(pen_bucket) * (float(b_min - b) / float(b_min))

    return float(obj)


def confusion_metrics_sold(
    p_fast_sold: Optional[np.ndarray] = None,
    dur_sold: Optional[np.ndarray] = None,
    thr: float = 0.5,
    slow_h: float = 168.0,
    *,
    p_fast: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute confusion metrics on SOLD-only rows.

    Note: Some meta-ensemble callers historically used `p_fast=` instead of
    `p_fast_sold=`. We accept both for backwards compatibility.
    """
    if p_fast_sold is None:
        p_fast_sold = p_fast
    if p_fast_sold is None or dur_sold is None:
        raise TypeError("confusion_metrics_sold requires p_fast_sold (or p_fast alias) and dur_sold")

    true_fast = dur_sold <= slow_h
    true_slow = ~true_fast
    pred_fast = p_fast_sold >= thr

    tp = int(np.sum(pred_fast & true_fast))
    fp = int(np.sum(pred_fast & true_slow))
    fn = int(np.sum((~pred_fast) & true_fast))
    tn = int(np.sum((~pred_fast) & true_slow))

    bucket = tp + fp
    precision = tp / bucket if bucket > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    contam = fp / bucket if bucket > 0 else float("nan")
    sac = fn / (tp + fn) if (tp + fn) > 0 else float("nan")

    # Objective-weighted contamination: overweight 10-21d and >21d false-positives if configured.
    # This targets the business cost of capital lock-up in the medium/long tail.
    fp_mask = pred_fast & true_slow
    fp_710 = int(np.sum(fp_mask & (dur_sold <= float(CONTAM_710_CUT_H))))
    fp_1021 = int(np.sum(fp_mask & (dur_sold > float(CONTAM_710_CUT_H)) & (dur_sold <= float(CONTAM_1021_CUT_H))))
    fp_gt21 = int(np.sum(fp_mask & (dur_sold > float(CONTAM_1021_CUT_H))))
    contam_obj = (
        (float(fp_710) + float(CONTAM_1021_MULT) * float(fp_1021) + float(CONTAM_GT21_MULT) * float(fp_gt21)) / float(bucket)
        if bucket > 0 else float("nan")
    )

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "contamination": float(contam),
        "contamination_obj": float(contam_obj),
        "fp_710": int(fp_710),
        "fp_1021": int(fp_1021),
        "fp_gt21": int(fp_gt21),
        "sacrifice": float(sac),
        "n_eval": int(len(dur_sold)),
        "true_fast": int(np.sum(true_fast)),
        "true_slow": int(np.sum(true_slow)),
        "bucket": int(bucket),
    }


def threshold_search_best_prcs_score_exact(
    p_fast_sold: np.ndarray,
    dur_sold: np.ndarray,
    slow_h: float,
    min_bucket_total: int,
    precision_min: float,
    sac_max: float,
    wP: float,
    wR: float,
    wC: float,
    wS: float,
) -> Tuple[float, Dict[str, float], float]:
    """
    Exact threshold search over all unique p_fast values (plus 0 and 1).

    Hard gates (feasible):
      - bucket >= min_bucket_total
      - precision >= precision_min   (if precision_min >= 0)
      - sacrifice <= sac_max         (if sac_max >= 0)

    Objective (feasible):
      PRCS score = wP*logit(P) + wR*R - wC*C - wS*S

    Important:
      If *no* threshold is feasible, we return the "closest" threshold (minimum total constraint
      violation) with a *negative* objective value. This avoids NaN thresholds (which collapse to
      predicting everything SLOW) and gives Optuna a gradient signal instead of repeating -1e18.
    """
    if len(p_fast_sold) == 0:
        return float("nan"), {}, -1e18

    # Candidate thresholds = unique probs + endpoints
    thrs = np.unique(p_fast_sold.astype(float))
    thrs = np.concatenate([np.array([0.0], dtype=float), thrs, np.array([1.0], dtype=float)])
    thrs = np.unique(thrs)

    best_score = -1e18
    best_thr = float("nan")
    best_cm: Dict[str, float] = {}

    # Best "closest" infeasible (min violation)
    best_thr_v = float("nan")
    best_cm_v: Dict[str, float] = {}
    best_joint_v = float("inf")
    best_recall_v = -1.0

    min_bucket_total = int(max(1, int(min_bucket_total)))

    for thr in thrs:
        cm = confusion_metrics_sold(p_fast_sold, dur_sold, float(thr), slow_h)
        P = cm.get("precision", float("nan"))
        R = cm.get("recall", float("nan"))
        C = cm.get("contamination", float("nan"))
        S = cm.get("sacrifice", float("nan"))

        if math.isnan(P) or math.isnan(R) or math.isnan(C) or math.isnan(S):
            continue

        bucket = int(cm.get("bucket", 0))

        # Feasible check
        if bucket >= min_bucket_total and (precision_min < 0.0 or P >= precision_min) and (sac_max < 0.0 or S <= sac_max):
            C_obj = float(cm.get('contamination_obj', C))
            score = prcs_score(P, R, C_obj, S, wP=wP, wR=wR, wC=wC, wS=wS)
            if score > best_score:
                best_score = float(score)
                best_thr = float(thr)
                best_cm = cm

        # Violation (always tracked for fallback)
        v_bucket = max(0.0, float(min_bucket_total - bucket) / float(max(min_bucket_total, 1)))
        v_prec = 0.0 if precision_min < 0.0 else max(0.0, float(precision_min - P) / float(max(precision_min, 1e-6)))
        v_sac = 0.0 if sac_max < 0.0 else max(0.0, float(S - sac_max) / float(max(sac_max, 1e-6)))
        joint = v_bucket + v_prec + v_sac

        if (joint < best_joint_v - 1e-12) or (abs(joint - best_joint_v) <= 1e-12 and R > best_recall_v):
            best_joint_v = float(joint)
            best_thr_v = float(thr)
            best_cm_v = cm
            best_recall_v = float(R)

    if best_cm:
        return best_thr, best_cm, float(best_score)

    if best_cm_v:
        # Negative objective encodes how far we are from feasibility.
        return best_thr_v, best_cm_v, -float(best_joint_v)

    return float("nan"), {}, -1e18


def threshold_search_min_sac_under_contam_cap(
    p_fast_s_sold: np.ndarray,
    dur_s_sold: np.ndarray,
    slow_h: float,
    policy: Policy,
    contam_cap: float,
    n_thr: int = 400,
) -> Tuple[float, Dict[str, float], float]:
    """Phase-3 threshold search: maximize recall (minimize sacrifice) under a hard contamination cap.

    Feasible constraints:
      - bucket >= policy.min_bucket_total
      - precision >= policy.precision_min
      - contamination <= contam_cap

    Optuna efficiency note:
      If *no* threshold is feasible, we still return the "closest" threshold (minimum
      contamination violation; precision violation as a secondary soft penalty) and a
      *negative* objective value:
          obj = -(violation / max(contam_cap, 1e-6))

      This prevents Phase-3 from burning hundreds of trials returning the same -1e18.
      Any feasible trial returns a non-negative objective (recall in [0,1]) and will
      automatically dominate infeasible trials (negative objective).
    """
    eps = 1e-12
    contam_cap = float(contam_cap)
    denom_cap = max(contam_cap, 1e-6)

    thrs = np.linspace(0.0, 1.0, int(n_thr), dtype=float)

    # Best feasible
    best_thr_f = float("nan")
    best_cm_f: Dict[str, float] = {}
    best_recall_f = -1.0

    # Best "closest" infeasible (min joint violation)
    best_thr_v = float("nan")
    best_cm_v: Dict[str, float] = {}
    best_joint_v = float("inf")
    best_recall_v = -1.0

    for thr in thrs:
        cm = confusion_metrics_sold(p_fast_s_sold, dur_s_sold, thr, slow_h=slow_h)

        # Hard guard against tiny buckets
        if cm["bucket"] < policy.min_bucket_total:
            continue

        # Compute violations (contamination is primary; precision secondary)
        contam_violation = max(0.0, cm["contamination"] - contam_cap)
        prec_violation = max(0.0, policy.precision_min - cm["precision"])

        feasible = (contam_violation <= 0.0 + eps) and (prec_violation <= 0.0 + eps)

        if feasible:
            if cm["recall"] > best_recall_f:
                best_recall_f = cm["recall"]
                best_thr_f = float(thr)
                best_cm_f = cm
        else:
            # Joint: contamination dominates; precision secondary (scaled down)
            joint = contam_violation + 0.1 * prec_violation
            if (joint + eps) < best_joint_v or (abs(joint - best_joint_v) <= eps and cm["recall"] > best_recall_v):
                best_joint_v = joint
                best_recall_v = cm["recall"]
                best_thr_v = float(thr)
                best_cm_v = cm

    if best_cm_f:
        best_cm_f["feasible"] = 1.0
        best_cm_f["contam_cap"] = float(contam_cap)
        best_cm_f["contam_violation"] = 0.0
        return best_thr_f, best_cm_f, float(best_recall_f)

    if best_cm_v:
        best_cm_v["feasible"] = 0.0
        best_cm_v["contam_cap"] = float(contam_cap)
        best_cm_v["contam_violation"] = float(best_joint_v)
        obj = -float(best_joint_v) / denom_cap  # negative, closer to 0 is better
        return best_thr_v, best_cm_v, float(obj)

    # Extremely rare: if even bucket constraint can't be met
    return float("nan"), {}, -1e18


def _fit_lda_logit(X: np.ndarray, y: np.ndarray, lda_lambda: float) -> Tuple[np.ndarray, float]:
    """
    X: (n, d) logits, y: (n,) in {0,1}
    Returns w, b such that score = X @ w + b
    Uses a shrinked pooled covariance (ridge) like in v7_smart.
    """
    d = X.shape[1]
    lam = float(lda_lambda)
    # class means
    m0 = X[y == 0].mean(axis=0) if np.any(y == 0) else X.mean(axis=0)
    m1 = X[y == 1].mean(axis=0) if np.any(y == 1) else X.mean(axis=0)

    # pooled covariance
    Xc = X - X.mean(axis=0, keepdims=True)
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    Sigma_reg = Sigma + lam * np.eye(d, dtype=float)
    try:
        inv = np.linalg.inv(Sigma_reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_reg)

    w = inv @ (m1 - m0)
    # intercept that centers at midpoint
    b = -0.5 * float((m1 + m0).dot(w))
    return w.astype(float), float(b)


def ensemble_p_fast(
    P: np.ndarray,
    method: str,
    *,
    trim_ratio: float = 0.0,
    kappa: float = 0.0,
    mv_lambda: float = 0.0,
    lda_lambda: float = 1e-4,
    wlogit_C: float = 1.0,
    y_fast_sold: Optional[np.ndarray] = None,
    sold_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Combine P (n_models, n_rows) into p_fast (n_rows).

    Returns:
      p_fast_all
      learned_w (or None)
      extra dict (intercept, etc)
    """
    if P.shape[0] == 1:
        return P[0], None, {"intercept": 0.0}

    if method == "mean":
        return P.mean(axis=0), None, {"intercept": 0.0}

    if method == "logit_lcb":
        logits = _logit(P)
        n = logits.shape[0]
        k = int(math.floor(float(trim_ratio) * n))
        logits_sorted = np.sort(logits, axis=0)
        if 2 * k < n:
            core = logits_sorted[k:n-k, :]
        else:
            core = logits_sorted
        mu = core.mean(axis=0)
        sd = core.std(axis=0, ddof=0)
        return _sigmoid(mu - float(kappa) * sd), None, {"intercept": 0.0}

    if method == "mv_logit":
        # Markowitz / min-variance weights on logits
        X = _logit(P).T  # (n_rows, n_models)
        Xc = X - X.mean(axis=0, keepdims=True)
        Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
        lam = float(mv_lambda)
        Sigma_reg = Sigma + lam * np.eye(Sigma.shape[0], dtype=float)
        try:
            inv = np.linalg.inv(Sigma_reg)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(Sigma_reg)
        ones = np.ones((Sigma.shape[0], 1), dtype=float)
        w = inv @ ones
        w = w / (ones.T @ w)
        w = w.ravel()
        z = X @ w
        return _sigmoid(z), w.astype(float), {"intercept": 0.0}

    if method == "lda_logit":
        if y_fast_sold is None or sold_mask is None:
            raise ValueError("lda_logit requires y_fast_sold and sold_mask")
        logits_all = _logit(P).T  # (n_rows, n_models)
        X = logits_all[sold_mask, :]
        y = y_fast_sold.astype(int)
        w, b = _fit_lda_logit(X, y, lda_lambda=float(lda_lambda))
        z = logits_all @ w + b
        return _sigmoid(z), w.astype(float), {"intercept": float(b), "lda_lambda": float(lda_lambda)}

    if method == "wlogit":
        if y_fast_sold is None or sold_mask is None:
            raise ValueError("wlogit requires y_fast_sold and sold_mask")
        logits_all = _logit(P).T  # (n_rows, n_models)
        X = logits_all[sold_mask, :]
        y = y_fast_sold.astype(int)
        # C is inverse regularization strength in sklearn; enforce positivity
        C = float(max(1e-6, wlogit_C))
        lr = LogisticRegression(
            penalty="l2",
            C=C,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=2000,
        )
        lr.fit(X, y)
        w = lr.coef_.ravel().astype(float)
        b = float(lr.intercept_.ravel()[0])
        z = logits_all @ w + b
        return _sigmoid(z), w.astype(float), {"intercept": float(b), "wlogit_C": float(C)}

    raise ValueError(f"Unknown method: {method}")
