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

def _fast_labels_and_mask(y_dur_rawours: np.ndarray, y_sold_event: np.ndarray, H: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAST label at horizon H, censoring-aware:
      - y_fast = 1 if sold and duration<=H
      - y_fast = 0 if sold and duration>H
      - if censored:
           known negative ONLY if follow-up>=H (y_fast=0),
           otherwise unknown (mask_known=False)
    """
    y_dur = y_dur_rawours.astype(float)
    sold = (y_sold_event.astype(float) > 0.5)
    cens = ~sold

    y_fast = np.zeros_like(y_dur, dtype=np.int64)
    y_fast[sold & (y_dur <= float(H))] = 1
    y_fast[sold & (y_dur > float(H))] = 0
    y_fast[cens & (y_dur >= float(H))] = 0  # known negative
    # else: unknown for FAST

    mask_known = sold | (cens & (y_dur >= float(H)))
    return y_fast, mask_known.astype(bool)


def _slow_labels_and_mask(y_dur_rawours: np.ndarray, y_sold_event: np.ndarray, H: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    SLOW label at horizon H, censoring-aware (slow is positive):
      - y_slow = 1 if sold and duration>H
      - y_slow = 0 if sold and duration<=H
      - if censored:
           known positive ONLY if follow-up>=H (y_slow=1),
           otherwise unknown (mask_known=False)
    """
    y_dur = y_dur_rawours.astype(float)
    sold = (y_sold_event.astype(float) > 0.5)
    cens = ~sold

    y_slow = np.zeros_like(y_dur, dtype=np.int64)
    y_slow[sold & (y_dur > float(H))] = 1
    y_slow[sold & (y_dur <= float(H))] = 0
    y_slow[cens & (y_dur >= float(H))] = 1  # known positive
    # else: unknown for SLOW

    mask_known = sold | (cens & (y_dur >= float(H)))
    return y_slow, mask_known.astype(bool)


def cls_counts(y_true01: np.ndarray, prob: np.ndarray, thr: float, mask_known: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
    """
    Confusion counts for positive class y=1 with decision prob>=thr -> pred=1.
    Returns tp, fp, fn, tn.
    """
    y = y_true01.astype(np.int64)
    p = prob.astype(float)
    m = np.ones_like(y, dtype=bool) if mask_known is None else mask_known.astype(bool)
    y = y[m]
    p = p[m]
    h = (p >= float(thr)).astype(np.int64)
    tp = int(((h == 1) & (y == 1)).sum())
    fp = int(((h == 1) & (y == 0)).sum())
    fn = int(((h == 0) & (y == 1)).sum())
    tn = int(((h == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def cls_metrics(y_true01: np.ndarray, prob: np.ndarray, thr: float, mask_known: Optional[np.ndarray] = None) -> Dict[str, float]:
    tp, fp, fn, tn = cls_counts(y_true01, prob, thr, mask_known)
    f1, p, r = f1_pr_re_from_counts(tp, fp, fn)

    bucket = int(tp + fp)
    denom_pos = int(tp + fn)

    # contamination = FP / (TP+FP)   ; if bucket=0 => empty accept set, treat contam as 0.0
    contam = float(fp) / float(bucket) if bucket > 0 else 0.0
    # sacrifice = FN / (TP+FN)       ; if denom=0 => no positives, treat sac as 1.0
    sac = float(fn) / float(denom_pos) if denom_pos > 0 else 1.0

    return dict(
        thr=float(thr),
        f1=float(f1),
        p=float(p),
        r=float(r),
        contam=float(contam),
        sac=float(sac),
        bucket=bucket,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF (quantile) of standard normal distribution.
    Uses the Peter John Acklam rational approximation (good accuracy for ML metrics).
    """
    # Guard against extreme values
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return -num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return num / den


def wilson_lcb(k: int, n: int, alpha: float = 0.05) -> float:
    """Wilson score lower confidence bound for a binomial proportion.

    Returns 0 if n==0. Alpha is one-sided (e.g. 0.05 => ~95% LCB).
    """
    if n <= 0:
        return 0.0
    k = int(k)
    n = int(n)
    if k < 0:
        k = 0
    if k > n:
        k = n
    z = _norm_ppf(1.0 - alpha)
    if not math.isfinite(z):
        return 0.0

    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    lcb = (centre - margin) / denom
    if lcb < 0.0:
        return 0.0
    if lcb > 1.0:
        return 1.0
    return float(lcb)


def _precision_lcb(tp: int, fp: int, alpha: float = 0.05) -> float:
    return wilson_lcb(tp, tp + fp, alpha=alpha)


def _recall_lcb(tp: int, fn: int, alpha: float = 0.05) -> float:
    return wilson_lcb(tp, tp + fn, alpha=alpha)


def _safe_logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def aggregate_probs_matrix(
    probs: np.ndarray,
    method: str = "mean",
    trim_ratio: float = 0.1,
    kappa: float = 1.0,
) -> np.ndarray:
    """Aggregate an (n_models, n_rows) probability matrix into (n_rows,).

    Methods:
      - mean
      - median
      - trimmed_mean (drops trim_ratio on both sides)
      - winsorized_mean (caps to quantiles then mean)
      - logit_mean
      - logit_lcb (sigmoid(mean_logit - kappa*std_logit))

    Notes:
      - For drift robustness, logit_lcb is often the best default.
      - All operations are row-wise across models.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (n_models,n_rows), got shape={probs.shape}")
    method = (method or "mean").lower().strip()

    if method == "mean":
        return np.nanmean(probs, axis=0)

    if method == "median":
        return np.nanmedian(probs, axis=0)

    if method in ("trimmed_mean", "trim"):
        m = probs.shape[0]
        k = int(round(float(trim_ratio) * m))
        if k <= 0:
            return np.nanmean(probs, axis=0)
        s = np.sort(probs, axis=0)
        return np.nanmean(s[k:m-k, :], axis=0)

    if method in ("winsorized_mean", "winsor"):
        m = probs.shape[0]
        k = int(round(float(trim_ratio) * m))
        if k <= 0:
            return np.nanmean(probs, axis=0)
        s = np.sort(probs, axis=0)
        lo = s[k, :]
        hi = s[m-k-1, :]
        capped = np.clip(probs, lo[None, :], hi[None, :])
        return np.nanmean(capped, axis=0)

    # logit-space aggregators
    logits = _safe_logit_np(probs)
    mu = np.nanmean(logits, axis=0)

    if method in ("logit_mean", "logit"):
        return _sigmoid_np(mu)

    if method in ("logit_lcb", "logit_mean_minus_std", "logit_mean_minus_kappa_std"):
        sd = np.nanstd(logits, axis=0)
        score = mu - float(kappa) * sd
        return _sigmoid_np(score)

    raise ValueError(f"Unknown ensemble aggregation method: {method}")


def sweep_threshold_best_f1(y_true01: np.ndarray, prob: np.ndarray, mask_known: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Best-F1 threshold sweep, implemented efficiently (O(N log N)).

    Decision rule: pred = 1 if prob >= thr.
    We only evaluate rows where mask_known is True (or all if mask_known is None).
    """
    m = np.ones_like(y_true01, dtype=bool) if mask_known is None else mask_known.astype(bool)
    y = y_true01[m].astype(np.int64)
    p = prob[m].astype(float)
    if p.size == 0:
        return dict(thr=float("nan"), f1=float("nan"), p=float("nan"), r=float("nan"), tp=0, fp=0, fn=0, tn=0)

    # Sort by score descending
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    # Group by unique threshold values (scores)
    if p_sorted.size == 1:
        ends = np.array([0], dtype=np.int64)
    else:
        ends = np.where(np.diff(p_sorted) != 0)[0]
        ends = np.concatenate([ends, [p_sorted.size - 1]])

    thr = p_sorted[ends]
    tp = tp_cum[ends].astype(np.int64)
    fp = fp_cum[ends].astype(np.int64)

    pos_total = int(tp_cum[-1])
    fn = (pos_total - tp).astype(np.int64)
    tn = (p_sorted.size - (tp + fp)).astype(np.int64)

    # Precision/Recall/F1
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)
    recall = np.where(pos_total > 0, tp / pos_total, 0.0)
    den = precision + recall
    f1 = np.zeros_like(den, dtype=np.float64)
    np.divide(2 * precision * recall, den, out=f1, where=den > 0)

    best_i = int(np.nanargmax(f1))
    return dict(
        thr=float(thr[best_i]),
        f1=float(f1[best_i]),
        p=float(precision[best_i]),
        r=float(recall[best_i]),
        tp=int(tp[best_i]),
        fp=int(fp[best_i]),
        fn=int(fn[best_i]),
        tn=int(tn[best_i]),
    )


def threshold_at_target_precision(
    y_true01: np.ndarray,
    prob_pos: np.ndarray,
    target_p: float = 0.95,
    *,
    use_precision_lcb: bool = False,
    precision_lcb_alpha: float = 0.05,
    use_recall_lcb: bool = False,
    recall_lcb_alpha: float = 0.05,
    n_grid: int = 400,
) -> Tuple[float, Dict[str, Any]]:
    """Select threshold maximizing recall subject to a (possibly conservative) precision constraint.

    If use_precision_lcb=True, the constraint uses Wilson LCB on precision instead of the point estimate.
    Optionally, use_recall_lcb=True will maximize Wilson LCB on recall (risk-averse) rather than point recall.

    Returns: (thr, info_dict)
    """
    y = np.asarray(y_true01).astype(np.int64)
    p = np.asarray(prob_pos).astype(np.float64)
    assert y.shape == p.shape

    # Candidate thresholds: quantile grid + endpoints
    qs = np.linspace(0.0, 1.0, int(n_grid))
    thr_grid = np.unique(np.quantile(p, qs))
    if thr_grid.size == 0:
        return float('inf'), {"note": "empty_prob_array"}

    best = None
    best_thr = float(thr_grid[-1])

    total_pos = int(y.sum())
    if total_pos <= 0:
        # No positives => recall undefined; pick conservative threshold
        return float(thr_grid[-1]), {"note": "no_positive_rows"}

    for thr in thr_grid:
        tp, fp, fn, tn = cls_counts(y, p, float(thr))
        if tp + fp <= 0:
            continue

        prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec_pt
        rec_eff = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec_pt

        if prec_eff < float(target_p):
            continue

        cand = (rec_eff, prec_eff, rec_pt, prec_pt, -thr)  # maximize recall/precision, then smaller thr? (higher bucket)
        if (best is None) or (cand > best):
            best = cand
            best_thr = float(thr)

    if best is None:
        # Could not meet target precision (even with LCB). Fall back to best precision-eff threshold.
        best_prec = -1.0
        best_rec = 0.0
        best_thr = float(thr_grid[-1])
        for thr in thr_grid:
            tp, fp, fn, tn = cls_counts(y, p, float(thr))
            if tp + fp <= 0:
                continue
            prec_pt = tp / (tp + fp)
            rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec_pt
            if prec_eff > best_prec:
                best_prec = float(prec_eff)
                best_rec = float(rec_pt)
                best_thr = float(thr)
        info = {
            "note": "target_precision_unreachable",
            "target_p": float(target_p),
            "precision_eff_best": float(best_prec),
            "recall_point_at_best": float(best_rec),
        }
        return best_thr, info

    # Report metrics at chosen threshold
    tp, fp, fn, tn = cls_counts(y, p, best_thr)
    prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    info = {
        "thr": float(best_thr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(prec_pt),
        "recall": float(rec_pt),
        "precision_lcb": float(_precision_lcb(tp, fp, alpha=precision_lcb_alpha)),
        "recall_lcb": float(_recall_lcb(tp, fn, alpha=recall_lcb_alpha)),
        "target_p": float(target_p),
        "use_precision_lcb": bool(use_precision_lcb),
        "use_recall_lcb": bool(use_recall_lcb),
        "precision_lcb_alpha": float(precision_lcb_alpha),
        "recall_lcb_alpha": float(recall_lcb_alpha),
    }
    return float(best_thr), info


def sweep_threshold_best_prrec50(
    y_true01: np.ndarray,
    prob_pos: np.ndarray,
    *,
    target_p: float = 0.95,
    max_sacrifice: float = 0.25,
    sacrifice_cap_penalty: float = 10.0,
    precision_shortfall_penalty: Optional[float] = None,
    use_precision_lcb: bool = False,
    precision_lcb_alpha: float = 0.05,
    use_recall_lcb: bool = False,
    recall_lcb_alpha: float = 0.05,
    n_grid: int = 600,
) -> Dict[str, Any]:
    """Sweep thresholds to pick a *deployment-relevant* operating point.

    Objective (risk-aware):
      - Prefer thresholds that satisfy both:
          precision_eff >= target_p
          sacrifice_eff <= max_sacrifice
      - Among feasible thresholds, maximize recall_eff.
      - If infeasible, maximize a penalized score.

    precision_eff is either point precision or Wilson LCB precision.
    recall_eff is either point recall or Wilson LCB recall.

    Returns a dict with selected thr + diagnostics, including score_excess (sacrifice overflow).
    """
    y = np.asarray(y_true01).astype(np.int64)
    p = np.asarray(prob_pos).astype(np.float64)
    assert y.shape == p.shape

    if precision_shortfall_penalty is None:
        precision_shortfall_penalty = float(sacrifice_cap_penalty)

    qs = np.linspace(0.0, 1.0, int(n_grid))
    thr_grid = np.unique(np.quantile(p, qs))
    if thr_grid.size == 0:
        return {"thr": float("inf"), "note": "empty_prob_array"}

    total_pos = int(y.sum())
    if total_pos <= 0:
        return {"thr": float(thr_grid[-1]), "note": "no_positive_rows"}

    best_safe = None
    best_safe_thr = float(thr_grid[-1])

    best_any = None
    best_any_thr = float(thr_grid[-1])

    for thr in thr_grid:
        tp, fp, fn, tn = cls_counts(y, p, float(thr))
        if tp + fp <= 0:
            continue

        prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec_pt
        rec_eff = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec_pt

        sac_pt = 1.0 - rec_pt
        sac_eff = 1.0 - rec_eff

        # Feasibility (risk-aware)
        feasible = (prec_eff >= float(target_p)) and (sac_eff <= float(max_sacrifice))

        # Penalized score when infeasible (still monotone for Optuna)
        p_short = max(0.0, float(target_p) - float(prec_eff))
        s_excess = max(0.0, float(sac_eff) - float(max_sacrifice))
        penalized = float(rec_eff) - float(precision_shortfall_penalty) * p_short - float(sacrifice_cap_penalty) * s_excess

        cand_any = (penalized, prec_eff, rec_eff, -thr)
        if (best_any is None) or (cand_any > best_any):
            best_any = cand_any
            best_any_thr = float(thr)

        if feasible:
            cand_safe = (rec_eff, prec_eff, -thr)
            if (best_safe is None) or (cand_safe > best_safe):
                best_safe = cand_safe
                best_safe_thr = float(thr)

    # Prefer feasible threshold if any exist
    thr_sel = best_safe_thr if best_safe is not None else best_any_thr

    tp, fp, fn, tn = cls_counts(y, p, thr_sel)
    prec_pt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec_pt
    rec_eff = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec_pt
    sac_pt = 1.0 - rec_pt
    sac_eff = 1.0 - rec_eff

    p_short = max(0.0, float(target_p) - float(prec_eff))
    s_excess = max(0.0, float(sac_eff) - float(max_sacrifice))
    score = float(rec_eff) if (prec_eff >= target_p and sac_eff <= max_sacrifice) else float(rec_eff) - float(precision_shortfall_penalty) * p_short - float(sacrifice_cap_penalty) * s_excess

    return {
        "thr": float(thr_sel),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "p": float(prec_pt),
        "r": float(rec_pt),
        "p_eff": float(prec_eff),
        "r_eff": float(rec_eff),
        "sac": float(sac_pt),
        "sac_eff": float(sac_eff),
        "score": float(score),
        "score_excess": float(s_excess),
        "precision_shortfall": float(p_short),
        "target_p": float(target_p),
        "max_sacrifice": float(max_sacrifice),
        "use_precision_lcb": bool(use_precision_lcb),
        "use_recall_lcb": bool(use_recall_lcb),
    }


def sweep_threshold_mincontam_sac(
    y_true01: np.ndarray,
    prob_pos: np.ndarray,
    *,
    max_sacrifice: float = 0.25,
    sacrifice_cap_penalty: float = 10.0,
    sac_tradeoff: float = 0.10,
    use_precision_lcb: bool = False,
    precision_lcb_alpha: float = 0.05,
    use_recall_lcb: bool = False,
    recall_lcb_alpha: float = 0.05,
    n_grid: int = 800,
) -> Dict[str, Any]:
    """
    Contamination-first threshold selection for FAST gate (y=1 is FAST):
      - Primary: minimize contamination_eff = 1 - precision_eff
      - Constraint: sac_eff <= max_sacrifice  (sac_eff = 1 - recall_eff)
      - Secondary: minimize sac_eff (i.e. maximize recall) within same contamination
      - Tertiary: prefer larger bucket (more operational utility)
      - Uses Wilson LCBs if enabled.

    Returns dict with thr + counts + p/r/contam/sac and risk-aware p_eff/r_eff/contam_eff/sac_eff.
    """
    y = np.asarray(y_true01).astype(np.int64)
    p = np.asarray(prob_pos).astype(np.float64)
    assert y.shape == p.shape

    if y.size == 0:
        return {"thr": float("inf"), "note": "empty"}

    qs = np.linspace(0.0, 1.0, int(n_grid))
    thr_grid = np.unique(np.quantile(p, qs))
    if thr_grid.size == 0:
        return {"thr": float("inf"), "note": "empty_prob_array"}

    total_pos = int(y.sum())
    if total_pos <= 0:
        return {"thr": float(thr_grid[-1]), "note": "no_positive_rows"}

    best_feas = None  # (contam_eff, sac_eff, -bucket, thr)
    best_feas_thr = float(thr_grid[-1])

    best_any = None  # (penalized_loss, contam_eff, sac_eff, -bucket, thr)
    best_any_thr = float(thr_grid[-1])

    for thr in thr_grid:
        tp, fp, fn, tn = cls_counts(y, p, float(thr))
        bucket = tp + fp
        if bucket <= 0:
            continue

        prec_pt = tp / bucket
        rec_pt = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else float(prec_pt)
        rec_eff  = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else float(rec_pt)

        contam_eff = 1.0 - float(prec_eff)
        sac_eff = 1.0 - float(rec_eff)

        feasible = (sac_eff <= float(max_sacrifice))

        cand_feas = (contam_eff, sac_eff, -bucket, float(thr))
        if feasible:
            if (best_feas is None) or (cand_feas < best_feas):
                best_feas = cand_feas
                best_feas_thr = float(thr)

        overflow = max(0.0, sac_eff - float(max_sacrifice))
        penalized_loss = contam_eff + float(sac_tradeoff) * sac_eff + float(sacrifice_cap_penalty) * overflow
        cand_any = (penalized_loss, contam_eff, sac_eff, -bucket, float(thr))
        if (best_any is None) or (cand_any < best_any):
            best_any = cand_any
            best_any_thr = float(thr)

    thr_sel = best_feas_thr if best_feas is not None else best_any_thr

    tp, fp, fn, tn = cls_counts(y, p, float(thr_sel))
    bucket = tp + fp
    prec_pt = tp / bucket if bucket > 0 else 0.0
    rec_pt  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    contam_pt = fp / bucket if bucket > 0 else 0.0
    sac_pt = fn / (tp + fn) if (tp + fn) > 0 else 1.0

    prec_eff = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else float(prec_pt)
    rec_eff  = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else float(rec_pt)
    sac_eff = 1.0 - float(rec_eff)
    contam_eff = 1.0 - float(prec_eff)

    overflow = max(0.0, sac_eff - float(max_sacrifice))
    loss = contam_eff + float(sac_tradeoff) * sac_eff + float(sacrifice_cap_penalty) * overflow

    return {
        "thr": float(thr_sel),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "bucket": int(bucket),
        "p": float(prec_pt), "r": float(rec_pt),
        "contam": float(contam_pt), "sac": float(sac_pt),
        "p_eff": float(prec_eff), "r_eff": float(rec_eff),
        "contam_eff": float(contam_eff), "sac_eff": float(sac_eff),
        "loss": float(loss),
        "overflow": float(overflow),
        "max_sacrifice": float(max_sacrifice),
    }


def bce_with_logits_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_w: torch.Tensor,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """
    Masked + weighted BCE-with-logits with:
      - per-example weights (sample_w)
      - pos_weight (positive class upweight)
      - optional focal scaling: (1 - p_t)^gamma
    Normalized by sum of weights over the mask.
    """
    assert logits.shape == targets.shape
    assert logits.shape == mask.shape
    assert logits.shape == sample_w.shape

    if pos_weight != 1.0:
        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    else:
        pw = None

    loss_vec = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pw
    )

    if focal_gamma and float(focal_gamma) > 0.0:
        p = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, p, 1.0 - p)
        loss_vec = loss_vec * (1.0 - p_t).pow(float(focal_gamma))

    w = sample_w * mask.float()
    denom = w.sum().clamp_min(1e-8)
    return (loss_vec * w).sum() / denom


def bce_prob_masked(
    prob: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_w: torch.Tensor,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Masked + weighted BCE on probabilities (0..1) with:
      - per-example weights (sample_w)
      - pos_weight (positive class upweight, applied to the y=1 term)
      - optional focal scaling: (1 - p_t)^gamma
    Normalized by sum of weights over the mask.
    """
    assert prob.shape == targets.shape
    assert prob.shape == mask.shape
    assert prob.shape == sample_w.shape

    p = prob.clamp(min=eps, max=1.0 - eps)

    # weighted BCE (pos_weight only on the y=1 term, matching your original implementation)
    loss_vec = -(pos_weight * targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))

    if focal_gamma and float(focal_gamma) > 0.0:
        p_t = torch.where(targets > 0.5, p, 1.0 - p)
        loss_vec = loss_vec * (1.0 - p_t).pow(float(focal_gamma))

    w = sample_w * mask.float()
    denom = w.sum().clamp_min(1e-8)
    return (loss_vec * w).sum() / denom


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def f1_pr_re_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return f1, p, r
