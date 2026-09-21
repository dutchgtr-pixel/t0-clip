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
    target_p: float = 0.90,
    max_sacrifice: float = 0.25,
    sacrifice_cap_penalty: float = 10.0,
    precision_shortfall_penalty: float = 2.0,
    use_precision_lcb: bool = False,
    precision_lcb_alpha: float = 0.05,
    use_recall_lcb: bool = False,
    recall_lcb_alpha: float = 0.05,
    n_grid: int = 600,
    min_bucket: int = 1,
    # --- NEW: bucket-aware FP penalties (optional) ---
    dur_hours: Optional[np.ndarray] = None,
    fp_minor_weight: float = 0.0,
    fp_mid_weight: float = 0.0,
    fp_major_weight: float = 0.0,
    minor_fp_max_hours: float = 168.0,
    mid_fp_max_hours: float = 240.0,
    max_fp_gt168_rate: float = -1.0,
    fp_gt168_cap_penalty: float = 0.0,
    max_fp_gt240_rate: float = -1.0,
    fp_gt240_cap_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Choose threshold to maximize recall (effective recall if LCBs enabled), subject to:
      precision >= target_p and sacrifice <= max_sacrifice if feasible.

    v2 addition: If dur_hours is provided, you can apply bucket-aware FP penalties so
    that late-sale false positives are explicitly discouraged.
    """
    y = np.asarray(y_true01).astype(np.int32)
    p = np.asarray(prob_pos).astype(np.float64)

    dur = None
    use_bucket = False
    if dur_hours is not None:
        dur = np.asarray(dur_hours).astype(np.float64)
        if dur.shape != y.shape:
            raise ValueError(f"dur_hours shape {dur.shape} != y shape {y.shape}")
        use_bucket = (
            (abs(fp_minor_weight) > 0)
            or (abs(fp_mid_weight) > 0)
            or (abs(fp_major_weight) > 0)
            or (max_fp_gt168_rate >= 0)
            or (max_fp_gt240_rate >= 0)
        )

    qs = np.linspace(0, 1, int(n_grid))
    thr_grid = np.unique(np.quantile(p, qs))
    thr_grid = np.clip(thr_grid, 0.0, 1.0)

    best_safe = None
    best_safe_key = None
    best_any = None
    best_any_key = None

    for thr in thr_grid:
        tp, fp, fn, tn = cls_counts(y, p, float(thr))
        bucket = tp + fp
        if bucket < min_bucket:
            continue

        prec = tp / bucket if bucket > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        prec_lcb = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec
        rec_lcb = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec

        p_eff = prec_lcb
        rec_eff = rec_lcb
        sac_eff = 1.0 - rec_eff

        feasible = (p_eff >= target_p) and (sac_eff <= max_sacrifice)

        # bucket penalties (normalized by accepted bucket)
        fp_minor_rate = fp_mid_rate = fp_major_rate = 0.0
        fp_gt168_rate = fp_gt240_rate = 0.0
        bucket_pen = 0.0
        cap_pen = 0.0
        if use_bucket and (dur is not None) and bucket > 0:
            fp_mask = (p >= thr) & (y == 0)
            fp_minor = int(np.sum(fp_mask & (dur <= minor_fp_max_hours)))
            fp_mid = int(np.sum(fp_mask & (dur > minor_fp_max_hours) & (dur <= mid_fp_max_hours)))
            fp_major = int(np.sum(fp_mask & (dur > mid_fp_max_hours)))

            fp_minor_rate = fp_minor / bucket
            fp_mid_rate = fp_mid / bucket
            fp_major_rate = fp_major / bucket
            fp_gt168_rate = (fp_mid + fp_major) / bucket
            fp_gt240_rate = fp_major / bucket

            bucket_pen = (
                fp_minor_weight * fp_minor_rate
                + fp_mid_weight * fp_mid_rate
                + fp_major_weight * fp_major_rate
            )
            if (max_fp_gt168_rate >= 0) and (fp_gt168_cap_penalty > 0):
                cap_pen += fp_gt168_cap_penalty * max(0.0, fp_gt168_rate - max_fp_gt168_rate)
            if (max_fp_gt240_rate >= 0) and (fp_gt240_cap_penalty > 0):
                cap_pen += fp_gt240_cap_penalty * max(0.0, fp_gt240_rate - max_fp_gt240_rate)

        p_short = max(0.0, target_p - p_eff)
        s_excess = max(0.0, sac_eff - max_sacrifice)

        # Base penalized score (higher is better)
        penalized = (
            rec_eff
            - precision_shortfall_penalty * p_short
            - sacrifice_cap_penalty * s_excess
            - bucket_pen
            - cap_pen
        )

        score_excess = (p_short > 0.0) or (s_excess > 0.0)

        if feasible:
            # For feasible points, still prefer fewer catastrophic FP via (bucket_pen + cap_pen)
            safe_utility = rec_eff - bucket_pen - cap_pen
            cand_safe = (safe_utility, rec_eff, p_eff, -float(thr))
            if (best_safe_key is None) or (cand_safe > best_safe_key):
                best_safe_key = cand_safe
                best_safe = float(thr)

        cand_any = (penalized, p_eff, rec_eff, -float(thr))
        if (best_any_key is None) or (cand_any > best_any_key):
            best_any_key = cand_any
            best_any = float(thr)

    thr_sel = best_safe if best_safe is not None else best_any
    if thr_sel is None:
        thr_sel = 1.0

    tp, fp, fn, tn = cls_counts(y, p, float(thr_sel))
    bucket = tp + fp
    prec = tp / bucket if bucket > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    prec_lcb = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec
    rec_lcb = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec

    p_eff = prec_lcb
    rec_eff = rec_lcb
    sac_eff = 1.0 - rec_eff
    contam = fp / bucket if bucket > 0 else 0.0
    sac = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    # recompute bucket penalty at selected thr
    fp_minor = fp_mid = fp_major = 0
    fp_minor_rate = fp_mid_rate = fp_major_rate = 0.0
    fp_gt168_rate = fp_gt240_rate = 0.0
    bucket_pen = 0.0
    cap_pen = 0.0
    if use_bucket and (dur is not None) and bucket > 0:
        fp_mask = (p >= thr_sel) & (y == 0)
        fp_minor = int(np.sum(fp_mask & (dur <= minor_fp_max_hours)))
        fp_mid = int(np.sum(fp_mask & (dur > minor_fp_max_hours) & (dur <= mid_fp_max_hours)))
        fp_major = int(np.sum(fp_mask & (dur > mid_fp_max_hours)))

        fp_minor_rate = fp_minor / bucket
        fp_mid_rate = fp_mid / bucket
        fp_major_rate = fp_major / bucket
        fp_gt168_rate = (fp_mid + fp_major) / bucket
        fp_gt240_rate = fp_major / bucket

        bucket_pen = (
            fp_minor_weight * fp_minor_rate
            + fp_mid_weight * fp_mid_rate
            + fp_major_weight * fp_major_rate
        )
        if (max_fp_gt168_rate >= 0) and (fp_gt168_cap_penalty > 0):
            cap_pen += fp_gt168_cap_penalty * max(0.0, fp_gt168_rate - max_fp_gt168_rate)
        if (max_fp_gt240_rate >= 0) and (fp_gt240_cap_penalty > 0):
            cap_pen += fp_gt240_cap_penalty * max(0.0, fp_gt240_rate - max_fp_gt240_rate)

    p_short = max(0.0, target_p - p_eff)
    s_excess = max(0.0, sac_eff - max_sacrifice)
    score = (
        rec_eff
        - precision_shortfall_penalty * p_short
        - sacrifice_cap_penalty * s_excess
        - bucket_pen
        - cap_pen
    )
    score_excess = (p_short > 0.0) or (s_excess > 0.0)

    out = dict(
        thr=float(thr_sel),
        p=float(prec),
        r=float(rec),
        p_eff=float(p_eff),
        r_eff=float(rec_eff),
        p_lcb=float(prec_lcb),
        r_lcb=float(rec_lcb),
        contam=float(contam),
        sac=float(sac),
        bucket=int(bucket),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        score=float(score),
        score_excess=bool(score_excess),
        feasible=bool(best_safe is not None),
        # bucket extras
        bucket_penalty=float(bucket_pen),
        cap_penalty=float(cap_pen),
        fp_minor=int(fp_minor),
        fp_mid=int(fp_mid),
        fp_major=int(fp_major),
        fp_minor_rate=float(fp_minor_rate),
        fp_mid_rate=float(fp_mid_rate),
        fp_major_rate=float(fp_major_rate),
        fp_gt168_rate=float(fp_gt168_rate),
        fp_gt240_rate=float(fp_gt240_rate),
    )
    return out


def sweep_threshold_mincontam_sac(
    y_true01: np.ndarray,
    prob_pos: np.ndarray,
    *,
    max_sacrifice: float = 0.25,
    sacrifice_cap_penalty: float = 10.0,
    sac_tradeoff: float = 0.2,
    use_precision_lcb: bool = False,
    precision_lcb_alpha: float = 0.05,
    use_recall_lcb: bool = False,
    recall_lcb_alpha: float = 0.05,
    n_grid: int = 800,
    min_bucket: int = 1,
    # --- NEW: bucket-aware FP penalties (optional) ---
    dur_hours: Optional[np.ndarray] = None,
    fp_minor_weight: float = 0.0,
    fp_mid_weight: float = 0.0,
    fp_major_weight: float = 0.0,
    minor_fp_max_hours: float = 168.0,
    mid_fp_max_hours: float = 240.0,
    max_fp_gt168_rate: float = -1.0,
    fp_gt168_cap_penalty: float = 0.0,
    max_fp_gt240_rate: float = -1.0,
    fp_gt240_cap_penalty: float = 0.0,
) -> Dict[str, Any]:
    """
    Sweep thresholds and pick an operating point that minimizes contamination while
    respecting a sacrifice ceiling.

    v2 addition: If dur_hours is provided, you can apply bucket-aware FP penalties so
    that FP far beyond the FAST horizon (e.g. >168h / >240h) are punished more than
    near-miss FP (72–168h).
    """
    y = np.asarray(y_true01).astype(np.int32)
    p = np.asarray(prob_pos).astype(np.float64)

    dur = None
    use_bucket = False
    if dur_hours is not None:
        dur = np.asarray(dur_hours).astype(np.float64)
        if dur.shape != y.shape:
            raise ValueError(f"dur_hours shape {dur.shape} != y shape {y.shape}")
        use_bucket = (
            (abs(fp_minor_weight) > 0)
            or (abs(fp_mid_weight) > 0)
            or (abs(fp_major_weight) > 0)
            or (max_fp_gt168_rate >= 0)
            or (max_fp_gt240_rate >= 0)
        )

    # quantile grid => more resolution where model puts mass
    qs = np.linspace(0, 1, int(n_grid))
    thr_grid = np.unique(np.quantile(p, qs))
    thr_grid = np.clip(thr_grid, 0.0, 1.0)

    best_feas = None
    best_feas_key = None  # minimize
    best_any = None
    best_any_key = None   # minimize

    for thr in thr_grid:
        tp, fp, fn, tn = cls_counts(y, p, float(thr))
        bucket = tp + fp
        if bucket < min_bucket:
            continue

        prec = tp / bucket if bucket > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        prec_lcb = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec
        rec_lcb = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec

        contam_eff = 1.0 - prec_lcb
        sac_eff = 1.0 - rec_lcb

        # --- bucket-aware FP penalties ---
        fp_minor = fp_mid = fp_major = 0
        fp_minor_rate = fp_mid_rate = fp_major_rate = 0.0
        fp_gt168_rate = fp_gt240_rate = 0.0
        bucket_pen = 0.0
        cap_pen = 0.0
        if use_bucket and (dur is not None) and bucket > 0:
            fp_mask = (p >= thr) & (y == 0)
            fp_minor = int(np.sum(fp_mask & (dur <= minor_fp_max_hours)))
            fp_mid = int(np.sum(fp_mask & (dur > minor_fp_max_hours) & (dur <= mid_fp_max_hours)))
            fp_major = int(np.sum(fp_mask & (dur > mid_fp_max_hours)))

            fp_minor_rate = fp_minor / bucket
            fp_mid_rate = fp_mid / bucket
            fp_major_rate = fp_major / bucket
            fp_gt168_rate = (fp_mid + fp_major) / bucket
            fp_gt240_rate = fp_major / bucket

            bucket_pen = (
                fp_minor_weight * fp_minor_rate
                + fp_mid_weight * fp_mid_rate
                + fp_major_weight * fp_major_rate
            )

            if (max_fp_gt168_rate >= 0) and (fp_gt168_cap_penalty > 0):
                cap_pen += fp_gt168_cap_penalty * max(0.0, fp_gt168_rate - max_fp_gt168_rate)
            if (max_fp_gt240_rate >= 0) and (fp_gt240_cap_penalty > 0):
                cap_pen += fp_gt240_cap_penalty * max(0.0, fp_gt240_rate - max_fp_gt240_rate)

        core_loss = contam_eff + bucket_pen + cap_pen

        feasible = (sac_eff <= max_sacrifice)

        # If not feasible, apply overflow penalty and tradeoff
        s_excess = max(0.0, sac_eff - max_sacrifice)
        penalized_loss = core_loss + sac_tradeoff * sac_eff + sacrifice_cap_penalty * s_excess

        if feasible:
            cand = (core_loss, sac_eff, -bucket, float(thr))
            if (best_feas_key is None) or (cand < best_feas_key):
                best_feas_key = cand
                best_feas = float(thr)

        cand_any = (penalized_loss, sac_eff, -bucket, float(thr))
        if (best_any_key is None) or (cand_any < best_any_key):
            best_any_key = cand_any
            best_any = float(thr)

    thr_sel = best_feas if best_feas is not None else best_any
    if thr_sel is None:
        thr_sel = 1.0

    tp, fp, fn, tn = cls_counts(y, p, float(thr_sel))
    bucket = tp + fp
    prec = tp / bucket if bucket > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    prec_lcb = _precision_lcb(tp, fp, alpha=precision_lcb_alpha) if use_precision_lcb else prec
    rec_lcb = _recall_lcb(tp, fn, alpha=recall_lcb_alpha) if use_recall_lcb else rec

    contam_eff = 1.0 - prec_lcb
    sac_eff = 1.0 - rec_lcb

    # recompute bucket stats at selected thr
    fp_minor = fp_mid = fp_major = 0
    fp_minor_rate = fp_mid_rate = fp_major_rate = 0.0
    fp_gt168_rate = fp_gt240_rate = 0.0
    bucket_pen = 0.0
    cap_pen = 0.0
    if use_bucket and (dur is not None) and bucket > 0:
        fp_mask = (p >= thr_sel) & (y == 0)
        fp_minor = int(np.sum(fp_mask & (dur <= minor_fp_max_hours)))
        fp_mid = int(np.sum(fp_mask & (dur > minor_fp_max_hours) & (dur <= mid_fp_max_hours)))
        fp_major = int(np.sum(fp_mask & (dur > mid_fp_max_hours)))

        fp_minor_rate = fp_minor / bucket
        fp_mid_rate = fp_mid / bucket
        fp_major_rate = fp_major / bucket
        fp_gt168_rate = (fp_mid + fp_major) / bucket
        fp_gt240_rate = fp_major / bucket

        bucket_pen = (
            fp_minor_weight * fp_minor_rate
            + fp_mid_weight * fp_mid_rate
            + fp_major_weight * fp_major_rate
        )

        if (max_fp_gt168_rate >= 0) and (fp_gt168_cap_penalty > 0):
            cap_pen += fp_gt168_cap_penalty * max(0.0, fp_gt168_rate - max_fp_gt168_rate)
        if (max_fp_gt240_rate >= 0) and (fp_gt240_cap_penalty > 0):
            cap_pen += fp_gt240_cap_penalty * max(0.0, fp_gt240_rate - max_fp_gt240_rate)

    s_excess = max(0.0, sac_eff - max_sacrifice)
    loss = (contam_eff + bucket_pen + cap_pen) + sac_tradeoff * sac_eff + sacrifice_cap_penalty * s_excess

    out = dict(
        thr=float(thr_sel),
        p=float(prec),
        r=float(rec),
        p_eff=float(prec_lcb),
        r_eff=float(rec_lcb),
        contam_eff=float(contam_eff),
        sac_eff=float(sac_eff),
        contam=float(fp / bucket if bucket > 0 else 0.0),
        sac=float(fn / (tp + fn) if (tp + fn) > 0 else 0.0),
        bucket=int(bucket),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        loss=float(loss),
        overflow=float(s_excess),
        max_sacrifice=float(max_sacrifice),
        precision_lcb=float(prec_lcb),
        recall_lcb=float(rec_lcb),
        # bucket extras
        bucket_penalty=float(bucket_pen),
        cap_penalty=float(cap_pen),
        fp_minor=int(fp_minor),
        fp_mid=int(fp_mid),
        fp_major=int(fp_major),
        fp_minor_rate=float(fp_minor_rate),
        fp_mid_rate=float(fp_mid_rate),
        fp_major_rate=float(fp_major_rate),
        fp_gt168_rate=float(fp_gt168_rate),
        fp_gt240_rate=float(fp_gt240_rate),
    )
    return out


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
