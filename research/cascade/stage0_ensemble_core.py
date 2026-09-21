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

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _nll_binary(y01: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    y = y01.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature_scaler(p: np.ndarray, y01: np.ndarray, t_min: float, t_max: float, grid: int) -> float:
    if p.size == 0 or np.unique(y01).size < 2:
        return 1.0
    l = _logit(p, eps=1e-9)
    if float(np.nanstd(l)) < 1e-9:
        return 1.0
    t_min = max(1e-3, float(t_min))
    t_max = max(t_min * 1.0001, float(t_max))
    grid = int(max(10, grid))
    Ts = np.logspace(np.log10(t_min), np.log10(t_max), grid)
    best_T, best_nll = 1.0, float("inf")
    for T in Ts:
        pT = _sigmoid(l / float(T))
        nll = _nll_binary(y01, pT)
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return float(best_T)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    T = max(1e-6, float(T))
    return _sigmoid(_logit(p, eps=1e-9) / T)


def fit_isotonic(p: np.ndarray, y01: np.ndarray) -> Optional[IsotonicRegression]:
    if p.size == 0 or np.unique(y01).size < 2:
        return None
    if float(np.nanstd(p)) < 1e-12:
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p.astype(float), y01.astype(float))
    return iso


def calibrate_matrix(P_s: np.ndarray, P_h: np.ndarray, y_s: np.ndarray, calib_mask: np.ndarray,
                     kind: str, temp_min: float, temp_max: float, temp_grid: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    kind = str(kind).lower()
    if kind in ("none", "off", "0", "false"):
        return P_s, P_h, {"kind": "none"}, [{"kind": "none"} for _ in range(int(P_s.shape[0]))]

    P_s = np.clip(P_s.astype(float), 1e-6, 1.0 - 1e-6)
    P_h = np.clip(P_h.astype(float), 1e-6, 1.0 - 1e-6)
    calib_mask = np.asarray(calib_mask, dtype=bool)
    y_cal = y_s[calib_mask].astype(int)

    P_sc = P_s.copy()
    P_hc = P_h.copy()
    temps: List[float] = []
    iso_used = 0
    state_rows: List[Dict[str, Any]] = []

    for i in range(P_s.shape[0]):
        p_cal = P_s[i, calib_mask]
        if kind.startswith("temp"):
            T = fit_temperature_scaler(p_cal, y_cal, temp_min, temp_max, temp_grid)
            temps.append(T)
            P_sc[i, :] = apply_temperature(P_s[i, :], T)
            P_hc[i, :] = apply_temperature(P_h[i, :], T)
            state_rows.append({"kind": "temp", "temperature": float(T)})
        elif kind.startswith("iso"):
            iso = fit_isotonic(p_cal, y_cal)
            if iso is None:
                P_sc[i, :] = P_s[i, :]
                P_hc[i, :] = P_h[i, :]
                state_rows.append({"kind": "identity"})
            else:
                iso_used += 1
                P_sc[i, :] = iso.predict(P_s[i, :])
                P_hc[i, :] = iso.predict(P_h[i, :])
                state_rows.append(
                    {
                        "kind": "isotonic",
                        "x_thresholds": [float(x) for x in np.asarray(iso.X_thresholds_, dtype=float).tolist()],
                        "y_thresholds": [float(y) for y in np.asarray(iso.y_thresholds_, dtype=float).tolist()],
                    }
                )
        else:
            raise ValueError(f"Unknown calibrate kind: {kind}")

    summary: Dict[str, Any] = {"kind": kind, "n_trials": int(P_s.shape[0])}
    if temps:
        ta = np.asarray(temps, dtype=float)
        summary.update({
            "temp_min": float(np.min(ta)),
            "temp_median": float(np.median(ta)),
            "temp_mean": float(np.mean(ta)),
            "temp_max": float(np.max(ta)),
            "temp_grid": int(temp_grid),
            "temp_range": [float(temp_min), float(temp_max)],
        })
    if kind.startswith("iso"):
        summary["iso_used"] = int(iso_used)
    return P_sc, P_hc, summary, state_rows


def cm_from_scores(p: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(int)
    pred = (p >= float(thr)).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": float(prec), "recall": float(rec), "f1": float(f1), "n": int(len(y))}


def threshold_search_best_f1(p: np.ndarray, y: np.ndarray, min_bucket: int, precision_min: float) -> Tuple[float, Dict[str, float]]:
    """Exact-ish threshold scan over unique p values (fast enough for n~1k)."""
    p = p.astype(float)
    y = y.astype(int)
    uniq = np.unique(p)
    # include endpoints
    thrs = np.unique(np.concatenate([np.array([0.0], float), uniq, np.array([1.0], float)]))
    best_thr = 0.5
    best = {"f1": -1.0}
    for thr in thrs:
        cm = cm_from_scores(p, y, float(thr))
        bucket = cm["tp"] + cm["fp"]
        if bucket < int(min_bucket):
            continue
        if precision_min >= 0.0 and cm["precision"] < float(precision_min):
            continue
        if cm["f1"] > best["f1"]:
            best_thr = float(thr)
            best = cm
    if best["f1"] < 0:
        best_thr = 0.5
        best = cm_from_scores(p, y, best_thr)
    best["thr"] = float(best_thr)
    best["bucket"] = int(best["tp"] + best["fp"])
    return best_thr, best


def _fit_lda_logit(X: np.ndarray, y: np.ndarray, lda_lambda: float) -> Tuple[np.ndarray, float]:
    d = X.shape[1]
    lam = float(lda_lambda)
    m0 = X[y == 0].mean(axis=0) if np.any(y == 0) else X.mean(axis=0)
    m1 = X[y == 1].mean(axis=0) if np.any(y == 1) else X.mean(axis=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    Sigma_reg = Sigma + lam * np.eye(d, dtype=float)
    try:
        inv = np.linalg.inv(Sigma_reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sigma_reg)
    w = inv @ (m1 - m0)
    b = -0.5 * float((m1 + m0).dot(w))
    return w.astype(float), float(b)


def ensemble_p_tail(P: np.ndarray,
                    method: str,
                    *,
                    y_tail_inner: Optional[np.ndarray] = None,
                    inner_mask: Optional[np.ndarray] = None,
                    trim_ratio: float = 0.0,
                    kappa: float = 0.0,
                    mv_lambda: float = 0.0,
                    lda_lambda: float = 1e-4,
                    wlogit_C: float = 1.0,
                    wlogit_penalty: str = "l2") -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    P: (K, n_rows) of p_tail
    Returns: p_tail_all, weights(optional), extra
    """
    if P.shape[0] == 1:
        return P[0], None, {"intercept": 0.0}

    method = str(method).lower().strip()

    if method == "mean_prob":
        return np.mean(P, axis=0), None, {"intercept": 0.0}

    if method == "mean_logit":
        z = np.mean(_logit(P), axis=0)
        return _sigmoid(z), None, {"intercept": 0.0}

    if method == "logit_lcb":
        logits = _logit(P)
        n = logits.shape[0]
        k = int(math.floor(float(trim_ratio) * n))
        logits_sorted = np.sort(logits, axis=0)
        core = logits_sorted[k:n-k, :] if (2*k < n) else logits_sorted
        mu = core.mean(axis=0)
        sd = core.std(axis=0, ddof=0)
        return _sigmoid(mu - float(kappa) * sd), None, {"intercept": 0.0, "trim_ratio": float(trim_ratio), "kappa": float(kappa)}

    if method == "mv_logit":
        X = _logit(P).T  # (n_rows, K)
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
        w = (w / (ones.T @ w)).ravel()
        z = X @ w
        return _sigmoid(z), w.astype(float), {"intercept": 0.0, "mv_lambda": float(mv_lambda)}

    if method == "lda_logit":
        if y_tail_inner is None or inner_mask is None:
            raise ValueError("lda_logit requires y_tail_inner and inner_mask")
        logits_all = _logit(P).T
        X = logits_all[inner_mask, :]
        y = y_tail_inner.astype(int)
        w, b = _fit_lda_logit(X, y, lda_lambda=float(lda_lambda))
        z = logits_all @ w + b
        return _sigmoid(z), w.astype(float), {"intercept": float(b), "lda_lambda": float(lda_lambda)}

    if method == "wlogit":
        if y_tail_inner is None or inner_mask is None:
            raise ValueError("wlogit requires y_tail_inner and inner_mask")
        logits_all = _logit(P).T
        X = logits_all[inner_mask, :]
        y = y_tail_inner.astype(int)

        C = float(max(1e-6, wlogit_C))
        pen = str(wlogit_penalty).lower().strip()
        if pen not in ("l1", "l2"):
            pen = "l2"

        # liblinear supports l1/l2 and is deterministic-ish
        lr = LogisticRegression(
            penalty=pen,
            C=C,
            fit_intercept=True,
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
        )
        lr.fit(X, y)
        w = lr.coef_.ravel().astype(float)
        b = float(lr.intercept_.ravel()[0])
        z = logits_all @ w + b
        return _sigmoid(z), w.astype(float), {"intercept": float(b), "wlogit_C": float(C), "wlogit_penalty": pen}

    raise ValueError(f"Unknown method: {method}")


def make_inner_outer_time_split_masks(time_vals: np.ndarray, mask: np.ndarray, outer_frac: float, min_outer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split masked rows into inner (older) and outer (newer)."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    inner = np.zeros(n, dtype=bool)
    outer = np.zeros(n, dtype=bool)

    idx = np.flatnonzero(mask)
    n_mask = int(idx.size)
    outer_frac = float(outer_frac)

    if outer_frac <= 0.0 or n_mask < 50:
        inner[idx] = True
        return inner, outer

    outer_frac = max(0.0, min(0.49, outer_frac))
    n_outer = int(math.floor(n_mask * outer_frac))
    if n_mask >= (min_outer + 10):
        n_outer = max(n_outer, int(min_outer))
    n_outer = max(1, min(n_outer, n_mask - 1))
    n_inner = n_mask - n_outer

    t = np.asarray(time_vals, dtype=float)
    t_masked = t[mask]
    order = np.argsort(t_masked, kind="mergesort")  # stable
    inner_idx = idx[order[:n_inner]]
    outer_idx = idx[order[n_inner:]]
    inner[inner_idx] = True
    outer[outer_idx] = True
    return inner, outer
