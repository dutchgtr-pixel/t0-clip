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

def _raw_bucket_codes(
    *,
    p_tail: np.ndarray,
    p_fast168: np.ndarray,
    p_fast72: np.ndarray,
    thr_tail: float,
    thr_fast168: float,
    thr_fast72: float,
) -> np.ndarray:
    pred_tail = np.asarray(p_tail, dtype=float) >= float(thr_tail)
    stage1_fast = np.asarray(p_fast168, dtype=float) >= float(thr_fast168)
    pred_fast72 = (~pred_tail) & stage1_fast & (np.asarray(p_fast72, dtype=float) >= float(thr_fast72))
    pred_slow168 = (~pred_tail) & (~stage1_fast)
    return np.select(
        [pred_tail, pred_slow168, pred_fast72],
        ["TAIL_21PLUS", "SLOW_168PLUS", "FAST_72H"],
        default="MID_72_168H",
    )


def _safe_logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _mean_logit_meta(seed_probs: np.ndarray) -> np.ndarray:
    return _sigmoid(_safe_logit(seed_probs).mean(axis=0))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
