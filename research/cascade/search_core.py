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

def _suggest_phase_A(args, trial, cfg_t, max_bce_points: int):
    # Optim
    cfg_t.lr = float(trial.suggest_float("lr", 1e-4, 8e-4, log=True))
    cfg_t.weight_decay = float(trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True))

    # Recency weighting
    _hl_choices = [10.0, 14.0, 17.0, 21.0, 25.0, 30.0]
    _hl_choices = [x for x in _hl_choices if (x >= float(args.half_life_min) and x <= float(args.half_life_max))]
    if not _hl_choices:
        _hl_choices = [float(args.half_life_min)]
    cfg_t.half_life_days = float(trial.suggest_categorical("half_life_days", _hl_choices))

    # Global multi-horizon BCE balance
    cfg_t.bce_weight = float(trial.suggest_float("bce_weight", 0.05, 3.0, log=True))
    cfg_t.max_horizon_bce_points = int(trial.suggest_int("max_horizon_bce_points", 1, max_bce_points))

    # Boundary focus around the Slow21 cutoff
    cfg_t.boundary_focus_k = float(trial.suggest_float("boundary_focus_k", 0.0, 1.0))
    cfg_t.boundary_focus_sigma_days = float(trial.suggest_float("boundary_focus_sigma_days", 3.0, 14.0))

    # Optional: allow Phase-A to tune censor floor (keeps negative examples; default fixed at args.censored_min_days)
    if bool(getattr(args, "tune_censored_min_days", False)):
        lo = float(args.censored_min_days_min)
        hi = float(args.censored_min_days_max)
        if hi < lo:
            lo, hi = hi, lo
        cfg_t.censored_min_days = float(trial.suggest_float("censored_min_days", lo, hi))


def _suggest_phase_B(trial, cfg_t):
    # Tail/slow emphasis weights
    cfg_t.slow_tail_weight = float(trial.suggest_float("slow_tail_weight", 0.8, 6.0, log=True))
    cfg_t.very_slow_tail_weight = float(trial.suggest_float("very_slow_tail_weight", 0.8, 6.0, log=True))

    # Fast-guard (reduce false-slow on very-fast listings)
    cfg_t.fast_guard_k = float(trial.suggest_float("fast_guard_k", 0.0, 4.0))
    cfg_t.fast_guard_hours = float(trial.suggest_categorical("fast_guard_hours", [72.0, 120.0, 168.0, 240.0, 336.0]))

    # Dedicated tail head loss weight (Patch 5 in the trainer)
    cfg_t.tail_bce_weight = float(trial.suggest_categorical("tail_bce_weight", [0.0, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]))


def _suggest_phase_C(trial, cfg_t, arch_choices: Dict[str, List[Any]]):
    # Core architecture
    cfg_t.d_model = int(trial.suggest_categorical("d_model", arch_choices["d_model"]))
    cfg_t.n_latents = int(trial.suggest_categorical("n_latents", arch_choices["n_latents"]))
    cfg_t.fusion_layers = int(trial.suggest_categorical("fusion_layers", arch_choices["fusion_layers"]))
    cfg_t.n_heads = int(trial.suggest_categorical("n_heads", arch_choices["n_heads"]))
    cfg_t.n_experts = int(trial.suggest_categorical("n_experts", arch_choices["n_experts"]))

    # Tokenization choices
    cfg_t.text_tokens = int(trial.suggest_categorical("text_tokens", arch_choices["text_tokens"]))
    cfg_t.img_tokens = int(trial.suggest_categorical("img_tokens", arch_choices["img_tokens"]))

    cfg_t.tab_token_mode = str(trial.suggest_categorical("tab_token_mode", arch_choices["tab_token_mode"]))
    if cfg_t.tab_token_mode == "compact":
        cfg_t.tab_num_tokens = int(trial.suggest_categorical("tab_num_tokens", arch_choices["tab_num_tokens"]))
        cfg_t.tab_cat_tokens = int(trial.suggest_categorical("tab_cat_tokens", arch_choices["tab_cat_tokens"]))
        cfg_t.tab_pool_heads = int(trial.suggest_categorical("tab_pool_heads", arch_choices["tab_pool_heads"]))
        cfg_t.tab_pool_dropout = float(trial.suggest_categorical("tab_pool_dropout", arch_choices["tab_pool_dropout"]))
    else:
        # keep deterministic values when using feature mode
        cfg_t.tab_num_tokens = int(getattr(cfg_t, "tab_num_tokens", 8))
        cfg_t.tab_cat_tokens = int(getattr(cfg_t, "tab_cat_tokens", 8))

    # Regularization
    cfg_t.dropout = float(trial.suggest_categorical("dropout", arch_choices["dropout"]))
    cfg_t.attn_dropout = float(trial.suggest_categorical("attn_dropout", arch_choices["attn_dropout"]))


def _baseline_arch_choices(base_cfg) -> Dict[str, List[Any]]:
    """Return a *single-value* choice list for every arch key (baseline only).

    This is ideal for smoke tests (trials_per_phase=1) to avoid random OOM configs.
    """
    return {
        "d_model": [int(getattr(base_cfg, "d_model", 256))],
        "n_latents": [int(getattr(base_cfg, "n_latents", 32))],
        "fusion_layers": [int(getattr(base_cfg, "fusion_layers", 9))],
        "n_heads": [int(getattr(base_cfg, "n_heads", 8))],
        "n_experts": [int(getattr(base_cfg, "n_experts", 7))],
        "text_tokens": [int(getattr(base_cfg, "text_tokens", 8))],
        "img_tokens": [int(getattr(base_cfg, "img_tokens", 8))],
        "tab_token_mode": [str(getattr(base_cfg, "tab_token_mode", "feature"))],
        "tab_num_tokens": [int(getattr(base_cfg, "tab_num_tokens", 8))],
        "tab_cat_tokens": [int(getattr(base_cfg, "tab_cat_tokens", 8))],
        "tab_pool_heads": [int(getattr(base_cfg, "tab_pool_heads", 4))],
        "tab_pool_dropout": [float(getattr(base_cfg, "tab_pool_dropout", 0.0))],
        "dropout": [float(getattr(base_cfg, "dropout", 0.10))],
        "attn_dropout": [float(getattr(base_cfg, "attn_dropout", 0.10))],
    }


def _safe_arch_choices(base_cfg) -> Dict[str, List[Any]]:
    """Conservative architecture search space (VRAM-friendly defaults).

    This intentionally avoids big configs like n_latents>=48, fusion_layers>=12,
    and token counts>=12 that can trigger CUDA/CUBLAS failures at batch_size=128.
    """
    def _uniq(xs: List[Any]) -> List[Any]:
        out: List[Any] = []
        for x in xs:
            if x not in out:
                out.append(x)
        return out

    d_model0 = int(getattr(base_cfg, "d_model", 256))
    lat0 = int(getattr(base_cfg, "n_latents", 32))
    layers0 = int(getattr(base_cfg, "fusion_layers", 9))
    heads0 = int(getattr(base_cfg, "n_heads", 8))
    exp0 = int(getattr(base_cfg, "n_experts", 7))
    tt0 = int(getattr(base_cfg, "text_tokens", 8))
    it0 = int(getattr(base_cfg, "img_tokens", 8))
    mode0 = str(getattr(base_cfg, "tab_token_mode", "feature"))
    tn0 = int(getattr(base_cfg, "tab_num_tokens", 8))
    tc0 = int(getattr(base_cfg, "tab_cat_tokens", 8))
    ph0 = int(getattr(base_cfg, "tab_pool_heads", 4))
    pd0 = float(getattr(base_cfg, "tab_pool_dropout", 0.0))
    do0 = float(getattr(base_cfg, "dropout", 0.10))
    ado0 = float(getattr(base_cfg, "attn_dropout", 0.10))

    return {
        "d_model": _uniq([d_model0, 192, 256, 320]),  # avoid 384 by default
        "n_latents": _uniq([lat0, 16, 32]),          # avoid 48/64 by default
        "fusion_layers": _uniq([layers0, 6, 9]),     # avoid 12 by default
        "n_heads": _uniq([heads0, 4, 8]),            # avoid 12 by default
        "n_experts": _uniq([exp0, 3, 5, 7]),         # avoid 9 by default
        "text_tokens": _uniq([tt0, 4, 8]),           # avoid 12 by default
        "img_tokens": _uniq([it0, 4, 8]),            # avoid 12 by default
        "tab_token_mode": _uniq([mode0, "feature", "compact"]),
        "tab_num_tokens": _uniq([tn0, 4, 8, 12]),
        "tab_cat_tokens": _uniq([tc0, 4, 8, 12]),
        "tab_pool_heads": _uniq([ph0, 2, 4]),
        "tab_pool_dropout": _uniq([pd0, 0.0, 0.05, 0.10]),
        "dropout": _uniq([do0, 0.05, 0.10, 0.15, 0.20]),
        "attn_dropout": _uniq([ado0, 0.05, 0.10, 0.15, 0.20]),
    }


def _wide_arch_choices(base_cfg) -> Dict[str, List[Any]]:
    """Wider architecture search space (may OOM on some GPUs)."""
    def _uniq(xs: List[Any]) -> List[Any]:
        out: List[Any] = []
        for x in xs:
            if x not in out:
                out.append(x)
        return out

    return {
        "d_model": _uniq([int(getattr(base_cfg, "d_model", 256)), 192, 256, 320, 384]),
        "n_latents": _uniq([int(getattr(base_cfg, "n_latents", 16)), 16, 32, 48, 64]),
        "fusion_layers": _uniq([int(getattr(base_cfg, "fusion_layers", 4)), 4, 6, 9, 12]),
        "n_heads": _uniq([int(getattr(base_cfg, "n_heads", 8)), 4, 8, 12]),
        "n_experts": _uniq([int(getattr(base_cfg, "n_experts", 3)), 3, 5, 7, 9]),
        "text_tokens": _uniq([int(getattr(base_cfg, "text_tokens", 4)), 4, 8, 12]),
        "img_tokens": _uniq([int(getattr(base_cfg, "img_tokens", 4)), 4, 8, 12]),
        "tab_token_mode": _uniq([str(getattr(base_cfg, "tab_token_mode", "compact")), "compact", "feature"]),
        "tab_num_tokens": _uniq([int(getattr(base_cfg, "tab_num_tokens", 8)), 4, 8, 12, 16]),
        "tab_cat_tokens": _uniq([int(getattr(base_cfg, "tab_cat_tokens", 8)), 4, 8, 12, 16]),
        "tab_pool_heads": _uniq([int(getattr(base_cfg, "tab_pool_heads", 4)), 2, 4, 8]),
        "tab_pool_dropout": _uniq([float(getattr(base_cfg, "tab_pool_dropout", 0.0)), 0.0, 0.05, 0.10]),
        "dropout": _uniq([float(getattr(base_cfg, "dropout", 0.10)), 0.05, 0.10, 0.15, 0.20]),
        "attn_dropout": _uniq([float(getattr(base_cfg, "attn_dropout", 0.10)), 0.05, 0.10, 0.15, 0.20]),
    }


def _apply_frozen_params(cfg_t, frozen: Dict[str, Any]) -> None:
    for k, v in (frozen or {}).items():
        if not hasattr(cfg_t, k):
            # Ignore unknown keys (keeps forward-compatibility if config evolves)
            continue
        setattr(cfg_t, k, v)
