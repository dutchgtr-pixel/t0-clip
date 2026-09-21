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

@dataclass
class TrainConfig:
    # -----------------------------
    # Objective / label contract
    # -----------------------------
    horizon_hours: float = 504.0
    n_bins: int = 64
    binning: str = "log1p"  # log1p or linear

    # BCE auxiliary targets: P(T<=t) at these times (hours)
    thresholds_hours: Tuple[float, ...] = (24.0, 72.0, 168.0, 504.0)
    bce_weight: float = 0.25
    tail_bce_weight: float = 0.0  # additional BCE loss on a dedicated tail gate head (0.0 disables; see Patch 5)
    max_horizon_bce_points: int = 4

    # Point-estimate + MAE target
    point_estimator: str = "expected"   # expected | median | quantile
    point_quantile: float = 0.50        # used if point_estimator=quantile
    mae_max_hours: float = 504.0        # compute MAE on sold rows with duration<=mae_max_hours
    mae_loss_weight: float = 0.0        # add differentiable MAE/Huber loss on sold<=mae_max_hours during training
    mae_loss_type: str = "huber"        # huber | l1
    mae_huber_delta_hours: float = 24.0 # huber delta (hours)

    # Gate / classification reporting (fast vs slow at threshold)
    f1_threshold_hours: float = 504.0
    prob_threshold: float = 0.50        # threshold for probability-based gate metrics

    # -----------------------------
    # Recency / boundary weighting (matches tree knobs)
    # -----------------------------
    half_life_days: float = 0.0         # 0 disables time-decay weighting
    boundary_focus_k: float = 0.0       # 0 disables boundary focus
    boundary_focus_sigma_days: float = 7.0
    # AFT-style weighting knobs:
    # - slow_tail_weight applies to rows with duration > 7 days (168h)
    # - very_slow_tail_weight applies to rows with duration > f1_threshold_hours (default 21d / 504h)
    # - fast_guard_* upweights very fast rows to reduce false-slow predictions
    slow_tail_weight: float = 1.0
    very_slow_tail_weight: float = 1.0
    fast_guard_k: float = 0.0
    fast_guard_hours: float = 240.0
    # Drop short-lived censored rows (unsold / right-censored) to strengthen tail learning (AFT gate uses 21d)
    censored_min_days: float = 0.0

    # -----------------------------
    # Model
    # -----------------------------
    d_model: int = 256
    n_latents: int = 16
    fusion_layers: int = 4
    n_heads: int = 8
    n_experts: int = 3

    text_tokens: int = 4   # split text_vec768 into this many tokens
    img_tokens: int = 4    # split img_vec512 into this many tokens

    # Tab tokenization (compute-efficient alternative to per-feature tokens)
    #   - feature: one token per tabular feature (slow when you have 200+ features)
    #   - compact: pool tabular features into a small number of tokens before fusion (recommended)
    tab_token_mode: str = "compact"  # feature | compact
    tab_num_tokens: int = 8
    tab_cat_tokens: int = 8
    tab_pool_heads: int = 4
    tab_pool_dropout: float = 0.0

    # Probability-threshold sweep for the slow21 gate (uses p_fast at f1_threshold_hours)
    sweep_prob_threshold: bool = True
    sweep_threshold_points: int = 101

    # Checkpoint / artifact controls (disable for Optuna to reduce I/O)
    save_checkpoints: bool = True
    write_metrics_csv: bool = True

    # Optuna pruning (turns "full training for every trial" into multi-fidelity)
    optuna_pruner: str = "hyperband"  # none | median | hyperband
    optuna_pruner_warmup_steps: int = 3
    optuna_hyperband_reduction_factor: int = 3
    optuna_hyperband_min_resource: int = 4

    # Tabular handling
    drop_raw_text_cols: bool = True
    cat_max_unique: int = 200
    cat_hash_buckets: int = 2000
    keep_all_object_cols: bool = False

    dropout: float = 0.10
    attn_dropout: float = 0.10

    # -----------------------------
    # Optim / train
    # -----------------------------
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 25
    num_workers: int = 0
    amp: bool = True
    warmup_steps: int = 1000
    max_grad_norm: float = 2.0

    # Early stopping
    early_stop_patience: int = 5
    early_stop_metric: str = "eval_slow21_f1_prob_best"  # metric key (suggested: eval_slow21_f1_prob_best)

    # Runtime
    seed: int = 1337
    device: str = "auto"  # auto | cuda | cpu

    # Outputs
    dump_eval_predictions: bool = False
    dump_eval_predictions_split: str = "eval"  # eval | eval_sold | both
    dump_eval_predictions_limit: int = 0            # 0 = all rows
    enable_group_shap: bool = True
    group_shap_max_rows: int = 16
    group_shap_batch_size: int = 64



    # Optuna (optional hyperparameter search)
    optuna_trials: int = 0
    optuna_metric: str = "eval_slow21_f1_prob_best"     # metric key (suggested: eval_slow21_f1_prob_best)
    optuna_direction: str = "maximize"   # maximize | minimize
    optuna_epochs: int = 25              # epochs per trial (<= epochs)


def make_time_edges(horizon_hours: float, n_bins: int, mode: str = "log1p") -> np.ndarray:
    if mode == "linear":
        return np.linspace(0.0, horizon_hours, n_bins + 1, dtype=np.float32)
    if mode == "log1p":
        u = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        edges = np.expm1(u * np.log1p(horizon_hours)).astype(np.float32)
        edges[0] = 0.0
        edges[-1] = float(horizon_hours)
        return edges
    raise ValueError(f"Unknown binning mode: {mode}")


class FeedForward(nn.Module):
    """
    Transformer FFN with GEGLU.
    """
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * mult
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(a) * b
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AttentionBlock(nn.Module):
    """
    Pre-LN attention + FF.
    - If cross=True: attends from x_q -> x_kv
    - If cross=False: self-attn on x_q
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, cross: bool = False):
        super().__init__()
        self.cross = cross
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model) if cross else self.ln_q
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mult=4, dropout=dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.cross:
            x_kv = x_q

        q = self.ln_q(x_q)
        kv = self.ln_kv(x_kv)

        out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x_q + self.drop(out)
        x = x + self.drop(self.ff(self.ln_ff(x)))
        return x


class TabTokenizer(nn.Module):
    """
    FT-Transformer-style feature tokenization:
      numeric: x_i * w_i + b_i + id_emb_i
      cat: Emb(col)(idx) + id_emb_j
    Produces tokens: [CLS] + numeric_tokens + cat_tokens
    """
    def __init__(self, n_num: int, cat_cardinalities: List[int], d_model: int, dropout: float):
        super().__init__()
        self.n_num = int(n_num)
        self.n_cat = int(len(cat_cardinalities))

        if self.n_num > 0:
            self.num_weight = nn.Parameter(torch.randn(self.n_num, d_model) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(self.n_num, d_model))
            self.num_id = nn.Parameter(torch.randn(self.n_num, d_model) * 0.02)
        else:
            self.num_weight = None
            self.num_bias = None
            self.num_id = None

        self.cat_embeds = nn.ModuleList([nn.Embedding(int(card), d_model) for card in cat_cardinalities])
        self.cat_id = nn.Parameter(torch.randn(self.n_cat, d_model) * 0.02) if self.n_cat > 0 else None

        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        B = x_num.shape[0]
        toks = [self.cls.expand(B, -1, -1)]

        if self.n_num > 0:
            # (B,n_num,1) * (1,n_num,d) -> (B,n_num,d)
            num = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            num = num + self.num_id.unsqueeze(0)
            toks.append(num)

        if self.n_cat > 0:
            cat_tokens = []
            for j, emb in enumerate(self.cat_embeds):
                idx = x_cat[:, j]
                cat_tokens.append(emb(idx))
            cat = torch.stack(cat_tokens, dim=1)  # (B,n_cat,d)
            cat = cat + self.cat_id.unsqueeze(0)
            toks.append(cat)

        out = torch.cat(toks, dim=1)
        return self.dropout(out)


class DenseTokenizer(nn.Module):
    """
    Turns a dense vector into multiple tokens:
      vec -> Linear -> reshape (B,n_tokens,d_model) + positional embedding
    """
    def __init__(self, d_in: int, d_model: int, n_tokens: int, dropout: float):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.proj = nn.Linear(d_in, n_tokens * d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        t = self.proj(x).view(B, self.n_tokens, self.d_model)
        t = t + self.pos.unsqueeze(0)
        return self.dropout(t)


class PerceiverFusion(nn.Module):
    """
    Perceiver-style fusion:
      - learnable latent tokens
      - repeated: cross-attn(latents <- tokens) then self-attn(latents)
    """
    def __init__(self, d_model: int, n_latents: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross": AttentionBlock(d_model, n_heads, dropout=dropout, cross=True),
                "self":  AttentionBlock(d_model, n_heads, dropout=dropout, cross=False),
            })
            for _ in range(n_layers)
        ])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = tokens.shape[0]
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B,L,d)
        for layer in self.layers:
            lat = layer["cross"](lat, tokens, key_padding_mask=key_padding_mask)
            lat = layer["self"](lat)
        lat = self.out_ln(lat)
        pooled = lat.mean(dim=1)
        return pooled, lat


class MoEHazardHead(nn.Module):
    """
    Mixture-of-Experts hazard head:
      - experts each output hazard logits (n_bins)
      - gate outputs mixture weights over experts
    """
    def __init__(self, d_model: int, n_bins: int, n_experts: int, dropout: float):
        super().__init__()
        self.n_bins = int(n_bins)
        self.n_experts = int(n_experts)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_bins),
            )
            for _ in range(n_experts)
        ])

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.gate(h), dim=1)  # (B,E)
        logits = torch.stack([ex(h) for ex in self.experts], dim=1)  # (B,E,K)
        hazard = torch.sigmoid(logits)  # (B,E,K)
        return hazard, w


def _pick_n_heads(d_model: int, requested: int) -> int:
    """Pick a valid number of attention heads (must divide d_model)."""
    requested = int(max(1, requested))
    requested = min(requested, d_model)
    for h in range(requested, 0, -1):
        if d_model % h == 0:
            return h
    return 1


class TabTokenizerCompact(nn.Module):
    """Compact tabular tokenizer.

    The original tokenizer emits one token per numeric and categorical feature
    (e.g., 222 + 61 = 283 tokens), which makes Perceiver cross-attention expensive.

    This tokenizer:
      1) embeds per-feature numeric/categorical signals into d_model
      2) pools them into a small set of learned tokens using cross-attention

    Typical setting: 8 numeric tokens + 8 categorical tokens => 16 total tab tokens.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        d_model: int,
        num_tokens: int = 8,
        cat_tokens: int = 8,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_num = int(n_num)
        self.n_cat = int(len(cat_cardinalities))
        self.d_model = int(d_model)

        self.num_tokens = int(num_tokens) if self.n_num > 0 else 0
        self.cat_tokens = int(cat_tokens) if self.n_cat > 0 else 0

        self.pool_heads = _pick_n_heads(self.d_model, n_heads)
        self.drop = nn.Dropout(dropout)

        # Numeric: per-feature affine -> token, then pool
        if self.n_num > 0 and self.num_tokens > 0:
            self.num_w = nn.Parameter(torch.randn(self.n_num, self.d_model) * 0.02)
            self.num_b = nn.Parameter(torch.zeros(self.n_num, self.d_model))
            self.num_queries = nn.Parameter(torch.randn(self.num_tokens, self.d_model) * 0.02)
            self.num_pool = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.pool_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.register_parameter("num_w", None)
            self.register_parameter("num_b", None)
            self.register_parameter("num_queries", None)
            self.num_pool = None

        # Categorical: per-feature embedding, then pool
        if self.n_cat > 0 and self.cat_tokens > 0:
            self.cat_embs = nn.ModuleList([nn.Embedding(int(c), self.d_model) for c in cat_cardinalities])
            self.cat_queries = nn.Parameter(torch.randn(self.cat_tokens, self.d_model) * 0.02)
            self.cat_pool = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.pool_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.cat_embs = nn.ModuleList()
            self.register_parameter("cat_queries", None)
            self.cat_pool = None

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # x_num: (B, n_num) float
        # x_cat: (B, n_cat) long
        if x_num is not None:
            B = x_num.shape[0]
            device = x_num.device
        else:
            B = x_cat.shape[0]
            device = x_cat.device

        outs: List[torch.Tensor] = []

        if self.n_num > 0 and self.num_tokens > 0:
            # (B, n_num, d_model)
            num_tok = x_num.unsqueeze(-1) * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)
            num_tok = self.drop(num_tok)
            q = self.num_queries.unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.num_pool(q, num_tok, num_tok, need_weights=False)
            outs.append(pooled)

        if self.n_cat > 0 and self.cat_tokens > 0:
            # (B, n_cat, d_model)
            cat_tok = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
            cat_tok = self.drop(cat_tok)
            q = self.cat_queries.unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.cat_pool(q, cat_tok, cat_tok, need_weights=False)
            outs.append(pooled)

        if len(outs) == 0:
            return torch.zeros((B, 0, self.d_model), device=device)

        return torch.cat(outs, dim=1)


class MultiModalSurvModel(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: List[int], cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        # Tabular tokenization:
        #  - "feature": one token per numeric/categorical feature (expensive if you have hundreds of features)
        #  - "compact": learn a small number of numeric+categorical summary tokens (much cheaper)
        if cfg.tab_token_mode == "feature":
            self.tab_tok = TabTokenizer(n_num, cat_cardinalities, cfg.d_model, cfg.dropout)
            tab_tokens = 1 + n_num + len(cat_cardinalities)
        else:
            self.tab_tok = TabTokenizerCompact(
                n_num=n_num,
                cat_cardinalities=cat_cardinalities,
                d_model=cfg.d_model,
                num_tokens=cfg.tab_num_tokens,
                cat_tokens=cfg.tab_cat_tokens,
                n_heads=cfg.tab_pool_heads,
                dropout=cfg.tab_pool_dropout,
            )
            tab_tokens = cfg.tab_num_tokens + cfg.tab_cat_tokens

        self.text_tok = DenseTokenizer(768, cfg.d_model, cfg.text_tokens, cfg.dropout)
        self.img_tok = DenseTokenizer(512, cfg.d_model, cfg.img_tokens, cfg.dropout)

        self.tokens_total = tab_tokens + cfg.text_tokens + cfg.img_tokens

        self.fusion = PerceiverFusion(cfg.d_model, cfg.n_latents, cfg.fusion_layers, _pick_n_heads(cfg.d_model, cfg.n_heads), cfg.attn_dropout)
        self.head = MoEHazardHead(cfg.d_model, cfg.n_bins, cfg.n_experts, cfg.dropout)

        # (Patch 5) Dedicated tail gate head (enabled by cfg.tail_bce_weight > 0 during training).
        self.tail_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x_num, x_cat, x_text, x_img, m_text=None, m_img=None):
        tab_toks = self.tab_tok(x_num, x_cat)  # [B, T_tab, D]
        text_toks = self.text_tok(x_text)  # [B, T_text, D]
        img_toks = self.img_tok(x_img)  # [B, T_img, D]

        tokens = torch.cat([tab_toks, text_toks, img_toks], dim=1)
        pooled, _lat = self.fusion(tokens)  # [B, D]

        hazard_e, w_mix = self.head(pooled)  # [B,E,BINS], [B,E]
        tail_logit = self.tail_head(pooled).squeeze(-1)  # [B]
        return hazard_e, w_mix, tail_logit


def mixture_surv_prob_at_times(
    hazard_e: torch.Tensor,  # (B,E,K)
    w: torch.Tensor,         # (B,E)
    edges: torch.Tensor,     # (K+1,)
    times: torch.Tensor,     # (T,)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Returns S(t) for each sample at each time in `times`.
    Output shape: (B,T)
    """
    B, E, K = hazard_e.shape
    hazard_e = hazard_e.clamp(eps, 1.0 - eps)
    log_surv_e = torch.log1p(-hazard_e)  # (B,E,K)
    csum = torch.cumsum(log_surv_e, dim=2)

    w = w.clamp(min=eps)
    w = w / w.sum(dim=1, keepdim=True)

    outs = []
    for t0 in times:
        t = t0.expand(B)
        idx = torch.bucketize(t.contiguous(), edges, right=False) - 1
        idx = idx.clamp(0, K - 1)
        idxm1 = (idx - 1).clamp(0, K - 1)

        gather_idxm1 = idxm1.view(B, 1, 1).expand(B, E, 1)
        pre = torch.where(
            idx.view(B, 1, 1) > 0,
            csum.gather(2, gather_idxm1),
            torch.zeros((B, E, 1), device=hazard_e.device, dtype=csum.dtype),
        ).squeeze(2)

        start = edges.gather(0, idx)
        end = edges.gather(0, (idx + 1).clamp(0, K))
        denom = (end - start).clamp(min=1e-6)
        frac = ((t - start) / denom).clamp(0.0, 1.0).view(B, 1)

        gather_idx = idx.view(B, 1, 1).expand(B, E, 1)
        log_surv_k = log_surv_e.gather(2, gather_idx).squeeze(2)

        S_e = torch.exp(pre + frac * log_surv_k)  # (B,E)
        S_mix = (w * S_e).sum(dim=1)              # (B,)
        outs.append(S_mix.unsqueeze(1))

    return torch.cat(outs, dim=1)


@torch.no_grad()
def expected_hours_mixture(hazard_e: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Expected time within horizon for a mixture distribution.
    """
    B, E, K = hazard_e.shape
    hazard_e = hazard_e.clamp(eps, 1.0 - eps)

    w = w.clamp(min=eps)
    w = w / w.sum(dim=1, keepdim=True)

    mids = 0.5 * (edges[:-1] + edges[1:])  # (K,)
    exp_total = torch.zeros(B, device=hazard_e.device)

    for e in range(E):
        h = hazard_e[:, e, :]  # (B,K)
        surv = torch.cumprod(1.0 - h, dim=1)
        S_prev = torch.cat([torch.ones(B, 1, device=h.device), surv[:, :-1]], dim=1)
        p = h * S_prev
        exp_e = (p * mids.unsqueeze(0)).sum(dim=1) + surv[:, -1] * edges[-1]
        exp_total += w[:, e] * exp_e

    return exp_total


def _weighted_mean(loss_vec: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = w.clamp(min=0.0)
    denom = w.sum().clamp(min=eps)
    return (loss_vec * w).sum() / denom


def mixture_survival_nll_discrete_vec(
    hazard_e: torch.Tensor,
    w_mix: torch.Tensor,
    t: torch.Tensor,
    event: torch.Tensor,
    edges: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Same as mixture_survival_nll_discrete(), but returns per-sample NLL (B,).
    """
    B, E, K = hazard_e.shape
    hazard_e = hazard_e.clamp(eps, 1.0 - eps)

    # Ensure contiguous input for bucketize/searchsorted perf
    t = t.contiguous()

    log_surv = torch.log1p(-hazard_e)  # (B,E,K)
    # S_{k} at end of each bin
    csum = torch.cumsum(log_surv, dim=2)  # (B,E,K)

    idx = torch.bucketize(t.contiguous(), edges, right=False) - 1
    idx = idx.clamp(0, K - 1)

    idxm1 = (idx - 1).clamp(0, K - 1)
    pre = torch.where(
        (idx > 0).view(B, 1),
        csum.gather(2, idxm1.view(B, 1, 1).expand(B, E, 1)).squeeze(2),
        torch.zeros((B, E), device=t.device, dtype=csum.dtype),
    )  # (B,E)

    h_k = hazard_e.gather(2, idx.view(B, 1, 1).expand(B, E, 1)).squeeze(2)  # (B,E)
    log_event = torch.log(h_k)

    # fractional censor inside bin
    start = edges.gather(0, idx)
    end = edges.gather(0, (idx + 1).clamp(0, K))
    denom = (end - start).clamp(min=1e-6)
    frac = ((t - start) / denom).clamp(0.0, 1.0)  # (B,)

    log_surv_k = log_surv.gather(2, idx.view(B, 1, 1).expand(B, E, 1)).squeeze(2)  # (B,E)
    log_censor = pre + frac.view(B, 1) * log_surv_k  # (B,E)

    loglik_e = torch.where(event.view(B, 1) > 0.5, pre + log_event, log_censor)  # (B,E)

    # mix in probability space
    logw = torch.log(w_mix.clamp(min=eps))  # (B,E)
    loglik = torch.logsumexp(logw + loglik_e, dim=1)  # (B,)
    return -loglik


def mixture_survival_nll_discrete(
    hazard_e: torch.Tensor,
    w_mix: torch.Tensor,
    t: torch.Tensor,
    event: torch.Tensor,
    edges: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """Scalar mean NLL wrapper (for eval logging)."""
    return mixture_survival_nll_discrete_vec(hazard_e, w_mix, t, event, edges, eps=eps).mean()


def masked_bce_vec(p: torch.Tensor, y: torch.Tensor, known: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns per-sample masked BCE (B,).
    known=1 where label known, 0 else.
    """
    p = p.clamp(eps, 1.0 - eps)
    bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
    return bce * known


def horizon_bce_loss(
    hazard_e: torch.Tensor,
    w_mix: torch.Tensor,
    edges: torch.Tensor,
    dur_raw: torch.Tensor,
    sold_event: torch.Tensor,
    w_row: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Multi-horizon BCE on P(T <= h) with censoring-aware masking.

    For each horizon h:
      - y=1 (fast) if sold_event==1 and dur_raw <= h
      - y=0 (not-fast) is KNOWN only if dur_raw >= h (survived past h)
      - rows with dur_raw < h and not sold are unknown and masked out

    The per-row loss is weighted by w_row, and (optionally) by cfg.slow_tail_weight
    for known negatives (tail) to emphasize tail precision/recall.

    Returns a scalar (sum of horizon losses, consistent with prior behavior).
    """
    ths = cfg.thresholds_hours[: max(1, int(cfg.max_horizon_bce_points))]
    if len(ths) == 0:
        return torch.zeros((), device=dur_raw.device)

    ths_t = torch.tensor(ths, device=dur_raw.device, dtype=torch.float32)
    P = 1.0 - mixture_surv_prob_at_times(hazard_e, w_mix, edges, ths_t)  # [B, H]

    sold = sold_event > 0.5
    dur = dur_raw

    loss_total = torch.zeros((), device=dur_raw.device)

    for j, h in enumerate(ths):
        h = float(h)
        y1 = sold & (dur <= h)
        known0 = dur >= h
        known = (y1 | known0).float()
        y = y1.float()

        bce_vec = masked_bce_vec(P[:, j], y, known)

        w_eff = w_row
        if float(cfg.slow_tail_weight) != 1.0:
            slow_known = (known > 0.5) & (~y1)
            w_eff = w_eff * torch.where(
                slow_known,
                torch.tensor(float(cfg.slow_tail_weight), device=w_eff.device, dtype=w_eff.dtype),
                torch.tensor(1.0, device=w_eff.device, dtype=w_eff.dtype),
            )

        denom = (w_eff * known).sum().clamp(min=1e-6)
        loss_h = (bce_vec * w_eff).sum() / denom
        loss_total = loss_total + loss_h

    return loss_total


def huber_or_l1_vec(err: torch.Tensor, loss_type: str, delta: float) -> torch.Tensor:
    """
    Returns per-sample loss (same shape as err).
    """
    ae = err.abs()
    if loss_type == "l1":
        return ae
    # huber
    d = float(delta)
    quad = torch.minimum(ae, torch.tensor(d, device=err.device, dtype=err.dtype))
    lin = ae - quad
    return 0.5 * (quad ** 2) / d + lin


def quantile_hours_mixture(
    hazard_e: torch.Tensor,
    w_mix: torch.Tensor,
    edges: torch.Tensor,
    q: float
) -> torch.Tensor:
    """
    Approximate mixture quantile using bin masses (piecewise-uniform inside bin).

    q in (0,1). Returns hours in [0, edges[-1]].
    """
    B, E, K = hazard_e.shape
    eps = 1e-6
    q = float(q)
    q = min(max(q, eps), 1.0 - eps)

    # expert bin masses
    hazard_e = hazard_e.clamp(eps, 1.0 - eps)
    surv_e = torch.cumprod(1.0 - hazard_e, dim=2)  # (B,E,K)
    S_prev = torch.cat([torch.ones((B, E, 1), device=hazard_e.device, dtype=hazard_e.dtype), surv_e[:, :, :-1]], dim=2)
    p_e = hazard_e * S_prev  # (B,E,K)

    p_mix = (w_mix.unsqueeze(2) * p_e).sum(dim=1)  # (B,K)
    cdf = torch.cumsum(p_mix, dim=1)  # (B,K)

    qv = torch.full((B,), q, device=hazard_e.device, dtype=hazard_e.dtype)
    # idx = number of bins with cdf < q
    idx = torch.sum(cdf < qv.unsqueeze(1), dim=1).clamp(0, K - 1)  # (B,)
    idxm1 = (idx - 1).clamp(0, K - 1)

    cdf_prev = torch.where(
        idx > 0,
        cdf.gather(1, idxm1.unsqueeze(1)).squeeze(1),
        torch.zeros((B,), device=hazard_e.device, dtype=hazard_e.dtype),
    )
    mass_k = p_mix.gather(1, idx.unsqueeze(1)).squeeze(1).clamp(min=eps)
    frac = ((qv - cdf_prev) / mass_k).clamp(0.0, 1.0)

    start = edges.gather(0, idx)
    end = edges.gather(0, (idx + 1).clamp(0, K))
    t = start + frac * (end - start)
    return t


def point_hours_mixture(hazard_e: torch.Tensor, w_mix: torch.Tensor, edges: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.point_estimator == "expected":
        return expected_hours_mixture(hazard_e, w_mix, edges)
    if cfg.point_estimator == "median":
        return quantile_hours_mixture(hazard_e, w_mix, edges, 0.50)
    if cfg.point_estimator == "quantile":
        return quantile_hours_mixture(hazard_e, w_mix, edges, cfg.point_quantile)
    raise ValueError(f"Unknown point_estimator: {cfg.point_estimator}")


def boundary_focus_weights(duration_hours: np.ndarray, boundary_hours: float, k: float, sigma_days: float) -> np.ndarray:
    """
    Symmetric Gaussian bump around the boundary (in DAYS).
      w = 1 + k * exp(-0.5 * (( (t-boundary)/24 ) / sigma_days)^2)
    """
    n = len(duration_hours)
    if k is None or k <= 0 or sigma_days is None or sigma_days <= 0:
        return np.ones(n, dtype=np.float32)

    z = (duration_hours.astype(np.float32) - float(boundary_hours)) / 24.0
    w = 1.0 + float(k) * np.exp(-0.5 * (z / float(sigma_days)) ** 2)
    return w.astype(np.float32)


def time_decay_weights(edited_date_utc: pd.Series, ref_utc: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """
    Exponential time-decay weights with half-life in days (mean-normalized).
    w = 0.5 ** (age_days / half_life_days)
    """
    n = len(edited_date_utc)
    if half_life_days is None or half_life_days <= 0:
        return np.ones(n, dtype=np.float32)

    age_days = (ref_utc - edited_date_utc).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(age_days.max() if np.isfinite(age_days.max()) else 0.0)
    w = np.power(0.5, age_days.to_numpy(dtype=np.float32) / float(half_life_days)).astype(np.float32)

    m = float(np.mean(w)) if n else 1.0
    if m > 0:
        w /= m
    return w
