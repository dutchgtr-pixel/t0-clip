"""Small mathematical references, not copies of historical production models.

The network learns from precomputed features/embeddings. There are no encoder
downloads or pretrained checkpoints. Time is discretized into equal-width bins;
within-bin survival is interpolated with a constant hazard rate.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import (feature_signature, flatten_features, validate_features,
                     validate_outcomes, validate_times)


def _log_survival_at(logits, times, edges):
    """Expert log S(t), shape [batch, experts, times]."""
    log_surv = F.logsigmoid(-logits)
    prefix = torch.cat([torch.zeros_like(log_surv[..., :1]), log_surv.cumsum(-1)], dim=-1)
    index = torch.bucketize(times.contiguous(), edges[1:], right=False).clamp(max=logits.shape[-1] - 1)
    fraction = ((times - edges[index]) / (edges[index + 1] - edges[index])).clamp(0, 1)
    return prefix[..., index] + fraction * log_surv[..., index]


def survival_nll(logits, mix_logits, duration, event, edges):
    """Censored discrete-event likelihood; events beyond horizon are censored.

    Observed events use interval probability S(left)*hazard. Censoring uses
    S(t), with fractional exposure inside the last observed interval.
    """
    horizon = edges[-1]
    observed = event.bool() & (duration <= horizon)
    t = duration.clamp(min=0, max=horizon)
    index = torch.bucketize(t.contiguous(), edges[1:], right=False).clamp(max=logits.shape[-1] - 1)
    log_surv = F.logsigmoid(-logits)
    prefix = torch.cat([torch.zeros_like(log_surv[..., :1]), log_surv.cumsum(-1)], dim=-1)
    ix = index[:, None, None].expand(-1, logits.shape[1], 1)
    before = prefix.gather(-1, ix).squeeze(-1)
    log_event = before + F.logsigmoid(logits).gather(-1, ix).squeeze(-1)
    fraction = ((t - edges[index]) / (edges[index + 1] - edges[index])).clamp(0, 1)
    log_censor = before + fraction[:, None] * log_surv.gather(-1, ix).squeeze(-1)
    log_lik = torch.where(observed[:, None], log_event, log_censor)
    return -torch.logsumexp(F.log_softmax(mix_logits, dim=1) + log_lik, dim=1).mean()


class MLPHazard(nn.Module):
    def __init__(self, input_dim, width, n_bins, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(width, width), nn.ReLU(), nn.Linear(width, n_bins))

    def forward(self, x):
        logits = self.network(x["flat"])
        return logits[:, None, :], logits.new_zeros((len(logits), 1))


class FusionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents, tokens):
        query, kv = self.norms[0](latents), self.norms[1](tokens)
        latents = latents + self.dropout(self.cross(query, kv, kv, need_weights=False)[0])
        query = self.norms[2](latents)
        latents = latents + self.dropout(self.self_attn(query, query, query, need_weights=False)[0])
        return latents + self.dropout(self.ff(self.norms[3](latents)))


class PerceiverMoE(nn.Module):
    def __init__(self, signature, width, heads, n_latents, layers, experts, n_bins, dropout):
        super().__init__()
        n_tab = signature["tabular"][0]
        self.tab_weight = nn.Parameter(torch.randn(n_tab, width) * 0.02)
        self.tab_bias = nn.Parameter(torch.randn(n_tab, width) * 0.02)
        self.tab_queries = nn.Parameter(torch.randn(4, width) * 0.02)
        self.tab_pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.projections = nn.ModuleDict()
        self.positions = nn.ParameterDict()
        self.missing = nn.ParameterDict()
        self.text_tokens = 2
        for name in ("text", "image", "report"):
            if name not in signature:
                continue
            dim = signature[name][-1]
            slots = self.text_tokens if name == "text" else signature[name][0]
            self.projections[name] = nn.Linear(dim, width * self.text_tokens if name == "text" else width)
            self.positions[name] = nn.Parameter(torch.randn(slots, width) * 0.02)
            if name != "text":
                self.missing[name] = nn.Parameter(torch.randn(1, width) * 0.02)
        self.width = width
        self.latents = nn.Parameter(torch.randn(n_latents, width) * 0.02)
        self.blocks = nn.ModuleList([FusionBlock(width, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                                   nn.Linear(width, n_bins)) for _ in range(experts)])

    def forward(self, x):
        n = len(x["tabular"])
        tab = x["tabular"][..., None] * self.tab_weight + self.tab_bias
        query = self.tab_queries[None].expand(n, -1, -1)
        tokens = [self.tab_pool(query, tab, tab, need_weights=False)[0]]
        for name, projection in self.projections.items():
            a = x[name]
            if name == "text":
                tok = projection(a).reshape(n, self.text_tokens, self.width)
            else:
                present = x[name + "_mask"][..., None]
                tok = projection(a * present)
                tok = torch.where(present, tok, self.missing[name])
            tokens.append(tok + self.positions[name])
        tokens = torch.cat(tokens, dim=1)
        latents = self.latents[None].expand(n, -1, -1)
        for block in self.blocks:
            latents = block(latents, tokens)
        pooled = self.norm(latents).mean(dim=1)
        return torch.stack([expert(pooled) for expert in self.experts], dim=1), self.gate(pooled)


class NeuralSurvivalModel:
    def __init__(self, architecture="mlp", random_state=42, horizon=504.0, n_bins=128,
                 epochs=50, patience=8, batch_size=128, learning_rate=1e-3,
                 weight_decay=1e-4, width=64, heads=4, n_latents=8, layers=2,
                 experts=3, dropout=0.1, device="cpu"):
        if architecture not in ("mlp", "perceiver_moe"):
            raise ValueError("Unknown neural architecture")
        if not np.isfinite(horizon) or horizon <= 0 or n_bins < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("horizon, n_bins, epochs and batch_size must be positive")
        if patience < 1 or width < 1 or heads < 1 or width % heads or n_latents < 1 or layers < 1 or experts < 1:
            raise ValueError("Invalid attention/training dimensions")
        if not 0 <= dropout < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid optimization options")
        self.architecture, self.random_state = architecture, random_state
        self.horizon, self.n_bins = float(horizon), int(n_bins)
        self.epochs, self.patience, self.batch_size = int(epochs), int(patience), int(batch_size)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.width, self.heads, self.n_latents, self.layers = width, heads, n_latents, layers
        self.experts, self.dropout, self.device = experts, dropout, torch.device(device)

    def _tensor_features(self, x):
        x = validate_features(x)
        if feature_signature(x) != self.signature_:
            raise ValueError("Features differ from the fitted feature contract")
        if self.architecture == "mlp":
            x = {"flat": flatten_features(x)}
        return {k: torch.as_tensor(v, device=self.device) for k, v in x.items()}

    def fit(self, x, duration, event, *, validation=None):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True)
        x = validate_features(x)
        n = len(x["tabular"])
        duration, event = validate_outcomes(duration, event, n)
        self.signature_ = feature_signature(x)
        if self.architecture == "mlp":
            self.model_ = MLPHazard(flatten_features(x).shape[1], self.width, self.n_bins, self.dropout)
        else:
            self.model_ = PerceiverMoE(self.signature_, self.width, self.heads, self.n_latents,
                                       self.layers, self.experts, self.n_bins, self.dropout)
        self.model_.to(self.device)
        self.edges_ = torch.linspace(0, self.horizon, self.n_bins + 1, device=self.device)
        features = self._tensor_features(x)
        duration_t = torch.as_tensor(duration, dtype=torch.float32, device=self.device)
        event_t = torch.as_tensor(event, device=self.device)
        validation_t = None
        if validation is not None:
            vx, vd, ve = validation
            vx = self._tensor_features(vx)
            vd, ve = validate_outcomes(vd, ve, len(next(iter(vx.values()))))
            validation_t = (vx, torch.as_tensor(vd, dtype=torch.float32, device=self.device),
                            torch.as_tensor(ve, device=self.device))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        generator = torch.Generator().manual_seed(self.random_state)
        best, stale, best_state = float("inf"), 0, None
        self.history_ = []
        self.best_epoch_ = None
        for epoch in range(self.epochs):
            self.model_.train()
            order = torch.randperm(n, generator=generator).to(self.device)
            total = 0.0
            for index in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits, mix = self.model_({k: a[index] for k, a in features.items()})
                loss = survival_nll(logits, mix, duration_t[index], event_t[index], self.edges_)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training likelihood")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(index)
            record = {"epoch": epoch + 1, "train_nll": total / n}
            if validation_t is not None:
                self.model_.eval()
                vx, vd, ve = validation_t
                with torch.no_grad():
                    value = 0.0
                    for start in range(0, len(vd), self.batch_size):
                        sl = slice(start, start + self.batch_size)
                        logits, mix = self.model_({k: a[sl] for k, a in vx.items()})
                        loss = survival_nll(logits, mix, vd[sl], ve[sl], self.edges_)
                        value += float(loss) * len(vd[sl])
                    value /= len(vd)
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite validation likelihood")
                record["validation_nll"] = value
                if value < best - 1e-7:
                    best, stale = value, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
            self.history_.append(record)
            if validation_t is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        else:
            self.best_epoch_ = len(self.history_)
        self.model_.eval()
        return self

    def predict_survival(self, x, times):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the model before prediction")
        times = validate_times(times)
        if np.any(times > self.horizon):
            raise ValueError("Neural predictions cannot extrapolate beyond the fitted horizon")
        features = self._tensor_features(x)
        n = len(next(iter(features.values())))
        query = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        parts = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                sl = slice(start, start + self.batch_size)
                logits, mix = self.model_({k: a[sl] for k, a in features.items()})
                log_s = _log_survival_at(logits, query, self.edges_)
                parts.append(torch.logsumexp(F.log_softmax(mix, dim=1)[..., None] + log_s,
                                             dim=1).exp().cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, len(times)))
