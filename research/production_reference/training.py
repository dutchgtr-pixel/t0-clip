"""Generic tensor-training adapter around the archived model and loss functions.

This is new public adapter code, not the historical production trainer or a
reproduction of its unavailable trial-level loss coefficients. It has no data
connector, split selection, model selection, or holdout access.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Iterable

import torch

from .model_core import (
    MultiModalSurvModel,
    TrainConfig,
    bce_prob_masked,
    bce_with_logits_masked,
    make_time_edges,
    mixture_surv_prob_at_times,
    mixture_survival_nll_discrete_vec,
)


@dataclass(frozen=True)
class Objective:
    """Explicit public demonstration weights, not recovered production values."""

    nll_weight: float = 1.0
    head_weight: float = 1.0
    curve_weight: float = 1.0
    head_horizon: float = 72.0
    curve_horizons: tuple[float, ...] = (24.0, 72.0, 168.0, 240.0)


@dataclass
class TensorBatch:
    numeric: torch.Tensor
    categorical: torch.Tensor
    text: torch.Tensor
    slot_images: torch.Tensor
    slot_reports: torch.Tensor
    image_missing: torch.Tensor
    report_missing: torch.Tensor
    duration: torch.Tensor
    event: torch.Tensor
    weight: torch.Tensor

    def to(self, device: torch.device) -> "TensorBatch":
        return TensorBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def validate(self) -> None:
        n = self.duration.numel()
        if n == 0 or self.duration.ndim != 1:
            raise ValueError("A batch must have a nonempty one-dimensional duration vector")
        for field in fields(self):
            value = getattr(self, field.name)
            if value.shape[0] != n or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid batch dimension or nonfinite values: {field.name}")
        if self.event.shape != (n,) or self.weight.shape != (n,):
            raise ValueError("Event and weight vectors must align with duration")
        if bool((self.duration < 0).any()) or bool((self.weight < 0).any()) or float(self.weight.sum()) <= 0:
            raise ValueError("Durations/weights must be nonnegative and total weight positive")
        if not bool(((self.event == 0) | (self.event == 1)).all()):
            raise ValueError("Event must be binary: 1 observed, 0 right-censored")
        if self.text.shape != (n, 768):
            raise ValueError("Text vectors must have dimension 768")
        if self.slot_images.ndim != 3 or self.slot_images.shape[-1] != 512:
            raise ValueError("Slot image vectors must have dimension 512")
        if self.slot_reports.shape != (*self.slot_images.shape[:2], 768):
            raise ValueError("Slot report vectors must align and have dimension 768")
        if self.image_missing.shape != self.slot_images.shape[:2] or self.report_missing.shape != self.slot_images.shape[:2]:
            raise ValueError("Slot missingness masks must align with vector slots")


def historical_model() -> MultiModalSurvModel:
    """Instantiate the historical dimensions with random weights, no data load."""
    spec = json.loads(Path(__file__).with_name("historical_architecture.json").read_text())
    cfg = TrainConfig(**spec["model_config"])
    return MultiModalSurvModel(spec["n_num"], spec["cat_cardinalities"], cfg)


def objective_loss(outputs, batch: TensorBatch, edges: torch.Tensor, spec: Objective):
    """Censored likelihood plus known-horizon curve/head supervision.

    Follow-up shorter than a binary target horizon has no binary label unless an
    event was observed. Events after the model horizon become right-censored at
    that horizon for the survival likelihood.
    """
    hazard, mixture, head_logit = outputs
    horizon = float(edges[-1])
    horizons = (*spec.curve_horizons, spec.head_horizon)
    if any(h <= 0 or h > horizon for h in horizons):
        raise ValueError("Binary supervision horizons must lie within model support")
    if any(w < 0 for w in (spec.nll_weight, spec.head_weight, spec.curve_weight)):
        raise ValueError("Loss weights must be nonnegative")
    effective_time = batch.duration.clamp(max=horizon)
    effective_event = batch.event * (batch.duration <= horizon).to(batch.event.dtype)
    nll_rows = mixture_survival_nll_discrete_vec(hazard, mixture, effective_time, effective_event, edges)
    nll = (nll_rows * batch.weight).sum() / batch.weight.sum().clamp_min(1e-8)
    observed = batch.event > 0.5
    head_known = observed | (batch.duration >= spec.head_horizon)
    head_target = (batch.duration > spec.head_horizon) | (~observed & (batch.duration >= spec.head_horizon))
    head_loss = bce_with_logits_masked(head_logit, head_target.float(), head_known, batch.weight)
    curve_loss = hazard.sum() * 0.0
    if spec.curve_horizons:
        times = torch.tensor(spec.curve_horizons, device=hazard.device, dtype=hazard.dtype)
        fast_probability = 1.0 - mixture_surv_prob_at_times(hazard, mixture, edges, times)
        for column, t in enumerate(spec.curve_horizons):
            known = observed | (batch.duration >= t)
            fast_target = observed & (batch.duration <= t)
            curve_loss = curve_loss + bce_prob_masked(fast_probability[:, column], fast_target.float(), known, batch.weight)
    total = spec.nll_weight * nll + spec.head_weight * head_loss + spec.curve_weight * curve_loss
    return total, {"nll": nll, "head_bce": head_loss, "curve_bce": curve_loss}


def train_tensor_epoch(
    model: MultiModalSurvModel,
    batches: Iterable[TensorBatch],
    optimizer: torch.optim.Optimizer,
    objective: Objective = Objective(),
    max_grad_norm: float = 2.0,
) -> dict[str, float]:
    """Train on caller-supplied tensors; split construction belongs to the caller."""
    model.train()
    device = next(model.parameters()).device
    cfg = model.cfg
    if cfg.use_legacy_img_vector or not cfg.use_k8_vectors:
        raise ValueError("This public adapter supports the archived K8 input contract")
    edges = torch.as_tensor(make_time_edges(cfg.horizon_hours, cfg.n_bins, cfg.binning), device=device)
    totals = {"loss": 0.0, "nll": 0.0, "head_bce": 0.0, "curve_bce": 0.0}
    count = 0
    for raw_batch in batches:
        batch = raw_batch.to(device)
        batch.validate()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch.numeric, batch.categorical, batch.text, None,
            x_k8_img=batch.slot_images, x_k8_report=batch.slot_reports,
            m_k8_img=batch.image_missing, m_k8_report=batch.report_missing,
        )
        loss, components = objective_loss(outputs, batch, edges, objective)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        n = len(batch.duration)
        count += n
        for name, value in {"loss": loss, **components}.items():
            totals[name] += float(value.detach()) * n
    if not count:
        raise ValueError("No training batches supplied")
    return {name: value / count for name, value in totals.items()}


def synthetic_batch(n: int = 4) -> TensorBatch:
    """Toy tensor fixture; never a substitute for a temporal benchmark."""
    return TensorBatch(
        torch.randn(n, 3), torch.randint(0, 4, (n, 2)), torch.randn(n, 768),
        torch.randn(n, 8, 512), torch.randn(n, 8, 768),
        torch.zeros(n, 8), torch.zeros(n, 8),
        torch.linspace(12.0, 360.0, n), (torch.arange(n) % 2).float(), torch.ones(n),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One tiny synthetic optimizer step")
    parser.add_argument("--count-historical", action="store_true", help="Instantiate random weights and verify historical dimensions")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    if args.count_historical:
        model = historical_model()
        print(json.dumps({"family": "historical_k8_survival_17m", "unique_parameters": sum(p.numel() for p in model.parameters()), "state_dict_elements": sum(t.numel() for t in model.state_dict().values()), "weights": "random_initialization"}))
    elif args.smoke:
        cfg = TrainConfig(d_model=32, n_latents=4, fusion_layers=1, n_heads=4, n_experts=2, n_bins=16, text_tokens=2, use_k8_vectors=True, use_legacy_img_vector=False)
        model = MultiModalSurvModel(3, [4, 4], cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = train_tensor_epoch(model, [synthetic_batch()], optimizer)
        print(json.dumps({"purpose": "synthetic_training_smoke_only", "unique_parameters": sum(p.numel() for p in model.parameters()), "metrics": result}))
    else:
        parser.error("Choose --smoke or --count-historical")


if __name__ == "__main__":
    main()
