"""New portable adapters around the preserved historical numerical model.

These adapters reproduce the stated model/loss equations, not private historical
fit rows, optimizer schedules or selected coefficients. Inputs are fixed vectors;
no upstream encoder, data service or source-specific schema is required.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import legacy_core as core
from . import stage1_core, stage2_core


GATE_HOURS = {0: 504.0, 1: 168.0, 2: 72.0}


@dataclass
class FeatureBatch:
    """Legacy backbone inputs: numeric, category IDs, text768 and image512.

    Missing whole vectors must be represented upstream under a fixed contract.
    The legacy backbone ignores its optional text/image masks; the later slot
    network has a different explicit missing-slot contract.
    """

    numeric: np.ndarray
    categorical: np.ndarray
    text: np.ndarray
    image: np.ndarray

    def __len__(self):
        return len(self.numeric)

    def take(self, indices):
        return FeatureBatch(**{name: np.asarray(getattr(self, name))[indices]
                               for name in self.__dataclass_fields__})

    def validate(self, cardinalities=None, n_numeric=None):
        n = len(self)
        for name in self.__dataclass_fields__:
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or len(value) != n or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite two-dimensional matrix with aligned rows")
        if self.text.shape != (n, 768) or self.image.shape != (n, 512):
            raise ValueError("Legacy text/image dimensions must be 768/512")
        if self.numeric.shape[1] == 0:
            raise ValueError("Supply at least one numerical feature")
        cat = np.asarray(self.categorical)
        if (cat < 0).any() or not np.equal(cat, np.floor(cat)).all():
            raise ValueError("Categorical inputs must be nonnegative integer IDs")
        if n_numeric is not None and self.numeric.shape[1] != n_numeric:
            raise ValueError("Numerical feature count differs from TRAIN")
        if cardinalities is not None:
            if cat.shape[1] != len(cardinalities):
                raise ValueError("Categorical feature count differs from TRAIN")
            if any((cat[:, j] >= size).any() for j, size in enumerate(cardinalities)):
                raise ValueError("Unknown category IDs must map to reserved ID zero")
        return self

    def tensors(self, device):
        return {name: torch.as_tensor(getattr(self, name), device=device,
                                     dtype=torch.long if name == "categorical" else torch.float32)
                for name in self.__dataclass_fields__}


class FrozenFeatureEncoder:
    """Fit numeric imputation/scaling and categorical vocabularies on TRAIN.

    This is a new platform-neutral adapter, not the archived feature selector.
    Learned vocabularies are intentionally never written by the public examples.
    """

    def fit(self, numeric, categorical):
        x, c = np.asarray(numeric, float), np.asarray(categorical, object)
        if x.ndim != 2 or c.ndim != 2 or len(x) != len(c) or not len(x):
            raise ValueError("Aligned nonempty TRAIN matrices are required")
        if np.isinf(x).any():
            raise ValueError("Numeric infinity is invalid")
        self.median_ = np.array([np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.
                                 for a in x.T])
        complete = np.where(np.isnan(x), self.median_, x)
        self.mean_, self.scale_ = complete.mean(0), complete.std(0)
        self.scale_[self.scale_ == 0] = 1
        self.vocabulary_ = [{v: j + 1 for j, v in enumerate(sorted(set(map(str, column))))}
                            for column in c.T]
        self.cardinalities_ = [len(v) + 1 for v in self.vocabulary_]
        return self

    def transform(self, numeric, categorical, text, image):
        x, c = np.asarray(numeric, float), np.asarray(categorical, object)
        if x.ndim != 2 or x.shape[1] != len(self.median_) or c.shape != (len(x), len(self.vocabulary_)):
            raise ValueError("Feature columns differ from fitted preprocessing")
        x = (np.where(np.isnan(x), self.median_, x) - self.mean_) / self.scale_
        ids = np.empty(c.shape, dtype=np.int64)
        for j, vocabulary in enumerate(self.vocabulary_):
            ids[:, j] = [vocabulary.get(str(v), 0) for v in c[:, j]]
        return FeatureBatch(x.astype(np.float32), ids, np.asarray(text, np.float32),
                            np.asarray(image, np.float32)).validate(self.cardinalities_)


@dataclass(frozen=True)
class StageObjective:
    """Explicit coefficients; defaults are public demonstration choices.

    Head and gate-curve targets are slow-positive at the stage's own horizon.
    An observed event exactly at that horizon is fast; a censor exactly there
    is known slow under the archived convention.
    """

    nll: float = 1.0
    curve: float = 1.0
    head: float = 1.0
    positive_weight: float = 1.0
    curve_focal_gamma: float = 0.0
    head_focal_gamma: float = 0.0
    fast_aux: tuple[tuple[float, float], ...] = ()
    slow_aux: tuple[tuple[float, float], ...] = ()
    consistency: float = 0.0
    ranking: float = 0.0
    rank_margin: float = 0.0
    rank_pairs: int = 2048

    def validate(self, horizon):
        weights = [self.nll, self.curve, self.head, self.positive_weight,
                   self.curve_focal_gamma, self.head_focal_gamma, self.consistency, self.ranking]
        if not np.isfinite(weights).all() or min(weights) < 0 or self.positive_weight <= 0:
            raise ValueError("Objective weights/gammas must be finite and nonnegative")
        if self.nll + self.curve + self.head + self.consistency + self.ranking + sum(w for _, w in self.fast_aux + self.slow_aux) <= 0:
            raise ValueError("At least one objective term must be enabled")
        if not isinstance(self.rank_pairs, (int, np.integer)) or self.rank_pairs < 1 or not np.isfinite(self.rank_margin):
            raise ValueError("Invalid pair-ranking options")
        for h, weight in self.fast_aux + self.slow_aux:
            if not np.isfinite([h, weight]).all() or not 0 < h <= horizon or weight < 0:
                raise ValueError("Auxiliary horizons/weights must lie within model support")


def _outcomes(duration, event, n, weight=None):
    d, e = np.asarray(duration, float), np.asarray(event)
    w = np.ones(n) if weight is None else np.asarray(weight, float)
    if n == 0 or any(a.shape != (n,) for a in (d, e, w)):
        raise ValueError("Nonempty aligned outcome and weight vectors are required")
    if not np.isfinite(d).all() or (d < 0).any() or not np.isin(e, [0, 1]).all():
        raise ValueError("Durations must be finite/nonnegative and events binary")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be finite/nonnegative with positive total")
    return d, e.astype(bool), w


def stage_loss(outputs, duration, event, weight, edges, stage, objective):
    """Port of the archived stage objectives with explicit horizon recensoring.

    The differentiable functions are exact preserved definitions. The surrounding
    tensor assembly is new. Optional expected-time MAE is intentionally not added:
    the historical expected-time helper is decorated with no_grad.
    """
    hazard, mixture, head_logit = outputs
    alg = stage2_core if stage == 2 else stage1_core
    gate = GATE_HOURS[stage]
    effective = duration.clamp(max=edges[-1])
    observed = event.bool()
    effective_event = observed & (duration <= edges[-1])
    nll_rows = core.mixture_survival_nll_discrete_vec(hazard, mixture, effective, effective_event, edges)
    terms = {"nll": (nll_rows * weight).sum() / weight.sum().clamp_min(1e-8)}
    known = observed | (duration >= gate)
    slow_target = ((observed & (duration > gate)) | (~observed & (duration >= gate))).float()
    probability = core.mixture_surv_prob_at_times(hazard, mixture, edges, edges.new_tensor([gate]))[:, 0]
    terms["curve"] = alg.bce_prob_masked(probability, slow_target, known, weight,
                                        objective.positive_weight, objective.curve_focal_gamma)
    terms["head"] = alg.bce_with_logits_masked(head_logit, slow_target, known, weight,
                                              objective.positive_weight, objective.head_focal_gamma)
    zero = hazard.sum() * 0
    terms["auxiliary"] = zero
    for direction, points in (("fast", objective.fast_aux), ("slow", objective.slow_aux)):
        for h, coefficient in points:
            valid = observed | (duration >= h)
            fast = observed & (duration <= h)
            p = core.mixture_surv_prob_at_times(hazard, mixture, edges, edges.new_tensor([h]))[:, 0]
            target = (~fast).float() if direction == "slow" else fast.float()
            p = p if direction == "slow" else 1 - p
            terms["auxiliary"] = terms["auxiliary"] + coefficient * alg.bce_prob_masked(p, target, valid, weight)
    masked_weight = weight * known
    terms["consistency"] = ((probability - torch.sigmoid(head_logit)).square() * masked_weight).sum() / masked_weight.sum().clamp_min(1e-9)
    terms["ranking"] = zero
    if objective.ranking:
        fast_idx = torch.where(known & (slow_target == 0))[0]
        slow_idx = torch.where(known & (slow_target == 1))[0]
        count = min(objective.rank_pairs, len(fast_idx) * len(slow_idx))
        if count:
            p_fast = (1 - .5 * (probability + torch.sigmoid(head_logit))).clamp(1e-5, 1-1e-5)
            scores = torch.logit(p_fast)
            i = fast_idx[torch.randint(len(fast_idx), (count,), device=hazard.device)]
            j = slow_idx[torch.randint(len(slow_idx), (count,), device=hazard.device)]
            pair_weight = (weight[i] * weight[j]).detach()
            terms["ranking"] = (F.softplus(-(scores[i] - scores[j] - objective.rank_margin)) * pair_weight).sum() / pair_weight.sum().clamp_min(1e-9)
    total = (objective.nll * terms["nll"] + objective.curve * terms["curve"] +
             objective.head * terms["head"] + terms["auxiliary"] +
             objective.consistency * terms["consistency"] + objective.ranking * terms["ranking"])
    return total, terms


class StageEstimator:
    """Train, validate and score one historical-backbone stage on caller tensors.

    Configuration and preprocessing are supplied explicitly. Early stopping uses
    only the passed validation cohort. This adapter uses AdamW and a cosine epoch
    schedule; it does not reconstruct the historical search controller.
    """

    def __init__(self, stage, config=None, objective=None, *, channel="combined", seed=42,
                 epochs=25, patience=5, batch_size=128, learning_rate=3e-4,
                 weight_decay=1e-4, device="cpu"):
        if stage not in GATE_HOURS or channel not in ("curve", "head", "combined"):
            raise ValueError("Choose stage 0/1/2 and curve/head/combined channel")
        self.stage, self.channel, self.seed = stage, channel, seed
        self.config = copy.deepcopy(config) if config else core.TrainConfig()
        if not np.isfinite(self.config.horizon_hours) or self.config.horizon_hours < GATE_HOURS[stage]:
            raise ValueError("Model support must include its stage gate horizon")
        sizes = ("d_model", "n_latents", "fusion_layers", "n_heads", "n_experts", "n_bins",
                 "text_tokens", "img_tokens", "tab_num_tokens", "tab_cat_tokens", "tab_pool_heads")
        if any(not isinstance(getattr(self.config, k), (int, np.integer)) or getattr(self.config, k) < 1 for k in sizes):
            raise ValueError("Architecture sizes must be positive integers")
        if self.config.binning not in ("linear", "log1p") or self.config.tab_token_mode not in ("feature", "compact"):
            raise ValueError("Unsupported time-grid or tabular tokenization mode")
        drops = [self.config.dropout, self.config.attn_dropout, self.config.tab_pool_dropout]
        if not np.isfinite(drops).all() or min(drops) < 0 or max(drops) >= 1:
            raise ValueError("Dropout rates must be finite in [0,1)")
        if not np.isfinite(self.config.max_grad_norm) or self.config.max_grad_norm <= 0:
            raise ValueError("Gradient clipping norm must be finite and positive")
        self.objective = objective or StageObjective()
        self.objective.validate(self.config.horizon_hours)
        if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (epochs, patience, batch_size)) or not np.isfinite([learning_rate, weight_decay]).all() or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid training controls")
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.learning_rate, self.weight_decay, self.device = learning_rate, weight_decay, torch.device(device)

    @staticmethod
    def _forward(model, features):
        return model(features["numeric"], features["categorical"], features["text"], features["image"])

    def _loss_on(self, features, d, e, w):
        total, mass = 0., 0.
        for start in range(0, len(d), self.batch_size):
            sl = slice(start, start + self.batch_size)
            if float(w[sl].sum()) == 0:
                continue
            value, _ = stage_loss(self._forward(self.model_, {k: a[sl] for k, a in features.items()}),
                                  d[sl], e[sl], w[sl], self.edges_, self.stage, self.objective)
            batch_mass = float(w[sl].sum())
            total += float(value) * batch_mass
            mass += batch_mass
        return total / mass

    def fit(self, features, duration, event, *, validation=None, weight=None, cardinalities=None):
        features.validate()
        d, e, w = _outcomes(duration, event, len(features), weight)
        self.n_numeric_ = features.numeric.shape[1]
        self.cardinalities_ = list(cardinalities) if cardinalities is not None else [int(a.max()) + 1 for a in features.categorical.T]
        features.validate(self.cardinalities_, self.n_numeric_)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.model_ = core.MultiModalSurvModel(self.n_numeric_, self.cardinalities_, self.config).to(self.device)
        self.edges_ = torch.as_tensor(core.make_time_edges(self.config.horizon_hours, self.config.n_bins, self.config.binning), device=self.device)
        x = features.tensors(self.device)
        d, e, w = (torch.as_tensor(v, device=self.device, dtype=torch.float32) for v in (d, e, w))
        val = None
        if validation is not None:
            vx, vd, ve = validation[:3]
            vw = validation[3] if len(validation) == 4 else None
            vx.validate(self.cardinalities_, self.n_numeric_)
            vd, ve, vw = _outcomes(vd, ve, len(vx), vw)
            val = (vx.tensors(self.device), *(torch.as_tensor(v, device=self.device, dtype=torch.float32) for v in (vd, ve, vw)))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        generator = torch.Generator().manual_seed(self.seed)
        best, stale, best_state = float("inf"), 0, None
        self.history_, self.best_epoch_ = [], None
        for epoch in range(self.epochs):
            self.model_.train()
            total, mass = 0., 0.
            order = torch.randperm(len(d), generator=generator).to(self.device)
            for indices in order.split(self.batch_size):
                batch_mass = float(w[indices].sum())
                if batch_mass == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                outputs = self._forward(self.model_, {k: a[indices] for k, a in x.items()})
                loss, _ = stage_loss(outputs, d[indices], e[indices], w[indices], self.edges_, self.stage, self.objective)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite stage training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * batch_mass
                mass += batch_mass
            scheduler.step()
            record = {"epoch": epoch + 1, "train_objective": total / mass}
            if val is not None:
                self.model_.eval()
                with torch.no_grad():
                    value = self._loss_on(*val)
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite validation objective")
                record["validation_objective"] = value
                if value < best:
                    best, stale = value, 0
                    best_state = copy.deepcopy(self.model_.state_dict())
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
            self.history_.append(record)
            if val is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        else:
            self.best_epoch_ = len(self.history_)
        self.model_.eval()
        return self

    def predict(self, features, times=None):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the stage before scoring")
        features.validate(self.cardinalities_, self.n_numeric_)
        if times is None:
            times = [t for t in (24., 72., 168., 504.) if t <= self.config.horizon_hours]
        times = np.asarray(times, float)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any() or (times > self.config.horizon_hours).any():
            raise ValueError("Query times must lie within model support")
        query = self.edges_.new_tensor(times)
        gate = self.edges_.new_tensor([GATE_HOURS[self.stage]])
        x = features.tensors(self.device)
        survival, slow_curve, slow_head = [], [], []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                sl = slice(start, start+self.batch_size)
                hazards, mixture, logit = self._forward(self.model_, {k: a[sl] for k, a in x.items()})
                survival.append(core.mixture_surv_prob_at_times(hazards, mixture, self.edges_, query).cpu().numpy())
                slow_curve.append(core.mixture_surv_prob_at_times(hazards, mixture, self.edges_, gate)[:, 0].cpu().numpy())
                slow_head.append(torch.sigmoid(logit).cpu().numpy())
        cat = lambda pieces: np.concatenate(pieces) if pieces else np.empty(0)
        curve, head = cat(slow_curve), cat(slow_head)
        combined = .5 * (curve + head)
        selected = {"curve": curve, "head": head, "combined": combined}[self.channel]
        return {"survival": np.concatenate(survival) if survival else np.empty((0, len(times))),
                "times": times, "slow_curve": curve, "slow_head": head, "slow_combined": combined,
                "score": selected if self.stage == 0 else 1 - selected,
                "score_semantics": "tail_probability" if self.stage == 0 else "fast_probability",
                "gate_hours": GATE_HOURS[self.stage], "channel": self.channel}

    def predict_survival(self, features, times):
        return self.predict(features, times)["survival"]

    def evaluate(self, features, duration, event, *, threshold=.5):
        """Report fixed-threshold gate metrics and curve NLL without fitting."""
        d, e, _ = _outcomes(duration, event, len(features))
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Evaluation threshold must be a finite probability")
        prediction = self.predict(features)
        label_fn = stage1_core._slow_labels_and_mask if self.stage == 0 else stage1_core._fast_labels_and_mask
        labels, known = label_fn(d, e, GATE_HOURS[self.stage])
        metrics = stage1_core.cls_metrics(labels, prediction["score"], threshold, known)
        tensors = features.tensors(self.device)
        loss = 0.
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                sl = slice(start, start+self.batch_size)
                hazard, mixture, _ = self._forward(self.model_, {k: a[sl] for k, a in tensors.items()})
                duration_t = self.edges_.new_tensor(d[sl])
                event_t = self.edges_.new_tensor(e[sl]) * (duration_t <= self.edges_[-1])
                rows = core.mixture_survival_nll_discrete_vec(hazard, mixture, duration_t.clamp(max=self.edges_[-1]), event_t, self.edges_)
                loss += float(rows.sum())
        return {"survival_nll": loss / len(features), "gate": metrics,
                "known_rows": int(known.sum()), "unresolved_rows": int((~known).sum()),
                "threshold": float(threshold), "gate_hours": GATE_HOURS[self.stage],
                "channel": self.channel, "stage": self.stage}

    def save(self, path):
        """Save locally trained weights; never include private fitted bundles in release."""
        torch.save({"model": self.model_.state_dict(), "config": asdict(self.config),
                    "objective": asdict(self.objective), "stage": self.stage, "channel": self.channel,
                    "n_numeric": self.n_numeric_, "cardinalities": self.cardinalities_,
                    "history": self.history_, "best_epoch": self.best_epoch_}, Path(path))

    @classmethod
    def load(cls, path, *, device="cpu"):
        bundle = torch.load(Path(path), map_location=device, weights_only=True)
        obj = cls(bundle["stage"], core.TrainConfig(**bundle["config"]), StageObjective(**bundle["objective"]),
                  channel=bundle["channel"], device=device)
        obj.n_numeric_, obj.cardinalities_ = bundle["n_numeric"], bundle["cardinalities"]
        obj.model_ = core.MultiModalSurvModel(obj.n_numeric_, obj.cardinalities_, obj.config).to(device)
        obj.model_.load_state_dict(bundle["model"], strict=True)
        obj.edges_ = torch.as_tensor(core.make_time_edges(obj.config.horizon_hours, obj.config.n_bins, obj.config.binning), device=device)
        obj.history_, obj.best_epoch_ = bundle["history"], bundle["best_epoch"]
        obj.model_.eval()
        return obj
