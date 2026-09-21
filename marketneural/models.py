"""Portable survival estimators sharing one multimodal feature contract.

All transforms must be fitted by the caller on training data only. Classical
models and the MLP flatten slots (including availability indicators), preserving
the same supplied information as the attention reference.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

Features = Mapping[str, np.ndarray]
GROUPS = ("tabular", "text", "image", "report")


def validate_features(x: Features) -> dict[str, np.ndarray]:
    if "tabular" not in x:
        raise ValueError("Features require a two-dimensional 'tabular' array")
    unknown = set(x) - set(GROUPS) - {"image_mask", "report_mask"}
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    out = {}
    n = None
    for key in GROUPS:
        if key not in x:
            continue
        a = np.asarray(x[key], dtype=np.float32)
        ndim = 3 if key in ("image", "report") else 2
        if a.ndim != ndim or any(size == 0 for size in a.shape[1:]):
            raise ValueError(f"{key} must have {ndim} dimensions with nonempty features")
        if n is None:
            n = len(a)
        if len(a) != n or not np.isfinite(a).all():
            raise ValueError(f"{key} must have matching rows and finite values")
        out[key] = a
        if ndim == 3:
            mask_key = key + "_mask"
            mask = np.asarray(x.get(mask_key, np.ones(a.shape[:2], dtype=bool)))
            if mask.shape != a.shape[:2] or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"{mask_key} must be a binary array matching slots; True=present")
            out[mask_key] = mask.astype(bool)
    for key in ("image", "report"):
        if key + "_mask" in x and key not in x:
            raise ValueError(f"{key}_mask supplied without {key}")
    return out


def feature_signature(x: Features) -> dict[str, tuple[int, ...]]:
    return {key: tuple(a.shape[1:]) for key, a in x.items()}


def flatten_features(x: Features) -> np.ndarray:
    """Flatten supplied modalities in fixed order; missing slots become zero."""
    x = validate_features(x)
    parts = []
    for key in GROUPS:
        if key not in x:
            continue
        a = x[key]
        if a.ndim == 3:
            mask = x[key + "_mask"]
            parts.extend([(a * mask[..., None]).reshape(len(a), int(np.prod(a.shape[1:]))),
                          mask.astype(np.float32)])
        else:
            parts.append(a)
    return np.concatenate(parts, axis=1)


def validate_outcomes(duration, event, n: int) -> tuple[np.ndarray, np.ndarray]:
    duration = np.asarray(duration, dtype=np.float64)
    event = np.asarray(event)
    if duration.shape != (n,) or event.shape != (n,) or n == 0:
        raise ValueError("Nonempty one-dimensional duration/event arrays must match feature rows")
    if not np.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("Durations must be finite and nonnegative")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("Event indicators must be binary")
    return duration, event.astype(bool)


def validate_times(times) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Prediction times must be a one-dimensional finite nonnegative array")
    return times


def _step_values(knots, values, times):
    indices = np.searchsorted(knots, times, side="right") - 1
    return np.where(indices >= 0, values[np.maximum(indices, 0)], 1.0)


class KaplanMeierModel:
    def fit(self, x, duration, event, *, validation=None):
        x = validate_features(x)
        duration, event = validate_outcomes(duration, event, len(x["tabular"]))
        self.signature_ = feature_signature(x)
        self.times_, inverse, counts = np.unique(duration, return_inverse=True, return_counts=True)
        deaths = np.bincount(inverse, weights=event, minlength=len(self.times_))
        risk = len(duration) - np.r_[0, np.cumsum(counts)[:-1]]
        self.survival_ = np.cumprod(1.0 - deaths / risk)
        return self

    def predict_survival(self, x, times):
        if not hasattr(self, "survival_"):
            raise RuntimeError("Fit the model before prediction")
        x = validate_features(x)
        if feature_signature(x) != self.signature_:
            raise ValueError("Prediction features differ from the fitted feature contract")
        s = _step_values(self.times_, self.survival_, validate_times(times))
        return np.broadcast_to(s, (len(x["tabular"]), len(s))).copy()


class SkSurvivalModel:
    def __init__(self, name: str, random_state: int = 42, **kwargs):
        # Optional dependencies are imported only for the selected estimator.
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        constructors = {
            "coxph": (CoxPHSurvivalAnalysis, {"alpha": 1.0, "n_iter": 100}),
            "rsf": (RandomSurvivalForest, {"n_estimators": 100, "min_samples_leaf": 5,
                    "max_depth": 8, "n_jobs": 1, "random_state": random_state}),
            "gbsa": (GradientBoostingSurvivalAnalysis, {"n_estimators": 100,
                     "learning_rate": 0.05, "max_depth": 2, "random_state": random_state}),
        }
        cls, defaults = constructors[name]
        self.estimator = cls(**(defaults | kwargs))

    def fit(self, x, duration, event, *, validation=None):
        from sksurv.util import Surv

        x = validate_features(x)
        duration, event = validate_outcomes(duration, event, len(x["tabular"]))
        if not event.any():
            raise ValueError("This estimator requires at least one observed event")
        self.signature_ = feature_signature(x)
        self.estimator.fit(flatten_features(x), Surv.from_arrays(event, duration))
        return self

    def predict_survival(self, x, times):
        if not hasattr(self, "signature_"):
            raise RuntimeError("Fit the model before prediction")
        x, times = validate_features(x), validate_times(times)
        if feature_signature(x) != self.signature_:
            raise ValueError("Prediction features differ from the fitted feature contract")
        if not len(x["tabular"]):
            return np.empty((0, len(times)))
        funcs = self.estimator.predict_survival_function(flatten_features(x))
        return np.vstack([_step_values(f.x, f.y, times) for f in funcs])


def create_model(name: str, random_state: int = 42, **kwargs):
    """Construct km, coxph, rsf, gbsa, mlp or perceiver_moe."""
    if name == "km":
        if kwargs:
            raise TypeError(f"Kaplan-Meier accepts no options: {sorted(kwargs)}")
        return KaplanMeierModel()
    if name in ("coxph", "rsf", "gbsa"):
        return SkSurvivalModel(name, random_state, **kwargs)
    if name in ("mlp", "perceiver_moe"):
        from .neural import NeuralSurvivalModel

        return NeuralSurvivalModel(architecture=name, random_state=random_state, **kwargs)
    raise ValueError(f"Unknown model: {name}")
