"""Censor-aware metrics and validation-only operating-point selection.

Marginal IPCW uses the TRAIN censoring distribution. Its transfer to later
cohorts assumes independent censoring and stability of that distribution;
temporal drift or feature-dependent censoring requires a different estimator.
"""
from __future__ import annotations

import numpy as np
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


def survival_target(duration, event):
    duration, event = np.asarray(duration, float), np.asarray(event)
    if duration.ndim != 1 or event.shape != duration.shape or not len(duration):
        raise ValueError("Survival outcomes must be nonempty aligned vectors")
    if not np.isfinite(duration).all() or np.any(duration <= 0) or not np.isin(event, [0, 1]).all():
        raise ValueError("Survival outcomes need positive finite times and binary events")
    return Surv.from_arrays(event.astype(bool), duration)


def check_predictions(predictions, n, times):
    times = np.asarray(times, float)
    predictions = np.asarray(predictions, float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Evaluation times must be at least two increasing positive finite values")
    if predictions.shape != (n, len(times)) or not np.isfinite(predictions).all():
        raise ValueError("Invalid survival prediction shape or nonfinite predictions")
    if (predictions < -1e-6).any() or (predictions > 1 + 1e-6).any() or (np.diff(predictions, axis=1) > 1e-6).any():
        raise ValueError("Survival predictions must be probabilities nonincreasing in time")
    return np.clip(predictions, 0, 1), times


class SurvivalMetrics:
    def __init__(self, train_duration, train_event, times, min_censor_survival=0.05):
        self.train = survival_target(train_duration, train_event)
        self.times = np.asarray(times, float)
        check_predictions(np.ones((1, len(self.times))), 1, self.times)
        if not 0 < min_censor_survival <= 1:
            raise ValueError("min_censor_survival must be in (0,1]")
        if self.times[-1] >= max(train_duration):
            raise ValueError("Evaluation grid exceeds TRAIN follow-up support")
        self.censor = CensoringDistributionEstimator().fit(self.train)
        self.g_times = self.censor.predict_proba(self.times)
        self.min_censor_survival = min_censor_survival
        if np.min(self.g_times) < min_censor_survival:
            raise ValueError("Insufficient TRAIN censoring support on the prespecified evaluation grid")

    def brier_contributions(self, duration, event, prediction):
        target = survival_target(duration, event)
        duration, event = target["time"], target["event"]
        prediction, _ = check_predictions(prediction, len(duration), self.times)
        if max(duration) <= self.times[-1]:
            raise ValueError("Evaluation cohort lacks follow-up beyond the fixed grid")
        cases = event[:, None] & (duration[:, None] <= self.times[None, :])
        controls = duration[:, None] > self.times[None, :]
        # Only observed events at/before the grid need G(Y); later follow-up
        # never requires extrapolating the censoring distribution.
        relevant = event & (duration <= self.times[-1])
        g_event = np.ones(len(duration))
        if relevant.any():
            g_event[relevant] = self.censor.predict_proba(duration[relevant])
        if (g_event[relevant] < self.min_censor_survival).any():
            raise ValueError("Insufficient censoring support at observed event times")
        return prediction**2 * cases / g_event[:, None] + (1-prediction)**2 * controls / self.g_times[None, :]

    def per_row_ibs(self, duration, event, prediction):
        brier = self.brier_contributions(duration, event, prediction)
        return np.trapezoid(brier, self.times, axis=1) / (self.times[-1] - self.times[0])

    def evaluate(self, duration, event, prediction):
        target = survival_target(duration, event)
        prediction, _ = check_predictions(prediction, len(duration), self.times)
        contributions = self.brier_contributions(duration, event, prediction)
        brier = contributions.mean(axis=0)
        result = {
            "integrated_brier_score": float(np.trapezoid(brier, self.times) / (self.times[-1]-self.times[0])),
            "brier_by_time": brier.tolist(), "time_grid_hours": self.times.tolist(),
            "censor_survival_at_grid_end": float(self.g_times[-1]),
        }
        try:
            cindex = concordance_index_ipcw(self.train, target, 1-prediction[:, -1], tau=self.times[-1])
            result["ipcw_concordance"] = float(cindex[0])
            if not np.isfinite(result["ipcw_concordance"]):
                result["ipcw_concordance"] = None
                result["concordance_unavailable_reason"] = "No positive-weight comparable pairs"
        except ValueError as error:
            result["ipcw_concordance"] = None
            result["concordance_unavailable_reason"] = str(error)
        return result


def horizon_labels(duration, event, horizon):
    target = survival_target(duration, event)
    duration, event = target["time"], target["event"]
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Horizon must be positive and finite")
    positive = event & (duration <= horizon)
    # A censor exactly at the horizon is conservatively considered unresolved.
    eligible = positive | (duration > horizon)
    return positive, eligible


def operating_metrics(duration, event, probability, horizon, threshold):
    positive, eligible = horizon_labels(duration, event, horizon)
    probability = np.asarray(probability, float)
    if probability.shape != positive.shape or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid horizon probabilities")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Threshold must be None or a finite probability")
    flagged = np.zeros(len(probability), dtype=bool) if threshold is None else probability >= threshold
    tp = int((eligible & flagged & positive).sum())
    fp = int((eligible & flagged & ~positive).sum())
    fn = int((eligible & ~flagged & positive).sum())
    tn = int((eligible & ~flagged & ~positive).sum())
    return {
        "eligible": int(eligible.sum()), "unresolved_censored": int((~eligible).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "flagged_all_rows": int(flagged.sum()), "flagged_unresolved": int((flagged & ~eligible).sum()),
        "threshold": threshold,
    }


def select_threshold(duration, event, probability, horizon, target_precision=0.8, min_flagged=10):
    """Maximize eligible SVAL recall subject to empirical precision/support.

    This is a validation operating point, not a guarantee of future precision.
    A missing feasible threshold explicitly selects a reject-all policy.
    """
    if not 0 < target_precision <= 1 or not isinstance(min_flagged, int) or min_flagged < 1:
        raise ValueError("Invalid operating-point constraints")
    operating_metrics(duration, event, probability, horizon, None)
    best = None
    for threshold in np.unique(probability):
        metrics = operating_metrics(duration, event, probability, horizon, float(threshold))
        if metrics["tp"]+metrics["fp"] < min_flagged or metrics["precision"] < target_precision:
            continue
        if best is None or (metrics["recall"], metrics["precision"], threshold) > (best["recall"], best["precision"], best["threshold"]):
            best = metrics
    if best is None:
        best = operating_metrics(duration, event, probability, horizon, None)
        best["selection_status"] = "no_feasible_validation_threshold_reject_all"
    else:
        best["selection_status"] = "selected_on_validation"
    best.update(target_precision=target_precision, min_flagged=min_flagged)
    return best


def paired_ibs_intervals(row_losses: dict[str, np.ndarray], *, reference: str, repetitions=1000, seed=2026):
    """Paired entity bootstrap conditional on fitted models and censor weights.

    Positive differences mean worse IBS than reference. These intervals do not
    include training/tuning uncertainty and do not correct multiple comparisons.
    """
    if not isinstance(repetitions, int) or repetitions < 1 or reference not in row_losses:
        raise ValueError("Bootstrap requires a valid reference and positive repetitions")
    row_losses = {key: np.asarray(value, dtype=float) for key, value in row_losses.items()}
    if any(value.ndim != 1 or not np.isfinite(value).all() for value in row_losses.values()):
        raise ValueError("Bootstrap losses must be finite one-dimensional vectors")
    rng = np.random.default_rng(seed)
    ref = row_losses[reference]
    n = len(ref)
    if n < 2 or any(len(x) != n for x in row_losses.values()):
        raise ValueError("Paired bootstrap requires matching entity rows")
    estimates = {key: [] for key in row_losses if key != reference}
    for _ in range(repetitions):
        indices = rng.integers(0, n, n)
        for name in estimates:
            estimates[name].append(float((row_losses[name][indices]-ref[indices]).mean()))
    return {
        name: {"delta_ibs_vs_reference": float((row_losses[name]-ref).mean()),
               "ci95": np.quantile(values, [.025, .975]).tolist()}
        for name, values in estimates.items()
    }
