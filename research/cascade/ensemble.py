"""Fit calibration and ensemble rules on explicitly supplied development rows.

Prediction never refits a calibrator, covariance matrix, stacker or threshold.
Exact archived combination functions live in the sibling *_ensemble_core files.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from . import stage0_ensemble_core as archived
from . import stage1_core


def probability_matrix(probability):
    p = np.asarray(probability, float)
    if p.ndim != 2 or p.shape[0] < 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected finite probabilities shaped (members, rows)")
    return p


def _labels(probability, duration, event, horizon, positive):
    p = probability_matrix(probability)
    d, e = np.asarray(duration, float), np.asarray(event)
    if d.shape != (p.shape[1],) or e.shape != d.shape or not len(d):
        raise ValueError("Nonempty aligned outcomes are required")
    if not np.isfinite(d).all() or (d < 0).any() or not np.isin(e, [0, 1]).all() or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Invalid ensemble outcomes/horizon")
    if positive not in ("fast", "slow"):
        raise ValueError("Positive class must be fast or slow")
    label_fn = stage1_core._fast_labels_and_mask if positive == "fast" else stage1_core._slow_labels_and_mask
    labels, known = label_fn(d, e, horizon)
    return p, labels, known


class ProbabilityEnsemble:
    """Reusable probability or logit ensemble, including the historical HGB stack.

    `fit_mask` must identify only development/calibration rows. Any learned
    weighting is fitted on that subset, including covariance-based weighting.
    Stacking/calibration results on those same rows are fitting diagnostics.
    """

    METHODS = {"mean_prob", "mean_logit", "logit_lcb", "mv_logit", "lda_logit", "wlogit", "stack_hgb"}

    def __init__(self, method="mean_logit", *, calibration="none", trim_ratio=0., kappa=0.,
                 regularization=1e-4, logistic_c=1., seed=42):
        if method not in self.METHODS or calibration not in ("none", "temperature", "isotonic"):
            raise ValueError("Unknown ensemble/calibration method")
        if not np.isfinite([trim_ratio, kappa, regularization, logistic_c]).all() or not 0 <= trim_ratio < .5 or kappa < 0 or regularization < 0 or logistic_c <= 0:
            raise ValueError("Invalid ensemble controls")
        self.method, self.calibration = method, calibration
        self.trim_ratio, self.kappa = trim_ratio, kappa
        self.regularization, self.logistic_c, self.seed = regularization, logistic_c, seed

    def fit(self, probability, duration, event, *, horizon, positive="fast", fit_mask=None):
        p, labels, known = _labels(probability, duration, event, horizon, positive)
        if fit_mask is not None:
            mask = np.asarray(fit_mask)
            if mask.shape != known.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError("Invalid development mask")
            known &= mask.astype(bool)
        if not known.any():
            raise ValueError("No known development labels for ensemble fitting")
        if (self.calibration != "none" or self.method in ("lda_logit", "wlogit", "stack_hgb")) and len(np.unique(labels[known])) != 2:
            raise ValueError("Learned calibration/stacking requires both development classes")
        self.n_members_, self.fit_rows_ = p.shape[0], int(known.sum())
        self.calibrators_ = []
        for member in p:
            if self.calibration == "temperature":
                self.calibrators_.append(archived.fit_temperature_scaler(member[known], labels[known], .5, 5., 50))
            elif self.calibration == "isotonic":
                self.calibrators_.append(IsotonicRegression(out_of_bounds="clip").fit(member[known], labels[known]))
            else:
                self.calibrators_.append(None)
        calibrated = self._calibrate(p)[:, known]
        self.weights_, self.intercept_, self.stacker_ = None, 0., None
        if self.method == "stack_hgb":
            self.stacker_ = HistGradientBoostingClassifier(learning_rate=.05, max_iter=200,
                                                          max_depth=None, max_leaf_nodes=None,
                                                          min_samples_leaf=20, l2_regularization=0.,
                                                          random_state=self.seed)
            self.stacker_.fit(archived._logit(calibrated).T, labels[known])
        elif self.method in ("mv_logit", "lda_logit", "wlogit"):
            _, self.weights_, extra = archived.ensemble_p_tail(
                calibrated, self.method, y_tail_inner=labels[known],
                inner_mask=np.ones(self.fit_rows_, bool), mv_lambda=self.regularization,
                lda_lambda=self.regularization, wlogit_C=self.logistic_c)
            self.intercept_ = extra.get("intercept", 0.)
            # The original singleton shortcut returns no learned weights.
            if self.weights_ is None:
                self.weights_ = np.ones(1)
        self.positive_, self.horizon_ = positive, float(horizon)
        return self

    def _calibrate(self, p):
        rows = []
        for member, calibrator in zip(p, self.calibrators_):
            if self.calibration == "temperature":
                rows.append(archived.apply_temperature(member, calibrator))
            elif self.calibration == "isotonic":
                rows.append(calibrator.predict(member) if len(member) else member)
            else:
                rows.append(member)
        return np.asarray(rows)

    def predict(self, probability):
        if not hasattr(self, "n_members_"):
            raise RuntimeError("Fit the ensemble before prediction")
        p = probability_matrix(probability)
        if p.shape[0] != self.n_members_:
            raise ValueError("Member ordering/count must match the fitted ensemble")
        if p.shape[1] == 0:
            return np.empty(0)
        p = self._calibrate(p)
        if self.stacker_ is not None:
            result = self.stacker_.predict_proba(archived._logit(p).T)[:, 1]
        elif self.weights_ is not None:
            result = expit(archived._logit(p).T @ self.weights_ + self.intercept_)
        else:
            result, _, _ = archived.ensemble_p_tail(p, self.method, trim_ratio=self.trim_ratio, kappa=self.kappa)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite ensemble prediction")
        return np.clip(result, 0, 1)


def choose_ensemble(candidates, inner_probability, inner_duration, inner_event,
                    outer_probability, outer_duration, outer_event, *, horizon, positive="fast"):
    """Fit candidates on inner SVAL and select by outer known-label log loss.

    This finite candidate controller is new. It is not the archived Optuna search
    or evidence that its selected candidate generalizes to a final test cohort.
    """
    if not candidates:
        raise ValueError("At least one registered candidate is required")
    outer_probability, labels, known = _labels(outer_probability, outer_duration, outer_event, horizon, positive)
    if not known.any():
        raise ValueError("Outer selection cohort has no known outcomes")
    fitted, records = [], []
    for index, options in enumerate(candidates):
        model = ProbabilityEnsemble(**options).fit(inner_probability, inner_duration, inner_event,
                                                   horizon=horizon, positive=positive)
        p = model.predict(outer_probability)
        loss = archived._nll_binary(labels[known], p[known])
        fitted.append(model)
        records.append({"candidate": index, "outer_binary_nll": float(loss), "options": dict(options)})
    selected = min(range(len(records)), key=lambda i: (records[i]["outer_binary_nll"], i))
    return fitted[selected], {"selected_candidate": selected, "candidates": records,
                              "selection_scope": "outer development only"}
