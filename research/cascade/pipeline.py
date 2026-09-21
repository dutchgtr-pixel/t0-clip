"""Complete portable training and sequential routing for the legacy cascade.

Historical input construction label-capped TRAIN at 504 hours for both later
stages but prediction-gated development/holdout rows. This adapter makes that
population distinction explicit. It does not convert the three scores into an
unconditional event-time distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import joblib
import numpy as np

from .ensemble import ProbabilityEnsemble
from .portable import GATE_HOURS, StageEstimator, _outcomes
from . import stage0_ensemble_core, stage1_core
from .routing_core import _raw_bucket_codes


@dataclass(frozen=True)
class RoutingPolicy:
    tail: float
    fast168: float
    fast72: float

    def __post_init__(self):
        values = [self.tail, self.fast168, self.fast72]
        if not np.isfinite(values).all() or any(not 0 <= v <= 1 for v in values):
            raise ValueError("Routing thresholds must be finite probabilities")


def route_scores(p_tail, p_fast168, p_fast72, policy: RoutingPolicy):
    """Preserve archived gate order and inclusive score-threshold comparisons.

    NaNs are allowed only in stages a row never reaches. Missing required scores
    fail closed instead of substituting a different horizon's probability.
    """
    p0, p1, p2 = (np.asarray(p, float) for p in (p_tail, p_fast168, p_fast72))
    if p0.ndim != 1 or p1.shape != p0.shape or p2.shape != p0.shape:
        raise ValueError("Aligned one-dimensional stage probabilities are required")
    reach1 = p0 < policy.tail
    reach2 = reach1 & (p1 >= policy.fast168)
    for p, required in ((p0, np.ones(len(p0), bool)), (p1, reach1), (p2, reach2)):
        if not np.isfinite(p[required]).all() or ((p[required] < 0) | (p[required] > 1)).any():
            raise ValueError("Missing or invalid probability at a required cascade stage")
    codes = _raw_bucket_codes(p_tail=p0, p_fast168=p1, p_fast72=p2,
                              thr_tail=policy.tail, thr_fast168=policy.fast168, thr_fast72=policy.fast72)
    return {"bucket": codes, "reached_stage1": reach1, "reached_stage2": reach2,
            "selected_fast72": codes == "FAST_72H", "p_tail": p0,
            "p_fast168": p1, "p_fast72": p2,
            "probability_semantics": "separate fitted gate scores; not normalized bucket probabilities"}


def select_stage_threshold(probability, duration, event, stage, *, min_precision=0., min_bucket=1,
                           max_sacrifice=1., observed_events_only=True):
    """Explicit development-only policy selection with feasibility reporting.

    Stage0 maximizes F1; later stages maximize recall under precision/support and
    sacrifice constraints. The archived richer penalty searches are also exported
    verbatim. No infeasible policy is silently labeled feasible by this adapter.
    """
    d, e, _ = _outcomes(duration, event, len(probability))
    p = np.asarray(probability, float)
    if p.shape != d.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid policy probabilities")
    label_fn = stage1_core._slow_labels_and_mask if stage == 0 else stage1_core._fast_labels_and_mask
    labels, known = label_fn(d, e, GATE_HOURS[stage])
    if observed_events_only:
        known &= e
    if not known.any() or not 0 <= min_precision <= 1 or not 0 <= max_sacrifice <= 1 or min_bucket < 1:
        raise ValueError("Invalid or empty policy-selection population")
    best = None
    for threshold in np.unique(p[known]):
        cm = stage0_ensemble_core.cm_from_scores(p[known], labels[known], float(threshold))
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        if tp + fp < min_bucket or precision < min_precision or 1 - recall > max_sacrifice:
            continue
        value = (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.) if stage == 0 else recall
        rank = (value, precision, float(threshold))
        if best is None or rank > best[0]:
            best = (rank, {"threshold": float(threshold), "precision": precision, "recall": recall,
                           "known_rows": int(known.sum()), "unresolved_rows": int((~known).sum()),
                           "observed_events_only": observed_events_only, "feasible": True})
    if best is None:
        raise ValueError("No feasible development threshold; revise the registered policy before TEST")
    return best[1]


class CascadeEstimator:
    """Fit all three stages and score only rows that pass predecessor gates.

    Each stage may contain multiple independently fitted members followed by a
    learned probability ensemble. This controller's seeds describe actual member
    fits, unlike historical search seeds over an already trained checkpoint pool.
    """

    def __init__(self, stages=None, ensembles=None, *, training_cap_hours=504., require_observed_train=False):
        self.stages = {s: [StageEstimator(s)] for s in GATE_HOURS} if stages is None else stages
        if set(self.stages) != set(GATE_HOURS) or any(not m or any(x.stage != s for x in m) for s, m in self.stages.items()):
            raise ValueError("Supply nonempty correctly numbered members for stages 0,1,2")
        self.ensembles = {s: ProbabilityEnsemble() for s in GATE_HOURS} if ensembles is None else ensembles
        if set(self.ensembles) != set(GATE_HOURS):
            raise ValueError("Supply one ensemble rule for each stage")
        if not np.isfinite(training_cap_hours) or training_cap_hours <= 0:
            raise ValueError("Training label cap must be positive and finite")
        self.training_cap_hours, self.require_observed_train = training_cap_hours, require_observed_train

    def fit(self, train, duration, event, *, validation, policy=None, threshold_options=None, cardinalities=None, weight=None):
        duration, event, weight = _outcomes(duration, event, len(train), weight)
        vx, vd, ve = validation
        vd, ve, _ = _outcomes(vd, ve, len(vx))
        train.validate()
        vx.validate()
        cards = cardinalities if cardinalities is not None else [int(a.max())+1 for a in train.categorical.T]
        later_train = duration <= self.training_cap_hours
        if self.require_observed_train:
            later_train &= event
        if not later_train.any():
            raise ValueError("No downstream TRAIN rows satisfy the declared label cap")
        active = np.ones(len(vx), bool)
        thresholds, self.fit_audit_ = {}, {}
        for stage in (0, 1, 2):
            train_mask = np.ones(len(train), bool) if stage == 0 else later_train
            if not active.any():
                raise ValueError(f"No development rows reach stage {stage}; downstream fitting cannot be validated")
            sx, sd, se = vx.take(active), vd[active], ve[active]
            for model in self.stages[stage]:
                model.fit(train.take(train_mask), duration[train_mask], event[train_mask],
                          validation=(sx, sd, se), cardinalities=cards, weight=weight[train_mask])
            matrix = np.vstack([m.predict(sx)["score"] for m in self.stages[stage]])
            self.ensembles[stage].fit(matrix, sd, se, horizon=GATE_HOURS[stage], positive="slow" if stage == 0 else "fast")
            probability = self.ensembles[stage].predict(matrix)
            if policy is None:
                selected = select_stage_threshold(probability, sd, se, stage, **(threshold_options or {}).get(stage, {}))
                threshold = selected["threshold"]
            else:
                threshold = (policy.tail, policy.fast168, policy.fast72)[stage]
                selected = {"threshold": threshold, "selection": "caller_prespecified"}
            thresholds[stage] = threshold
            self.fit_audit_[stage] = {"train_rows": int(train_mask.sum()), "train_events": int(event[train_mask].sum()),
                                      "development_rows": int(active.sum()), "development_events": int(se.sum()),
                                      "neural_member_fits": len(self.stages[stage]), "policy": selected}
            if stage < 2:
                indices = np.flatnonzero(active)
                keep = probability < threshold if stage == 0 else probability >= threshold
                active[indices[~keep]] = False
        self.policy_ = RoutingPolicy(thresholds[0], thresholds[1], thresholds[2])
        return self

    def predict(self, features):
        if not hasattr(self, "policy_"):
            raise RuntimeError("Fit the complete cascade before scoring")
        active = np.ones(len(features), bool)
        scores = [np.full(len(features), np.nan) for _ in GATE_HOURS]
        for stage in (0, 1, 2):
            if not active.any():
                break
            x = features.take(active)
            matrix = np.vstack([m.predict(x)["score"] for m in self.stages[stage]])
            p = self.ensembles[stage].predict(matrix)
            scores[stage][active] = p
            if stage < 2:
                indices = np.flatnonzero(active)
                keep = p < self.policy_.tail if stage == 0 else p >= self.policy_.fast168
                active[indices[~keep]] = False
        return route_scores(*scores, self.policy_)

    def save(self, directory):
        """Write a locally fitted bundle; release examples never include weights."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Choose an empty output directory")
        directory.mkdir(parents=True, exist_ok=True)
        members = {}
        for stage, models in self.stages.items():
            members[stage] = len(models)
            for index, model in enumerate(models):
                model.save(directory / f"stage_{stage}_member_{index}.pt")
        joblib.dump(self.ensembles, directory / "ensembles.joblib")
        payload = {"schema_version": 1, "members": members, "policy": asdict(self.policy_),
                   "fit_audit": self.fit_audit_, "training_cap_hours": self.training_cap_hours,
                   "require_observed_train": self.require_observed_train,
                   "preprocessing": "Caller must retain its fitted feature encoder separately"}
        (directory / "manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, directory, *, trusted=False, device="cpu"):
        """Load a caller-created trusted bundle; joblib contains fitted estimators."""
        if not trusted:
            raise ValueError("Only load your own trusted locally fitted bundle; set trusted=True explicitly")
        directory = Path(directory)
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or set(payload["members"]) != {"0", "1", "2"}:
            raise ValueError("Invalid cascade bundle manifest")
        stages = {stage: [StageEstimator.load(directory / f"stage_{stage}_member_{index}.pt", device=device)
                           for index in range(payload["members"][str(stage)])] for stage in (0, 1, 2)}
        obj = cls(stages, joblib.load(directory / "ensembles.joblib"),
                  training_cap_hours=payload["training_cap_hours"], require_observed_train=payload["require_observed_train"])
        obj.policy_ = RoutingPolicy(**payload["policy"])
        obj.fit_audit_ = {int(k): v for k, v in payload["fit_audit"].items()}
        return obj
