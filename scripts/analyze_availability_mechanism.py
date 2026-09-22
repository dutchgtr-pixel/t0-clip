"""Audit a coverage cutoff under outcome-time cohort selection.

Outputs are aggregate-only. The default experiment is a deterministic constructed
counterexample, not an estimate of any historical model's performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _array(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def confusion(y, predicted) -> dict:
    y, predicted = np.asarray(y, dtype=bool), np.asarray(predicted, dtype=bool)
    if y.shape != predicted.shape:
        raise ValueError("Target and prediction shapes differ")
    tp = int(np.sum(y & predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    tn = int(np.sum(~y & ~predicted))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "balanced_accuracy": ((tp / (tp + fn) + tn / (tn + fp)) / 2
                              if tp + fn and tn + fp else None),
    }


def audit_cutoff(origin, event, duration, missing, *, cutoff: float,
                 horizon: float, tolerance: float = 1e-6) -> dict:
    """Audit observed-event records; clocks/durations share one explicit unit.

    Missingness must be defined without consulting the target. This descriptive
    audit takes the cutoff as an input; it does not optimize it on outcomes.
    """
    arrays = [_array(x, n) for x, n in zip(
        (origin, event, duration, missing), ("origin", "event", "duration", "missing"))]
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Input lengths differ")
    origin, event, duration, missing = arrays
    if not np.isin(missing, (0, 1)).all():
        raise ValueError("Missingness must be binary")
    if not np.isfinite([cutoff, horizon, tolerance]).all() or horizon < 0 or tolerance < 0:
        raise ValueError("Cutoff must be finite; horizon and tolerance must be nonnegative")
    if np.any(duration < 0) or np.any(event < origin - tolerance):
        raise ValueError("Negative durations are incompatible with this mechanism")
    residual = np.abs(event - origin - duration)
    if np.any(residual > tolerance):
        raise ValueError("Stored duration does not equal event minus origin")
    y, missing = duration <= horizon, missing.astype(bool)
    expected_missing = origin >= cutoff
    mismatches = int(np.sum(missing != expected_missing))
    a, b = float(event.min()), float(event.max())
    # With selection a <= E <= b, O >= c iff D <= E-c. Furthermore
    # c <= a-h implies D <= h => O >= c, and D > b-c => O < c.
    long_zone = duration > b - cutoff + tolerance
    forced_fast_zone = origin >= b - horizon
    available = ~missing
    return {
        "rows": len(origin), "positive_rows": int(y.sum()),
        "missing_rows": int(missing.sum()),
        "cutoff_mismatch_rows": mismatches,
        "cutoff_agreement": float(np.mean(missing == expected_missing)),
        "duration_identity_max_abs_error": float(residual.max()),
        "event_lower_minus_cutoff": a - cutoff,
        "event_upper_minus_cutoff": b - cutoff,
        "horizon": horizon,
        "geometry_all_fast_origins_after_cutoff": bool(cutoff <= a - horizon),
        "sufficient_condition_all_fast_missing": bool(mismatches == 0 and cutoff <= a - horizon),
        "fast_but_covered_rows": int(np.sum(y & available)),
        "beyond_window_bound_rows": int(long_zone.sum()),
        "beyond_window_bound_but_missing_rows": int(np.sum(long_zone & missing)),
        "forced_fast_zone_rows": int(forced_fast_zone.sum()),
        "forced_fast_zone_nonfast_rows": int(np.sum(forced_fast_zone & ~y)),
        "positive_rate_when_missing": float(y[missing].mean()) if missing.any() else None,
        "positive_rate_when_covered": float(y[available].mean()) if available.any() else None,
        "fixed_missingness_rule": confusion(y, missing),
        "all_positive_reference": confusion(y, np.ones(len(y), dtype=bool)),
        "interpretation": "Descriptive mechanism audit; association alone does not prove future access or causal model dependence.",
    }


def constructed_experiment() -> dict:
    """Exact finite population with independent origin and duration, no sampling."""
    origin_days, duration_days = np.meshgrid(np.arange(121), np.arange(1, 31))
    origin, duration = origin_days.ravel() * 24.0, duration_days.ravel() * 24.0
    event = origin + duration
    cutoff, horizon = 100 * 24.0, 72.0
    missing = origin >= cutoff
    outcome_selected = (event >= 104 * 24) & (event <= 107 * 24)
    origin_selected = (origin >= 96 * 24) & (origin <= 107 * 24)
    scenarios = {}
    for name, selected, observed_missing in (
        ("outcome_time_selected_frozen", outcome_selected, missing),
        ("origin_time_selected_fully_matured", origin_selected, missing),
        ("same_outcome_rows_after_coverage_refresh", outcome_selected,
         np.zeros(len(origin), dtype=bool)),
    ):
        scenarios[name] = audit_cutoff(
            origin[selected], event[selected], duration[selected], observed_missing[selected],
            cutoff=cutoff, horizon=horizon)
    return {
        "evidence_class": "new constructed mechanism experiment; not historical data or model validation",
        "design": {
            "population_rows": len(origin), "origin_days": [0, 120],
            "duration_days": [1, 30], "horizon_days": 3, "coverage_cutoff_day": 100,
            "outcome_selection_days_inclusive": [104, 107],
            "origin_selection_days_inclusive": [96, 107],
            "duration_and_origin_independent_by_construction": True,
            "followup_complete_for_all_rows": True,
            "rule_fitted_or_tuned": False,
            "feature_values": "constant historical value; only its calendar-regime availability varies",
            "availability_mechanism": "Known-at-origin mask M=1[origin>=100 days]; no outcome is used to assign M",
        },
        "scenarios": scenarios,
    }


def audit_file(args) -> dict:
    import pandas as pd

    required = [args.origin_column, args.event_column, args.duration_column,
                args.observed_column, args.count_column, args.share_column]
    path = Path(args.input)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=required)
    else:
        frame = pd.read_csv(path, usecols=required)
    observed = frame[args.observed_column]
    if observed.isna().any() or not observed.isin([True, 1]).all():
        raise ValueError("This audit requires observed events; censored/unknown events need a separate design")
    def utc_times(series):
        for item in series:
            parsed = pd.Timestamp(item)
            if pd.isna(parsed):
                raise ValueError("Missing timestamps are not allowed")
            if parsed.tzinfo is None:
                raise ValueError("Every timestamp must include an explicit timezone")
        return pd.to_datetime(series, utc=True, errors="raise", format="mixed")

    origin = utc_times(frame[args.origin_column])
    event = utc_times(frame[args.event_column])
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff must include an explicit timezone")
    count = pd.to_numeric(frame[args.count_column], errors="raise")
    share = pd.to_numeric(frame[args.share_column], errors="raise")
    duration = pd.to_numeric(frame[args.duration_column], errors="raise")
    result = audit_cutoff(
        (origin - cutoff).dt.total_seconds().to_numpy() / 3600,
        (event - cutoff).dt.total_seconds().to_numpy() / 3600,
        duration.to_numpy(), ((count == 0) & share.isna()).to_numpy(),
        cutoff=0, horizon=args.horizon_hours)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "evidence_class": "aggregate recomputation from supplied event records",
        "source_sha256": digest.hexdigest(), "time_unit": "hours",
        "cutoff_utc": cutoff.tz_convert("UTC").isoformat(),
        "missing_rule": "count equals zero and share is null", "audit": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Optional local CSV or Parquet; never copied to output")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--origin-column", default="origin_at")
    parser.add_argument("--event-column", default="event_at")
    parser.add_argument("--duration-column", default="duration_hours")
    parser.add_argument("--observed-column", default="event_observed")
    parser.add_argument("--count-column", default="context_count")
    parser.add_argument("--share-column", default="context_share")
    parser.add_argument("--cutoff", help="Explicit timezone required for historical input")
    parser.add_argument("--horizon-hours", type=float, default=72)
    args = parser.parse_args()
    if args.input and not args.cutoff:
        parser.error("--cutoff is required with --input")
    result = audit_file(args) if args.input else constructed_experiment()
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
