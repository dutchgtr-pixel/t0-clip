"""Reproduce an aggregate-only, conditional cohort-deletion sensitivity bound.

This script never loads observation-level data or fits a model. It infers an
integer confusion matrix from rounded published metrics, then exhaustively
deletes a specified number of records while holding labels and predictions
fixed. The result is neither a paired score nor a confidence bound.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "research/thesis_evidence/historical_neural_comparison.json"


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    def f1(self) -> Fraction:
        denominator = 2 * self.tp + self.fp + self.fn
        return Fraction(2 * self.tp, denominator) if denominator else Fraction(0)


def integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def rounded_ratio(numerator: int, denominator: int, digits: int) -> Decimal:
    """Use explicit decimal half-even rounding, with zero_division=0."""
    with localcontext() as context:
        context.prec = 40
        value = Decimal(numerator) / Decimal(denominator) if denominator else Decimal(0)
        return value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)


def infer_confusion(row: dict, digits: int = 4) -> Confusion:
    n = integer(row["cohort_n"], "cohort_n")
    positives = integer(row["positives"], "positives")
    if not 0 < positives <= n:
        raise ValueError("The inference requires 0 < positives <= cohort_n")
    targets = {key: Decimal(str(row[key])) for key in ("precision", "recall", "f1")}
    if any(not value.is_finite() or not 0 <= value <= 1 for value in targets.values()):
        raise ValueError("Published metrics must be finite probabilities")
    if any(value != value.quantize(Decimal(1).scaleb(-digits)) for value in targets.values()):
        raise ValueError("Published metrics must have at most the specified decimal places")
    compatible = []
    for tp in range(positives + 1):
        if rounded_ratio(tp, positives, digits) != targets["recall"]:
            continue
        for fp in range(n - positives + 1):
            if rounded_ratio(tp, tp + fp, digits) != targets["precision"]:
                continue
            fn = positives - tp
            if rounded_ratio(2 * tp, 2 * tp + fp + fn, digits) == targets["f1"]:
                compatible.append(Confusion(tp, fp, fn, n - positives - fp))
    if len(compatible) != 1:
        raise ValueError(f"Expected exactly one compatible confusion matrix; found {len(compatible)}")
    return compatible[0]


def exact_value(value: Fraction) -> dict:
    return {"fraction": str(value), "numerator": value.numerator,
            "denominator": value.denominator, "decimal": float(value)}


def deletion_bounds(confusion: Confusion, deleted: int) -> dict:
    deleted = integer(deleted, "deleted")
    n = sum(asdict(confusion).values())
    if not 0 <= deleted < n:
        raise ValueError("Deletion count must leave at least one record")
    errors = confusion.fp + confusion.fn
    count = 0
    minimum = maximum = None
    for remove_tp in range(min(deleted, confusion.tp) + 1):
        for remove_fp in range(min(deleted - remove_tp, confusion.fp) + 1):
            for remove_fn in range(min(deleted - remove_tp - remove_fp, confusion.fn) + 1):
                remove_tn = deleted - remove_tp - remove_fp - remove_fn
                if remove_tn > confusion.tn:
                    continue
                removed = Confusion(remove_tp, remove_fp, remove_fn, remove_tn)
                remaining = Confusion(confusion.tp - remove_tp, confusion.fp - remove_fp,
                                      confusion.fn - remove_fn, confusion.tn - remove_tn)
                score = remaining.f1()
                item = (score, removed, remaining)
                if minimum is None or score < minimum[0]:
                    minimum = item
                if maximum is None or score > maximum[0]:
                    maximum = item
                count += 1
    assert minimum is not None and maximum is not None
    minimum_tp = max(0, confusion.tp - deleted)
    minimum_denominator = 2 * minimum_tp + errors
    analytic_min = Fraction(2 * minimum_tp, minimum_denominator) if minimum_denominator else Fraction(0)
    maximum_errors = max(0, errors - deleted)
    maximum_denominator = 2 * confusion.tp + maximum_errors
    analytic_max = Fraction(2 * confusion.tp, maximum_denominator) if maximum_denominator else Fraction(0)
    assert minimum[0] == analytic_min, "Enumerated minimum disagrees with analytic bound"
    assert maximum[0] == analytic_max, "Enumerated maximum disagrees with analytic bound"

    def extremum(item: tuple) -> dict:
        score, removed, remaining = item
        return {"f1": exact_value(score), "removed": asdict(removed), "remaining": asdict(remaining)}

    return {"records_deleted": deleted, "records_remaining": n - deleted,
            "feasible_deletion_allocations": count,
            "minimum": extremum(minimum), "maximum": extremum(maximum),
            "analytic_checks": {
                "minimum_rule": "Remove true positives first. If all are removed, F1 is zero under the stated zero-division convention.",
                "maximum_rule": "Remove errors first, then true negatives, then true positives if required.",
                "enumeration_matches_both_extrema": True}}


def analyze(source: Path, delete_count: int = 23) -> dict:
    raw = source.read_bytes()
    document = json.loads(raw)
    records = document["historical_comparison"]["results"]

    def select(model_id: str) -> dict:
        matches = [row for row in records if row.get("model_id") == model_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one aggregate record for {model_id}")
        return matches[0]

    neural = select("neural_meta")
    tree = select("xgboost_aft")
    confusion = infer_confusion(neural)
    n = integer(neural["cohort_n"], "neural cohort_n")
    comparison_n = integer(tree["cohort_n"], "tree cohort_n")
    positives = integer(neural["positives"], "neural positives")
    subset_positives = integer(tree["positives"], "tree positives")
    if comparison_n <= 0 or subset_positives > comparison_n:
        raise ValueError("Invalid reported comparison population")
    deleted = integer(delete_count, "delete_count")
    bounds = deletion_bounds(confusion, deleted)
    subset_n = n - deleted
    min_positive = max(0, positives - deleted)
    max_positive = min(positives, subset_n)
    try:
        source_label = source.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        source_label = source.name
    baseline_f1 = Decimal(str(tree["f1"]))
    if not baseline_f1.is_finite() or not 0 <= baseline_f1 <= 1:
        raise ValueError("The tree F1 must be a finite probability")
    return {
        "schema_version": 1,
        "kind": "conditional_aggregate_cohort_deletion_sensitivity",
        "source": {"aggregate_path": source_label, "sha256": hashlib.sha256(raw).hexdigest()},
        "input": {"neural_model_id": "neural_meta", "comparison_model_id": "xgboost_aft",
                  "neural_n": n, "neural_positives": positives,
                  "comparison_n": comparison_n, "comparison_positives": subset_positives,
                  "hypothetical_subset_n": subset_n,
                  "reported_neural_metrics": {key: neural[key] for key in ("precision", "recall", "f1")},
                  "reported_comparison_f1": tree["f1"]},
        "confusion_inference": {"compatible_integer_matrices": 1, "matrix": asdict(confusion),
                                "status": "Derived from rounded metrics and the surrounding cohort counts; not a printed meta confusion matrix.",
                                "rounding": "Four decimal places, decimal round-half-even; undefined ratios use zero."},
        "bounds": bounds,
        "reported_positive_subset_check": {
            "minimum_possible_positive_count": min_positive,
            "maximum_possible_positive_count": max_positive,
            "reported_comparison_positive_count": subset_positives,
            "reported_comparison_n_matches_hypothetical_subset_n": comparison_n == subset_n,
            "satisfied": comparison_n == subset_n and min_positive <= subset_positives <= max_positive,
            "interpretation": "A necessary count-consistency check only. Even a passing result would not verify row identity, dates or labels."},
        "conditional_lower_bound_exceeds_reported_comparison_f1":
            Fraction(bounds["minimum"]["f1"]["numerator"], bounds["minimum"]["f1"]["denominator"])
            > Fraction(baseline_f1),
        "assumptions": [
            "The meta metrics and surrounding neural cohort counts describe the same evaluation population.",
            "The smaller cohort is obtained solely by deleting records from the larger neural cohort.",
            "Labels, predictions and the decision threshold remain fixed when records are deleted.",
            "The published four-decimal metrics are correctly transcribed and rounded."],
        "scope": [
            "This is a deterministic worst/best-case sensitivity calculation over aggregate confusion counts.",
            "No row identifiers, predictions, private data, model fitting or retraining are used.",
            "The bound is not a recomputed paired-cohort F1, confidence interval, significance test or causal effect.",
            "A positive-count inconsistency leaves the unchanged-label subset premise unverified; the calculation does not repair it.",
            "The default 23-row deletion is counterfactual. Recovered run logs place the 941-row AFT report in an earlier evaluation week; the later shared-window AFT export contains 971 records and matches 964 neural records. This calculation does not assert that the earlier 941 rows are a subset of the neural 964."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Published aggregate evidence JSON")
    parser.add_argument("--delete-count", type=int, default=23, help="Counterfactual fixed-label record deletions (default: 23)")
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print it to stdout")
    args = parser.parse_args()
    try:
        report = analyze(args.source, args.delete_count)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
