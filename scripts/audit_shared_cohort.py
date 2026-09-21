"""Reconcile two private prediction CSVs without exporting individual records.

The mapping JSON has exactly two objects, ``left`` and ``right``. Each maps
``id``, ``origin``, ``duration`` and ``prediction`` to input column names; an
``event`` column is optional. Durations and predictions are in hours. For example::

    {"left": {"id": "entity", "origin": "origin_time", "duration": "hours",
              "prediction": "estimated_hours", "event": "observed"},
     "right": {"id": "entity", "origin": "origin_time", "duration": "hours",
               "prediction": "estimated_hours", "event": "observed"}}

Identity is an opaque string; origins must carry an explicit UTC offset. A row
with no event column is assumed observed. With an event column, only 0/1 or
false/true are accepted. A censored duration strictly beyond the horizon has a
known positive tail label; censoring at or before the horizon is rejected.

This audits the fixed rule ``predicted_duration > horizon``. It is a comparison
of the supplied duration outputs, not a substitute for a survival-probability
or ensemble classifier comparison. Neither a threshold nor a model is fitted.
The output contains aggregate statistics and hashes, never paths, column names,
identities, or row-level predictions. Keep private input mappings outside Git.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REQUIRED_ROLES = {"id", "origin", "duration", "prediction"}
OPTIONAL_ROLES = {"event"}


class AuditError(ValueError):
    """An intentionally value-free failure suitable for aggregate-only output."""

    def __init__(self, code: str, side: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.side = side


@dataclass(frozen=True)
class Record:
    origin_ns: int
    duration: Decimal
    prediction: Decimal
    event: bool


@dataclass(frozen=True)
class InputData:
    sha256: str
    records: dict[tuple[str, int], Record]
    event_provided: bool


def _read_bytes(path: Path, code: str, side: str | None = None) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        raise AuditError(code, side) from None


def _number(value: str, code: str, side: str | None = None) -> Decimal:
    try:
        result = Decimal(value.strip())
        valid = result.is_finite() and math.isfinite(float(result))
    except (InvalidOperation, ValueError, OverflowError, AttributeError):
        valid = False
    if not valid:
        raise AuditError(code, side)
    return result


def _mapping(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    raw = _read_bytes(path, "mapping_unreadable")
    try:
        mapping = json.loads(raw)
    except (ValueError, UnicodeError):
        raise AuditError("mapping_invalid_json") from None
    if not isinstance(mapping, dict) or set(mapping) != {"left", "right"}:
        raise AuditError("mapping_requires_left_and_right")
    for side in ("left", "right"):
        fields = mapping[side]
        if not isinstance(fields, dict):
            raise AuditError("mapping_invalid_fields", side)
        if not REQUIRED_ROLES <= set(fields) <= REQUIRED_ROLES | OPTIONAL_ROLES:
            raise AuditError("mapping_invalid_roles", side)
        if any(not isinstance(name, str) or not name.strip() for name in fields.values()):
            raise AuditError("mapping_invalid_column_names", side)
        if len(set(fields.values())) != len(fields):
            raise AuditError("mapping_reuses_column", side)
    return mapping, hashlib.sha256(raw).hexdigest()


def _origin(value: str, side: str) -> int:
    try:
        # Scalar parsing retains nanoseconds and permits different explicit
        # offsets. It never silently localizes a naive timestamp to UTC.
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise AuditError("invalid_origin", side)
        if stamp.tzinfo is None:
            raise AuditError("naive_origin", side)
        return int(stamp.tz_convert("UTC").value)
    except AuditError:
        raise
    except (ValueError, TypeError, OverflowError):
        raise AuditError("invalid_origin", side) from None


def _load(path: Path, fields: dict[str, str], horizon: Decimal, side: str) -> InputData:
    raw = _read_bytes(path, "input_unreadable", side)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise AuditError("input_not_utf8", side) from None
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    records: dict[tuple[str, int], Record] = {}
    try:
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise AuditError("invalid_or_duplicate_headers", side)
        if not set(fields.values()) <= set(headers):
            raise AuditError("missing_mapped_column", side)
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise AuditError("malformed_csv_row", side)
            identity = row[fields["id"]]
            if not identity.strip() or identity != identity.strip():
                raise AuditError("invalid_identity", side)
            origin = _origin(row[fields["origin"]], side)
            key = (identity, origin)
            if key in records:
                raise AuditError("duplicate_entity_origin", side)
            duration = _number(row[fields["duration"]], "invalid_duration", side)
            prediction = _number(row[fields["prediction"]], "invalid_prediction", side)
            if duration < 0 or prediction < 0:
                raise AuditError("negative_duration_or_prediction", side)
            event = True
            if "event" in fields:
                flag = row[fields["event"]].strip().lower()
                if flag not in {"0", "1", "false", "true"}:
                    raise AuditError("unknown_event_label", side)
                event = flag in {"1", "true"}
            if not event and duration <= horizon:
                raise AuditError("censored_tail_label_unknown", side)
            records[key] = Record(origin, duration, prediction, event)
    except csv.Error:
        raise AuditError("malformed_csv", side) from None
    if not records:
        raise AuditError("empty_input", side)
    return InputData(hashlib.sha256(raw).hexdigest(), records, "event" in fields)


def _iso(ns: int) -> str:
    try:
        return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()
    except (ValueError, OverflowError):
        raise AuditError("derived_time_out_of_range") from None


def _summary(rows: list[Record], horizon: Decimal) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    # The reconstructed observation end is a diagnostic, conditional on the
    # caller's origin and duration describing the same time origin. Censoring
    # is retained in the counts and is never presented as an observed event.
    end_ns = [r.origin_ns + int(r.duration * Decimal(3_600_000_000_000)) for r in rows]
    return {
        "rows": len(rows),
        "observed_event_rows": sum(r.event for r in rows),
        "known_censored_tail_rows": sum(not r.event for r in rows),
        "tail_positive_rows": sum(r.duration > horizon for r in rows),
        "duration_equal_to_horizon_rows": sum(r.duration == horizon for r in rows),
        "zero_duration_rows": sum(r.duration == 0 for r in rows),
        "duration_min_hours": float(min(r.duration for r in rows)),
        "duration_max_hours": float(max(r.duration for r in rows)),
        "origin_min_utc": _iso(min(r.origin_ns for r in rows)),
        "origin_max_utc": _iso(max(r.origin_ns for r in rows)),
        "derived_observation_end_min_utc": _iso(min(end_ns)),
        "derived_observation_end_max_utc": _iso(max(end_ns)),
    }


def _score(rows: list[Record], horizon: Decimal) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        truth = row.duration > horizon
        decision = row.prediction > horizon
        if truth and decision:
            tp += 1
        elif decision:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return {
        "rows": len(rows),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def audit(left_path: Path, right_path: Path, columns_path: Path, horizon: str = "504") -> dict[str, Any]:
    """Return aggregates, or reject inputs without including their values."""
    threshold = _number(horizon, "invalid_horizon")
    if threshold <= 0:
        raise AuditError("invalid_horizon")
    mapping, mapping_hash = _mapping(columns_path)
    left = _load(left_path, mapping["left"], threshold, "left")
    right = _load(right_path, mapping["right"], threshold, "right")
    common = left.records.keys() & right.records.keys()
    if not common:
        raise AuditError("no_common_entity_origin_keys")
    if any(left.records[key].duration != right.records[key].duration for key in common):
        raise AuditError("common_duration_mismatch")
    common_left = [left.records[key] for key in common]
    common_right = [right.records[key] for key in common]
    left_only = [left.records[key] for key in left.records.keys() - common]
    right_only = [right.records[key] for key in right.records.keys() - common]
    return {
        "schema_version": 1,
        "scope": "fixed_duration_prediction_shared_cohort_audit",
        "interpretation": (
            "Scores apply only to the supplied duration predictions. Survival-probability "
            "scores, selected classifiers and ensemble decisions require their own artifacts."
        ),
        "horizon_hours": float(threshold),
        "truth_rule": "duration_hours > horizon_hours",
        "prediction_rule": "predicted_duration_hours > horizon_hours",
        "threshold_fitted": False,
        "undefined_metric_policy": "null when the metric denominator is zero",
        "join_contract": "unique opaque identity plus explicitly timezone-aware origin normalized to UTC",
        "source_sha256": {"left": left.sha256, "right": right.sha256},
        "mapping_sha256": mapping_hash,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "event_columns_provided": {"left": left.event_provided, "right": right.event_provided},
        "absent_event_column_policy": "all rows assumed observed; caller must verify this assumption",
        "common_rows": len(common),
        "exact_common_duration_matches": len(common),
        "exact_common_tail_label_matches": len(common),
        "common_event_flag_matches": sum(left.records[key].event == right.records[key].event for key in common),
        "cohorts": {
            "full_left": _summary(list(left.records.values()), threshold),
            "full_right": _summary(list(right.records.values()), threshold),
            "common": _summary(common_left, threshold),
            "left_only": _summary(left_only, threshold),
            "right_only": _summary(right_only, threshold),
        },
        "scores": {
            "full_left": _score(list(left.records.values()), threshold),
            "common_left": _score(common_left, threshold),
            "common_right": _score(common_right, threshold),
        },
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default error can echo user-supplied private paths.
        self.exit(2, '{"status":"error","code":"invalid_arguments_run_help"}\n')


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-csv", type=Path, required=True)
    parser.add_argument("--right-csv", type=Path, required=True)
    parser.add_argument("--columns-json", type=Path, required=True)
    parser.add_argument("--horizon", default="504")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Prevent a mistaken output argument from overwriting private input.
        if args.output.resolve() in {args.left_csv.resolve(), args.right_csv.resolve(), args.columns_json.resolve()}:
            raise AuditError("output_overwrites_input")
        result = audit(args.left_csv, args.right_csv, args.columns_json, args.horizon)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Never leave an old successful receipt behind under a new result.
            # Existing outputs are refused rather than silently replaced.
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
        except FileExistsError:
            raise AuditError("output_already_exists") from None
        except OSError:
            raise AuditError("output_unwritable") from None
    except AuditError as exc:
        error = {"status": "error", "code": exc.code}
        if exc.side is not None:
            error["side"] = exc.side
        print(json.dumps(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "common_rows": result["common_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
