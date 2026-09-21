"""Synthetic checks for the empirical cohort audit and aggregate sensitivity."""
import csv
import json
from fractions import Fraction

import pytest

from scripts.audit_shared_cohort import AuditError, audit
from scripts.analyze_cohort_sensitivity import Confusion, deletion_bounds, infer_confusion


def fixture_files(tmp_path, right_change=None):
    columns = {"id": "entity", "origin": "at", "duration": "hours",
               "prediction": "estimate", "event": "observed"}
    base = [
        {"entity": "synthetic-a", "at": "2020-01-01T00:00:00Z", "hours": "600", "estimate": "700", "observed": "1"},
        {"entity": "synthetic-b", "at": "2020-01-02T00:00:00Z", "hours": "100", "estimate": "700", "observed": "1"},
        {"entity": "synthetic-c", "at": "2020-01-03T00:00:00Z", "hours": "600", "estimate": "100", "observed": "1"},
    ]
    left = base + [{"entity": "synthetic-extra", "at": "2020-01-04T00:00:00Z", "hours": "600", "estimate": "700", "observed": "1"}]
    right = [dict(row) for row in base]
    right[0]["at"] = "2020-01-01T01:00:00+01:00"
    if right_change:
        right_change(right)
    paths = [tmp_path / "left.csv", tmp_path / "right.csv"]
    for path, rows in zip(paths, [left, right]):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(base[0]))
            writer.writeheader()
            writer.writerows(rows)
    mapping = tmp_path / "columns.json"
    mapping.write_text(json.dumps({"left": columns, "right": columns}), encoding="utf-8")
    return *paths, mapping


def test_common_keys_not_equal_row_counts_define_comparison(tmp_path):
    result = audit(*fixture_files(tmp_path))
    assert result["common_rows"] == 3
    assert result["exact_common_duration_matches"] == 3
    assert result["scores"]["full_left"]["f1"] == pytest.approx(2 / 3)
    assert result["scores"]["common_left"]["f1"] == pytest.approx(1 / 2)
    assert result["cohorts"]["left_only"]["rows"] == 1
    assert "synthetic-a" not in json.dumps(result)


@pytest.mark.parametrize("change,code", [
    (lambda rows: rows.append(dict(rows[0])), "duplicate_entity_origin"),
    (lambda rows: rows[0].update(hours="601"), "common_duration_mismatch"),
    (lambda rows: rows[0].update(at="2020-01-01T00:00:00"), "naive_origin"),
])
def test_unverifiable_pairing_is_rejected(tmp_path, change, code):
    with pytest.raises(AuditError) as exc:
        audit(*fixture_files(tmp_path, change))
    assert exc.value.code == code


def test_rounded_meta_metrics_have_one_compatible_integer_matrix():
    matrix = infer_confusion({"cohort_n": 964, "positives": 114,
                              "precision": .9802, "recall": .8684, "f1": .9209})
    assert matrix == Confusion(99, 2, 15, 848)
    with pytest.raises(ValueError):
        infer_confusion({"cohort_n": 964, "positives": 114,
                         "precision": .9802, "recall": .8684, "f1": .1})


def test_twenty_three_row_bound_is_attainable_and_not_a_paired_estimate():
    result = deletion_bounds(Confusion(99, 2, 15, 848), 23)
    assert result["feasible_deletion_allocations"] == 744
    assert Fraction(result["minimum"]["f1"]["fraction"]) == Fraction(152, 169)
    assert result["minimum"]["removed"] == {"tp": 23, "fp": 0, "fn": 0, "tn": 0}
    assert Fraction(result["maximum"]["f1"]["fraction"]) == 1
