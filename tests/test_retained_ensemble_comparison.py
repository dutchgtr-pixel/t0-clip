"""Pairing checks for retrospective fixed-decision comparisons."""
import csv
import json

import pytest

from scripts.analyze_retained_ensemble_comparison import compare


def files(tmp_path, change=None):
    rows = [
        {"id": "private-a", "time": "t1", "duration": 8, "label": 1, "pred": 1},
        {"id": "private-b", "time": "t2", "duration": 9, "label": 1, "pred": 0},
        {"id": "private-c", "time": "t3", "duration": 2, "label": 0, "pred": 1},
        {"id": "private-d", "time": "t4", "duration": 1, "label": 0, "pred": 0},
    ]
    candidate = [dict(row) for row in reversed(rows)]
    if change:
        change(candidate)
    paths = {}
    for name, contents in [("reference", rows), ("candidate", candidate)]:
        path = tmp_path / (name + ".csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(contents)
        paths[name] = path
    return paths


def run(paths):
    return compare(paths, "reference", keys=["id", "time"], label="label",
                   prediction="pred", invariants=["duration"], bootstrap=200, seed=6)


def test_pairing_uses_keys_not_row_order_and_identical_decisions_have_zero_interval(tmp_path):
    result = run(files(tmp_path))
    assert result["models"]["reference"]["f1"] == 0.5
    assert result["contrasts"]["candidate"]["f1_difference_interval"] == [0.0, 0.0]
    serialized = json.dumps(result)
    assert "private-a" not in serialized
    assert str(tmp_path) not in serialized


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.append(dict(r[0])), "unique"),
    (lambda r: r.pop(), "same keys"),
    (lambda r: r[0].update(label=1), "Labels or invariant"),
    (lambda r: r[0].update(duration=99), "Labels or invariant"),
    (lambda r: r[0].update(pred=0.5), "binary"),
])
def test_unsafe_pairing_or_nonbinary_predictions_rejected(tmp_path, change, reason):
    with pytest.raises(ValueError, match=reason):
        run(files(tmp_path, change))


def test_joint_disagreement_counts_and_seed_are_reproducible(tmp_path):
    paths = files(tmp_path, lambda rows: [row.update(pred=row["label"]) for row in rows])
    first, second = run(paths), run(paths)
    assert first == second
    contrast = first["contrasts"]["candidate"]
    assert contrast["candidate_only_correct"] == 2
    assert contrast["reference_only_correct"] == 0
    assert contrast["f1_difference"] == 0.5
