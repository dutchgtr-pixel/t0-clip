import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "availability_mechanism", Path(__file__).parents[1] / "scripts" / "analyze_availability_mechanism.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selection_creates_association_without_predictive_source():
    experiment = MODULE.constructed_experiment()
    selected = experiment["scenarios"]["outcome_time_selected_frozen"]
    origin = experiment["scenarios"]["origin_time_selected_fully_matured"]
    assert selected["rows"] == 120
    assert selected["fixed_missingness_rule"]["tp"] == 12
    assert selected["fixed_missingness_rule"]["fp"] == 10
    assert selected["fixed_missingness_rule"]["f1"] == pytest.approx(12 / 17)
    assert origin["positive_rate_when_missing"] == origin["positive_rate_when_covered"] == 0.1
    assert origin["fixed_missingness_rule"]["balanced_accuracy"] == 0.5
    assert selected["sufficient_condition_all_fast_missing"]


def test_same_rows_coverage_refresh_breaks_fixed_shortcut():
    cases = MODULE.constructed_experiment()["scenarios"]
    before, after = cases["outcome_time_selected_frozen"], cases["same_outcome_rows_after_coverage_refresh"]
    assert before["rows"] == after["rows"]
    assert before["positive_rows"] == after["positive_rows"] == 12
    assert after["fixed_missingness_rule"]["f1"] == 0
    assert after["cutoff_mismatch_rows"] == 22
    assert after["geometry_all_fast_origins_after_cutoff"]
    assert not after["sufficient_condition_all_fast_missing"]


def test_sufficient_conditions_across_window_geometries():
    for a, b in ((8, 12), (15, 15), (20, 27)):
        event, duration = np.meshgrid(np.arange(a, b + 1), np.arange(31))
        event, duration = event.ravel(), duration.ravel()
        origin = event - duration
        for horizon in (0, 1, 3, 7):
            for cutoff in (a - horizon - 1, a - horizon, a - horizon + 1):
                result = MODULE.audit_cutoff(origin, event, duration, origin >= cutoff,
                                             cutoff=cutoff, horizon=horizon)
                assert result["cutoff_mismatch_rows"] == 0
                assert result["beyond_window_bound_but_missing_rows"] == 0
                assert result["forced_fast_zone_nonfast_rows"] == 0
                if result["sufficient_condition_all_fast_missing"]:
                    assert result["fast_but_covered_rows"] == 0


@pytest.mark.parametrize("origin,event,duration,missing", [
    ([0], [1], [2], [0]),
    ([0], [1], [1], [2]),
    ([0], [1], [float("nan")], [0]),
    ([2], [1], [-1], [0]),
    ([0], [1, 2], [1], [0]),
    ([], [], [], []),
])
def test_incompatible_evidence_is_rejected(origin, event, duration, missing):
    with pytest.raises(ValueError):
        MODULE.audit_cutoff(origin, event, duration, missing, cutoff=0, horizon=3)


def test_historical_aggregates_preserve_original_counts():
    import json
    path = Path(__file__).parents[1] / "research" / "leakage" / "availability_mechanism_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    expected = [(408, 167, 52, 189), (503, 208, 127, 168)]
    for record, (n, tp, fp, tn) in zip(results["historical_recomputation"], expected, strict=True):
        audit = record["audit"]
        assert audit["rows"] == n
        assert audit["cutoff_mismatch_rows"] == 0
        assert audit["duration_identity_max_abs_error"] < 1e-6
        assert audit["fixed_missingness_rule"]["tp"] == tp
        assert audit["fixed_missingness_rule"]["fp"] == fp
        assert audit["fixed_missingness_rule"]["fn"] == 0
        assert audit["fixed_missingness_rule"]["tn"] == tn
    assert results["constructed_verification"] == MODULE.constructed_experiment()


def test_file_audit_normalizes_timezones_and_emits_no_records(tmp_path):
    import json
    from types import SimpleNamespace

    private = tmp_path / "sensitive_filename.csv"
    private.write_text(
        "origin,event,duration,observed,count,share,identifier\n"
        "2026-03-06T01:00:00+01:00,2026-03-08T00:00:00Z,48,1,0,,PRIVATE_A\n"
        "2026-03-05T00:00:00.000Z,2026-03-10T01:00:00+01:00,120,1,5,0.5,PRIVATE_B\n",
        encoding="utf-8")
    args = SimpleNamespace(input=str(private), origin_column="origin", event_column="event",
                           duration_column="duration", observed_column="observed", count_column="count",
                           share_column="share", cutoff="2026-03-06T01:00:00+01:00", horizon_hours=72)
    result = MODULE.audit_file(args)
    assert result["audit"]["duration_identity_max_abs_error"] == 0
    assert result["audit"]["cutoff_mismatch_rows"] == 0
    serialized = json.dumps(result)
    for forbidden in ("sensitive_filename", "PRIVATE_A", "PRIVATE_B", str(tmp_path)):
        assert forbidden not in serialized
    args.cutoff = "2026-03-06"
    with pytest.raises(ValueError, match="timezone"):
        MODULE.audit_file(args)
    args.cutoff = "2026-03-06T00:00:00Z"
    private.write_text(private.read_text().replace("2026-03-08T00:00:00Z", "2026-03-08T00:00:00"))
    with pytest.raises(ValueError, match="timezone"):
        MODULE.audit_file(args)


def test_file_audit_rejects_censored_events(tmp_path):
    from types import SimpleNamespace

    private = tmp_path / "observations.csv"
    private.write_text("origin,event,duration,observed,count,share\n2026-03-06Z,2026-03-08Z,48,0,0,\n")
    args = SimpleNamespace(input=str(private), origin_column="origin", event_column="event",
                           duration_column="duration", observed_column="observed", count_column="count",
                           share_column="share", cutoff="2026-03-06T00:00:00Z", horizon_hours=72)
    with pytest.raises(ValueError, match="observed events"):
        MODULE.audit_file(args)
