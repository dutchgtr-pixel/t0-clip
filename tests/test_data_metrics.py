"""Protocol failures and independent arithmetic checks for research evaluation."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from marketneural.data import Preprocessor, attach_features, read_table, split_temporally
from marketneural.metrics import (SurvivalMetrics, horizon_labels, operating_metrics,
                                 paired_ibs_intervals, select_threshold, survival_target)


PROTOCOL = {"train_start": "2024-01-01T00:00:00Z",
            "validation_start": "2024-01-10T00:00:00Z",
            "test_start": "2024-01-20T00:00:00Z",
            "test_end": "2024-01-30T00:00:00Z",
            "test_as_of": "2024-02-15T00:00:00Z"}


def source_frame():
    decisions = pd.to_datetime(["2024-01-01", "2024-01-09", "2024-01-10",
                                "2024-01-19", "2024-01-20", "2024-01-29"], utc=True)
    observed = pd.to_datetime(["2024-01-02", "2024-01-10", "2024-01-12",
                               "2024-01-22", "2024-01-22", "2024-02-20"], utc=True)
    return pd.DataFrame({"row_id": [f"r{i}" for i in range(6)],
                         "entity_id": [f"e{i}" for i in range(6)],
                         "decision_time": decisions,
                         "feature_observed_at": decisions - pd.Timedelta(hours=1),
                         "observed_until": observed,
                         "event": [1, 1, 0, 1, 1, 1],
                         "price": [10., 20., 99., 100., 200., 300.],
                         "condition": ["a", "b", "c", "a", "b", "c"]})


def read_csv(tmp_path, frame):
    path = tmp_path / "cohort.csv"
    frame.to_csv(path, index=False)
    return read_table(path, ["price"], ["condition"])


def test_cutoffs_recensor_outcomes_at_fit_and_selection_time(tmp_path):
    cohorts, audit = split_temporally(read_csv(tmp_path, source_frame()), PROTOCOL)
    np.testing.assert_array_equal(cohorts["train"].event, [True, False])
    np.testing.assert_array_equal(cohorts["train"].duration, [24, 24])
    np.testing.assert_array_equal(cohorts["validation"].event, [False, False])
    np.testing.assert_array_equal(cohorts["validation"].duration, [48, 24])
    np.testing.assert_array_equal(cohorts["test"].event, [True, False])
    np.testing.assert_array_equal(cohorts["test"].duration, [48, 17 * 24])
    assert audit["train"]["administratively_censored"] == 1
    assert audit["validation"]["administratively_censored"] == 1
    assert audit["test"]["administratively_censored"] == 1
    assert audit["excluded_outside_windows"] == 0


@pytest.mark.parametrize("column", ["decision_time", "feature_observed_at", "observed_until"])
def test_nat_literal_cannot_evade_timestamp_attestation(tmp_path, column):
    frame = source_frame()
    frame[column] = frame[column].astype(str)
    frame.loc[0, column] = "NaT"
    with pytest.raises(ValueError):
        read_csv(tmp_path, frame)


@pytest.mark.parametrize("column", ["row_id", "entity_id"])
def test_whitespace_identifiers_cannot_bypass_identity_checks(tmp_path, column):
    frame = source_frame()
    frame.loc[2, column] = " " + frame.loc[0, column]
    with pytest.raises(ValueError, match="whitespace"):
        read_csv(tmp_path, frame)


def test_future_features_invalid_followup_and_outcome_allowlist_fail(tmp_path):
    frame = source_frame()
    frame.loc[0, "feature_observed_at"] = frame.loc[0, "decision_time"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="Future feature"):
        read_csv(tmp_path, frame)
    frame = source_frame()
    frame.loc[0, "observed_until"] = frame.loc[0, "decision_time"]
    with pytest.raises(ValueError, match="follow-up"):
        read_csv(tmp_path, frame)
    frame = source_frame().assign(sold_price=100)
    path = tmp_path / "outcome.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Forbidden"):
        read_table(path, ["price", "sold_price"], [])


@pytest.mark.parametrize("duplicate_row", [1, 2, 4])
def test_one_entity_cannot_repeat_or_cross_partitions(tmp_path, duplicate_row):
    frame = source_frame()
    frame.loc[duplicate_row, "entity_id"] = frame.loc[0, "entity_id"]
    with pytest.raises(ValueError, match="Entity leakage|one landmark"):
        split_temporally(read_csv(tmp_path, frame), PROTOCOL)


def test_boundary_order_and_empty_cohorts_fail(tmp_path):
    frame = read_csv(tmp_path, source_frame())
    protocol = {**PROTOCOL, "test_start": PROTOCOL["validation_start"]}
    with pytest.raises(ValueError, match="Require"):
        split_temporally(frame, protocol)
    with pytest.raises(ValueError, match="Empty temporal validation"):
        split_temporally(frame.drop(index=[2, 3]), PROTOCOL)


def test_train_only_statistics_and_unknown_categories():
    train = pd.DataFrame({"price": [1., np.nan, 3.], "condition": ["a", "b", "a"]})
    prep = Preprocessor(["price"], ["condition"]).fit(train)
    metadata_before = prep.metadata()
    validation = pd.DataFrame({"price": [1e6, np.nan], "condition": ["never_seen", "a"]})
    transformed = prep.transform(validation)
    assert prep.imputer.statistics_[0] == 2
    assert prep.scaler.mean_[0] == 2
    assert transformed[0, 0] > 1e5
    np.testing.assert_array_equal(transformed[0, -2:], [0, 0])
    assert transformed[1, 0] == 0  # Missing value imputes TRAIN median.
    assert metadata_before == prep.metadata()


def test_vectors_align_by_row_id_not_archive_position(tmp_path):
    cohorts, _ = split_temporally(read_csv(tmp_path, source_frame()), PROTOCOL)
    archive = tmp_path / "vectors.npz"
    np.savez(archive, row_ids=np.array([f"r{i}" for i in range(5, -1, -1)]),
             text=np.array([[i, i + 1] for i in range(5, -1, -1)], dtype=np.float32))
    metadata = attach_features(cohorts, ["price"], ["condition"], archive)
    np.testing.assert_array_equal(cohorts["train"].features["text"], [[0, 1], [1, 2]])
    np.testing.assert_array_equal(cohorts["test"].features["text"], [[4, 5], [5, 6]])
    assert metadata["fit_partition"] == "train"
    np.savez(archive, row_ids=np.array(["r0", "r0"]), text=np.ones((2, 2)))
    with pytest.raises(ValueError, match="unique"):
        attach_features(cohorts, ["price"], ["condition"], archive)
    np.savez(archive, row_ids=np.array(["r0"]), text=np.ones((1, 2)))
    with pytest.raises(ValueError, match="missing from vector"):
        attach_features(cohorts, ["price"], ["condition"], archive)


def test_ipcw_brier_matches_independent_library_implementation():
    from sksurv.metrics import brier_score

    train_d = np.array([1., 2., 3., 4., 6., 8., 10.])
    train_e = np.array([1, 0, 1, 0, 1, 0, 1], dtype=bool)
    eval_d = np.array([1., 2.5, 4., 7., 9.])
    eval_e = np.array([1, 0, 1, 1, 0], dtype=bool)
    times = np.array([1.5, 3.5, 5.])
    prediction = np.exp(-np.array([.4, .1, .2, .3, .05])[:, None] * times)
    metrics = SurvivalMetrics(train_d, train_e, times)
    _, expected = brier_score(survival_target(train_d, train_e),
                              survival_target(eval_d, eval_e), prediction, times)
    result = metrics.evaluate(eval_d, eval_e, prediction)
    np.testing.assert_allclose(result["brier_by_time"], expected)
    expected_ibs = np.trapezoid(expected, times) / (times[-1] - times[0])
    assert result["integrated_brier_score"] == pytest.approx(expected_ibs)
    assert metrics.per_row_ibs(eval_d, eval_e, prediction).mean() == pytest.approx(expected_ibs)


def test_no_censor_brier_reduces_to_binary_squared_error():
    duration = np.array([1., 3., 6., 8.])
    event = np.ones(4, dtype=bool)
    times = np.array([2., 4.])
    prediction = np.array([[.1, .05], [.7, .2], [.9, .8], [.95, .9]])
    metrics = SurvivalMetrics(duration, event, times)
    expected = (prediction - (duration[:, None] > times)) ** 2
    np.testing.assert_allclose(metrics.brier_contributions(duration, event, prediction), expected)
    assert metrics.evaluate(duration, event, prediction)["integrated_brier_score"] == pytest.approx(expected.mean())


@pytest.mark.parametrize("event", [[0, 0, 0], [0, 1, 1]])
def test_brier_is_defined_without_events_inside_grid(event):
    metrics = SurvivalMetrics([1., 2., 6., 10.], [1, 1, 1, 1], [2., 4.])
    duration = np.array([1., 6., 8.])
    prediction = np.tile([.8, .7], (3, 1))
    result = metrics.evaluate(duration, event, prediction)
    expected = (1 - prediction) ** 2 * (duration[:, None] > metrics.times)
    np.testing.assert_allclose(result["brier_by_time"], expected.mean(axis=0))
    assert np.isfinite(result["integrated_brier_score"])


def test_censor_support_and_incoherent_survival_predictions_fail():
    with pytest.raises(ValueError, match="follow-up support"):
        SurvivalMetrics([1, 2, 3, 4], [1, 0, 1, 0], [2, 4])
    with pytest.raises(ValueError, match="censoring support"):
        SurvivalMetrics([1, 2, 3, 4], [0, 0, 0, 1], [1.5, 3.5], min_censor_survival=.5)
    metrics = SurvivalMetrics([1, 2, 3, 8], [1, 1, 1, 1], [2, 4])
    with pytest.raises(ValueError, match="nonincreasing"):
        metrics.evaluate([1, 3, 6, 8], [1, 1, 1, 0], np.tile([.2, .8], (4, 1)))
    with pytest.raises(ValueError, match="follow-up beyond"):
        metrics.evaluate([1, 2, 3, 4], [1, 1, 1, 0], np.tile([.8, .2], (4, 1)))


def test_horizon_eligibility_threshold_selection_and_fixed_transfer():
    duration = np.array([1., 3., 5., 2., 4., 2.])
    event = np.array([1, 1, 1, 0, 0, 1])
    probability = np.array([.9, .8, .7, .99, .1, .4])
    positive, eligible = horizon_labels(duration, event, 3)
    np.testing.assert_array_equal(positive, [1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(eligible, [1, 1, 1, 0, 1, 1])
    selection = select_threshold(duration, event, probability, 3, target_precision=1, min_flagged=2)
    assert selection["threshold"] == .8
    assert selection["tp"] == 2 and selection["fp"] == 0
    assert selection["flagged_all_rows"] == 3 and selection["flagged_unresolved"] == 1
    assert selection["recall"] == pytest.approx(2 / 3)
    # TEST is evaluated at the SVAL threshold even when the result degrades.
    transfer = operating_metrics([5., 1.], [1, 1], [.9, .4], 3, selection["threshold"])
    assert transfer["precision"] == 0 and transfer["threshold"] == .8
    rejected = select_threshold(duration, event, probability, 3, target_precision=1, min_flagged=10)
    assert rejected["threshold"] is None and rejected["flagged_all_rows"] == 0
    assert rejected["selection_status"] == "no_feasible_validation_threshold_reject_all"
    _, exactly_censored = horizon_labels([3.], [0], 3)
    assert not exactly_censored[0]


@pytest.mark.parametrize("duration,event,horizon", [([1, np.nan], [1, 0], 3),
                                                      ([1, 2], [1, 2], 3),
                                                      ([1, 2], [1, 0], np.nan)])
def test_horizon_outcomes_are_validated(duration, event, horizon):
    with pytest.raises(ValueError):
        operating_metrics(duration, event, [.8, .4], horizon, .5)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -.1, 1.1])
def test_invalid_operating_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        operating_metrics([1., 4.], [1, 0], [.8, .4], 3, threshold)


def test_paired_bootstrap_preserves_entity_pairing_and_seed():
    reference = np.array([.1, .4, .2, .8, .3])
    losses = {"km": reference, "shifted": reference + .05, "same": reference.copy()}
    first = paired_ibs_intervals(losses, reference="km", repetitions=100, seed=21)
    second = paired_ibs_intervals(losses, reference="km", repetitions=100, seed=21)
    assert first == second
    assert first["shifted"]["delta_ibs_vs_reference"] == pytest.approx(.05)
    np.testing.assert_allclose(first["shifted"]["ci95"], [.05, .05])
    np.testing.assert_array_equal(first["same"]["ci95"], [0, 0])


@pytest.mark.parametrize("bad", [[.1, np.nan], [[.1], [.2]], [.1]])
def test_invalid_bootstrap_rows_fail_closed(bad):
    with pytest.raises(ValueError):
        paired_ibs_intervals({"reference": [.1, .2], "candidate": bad}, reference="reference")


def test_benchmark_freezes_all_selection_before_test_and_test_changes_cannot_select(tmp_path, monkeypatch):
    from marketneural import benchmark

    config = {"data": {"table": "cohort.csv", "numeric_columns": ["price"],
                       "categorical_columns": ["condition"]},
              "protocol": PROTOCOL, "seeds": [5, 17],
              "evaluation": {"times_hours": [6., 12.], "decision_horizon_hours": 10.,
                             "operating_point": {"target_precision": .5, "min_flagged": 1},
                             "bootstrap": {"repetitions": 20}},
              "models": {"coxph": [{"rate": .01}, {"rate": .1}],
                         "mlp": [{"rate": .02}, {"rate": .2}]}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls, current_output = [], [None]

    class RecordingModel:
        def __init__(self, name, random_state, rate):
            self.name, self.seed, self.rate = name, random_state, rate

        def fit(self, features, duration, event, *, validation):
            assert not (current_output[0] / "selection.json").exists()
            np.testing.assert_array_equal(duration, [24., 24.])
            np.testing.assert_array_equal(event, [True, False])
            # The only early-stop outcomes supplied are the validation cohort.
            self.validation_features = validation[0]
            np.testing.assert_array_equal(validation[1], [8., 24.])
            np.testing.assert_array_equal(validation[2], [True, False])
            calls.append(("fit", self.name, self.seed, self.rate))
            return self

        def predict_survival(self, features, times):
            if features is self.validation_features:
                assert not (current_output[0] / "selection.json").exists()
                calls.append(("validation", self.name))
            else:
                assert (current_output[0] / "selection.json").exists()
                assert sum(call[0] == "fit" for call in calls) == 8
                calls.append(("test", self.name))
            return np.broadcast_to(np.exp(-self.rate * np.asarray(times)),
                                   (len(features["tabular"]), len(times))).copy()

    monkeypatch.setattr(benchmark, "create_model", RecordingModel)
    frame = source_frame()
    frame.loc[2, "event"] = 1
    frame.loc[2, "observed_until"] = frame.loc[2, "decision_time"] + pd.Timedelta(hours=8)
    frame.loc[4, "observed_until"] = frame.loc[4, "decision_time"] + pd.Timedelta(hours=9)
    selections, summaries, manifests = [], [], []
    for iteration in range(2):
        if iteration:
            # Neither held-out outcomes nor held-out feature scale may select a model.
            frame.loc[4:, "event"] = 0
            frame.loc[4:, "price"] = [999., 9999.]
        frame.to_csv(tmp_path / "cohort.csv", index=False)
        current_output[0] = tmp_path / f"run{iteration}"
        calls.clear()
        summaries.append(benchmark.run(config_path, current_output[0]))
        selection_bytes = (current_output[0] / "selection.json").read_bytes()
        assert summaries[-1]["frozen_selection_sha256"] == hashlib.sha256(selection_bytes).hexdigest()
        selection = json.loads(selection_bytes)
        manifests.append(json.loads((current_output[0] / "protocol.json").read_text()))
        for model in selection.values():
            for candidate in model["candidate_results"]:
                for seed in candidate["seeds"]:
                    seed.pop("fit_seconds")  # Wall-clock duration is intentionally variable.
        selections.append(selection)
        assert calls[-1][0] == "test"
        first_test = next(i for i, call in enumerate(calls) if call[0] == "test")
        assert all(call[0] == "test" for call in calls[first_test:])
        for name in config["models"]:
            chosen = selection[name]["candidate_results"][selection[name]["selected_candidate"]]
            for expected, actual in zip(chosen["seeds"], summaries[-1]["results"][name]["seed_results"]):
                assert actual["test_operating_point"]["threshold"] == expected["operating_point"]["threshold"]
    assert selections[0] == selections[1]
    assert manifests[0]["preprocessing"] == manifests[1]["preprocessing"]
    assert manifests[0]["provenance"]["table_sha256"] != manifests[1]["provenance"]["table_sha256"]
    assert summaries[0]["results"]["coxph"]["mean_test_ibs"] != summaries[1]["results"]["coxph"]["mean_test_ibs"]
    with pytest.raises(FileExistsError, match="Do not overwrite"):
        benchmark.run(config_path, current_output[0])
