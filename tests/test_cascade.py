"""Historical source integrity and portable cascade behavior, using artificial data."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research.cascade import (CascadeEstimator, FeatureBatch, FrozenFeatureEncoder,
                              ProbabilityEnsemble, RoutingPolicy, StageEstimator, StageObjective, route_scores)
from research.cascade import legacy_core, search_core, stage1_core, stage2_core
from research.cascade.ensemble import choose_ensemble
from research.cascade.pipeline import select_stage_threshold
from research.cascade.portable import stage_loss


@pytest.fixture(autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def features(n=20, seed=4):
    rng = np.random.default_rng(seed)
    return FeatureBatch(rng.normal(size=(n, 3)).astype(np.float32), rng.integers(0, 3, (n, 2)),
                         rng.normal(size=(n, 768)).astype(np.float32), rng.normal(size=(n, 512)).astype(np.float32))


def tiny_config():
    return legacy_core.TrainConfig(d_model=16, n_latents=2, fusion_layers=1, n_heads=2,
                                   n_experts=2, n_bins=8, text_tokens=1, img_tokens=1,
                                   tab_num_tokens=2, tab_cat_tokens=2, dropout=0., attn_dropout=0.)


def test_archived_sources_are_exact_and_all_symbols_are_preserved():
    folder = Path(__file__).parents[1] / "research" / "cascade"
    manifest = json.loads((folder / "provenance.json").read_text())
    count = 0
    for source in manifest["sources"]:
        raw = (folder / source["export_file"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == source["export_sha256"]
        definitions = {node.name: node for node in ast.parse(raw).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for symbol in source["symbols"]:
            digest = hashlib.sha256(ast.dump(definitions[symbol["name"]], include_attributes=False).encode()).hexdigest()
            assert digest == symbol["ast_sha256"], symbol["name"]
            count += 1
    assert count >= 95


@pytest.mark.parametrize("module", [stage1_core, stage2_core])
def test_historical_censor_labels_and_exact_boundary(module):
    duration, event = np.array([71., 72., 73., 71., 72., 73.]), np.array([1, 1, 1, 0, 0, 0])
    fast, known = module._fast_labels_and_mask(duration, event, 72)
    slow, slow_known = module._slow_labels_and_mask(duration, event, 72)
    np.testing.assert_array_equal(known, [1, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(slow_known, known)
    np.testing.assert_array_equal(fast[known], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(slow[known], 1-fast[known])
    assert module.cls_metrics(fast, fast.astype(float), .5, known)["f1"] == 1


def test_survival_likelihood_partial_censor_and_beyond_horizon_recensoring():
    h = torch.tensor([[[.2, .5]]]*3, requires_grad=True)
    outputs = (h, torch.ones(3, 1), torch.zeros(3))
    duration, event = torch.tensor([100., 84., 600.]), torch.tensor([1., 0., 1.])
    objective = StageObjective(nll=1, curve=0, head=0)
    loss, terms = stage_loss(outputs, duration, event, torch.ones(3), torch.tensor([0., 168., 504.]), 1, objective)
    expected = -np.log([.2, np.sqrt(.8), .4]).mean()
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert torch.isfinite(h.grad).all()


def test_short_censor_has_no_binary_head_gradient():
    logits = torch.zeros(3, requires_grad=True)
    outputs = (torch.full((3, 1, 2), .2), torch.ones(3, 1), logits)
    loss, _ = stage_loss(outputs, torch.tensor([20., 168., 168.]), torch.tensor([0., 0., 1.]),
                         torch.ones(3), torch.tensor([0., 168., 504.]), 1, StageObjective(nll=0, curve=0, head=1))
    loss.backward()
    assert logits.grad[0] == 0
    assert logits.grad[1] < 0 and logits.grad[2] > 0


def test_preprocessing_is_train_only_and_unknown_categories_use_zero():
    prep = FrozenFeatureEncoder().fit([[1., np.nan], [3., 8.]], [["a"], ["b"]])
    result = prep.transform([[999., np.nan]], [["unseen"]], np.zeros((1, 768)), np.zeros((1, 512)))
    assert prep.mean_[0] == 2 and prep.median_[1] == 8
    assert result.categorical[0, 0] == 0 and result.numeric[0, 0] == 997


def test_archived_recency_half_life_preserves_relative_weight():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-01"], utc=True))
    weights = legacy_core.time_decay_weights(dates, dates.iloc[-1], 30.)
    assert weights.mean() == pytest.approx(1.)
    assert weights[-1] / weights[0] == pytest.approx(4.)
    np.testing.assert_array_equal(legacy_core.time_decay_weights(dates, dates.iloc[-1], 0), np.ones(3))


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_each_actual_backbone_trains_predicts_and_roundtrips(tmp_path, stage):
    x, vx = features(), features(6, 9)
    d = np.linspace(12., 650., len(x)); e = np.arange(len(x)) % 3 != 0
    vd, ve = np.linspace(20., 600., len(vx)), np.arange(len(vx)) % 2
    config = tiny_config()
    config.horizon_hours = (504., 168., 72.)[stage]
    model = StageEstimator(stage, config, epochs=3, batch_size=8, seed=14).fit(x, d, e, validation=(vx, vd, ve), cardinalities=[3, 3])
    result = model.predict(vx, [t for t in (0, 24, 72, 168, 504) if t <= config.horizon_hours])
    assert np.isfinite(result["survival"]).all()
    np.testing.assert_allclose(result["survival"][:, 0], 1, atol=1e-6)
    assert (np.diff(result["survival"], axis=1) <= 1e-6).all()
    expected = result["slow_combined"] if stage == 0 else 1-result["slow_combined"]
    np.testing.assert_allclose(result["score"], expected)
    assert model.best_epoch_ == min(model.history_, key=lambda r: r["validation_objective"])["epoch"]
    evaluation = model.evaluate(vx, vd, ve)
    assert np.isfinite(evaluation["survival_nll"])
    assert evaluation["known_rows"] + evaluation["unresolved_rows"] == len(vx)
    path = tmp_path / "local_model.pt"
    model.save(path)
    restored = StageEstimator.load(path)
    np.testing.assert_array_equal(restored.predict(vx)["score"], model.predict(vx)["score"])
    assert model.predict(vx.take(slice(0, 0)))["score"].shape == (0,)
    assert model.predict(vx)["times"][-1] == config.horizon_hours


def test_historical_search_helpers_execute_and_freeze_only_known_fields():
    from types import SimpleNamespace

    class Trial:
        def suggest_float(self, name, low, high, **kwargs):
            return low

        def suggest_int(self, name, low, high, **kwargs):
            return low

        def suggest_categorical(self, name, values):
            return values[0]

    config, trial = tiny_config(), Trial()
    search_core._suggest_phase_A(SimpleNamespace(half_life_min=14., half_life_max=25.), trial, config, 4)
    assert config.half_life_days == 14.
    search_core._suggest_phase_B(trial, config)
    assert config.tail_bce_weight == 0
    search_core._suggest_phase_C(trial, config, search_core._baseline_arch_choices(config))
    assert config.d_model == 16
    search_core._apply_frozen_params(config, {"d_model": 24, "not_a_config_field": 99})
    assert config.d_model == 24 and not hasattr(config, "not_a_config_field")
    assert 192 in search_core._safe_arch_choices(config)["d_model"]
    assert 384 in search_core._wide_arch_choices(config)["d_model"]


@pytest.mark.parametrize("method", sorted(ProbabilityEnsemble.METHODS))
def test_ensemble_fit_excludes_nonfit_rows_and_predict_does_not_refit(method):
    p = np.array([[.1, .9, .2, .8, .3, .7, .4, .6], [.2, .8, .1, .9, .4, .6, .3, .7]])
    duration = np.array([200., 20., 200., 20., 200., 20., 200., 20.])
    event = np.ones(8)
    mask = np.arange(8) < 6
    kwargs = dict(method=method, calibration="temperature", regularization=.1)
    a = ProbabilityEnsemble(**kwargs).fit(p, duration, event, horizon=72, fit_mask=mask)
    changed = p.copy(); changed[:, ~mask] = 1-changed[:, ~mask]
    b = ProbabilityEnsemble(**kwargs).fit(changed, duration, event, horizon=72, fit_mask=mask)
    np.testing.assert_array_equal(a.predict(p), b.predict(p))
    assert a.fit_rows_ == 6
    before = a.predict(p)
    a.predict(changed)
    np.testing.assert_array_equal(a.predict(p), before)


def test_routing_order_boundaries_and_missing_unreached_scores():
    policy = RoutingPolicy(.7, .6, .8)
    out = route_scores([.7, .2, .1, .1], [np.nan, .59, .6, .6], [np.nan, np.nan, .8, .79], policy)
    np.testing.assert_array_equal(out["bucket"], ["TAIL_21PLUS", "SLOW_168PLUS", "FAST_72H", "MID_72_168H"])
    with pytest.raises(ValueError, match="required"):
        route_scores([.1], [np.nan], [np.nan], policy)
    with pytest.raises(ValueError, match="required"):
        route_scores([.1], [.8], [np.nan], policy)


def test_full_cascade_fits_both_downstream_train_caps_and_routes_test(tmp_path):
    x = features(32)
    duration = np.tile([24., 60., 100., 200., 480., 650., 80., 300.], 4)
    event = np.tile([1, 1, 1, 1, 1, 0, 0, 0], 4)
    stages = {s: [StageEstimator(s, tiny_config(), seed=s+2, epochs=1, batch_size=8)] for s in (0, 1, 2)}
    model = CascadeEstimator(stages).fit(x.take(slice(0, 24)), duration[:24], event[:24],
                                         validation=(x.take(slice(24, None)), duration[24:], event[24:]),
                                         policy=RoutingPolicy(.999, 0., .5), cardinalities=[3, 3])
    assert model.fit_audit_[0]["train_rows"] == 24
    assert model.fit_audit_[1]["train_rows"] == model.fit_audit_[2]["train_rows"] == 21
    assert model.fit_audit_[1]["development_rows"] == model.fit_audit_[2]["development_rows"] == 8
    result = model.predict(features(5, 12))
    assert len(result["bucket"]) == 5 and result["reached_stage2"].all()
    model.save(tmp_path / "local_bundle")
    with pytest.raises(ValueError, match="trusted"):
        CascadeEstimator.load(tmp_path / "local_bundle")
    restored = CascadeEstimator.load(tmp_path / "local_bundle", trusted=True)
    replay = restored.predict(features(5, 12))
    np.testing.assert_array_equal(result["bucket"], replay["bucket"])
    np.testing.assert_array_equal(result["p_fast72"], replay["p_fast72"])


def test_policy_feasibility_is_not_silently_relaxed():
    result = select_stage_threshold([.9, .8, .1], [24., 50., 300.], [1, 1, 1], 2, min_precision=1)
    assert result["precision"] == 1 and result["recall"] == 1
    with pytest.raises(ValueError, match="No feasible"):
        select_stage_threshold([.9, .8, .1], [300., 400., 500.], [1, 1, 1], 2, min_precision=1)


def test_ensemble_selection_is_development_only():
    p = np.array([[.1, .9, .2, .8], [.2, .8, .1, .9]])
    d, e = np.array([200., 20., 200., 20.]), np.ones(4)
    model, report = choose_ensemble([{"method": "mean_prob"}, {"method": "mean_logit"}], p, d, e, p, d, e, horizon=72)
    assert report["selection_scope"] == "outer development only"
    assert np.isfinite(model.predict(p)).all()


@pytest.mark.parametrize("change", [
    {"outer_event": [2, 2, 2, 2]},
    {"outer_duration": [np.nan, 20., 200., 20.]},
    {"outer_duration": [200., 20.]},
    {"outer_probability": [[np.nan, .9, .2, .8]]},
    {"horizon": np.nan},
    {"positive": "misspelled"},
])
def test_ensemble_selection_rejects_invalid_outer_outcomes(change):
    p = np.array([[.1, .9, .2, .8]])
    kwargs = dict(candidates=[{"method": "mean_prob"}], inner_probability=p,
                  inner_duration=[200., 20., 200., 20.], inner_event=np.ones(4),
                  outer_probability=p, outer_duration=[200., 20., 200., 20.],
                  outer_event=np.ones(4), horizon=72.)
    kwargs.update(change)
    with pytest.raises(ValueError):
        choose_ensemble(**kwargs)


def test_invalid_explicit_configuration_fails_before_fitting():
    for control in ("learning_rate", "weight_decay", "epochs", "batch_size"):
        with pytest.raises(ValueError):
            StageEstimator(0, **{control: np.nan})
    for control in ("horizon_hours", "n_bins", "dropout", "max_grad_norm"):
        config = tiny_config()
        setattr(config, control, np.nan)
        with pytest.raises(ValueError):
            StageEstimator(0, config)
    for control in ("kappa", "trim_ratio", "regularization", "logistic_c"):
        with pytest.raises(ValueError):
            ProbabilityEnsemble(**{control: np.nan})
    with pytest.raises(ValueError):
        CascadeEstimator(stages={})
    with pytest.raises(ValueError):
        CascadeEstimator(ensembles={})
