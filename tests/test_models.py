"""Analytical survival contracts and small, offline estimator integrations."""
import importlib.util

import numpy as np
import pytest
import torch

from marketneural.models import create_model, flatten_features
from marketneural.neural import _log_survival_at, survival_nll


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def multimodal(n=24):
    rng = np.random.default_rng(18)
    return {"tabular": rng.normal(size=(n, 3)).astype(np.float32),
            "text": rng.normal(size=(n, 4)).astype(np.float32),
            "image": rng.normal(size=(n, 2, 3)).astype(np.float32),
            "report": rng.normal(size=(n, 2, 4)).astype(np.float32),
            "image_mask": np.tile([1, 0], (n, 1)).astype(bool),
            "report_mask": np.tile([0, 1], (n, 1)).astype(bool)}


def test_kaplan_meier_ties_censoring_and_boundaries():
    x = {"tabular": np.ones((4, 1))}
    model = create_model("km").fit(x, [1, 2, 2, 3], [1, 1, 0, 1])
    # Both event and censor at t=2 are in its risk set of three.
    result = model.predict_survival(x, [0, 1, 1.5, 2, 3, 9])
    np.testing.assert_allclose(result, np.tile([1, .75, .75, .5, 0, 0], (4, 1)))
    censored = create_model("km").fit(x, [1, 2, 2, 3], [0, 0, 0, 0])
    np.testing.assert_array_equal(censored.predict_survival(x, [0, 100]), 1)


def test_flatten_retains_slots_and_explicit_missingness():
    x = {"tabular": np.array([[2]]), "image": np.array([[[3, 4], [9, 8]]]),
         "image_mask": np.array([[True, False]])}
    np.testing.assert_array_equal(flatten_features(x), [[2, 3, 4, 0, 0, 1, 0]])


def test_survival_likelihood_matches_hand_calculation():
    # Hazards .2 and .5, mixture has one expert; S(1)=.8, S(2)=.4.
    hazards = torch.tensor([[[.2, .5]]], dtype=torch.float64)
    logits = torch.logit(hazards).repeat(4, 1, 1).requires_grad_()
    edges = torch.tensor([0., 1., 2.], dtype=torch.float64)
    duration = torch.tensor([1., 2., 1.5, 9.], dtype=torch.float64)
    event = torch.tensor([True, True, False, True])
    loss = survival_nll(logits, torch.zeros(4, 1), duration, event, edges)
    # Event after horizon is censored at the horizon, not an event in last bin.
    expected = -np.log([.2, .8 * .5, .8 * np.sqrt(.5), .8 * .5]).mean()
    np.testing.assert_allclose(loss.detach(), expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    s = _log_survival_at(logits[:1], torch.tensor([0., 1., 1.5, 2.], dtype=torch.float64), edges).exp()
    np.testing.assert_allclose(s.detach(), [[[1, .8, .8 * np.sqrt(.5), .4]]])


def test_mixture_likelihood_and_extreme_logits_are_stable():
    logits = torch.tensor([[[0.], [np.log(3.)]]], dtype=torch.float64, requires_grad=True)
    mix = torch.tensor([[np.log(.25), np.log(.75)]], dtype=torch.float64)
    loss = survival_nll(logits, mix, torch.tensor([1.]), torch.tensor([True]), torch.tensor([0., 1.]))
    np.testing.assert_allclose(loss.detach(), -np.log(.25 * .5 + .75 * .75))
    extreme = torch.tensor([[[1000., -1000.]]], requires_grad=True)
    stable = survival_nll(extreme, torch.zeros(1, 1), torch.tensor([1.5]),
                          torch.tensor([False]), torch.tensor([0., 1., 2.]))
    stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()


@pytest.mark.parametrize("name", ["mlp", "perceiver_moe"])
def test_neural_survival_contract_masking_and_reproducibility(name):
    x = multimodal()
    duration = np.linspace(.2, 9, 24)
    event = np.arange(24) % 3 != 0
    options = dict(random_state=9, horizon=10, n_bins=5, epochs=3, width=8,
                   heads=2, n_latents=3, layers=1, experts=2, batch_size=12)
    a = create_model(name, **options).fit(x, duration, event)
    b = create_model(name, **options).fit(x, duration, event)
    times = np.array([0, 1, 3, 7, 10])
    predictions = a.predict_survival(x, times)
    assert predictions.shape == (24, 5)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1 + 1e-7)).all()
    assert (np.diff(predictions, axis=1) <= 1e-7).all()
    np.testing.assert_allclose(predictions[:, 0], 1, atol=1e-7)
    np.testing.assert_array_equal(predictions, b.predict_survival(x, times))
    masked_changed = {k: v.copy() for k, v in x.items()}
    masked_changed["image"][:, 1] = 9999
    masked_changed["report"][:, 0] = -9999
    np.testing.assert_array_equal(predictions, a.predict_survival(masked_changed, times))
    assert len(a.history_) == 3  # No SVAL: fixed epochs, no train-loss early stopping.
    with pytest.raises(ValueError, match="horizon"):
        a.predict_survival(x, [11])
    empty = {k: v[:0] for k, v in x.items()}
    assert a.predict_survival(empty, times).shape == (0, 5)


def test_mlp_learns_time_ordering_and_selects_validation_epoch():
    x = {"tabular": np.repeat(np.array([[-1.], [1.]], dtype=np.float32), 24, axis=0)}
    duration = np.repeat([1., 8.], 24)
    event = np.ones(48, dtype=bool)
    model = create_model("mlp", horizon=10, n_bins=10, epochs=30, patience=5,
                         width=16, learning_rate=.03, dropout=0, batch_size=48)
    # Independent validation rows, same synthetic mechanism.
    vx = {"tabular": np.array([[-1.], [1.]], dtype=np.float32)}
    model.fit(x, duration, event, validation=(vx, np.array([1., 8.]), np.ones(2, dtype=bool)))
    prediction = model.predict_survival(vx, [3])[:, 0]
    assert prediction[0] < .2 and prediction[1] > .8
    best = min(range(len(model.history_)), key=lambda i: model.history_[i]["validation_nll"])
    assert model.best_epoch_ == best + 1
    assert model.history_[-1]["train_nll"] < model.history_[0]["train_nll"]


@pytest.mark.skipif(importlib.util.find_spec("sksurv") is None, reason="scikit-survival optional dependency")
@pytest.mark.parametrize("name", ["coxph", "rsf", "gbsa"])
def test_classical_estimator_integration(name):
    rng = np.random.default_rng(11)
    x = {"tabular": rng.normal(size=(40, 2)).astype(np.float32)}
    duration = np.exp(2 - .7 * x["tabular"][:, 0])
    event = np.arange(40) % 4 != 0
    kwargs = {"n_estimators": 8} if name != "coxph" else {}
    model = create_model(name, **kwargs).fit(x, duration, event)
    s = model.predict_survival(x, [0, 2, 5, 10])
    assert s.shape == (40, 4) and np.isfinite(s).all()
    assert ((s >= 0) & (s <= 1)).all() and (np.diff(s, axis=1) <= 1e-12).all()


def test_invalid_outcomes_and_contract_changes_fail():
    x = {"tabular": np.ones((2, 1))}
    with pytest.raises(ValueError, match="nonnegative"):
        create_model("km").fit(x, [-1, 2], [1, 0])
    with pytest.raises(ValueError, match="binary"):
        create_model("km").fit(x, [1, 2], [1, 2])
    with pytest.raises(ValueError, match="binary"):
        flatten_features({**x, "image": np.ones((2, 1, 1)), "image_mask": np.full((2, 1), np.nan)})
    model = create_model("km").fit(x, [1, 2], [1, 0])
    with pytest.raises(ValueError, match="contract"):
        model.predict_survival({"tabular": np.ones((2, 2))}, [1])
