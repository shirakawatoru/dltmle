"""Categorical treatment (J=3) and categorical censoring (K=3)."""
import numpy as np
import pytest
import torch

import dltmle
from dltmle import Static, Dynamic, Stochastic
from dltmle.intervention import resolve_intervention
from dltmle.model.categorical import mix_logits, reference_softmax
from dltmle.model.dltmle import DeepLTMLE

from .conftest import (
    CAT_STOCHASTIC, LINEAR_CAT_HPARAMS, TINY_DEEP_HPARAMS, cat_dynamic_est,
)

INTERVENTIONS = {
    "s0": 0, "s1": 1, "s2": 2,
    "dyn": Dynamic(cat_dynamic_est),
    "sto": Stochastic(CAT_STOCHASTIC),
}


@pytest.fixture(scope="module")
def linear_result(cat_data):
    W, L, A, C, Y = cat_data
    return dltmle.estimate(W, L, A, C, Y, INTERVENTIONS, estimator="linear",
                           hparams=LINEAR_CAT_HPARAMS, device="cpu")


# ---------------------------------------------------------------------------
# Quality: close to the truth (G correctly specified for the linear model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(INTERVENTIONS))
def test_linear_within_3se_of_truth(linear_result, cat_truth, name):
    r = linear_result.ral(name)
    err = abs(r.est - cat_truth[name])
    assert err < 3 * r.se, f"{name}: est={r.est:.4f} true={cat_truth[name]:.4f} se={r.se:.4f}"


def test_linear_no_warnings_and_calibrated(linear_result):
    assert linear_result.diagnostics["warnings"] == []
    prop = linear_result.diagnostics["propensity"]
    np.testing.assert_allclose(prop["treatment_mean_predicted"], prop["treatment_observed_freq"], atol=0.02)
    np.testing.assert_allclose(prop["censoring_mean_predicted"], prop["censoring_observed_freq"], atol=0.01)
    assert len(prop["censoring_observed_freq"]) == 3


def test_before_treatment_timing_agrees(cat_data, linear_result, cat_truth):
    """P(C=0|H) g(A|H,C=0) = g(A|H) P(C=0|H,A): both orderings target the same
    parameter when A_t is observed on censoring rows."""
    W, L, A, C, Y = cat_data
    r = dltmle.estimate(W, L, A, C, Y, {"s2": 2, "sto": INTERVENTIONS["sto"]},
                        estimator="linear", hparams=LINEAR_CAT_HPARAMS, device="cpu",
                        censoring_timing="before_treatment")
    for name in ("s2", "sto"):
        assert abs(r.ral(name).est - linear_result.ral(name).est) < 0.5 * linear_result.ral(name).se
        assert abs(r.ral(name).est - cat_truth[name]) < 3 * r.ral(name).se


def test_deep_categorical_within_3se(cat_data, cat_truth):
    W, L, A, C, Y = cat_data
    hp = dict(TINY_DEEP_HPARAMS, max_epochs=15)
    r = dltmle.estimate(W, L, A, C, Y, {"s2": 2, "sto": INTERVENTIONS["sto"]},
                        hparams=hp, seed=0, device="cpu")
    for name in ("s2", "sto"):
        est = r.ral(name)
        assert abs(est.est - cat_truth[name]) < 3 * est.se, (name, est.est, cat_truth[name], est.se)


def test_ontarget_shape_j4_k5():
    """4 drug classes, 5 censoring categories (none + 4 reasons)."""
    W, L, A, C, Y = dltmle.example_dgp_categorical(
        np.random.default_rng(1), n=800, tau=3, n_treatment=4, n_censoring=5)
    levels = ["SU", "DPP4", "SGLT2i", "GLP1"]
    cens = ["none", "non_cvd_death", "unknown_death", "administrative", "disenrollment"]
    r = dltmle.estimate(W, L, A, C, Y, {lv: lv for lv in levels}, estimator="linear",
                        hparams={"max_epochs": 10}, treatment_levels=levels,
                        censoring_levels=cens, n_censoring=5, device="cpu", eval_times="all")
    assert r.n_treatment == 4 and r.n_censoring == 5
    assert len(r.estimates) == 4 * 3
    assert np.all(np.isfinite(r.estimates["psi"]))
    assert len(r.diagnostics["propensity"]["censoring_mean_predicted"]) == 5
    c = r.contrasts("SU")
    assert set(c["intervention"]) == {"DPP4", "SGLT2i", "GLP1"}


# ---------------------------------------------------------------------------
# Unit tests: interventions, clever covariate, mixtures
# ---------------------------------------------------------------------------

def test_resolve_intervention_forms(cat_data):
    W, L, A, C, Y = cat_data
    n, tau = A.shape[:2]
    g = resolve_intervention(2, W, L, A, 3)
    assert g.shape == (n, tau, 3) and np.all(g[..., 2] == 1)
    g = resolve_intervention(Static([0, 1, 2]), W, L, A, 3)
    np.testing.assert_array_equal(g[0].argmax(-1), [0, 1, 2])
    g = resolve_intervention(Stochastic(CAT_STOCHASTIC), W, L, A, 3)
    np.testing.assert_allclose(g[5, 1], CAT_STOCHASTIC)
    g = resolve_intervention(cat_dynamic_est, W, L, A, 3)
    np.testing.assert_array_equal(g[..., 2] == 1, cat_dynamic_est(W, L, A) == 2)
    g = resolve_intervention("b", W, L, A, 3, levels=["a", "b", "c"])
    assert np.all(g[..., 1] == 1)
    with pytest.raises(ValueError):
        resolve_intervention(3, W, L, A, 3)
    with pytest.raises(ValueError):
        resolve_intervention(Stochastic([0.5, 0.6, 0.1]), W, L, A, 3)


def test_clever_weights_hand_computed():
    p_A = torch.tensor([[[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]])     # (1, 2, 3)
    p_C = torch.tensor([[[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]])
    A = torch.tensor([[[1.0], [1.0]]])
    C = torch.tensor([[[0.0], [0.0]]])
    gstar = torch.tensor([[[0.0, 1.0, 0.0], [0.2, 0.5, 0.3]]])
    m = DeepLTMLE.from_models(None, None, tau=2)
    g = m.clever_weights(p_A, p_C, A, C, gstar)[0, :, 0]
    w1 = (1.0 / 0.3) / 0.9
    w2 = w1 * (0.5 / 0.8) / 0.8
    np.testing.assert_allclose(g.numpy(), [1.0, w1, w2], rtol=1e-6)
    C2 = torch.tensor([[[0.0], [2.0]]])             # censored (reason 2) at t=1
    g2 = m.clever_weights(p_A, p_C, A, C2, gstar)[0, :, 0]
    np.testing.assert_allclose(g2.numpy(), [1.0, w1, 0.0], rtol=1e-6)


def test_mix_logits():
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    onehot = torch.tensor([[[0.0, 1.0, 0.0]]])
    assert mix_logits(logits, onehot).item() == 1.0
    probs = torch.tensor([[[0.2, 0.3, 0.5]]])
    expected = (torch.sigmoid(logits) * probs).sum()
    assert torch.sigmoid(mix_logits(logits, probs)).item() == pytest.approx(expected.item(), rel=1e-6)


def test_reference_softmax_binary_is_sigmoid():
    z = torch.randn(4, 3, 1)
    p = reference_softmax(z)
    torch.testing.assert_close(p[..., 1:], torch.sigmoid(z))
    p3 = reference_softmax(torch.randn(4, 3, 2))
    torch.testing.assert_close(p3.sum(-1), torch.ones(4, 3))
