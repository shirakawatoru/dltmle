"""Backward compatibility of the binary (J=2, K=2) special case.

Golden values were produced by the ``dev-seperateGQ`` implementation
(commit 9675d21, CPU) on the same data / seed / hparams.  Point estimates
must match to numerical precision.  SEs differ slightly by design: the
influence curve is now centred at the full-sample estimate instead of the
per-batch mean (bug fix), so SEs are compared with a 2% relative tolerance.
"""
import numpy as np
import pytest

import dltmle
from dltmle import Static, Stochastic

GOLDEN_HPARAMS = {"dim_model": 16, "num_layers": 1, "nhead": 2, "dropout": 0.0,
                  "learning_rate": 1e-3, "alpha": 1.0, "beta": 1.0, "ema_decay": 0.99,
                  "max_epochs": 5, "batch_size": 64}
GOLDEN_LINEAR_HPARAMS = {"learning_rate": 1e-2, "alpha": 1.0, "beta": 1.0,
                         "max_epochs": 20, "batch_size": 64}
# (estimator, a) -> (est, se) from dev-seperateGQ @ 9675d21
GOLDEN = {
    ("deep", 1.0):   (0.009726051241159439, 0.035859037190675735),
    ("linear", 1.0): (0.00808076560497284, 0.032033417373895645),
    ("deep", 0.0):   (0.09775371849536896, 0.299314022064209),
    ("linear", 0.0): (0.03934136778116226, 0.30487897992134094),
}


@pytest.fixture(scope="module")
def golden_data():
    return dltmle.example_dgp(np.random.default_rng(123), n=400, tau=4)


@pytest.mark.parametrize("kind,a", list(GOLDEN))
def test_matches_previous_implementation(golden_data, kind, a):
    W, L, A, C, Y = golden_data
    f = dltmle.fit if kind == "deep" else dltmle.fit_linear
    hp = GOLDEN_HPARAMS if kind == "deep" else GOLDEN_LINEAR_HPARAMS
    r = f(7, hp, W, L, A, C, Y, a=np.full_like(A, a), device="cpu", verbose=False)
    est, se = GOLDEN[(kind, a)]
    assert r.est == pytest.approx(est, abs=1e-6)
    assert r.se == pytest.approx(se, rel=0.02)


@pytest.fixture(scope="module")
def small_binary():
    return dltmle.example_dgp(np.random.default_rng(5), n=300, tau=3)


@pytest.mark.parametrize("estimator", ["dltmle", "linear"])
def test_binary_equals_categorical_j2_k2(small_binary, estimator):
    """Legacy float-binary call == explicit categorical call with J=K=2."""
    W, L, A, C, Y = small_binary
    hp = dict(GOLDEN_HPARAMS if estimator == "dltmle" else GOLDEN_LINEAR_HPARAMS, max_epochs=3)
    f = dltmle.fit if estimator == "dltmle" else dltmle.fit_linear
    legacy = f(3, hp, W, L, A, C, Y, a=np.ones_like(A), device="cpu", verbose=False)

    gstar = np.zeros(A.shape[:2] + (2,)); gstar[..., 1] = 1.0
    for spec in (Static(1), 1, gstar, {"type": "static", "level": 1}):
        new = dltmle.estimate(
            W, L, A.astype(np.int64), C.astype(np.int64), Y, spec,
            hparams={**hp, "init_bias": False}, seed=3, estimator=estimator,
            n_treatment=2, n_censoring=2, device="cpu",
        )
        assert new.psi == legacy.est
        np.testing.assert_array_equal(new.ral().ic, legacy.ic)


def test_legacy_fractional_a_is_stochastic(small_binary):
    """a in (0,1) (legacy binary stochastic intervention) == Stochastic([1-p, p])."""
    W, L, A, C, Y = small_binary
    hp = dict(GOLDEN_LINEAR_HPARAMS, max_epochs=3)
    legacy = dltmle.fit_linear(0, hp, W, L, A, C, Y, a=np.full_like(A, 0.3),
                               device="cpu", verbose=False)
    new = dltmle.estimate(W, L, A, C, Y, Stochastic([0.7, 0.3]), estimator="linear",
                          hparams={**hp, "init_bias": False}, device="cpu")
    assert new.psi == pytest.approx(legacy.est, abs=1e-12)


def test_fit_returns_diagnostics(small_binary):
    W, L, A, C, Y = small_binary
    r = dltmle.fit_linear(0, dict(GOLDEN_LINEAR_HPARAMS, max_epochs=2), W, L, A, C, Y,
                          a=np.ones_like(A), device="cpu", verbose=False)
    assert {"targeting", "weights", "propensity"} <= set(r.diagnostics)


def test_tune_smoke(small_binary):
    W, L, A, C, Y = small_binary
    cand = {k: [v] for k, v in dict(GOLDEN_HPARAMS, max_epochs=1).items()}
    best = dltmle.tune(0, cand, W, L, A, C, Y, n_trials=1, log_hparams=False)
    assert best["dim_model"] == 16
