"""Shared fixtures for the dltmle test suite."""
import numpy as np
import pytest
import dltmle


@pytest.fixture(scope="session")
def dgp_data():
    """Small dataset (n=1000, tau=5) used across most tests."""
    rng = np.random.default_rng(0)
    return dltmle.example_dgp(rng, n=1000, tau=5)


@pytest.fixture(scope="session")
def true_psi_1():
    """Oracle counterfactual mean E[Y_tau | do(A=1)] by large-n simulation."""
    _, _, _, _, Y = dltmle.example_dgp(np.random.default_rng(1), n=500_000, tau=5, a_cf=1.0)
    return float(Y[:, -1].mean())


@pytest.fixture(scope="session")
def true_psi_0():
    """Oracle counterfactual mean E[Y_tau | do(A=0)] by large-n simulation."""
    _, _, _, _, Y = dltmle.example_dgp(np.random.default_rng(2), n=500_000, tau=5, a_cf=0.0)
    return float(Y[:, -1].mean())


# Fast hparams for DeepLTMLE (small model, few epochs — suitable for tests)
FAST_HPARAMS = {
    "dim_model":    16,
    "num_layers":   1,
    "nhead":        2,
    "dropout":      0.0,
    "learning_rate": 1e-3,
    "alpha":        1.0,
    "beta":         1.0,
    "ema_decay":    0.99,
    "max_epochs":   50,
    "batch_size":   128,
}

# Fast hparams for LinearTMLE
LINEAR_HPARAMS = {
    "learning_rate": 1e-2,
    "alpha":         1.0,
    "beta":          1.0,
    "max_epochs":    200,
    "batch_size":    256,
}
