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


# ---------------------------------------------------------------------------
# Categorical treatment / censoring fixtures
# ---------------------------------------------------------------------------

import torch

# tiny models: per-op threading overhead dominates; keep runs fast and stable
torch.set_num_threads(min(4, torch.get_num_threads()))

CAT_TAU = 3
CAT_N = 3000


def cat_dynamic_dgp(t, W, L_hist, A_hist):
    """Dynamic regime used in the DGP: class 2 if L_0 > 0 else class 0."""
    return np.where(L_hist[:, 0] > 0, 2, 0)


def cat_dynamic_est(W, L, A):
    """Same regime evaluated on observed panel arrays."""
    return np.repeat(np.where(L[:, 0, 0] > 0, 2, 0)[:, None], L.shape[1], axis=1)


CAT_STOCHASTIC = [0.2, 0.3, 0.5]


@pytest.fixture(scope="session")
def cat_data():
    """Observational data with J=3 treatment classes and K=3 censoring categories."""
    return dltmle.example_dgp_categorical(np.random.default_rng(0), n=CAT_N, tau=CAT_TAU)


@pytest.fixture(scope="session")
def cat_truth():
    """Oracle P(Y_{tau-1}=1) under each regime (no censoring), by simulation."""
    def _truth(regime):
        Y = dltmle.example_dgp_categorical(
            np.random.default_rng(99), n=300_000, tau=CAT_TAU, regime=regime)[4]
        return float(Y[:, -1].mean())
    return {
        "s0": _truth(0), "s1": _truth(1), "s2": _truth(2),
        "dyn": _truth(cat_dynamic_dgp), "sto": _truth(CAT_STOCHASTIC),
    }


LINEAR_CAT_HPARAMS = {
    "learning_rate": 1e-2,
    "max_epochs":    100,
    "batch_size":    256,
}

TINY_DEEP_HPARAMS = {
    "dim_model":     16,
    "num_layers":    1,
    "nhead":         2,
    "dropout":       0.0,
    "learning_rate": 1e-3,
    "max_epochs":    3,
    "batch_size":    128,
}
