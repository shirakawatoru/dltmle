"""Tests for dltmle.fit (DeepLTMLE estimator)."""
import numpy as np
import pytest
import dltmle
from dltmle.ral import RALEstimate

from .conftest import FAST_HPARAMS


# ---------------------------------------------------------------------------
# Smoke tests — check return type and basic sanity
# ---------------------------------------------------------------------------

def test_fit_returns_ral_estimate(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit(0, FAST_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert isinstance(result, RALEstimate)


def test_fit_estimate_is_probability(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit(0, FAST_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert 0.0 < result.est < 1.0


def test_fit_se_is_positive(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit(0, FAST_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert result.se > 0.0


# ---------------------------------------------------------------------------
# Quality test — estimate should be within 3 SEs of the oracle
# ---------------------------------------------------------------------------

def test_fit_estimate_a1_within_3se(dgp_data, true_psi_1):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit(0, FAST_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    err = abs(result.est - true_psi_1)
    assert err < 3 * result.se, (
        f"est={result.est:.4f}, true={true_psi_1:.4f}, "
        f"se={result.se:.4f}, |err|/se={err/result.se:.2f}"
    )


def test_fit_estimate_a0_within_3se(dgp_data, true_psi_0):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit(1, FAST_HPARAMS, W, L, A, C, Y, a=np.zeros_like(A))
    err = abs(result.est - true_psi_0)
    assert err < 3 * result.se, (
        f"est={result.est:.4f}, true={true_psi_0:.4f}, "
        f"se={result.se:.4f}, |err|/se={err/result.se:.2f}"
    )


# ---------------------------------------------------------------------------
# RALEstimate arithmetic
# ---------------------------------------------------------------------------

def test_fit_ate_ral_arithmetic(dgp_data, true_psi_1, true_psi_0):
    """ATE = psi_1 - psi_0 should contain the true ATE in its 95% CI."""
    W, L, A, C, Y = dgp_data
    psi_1 = dltmle.fit(0, FAST_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    psi_0 = dltmle.fit(1, FAST_HPARAMS, W, L, A, C, Y, a=np.zeros_like(A))
    ate = psi_1 - psi_0
    true_ate = true_psi_1 - true_psi_0
    assert isinstance(ate, RALEstimate)
    assert abs(ate.est - true_ate) < 3 * ate.se
