"""Tests for dltmle.fit_linear (LinearTMLE baseline)."""
import numpy as np
import pytest
import dltmle
from dltmle.ral import RALEstimate

from .conftest import LINEAR_HPARAMS


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_fit_linear_returns_ral_estimate(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit_linear(0, LINEAR_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert isinstance(result, RALEstimate)


def test_fit_linear_estimate_is_probability(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit_linear(0, LINEAR_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert 0.0 < result.est < 1.0


def test_fit_linear_se_is_positive(dgp_data):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit_linear(0, LINEAR_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    assert result.se > 0.0


# ---------------------------------------------------------------------------
# Quality tests — linear TMLE should be approximately unbiased for the DGP
# ---------------------------------------------------------------------------

def test_fit_linear_estimate_a1_within_3se(dgp_data, true_psi_1):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit_linear(0, LINEAR_HPARAMS, W, L, A, C, Y, a=np.ones_like(A))
    err = abs(result.est - true_psi_1)
    assert err < 3 * result.se, (
        f"est={result.est:.4f}, true={true_psi_1:.4f}, "
        f"se={result.se:.4f}, |err|/se={err/result.se:.2f}"
    )


def test_fit_linear_estimate_a0_within_3se(dgp_data, true_psi_0):
    W, L, A, C, Y = dgp_data
    result = dltmle.fit_linear(1, LINEAR_HPARAMS, W, L, A, C, Y, a=np.zeros_like(A))
    err = abs(result.est - true_psi_0)
    assert err < 3 * result.se, (
        f"est={result.est:.4f}, true={true_psi_0:.4f}, "
        f"se={result.se:.4f}, |err|/se={err/result.se:.2f}"
    )
