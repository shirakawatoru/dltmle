"""Unit tests for RALEstimate arithmetic (delta method)."""
import numpy as np
import pytest
from dltmle.ral import RALEstimate


def _make(est, ic_seed=0, n=100):
    """Helper: RALEstimate with given point estimate and random IC."""
    rng = np.random.default_rng(ic_seed)
    ic  = rng.normal(0, 0.1, n)
    ic  = ic - ic.mean() + est * 0.1   # centre around est-proportional value
    return RALEstimate(est, ic)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructor():
    r = _make(0.5)
    assert r.est == 0.5
    assert r.n   == 100
    assert r.se  > 0


def test_se_formula():
    n  = 200
    ic = np.ones(n) * 0.5
    r  = RALEstimate(0.3, ic)
    expected_se = np.sqrt((ic ** 2).mean() / n)
    assert np.isclose(r.se, expected_se)


# ---------------------------------------------------------------------------
# Scalar arithmetic
# ---------------------------------------------------------------------------

def test_add_scalar():
    r = _make(0.4)
    r2 = r + 0.1
    assert np.isclose(r2.est, 0.5)
    np.testing.assert_array_equal(r2.ic, r.ic)


def test_radd_scalar():
    r  = _make(0.4)
    r2 = 0.1 + r
    assert np.isclose(r2.est, 0.5)


def test_sub_scalar():
    r  = _make(0.6)
    r2 = r - 0.1
    assert np.isclose(r2.est, 0.5)


def test_rsub_scalar():
    r  = _make(0.3)
    r2 = 1.0 - r
    assert np.isclose(r2.est, 0.7)
    np.testing.assert_array_equal(r2.ic, r.ic)


def test_mul_scalar():
    r  = _make(0.4)
    r2 = r * 2
    assert np.isclose(r2.est, 0.8)
    np.testing.assert_array_equal(r2.ic, r.ic * 2)


def test_rmul_scalar():
    r  = _make(0.4)
    r2 = 2 * r
    assert np.isclose(r2.est, 0.8)


def test_div_scalar():
    r  = _make(0.8)
    r2 = r / 2
    assert np.isclose(r2.est, 0.4)
    np.testing.assert_array_equal(r2.ic, r.ic / 2)


def test_neg():
    r  = _make(0.5)
    r2 = -r
    assert np.isclose(r2.est, -0.5)
    np.testing.assert_array_equal(r2.ic, -r.ic)


# ---------------------------------------------------------------------------
# RALEstimate × RALEstimate (delta method)
# ---------------------------------------------------------------------------

def test_add_ral():
    a = _make(0.3, ic_seed=0)
    b = _make(0.2, ic_seed=1)
    c = a + b
    assert np.isclose(c.est, 0.5)
    np.testing.assert_array_equal(c.ic, a.ic + b.ic)


def test_sub_ral():
    a = _make(0.7, ic_seed=0)
    b = _make(0.2, ic_seed=1)
    c = a - b
    assert np.isclose(c.est, 0.5)
    np.testing.assert_array_equal(c.ic, a.ic - b.ic)


def test_mul_ral_delta_method():
    a = _make(2.0, ic_seed=0)
    b = _make(3.0, ic_seed=1)
    c = a * b
    assert np.isclose(c.est, 6.0)
    # delta method: IC_c = IC_a * b.est + a.est * IC_b
    expected_ic = a.ic * b.est + a.est * b.ic
    np.testing.assert_array_equal(c.ic, expected_ic)


def test_div_ral_delta_method():
    a = _make(6.0, ic_seed=0)
    b = _make(2.0, ic_seed=1)
    c = a / b
    assert np.isclose(c.est, 3.0)
    expected_ic = a.ic / b.est - a.est * b.ic / b.est ** 2
    np.testing.assert_allclose(c.ic, expected_ic)


def test_mismatched_n_raises():
    a = _make(0.5, n=100)
    b = RALEstimate(0.5, np.ones(50))
    with pytest.raises(AssertionError):
        _ = a + b


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_estimate():
    r = _make(0.42)
    s = repr(r)
    assert "0.42" in s or "estimate" in s.lower()
