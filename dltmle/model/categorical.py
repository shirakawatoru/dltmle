"""Helpers for categorical (multi-class) treatment / censoring nodes.

Conventions
-----------
* A categorical node with ``m`` classes is parametrised with ``m - 1`` logits
  against the reference class 0 (multinomial logistic / softmax with a fixed
  zero logit for class 0).  For ``m == 2`` this is *exactly* the previous
  binary parametrisation ``sigmoid(Linear(z))`` (same parameter shapes, same
  initialisation, same numerics), which keeps binary results backward
  compatible.
* Interventions are represented as ``gstar`` tensors of shape ``(..., J)``
  holding the intervention probabilities ``g*(a | history)`` per class.  A
  static or deterministic dynamic regime is a one-hot ``gstar``.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit, logit


def reference_softmax(z: torch.Tensor) -> torch.Tensor:
    """Class probabilities from ``m - 1`` reference-coded logits.

    Parameters
    ----------
    z : (..., m - 1) logits of classes 1..m-1 vs class 0.

    Returns
    -------
    (..., m) probabilities.
    """
    if z.shape[-1] == 1:
        # binary: identical numerics to the legacy sigmoid head
        p1 = torch.sigmoid(z)
        return torch.cat([1 - p1, p1], dim=-1)
    zeros = torch.zeros_like(z[..., :1])
    return torch.softmax(torch.cat([zeros, z], dim=-1), dim=-1)


def categorical_nll(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise negative log-likelihood of class ``target`` under ``p``.

    Parameters
    ----------
    p : (B, T, m) class probabilities.
    target : (B, T, 1) class indices (float or long).

    Returns
    -------
    (B, T, 1)
    """
    m = p.shape[-1]
    if m == 2:
        # binary: legacy BCE (bit-identical to the previous implementation)
        return F.binary_cross_entropy(p[..., 1:2], target.float(), reduction="none")
    onehot = F.one_hot(target.long().squeeze(-1), m).to(p.dtype)
    # one-hot product instead of gather → deterministic backward on CUDA
    logp = torch.log(p).clamp(min=-100)
    return -(logp * onehot).sum(dim=-1, keepdim=True)


def onehot_drop_first(x: torch.Tensor, m: int) -> torch.Tensor:
    """One-hot encode class indices dropping the reference class 0.

    ``x`` : (B, T, 1) → (B, T, m - 1).  For ``m == 2`` this equals ``x``.
    """
    return F.one_hot(x.long().squeeze(-1), m)[..., 1:].to(torch.get_default_dtype())


def is_deterministic(gstar) -> bool:
    """True if every ``g*`` row is one-hot (static / deterministic dynamic)."""
    return bool(((gstar == 0) | (gstar == 1)).all())


def mix_logits(logit_cls: torch.Tensor, gstar: torch.Tensor) -> torch.Tensor:
    """Logit of ``sum_j g*_j sigmoid(logit_cls_j)``.

    For a one-hot ``gstar`` this is the selected class logit (exactly the
    legacy computation).  For a stochastic ``gstar`` the mixture is taken on
    the probability scale, i.e. ``V = E_{a ~ g*}[Q(a)]``.

    Parameters
    ----------
    logit_cls : (B, T, J)
    gstar : (B, T, J)

    Returns
    -------
    (B, T, 1)
    """
    if is_deterministic(gstar):
        return (logit_cls * gstar).sum(dim=-1, keepdim=True)
    p = (torch.sigmoid(logit_cls) * gstar).sum(dim=-1, keepdim=True)
    return torch.logit(p.clamp(1e-7, 1 - 1e-7))


def mix_prob_np(logit_cls: np.ndarray, gstar: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`mix_logits` returning probabilities ``(..., )``."""
    if is_deterministic(gstar):
        return expit((logit_cls * gstar).sum(axis=-1))
    return (expit(logit_cls) * gstar).sum(axis=-1)


CENSORING_TIMINGS = ("after_treatment", "before_treatment")


def check_censoring_timing(timing: str) -> str:
    if timing not in CENSORING_TIMINGS:
        raise ValueError(f"censoring_timing must be one of {CENSORING_TIMINGS}, got {timing!r}")
    return timing
