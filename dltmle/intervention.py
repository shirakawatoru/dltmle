"""Intervention (treatment regime) specifications.

Every intervention is resolved to a ``gstar`` array of shape ``(n, tau, J)``
with ``gstar[i, t, j] = g*(A_t = j | observed history of subject i up to t)``.

* :class:`Static`     — a fixed class for all t, or one class per time point.
* :class:`Dynamic`    — a deterministic function of the observed history
  ``fn(W, L, A) -> (n, tau)`` class indices (``A`` is the *observed*
  treatment; at time t the function must only use ``L[:, :t+1]`` and
  ``A[:, :t]``).
* :class:`Stochastic` — class probabilities: constant ``(J,)``, per time
  ``(tau, J)``, per subject-time ``(n, tau, J)``, or a function
  ``fn(W, L, A) -> (n, tau, J)``.

Evaluating dynamic / stochastic regimes on the observed history is what the
sequential-regression (ICE) TMLE needs: only subjects whose observed past is
compatible with the regime carry weight in the clever covariate.

Plain values are accepted too (see :func:`resolve_intervention`); JSON-style
dicts (``{"type": "static", "level": 1}``, ``{"type": "stochastic",
"probs": [...]}``) are what the CLI / R wrapper use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


def _level_index(level, levels):
    """Map a treatment label (or index) to its class index."""
    if levels is not None:
        levels = list(levels)
        if level in levels:
            return levels.index(level)
        # allow numeric strings / numpy scalars matching labels
        for j, lv in enumerate(levels):
            if str(lv) == str(level):
                return j
    if isinstance(level, (int, np.integer)) or (
        isinstance(level, (float, np.floating)) and float(level).is_integer()
    ):
        return int(level)
    raise ValueError(f"unknown treatment level {level!r}; levels = {levels}")


def _onehot(idx: np.ndarray, J: int) -> np.ndarray:
    idx = np.asarray(idx)
    if not np.issubdtype(idx.dtype, np.integer):
        if not np.all(np.isfinite(idx)) or not np.all(np.equal(np.mod(idx, 1), 0)):
            raise ValueError("regime class indices must be integers")
        idx = idx.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= J):
        raise ValueError(f"regime class indices must lie in 0..{J - 1}")
    return np.eye(J, dtype=np.float64)[idx]


def _check_probs(p: np.ndarray, J: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] != J:
        raise ValueError(f"stochastic intervention needs {J} class probabilities, got shape {p.shape}")
    if np.any(p < -1e-9) or np.any(np.abs(p.sum(-1) - 1) > 1e-6):
        raise ValueError("stochastic intervention probabilities must be >= 0 and sum to 1")
    return np.clip(p, 0.0, 1.0)


@dataclass
class Static:
    """Static regime: ``level`` is a class (label or index) or a sequence of
    length ``tau`` giving the class at each time point."""
    level: Any

    def resolve(self, W, L, A, J, levels=None):
        n, tau = A.shape[0], A.shape[1]
        if isinstance(self.level, (list, tuple, np.ndarray)):
            lv = list(self.level)
            if len(lv) != tau:
                raise ValueError(f"Static regime has {len(lv)} levels but tau = {tau}")
            idx = np.array([_level_index(v, levels) for v in lv])
            idx = np.broadcast_to(idx, (n, tau))
        else:
            idx = np.full((n, tau), _level_index(self.level, levels))
        return _onehot(idx, J)


@dataclass
class Dynamic:
    """Deterministic dynamic regime ``fn(W, L, A) -> (n, tau)`` class indices
    (or labels when ``levels`` are given)."""
    fn: Callable

    def resolve(self, W, L, A, J, levels=None):
        out = np.asarray(self.fn(W, L, A))
        n, tau = A.shape[0], A.shape[1]
        if out.ndim == 3 and out.shape[-1] == 1:
            out = out[..., 0]
        if out.shape != (n, tau):
            raise ValueError(f"Dynamic regime must return shape {(n, tau)}, got {out.shape}")
        if out.dtype.kind in "OUS":
            out = np.vectorize(lambda v: _level_index(v, levels))(out)
        return _onehot(out, J)


@dataclass
class Stochastic:
    """Stochastic regime: class probabilities ``(J,)``, ``(tau, J)``,
    ``(n, tau, J)``, or ``fn(W, L, A) -> (n, tau, J)``."""
    probs: Any

    def resolve(self, W, L, A, J, levels=None):
        n, tau = A.shape[0], A.shape[1]
        p = self.probs(W, L, A) if callable(self.probs) else self.probs
        p = _check_probs(p, J)
        return np.broadcast_to(p, (n, tau, J)).copy()


def from_dict(spec: dict):
    """Build an intervention from a JSON-style dict.

    ``{"type": "static", "level": x}`` / ``{"type": "static", "levels": [...]}``
    ``{"type": "stochastic", "probs": [...]}``
    (column-based forms are handled by the long-format API.)
    """
    kind = spec.get("type")
    if kind == "static":
        return Static(spec["levels"] if "levels" in spec else spec["level"])
    if kind == "stochastic" and "probs" in spec:
        return Stochastic(spec["probs"])
    raise ValueError(f"unsupported intervention spec for array data: {spec!r}")


def resolve_intervention(spec, W, L, A, J: int, levels: Sequence | None = None) -> np.ndarray:
    """Resolve ``spec`` into ``gstar`` of shape ``(n, tau, J)``.

    Accepted ``spec`` values
    ------------------------
    * :class:`Static` / :class:`Dynamic` / :class:`Stochastic`
    * ``int`` or label → static class for all time points
    * callable ``fn(W, L, A)`` → :class:`Dynamic` (or stochastic if it returns
      ``(n, tau, J)`` probabilities with ``J > 1``)
    * array ``(n, tau)`` / ``(n, tau, 1)`` of integer class indices (legacy:
      ``a = np.ones_like(A)``); for binary treatment non-integer values in
      ``[0, 1]`` are read as ``P(A_t = 1)`` (legacy stochastic intervention)
    * array ``(n, tau, J)`` of class probabilities
    * dict (see :func:`from_dict`)
    """
    A = np.asarray(A)
    n, tau = A.shape[0], A.shape[1]

    if isinstance(spec, (Static, Dynamic, Stochastic)):
        return spec.resolve(W, L, A, J, levels)
    if isinstance(spec, dict):
        return from_dict(spec).resolve(W, L, A, J, levels)
    if callable(spec):
        out = np.asarray(spec(W, L, A))
        if out.ndim == 3 and out.shape[-1] == J:
            return Stochastic(out).resolve(W, L, A, J, levels)
        return Dynamic(lambda *_: out).resolve(W, L, A, J, levels)
    if np.isscalar(spec) or isinstance(spec, str):
        return Static(spec).resolve(W, L, A, J, levels)

    a = np.asarray(spec, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] == J:      # J >= 2, so never the (n, tau, 1) form
        if a.shape[:2] != (n, tau):
            raise ValueError(f"intervention probabilities must have shape {(n, tau, J)}")
        return Stochastic(a).resolve(W, L, A, J, levels)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.shape != (n, tau):
        raise ValueError(
            f"intervention array must have shape (n, tau), (n, tau, 1) or (n, tau, J); got {np.shape(spec)}"
        )
    if np.all(np.equal(np.mod(a, 1), 0)):
        return _onehot(a.astype(np.int64), J)
    if J == 2 and np.all((a >= 0) & (a <= 1)):
        # legacy binary stochastic intervention: a = P(A_t = 1)
        return np.stack([1 - a, a], axis=-1)
    raise ValueError("non-integer intervention values are only allowed for binary treatment")
