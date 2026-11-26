"""Stable high-level API (intended for the stremr / R wrapper).

* :func:`estimate`       — panel arrays ``W, L, A, C, Y`` → :class:`DLTMLEResult`
* :func:`estimate_long`  — long-format DataFrame with named columns
* :func:`estimate_file`  — Parquet / CSV file with named columns

All three share the same estimation options:

``intervention``
    One intervention spec, or a ``dict`` ``{name: spec}`` for several
    regimes estimated with one shared propensity model (contrasts via
    :meth:`DLTMLEResult.contrasts`).  See :func:`resolve_long_intervention`
    for the accepted specs.
``eval_times``
    Time points at which ``psi(t) = P(Y_t = 1)`` is estimated: ``None`` (the
    last time point), ``"all"``, or a list of time values.
``estimator``
    ``"dltmle"`` (transformer G / Q models) or ``"linear"`` (pooled logistic
    regression baseline).
``hparams``
    Overrides of :data:`dltmle.function.fit.DEFAULT_HPARAMS` (or the linear
    defaults); the full set used is returned in the result.  Unlike the
    legacy :func:`dltmle.fit`, ``init_bias`` defaults to ``True`` here.
``seed``, ``device`` (``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``),
``deterministic``, ``num_threads``
    Reproducibility / execution controls.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import PanelData, long_to_panel, read_table
from .function.fit import run_estimation, _as_panel, _n_classes
from .intervention import (
    Static, Dynamic, Stochastic, resolve_intervention, _check_probs, _level_index, _onehot,
)
from .model.categorical import check_censoring_timing
from .result import DLTMLEResult, z_value


_SPEC_TYPES = (Static, Dynamic, Stochastic)


def _is_single_spec(spec) -> bool:
    if isinstance(spec, Mapping):
        return "type" in spec
    return True


def _named_interventions(intervention) -> dict:
    if intervention is None:
        raise ValueError("an intervention is required")
    if isinstance(intervention, Mapping) and not _is_single_spec(intervention):
        if not intervention:
            raise ValueError("empty intervention dict")
        return {str(k): v for k, v in intervention.items()}
    return {_default_name(intervention): intervention}


def _default_name(spec) -> str:
    if isinstance(spec, (str, int, np.integer)):
        return str(spec)
    if isinstance(spec, Static) and not isinstance(spec.level, (list, tuple, np.ndarray)):
        return str(spec.level)
    if isinstance(spec, Mapping):
        if spec.get("type") == "static" and "level" in spec:
            return str(spec["level"])
        if "column" in spec:
            return str(spec["column"])
    return "regime"


def _eval_indices(eval_times, times) -> list:
    tau = len(times)
    if eval_times is None:
        return [tau - 1]
    if isinstance(eval_times, str):
        if eval_times == "all":
            return list(range(tau))
        if eval_times == "last":
            return [tau - 1]
        eval_times = [eval_times]
    if np.isscalar(eval_times):
        eval_times = [eval_times]
    out = []
    times_list = list(times)
    for t in eval_times:
        if t in times_list:
            out.append(times_list.index(t))
            continue
        match = [i for i, v in enumerate(times_list) if str(v) == str(t)]
        if not match:
            raise ValueError(f"eval time {t!r} not in time grid {times_list}")
        out.append(match[0])
    return sorted(set(out))


def _build_result(out, *, ids, times, names, estimator, seed, deterministic, ci_level,
                  treatment_levels, censoring_levels, censoring_timing, extra=None):
    z = z_value(ci_level)
    rows, ic, jobs_diag = [], {}, []
    for job in out["jobs"]:
        est, se = job["est"], job["se"]
        rows.append({
            "intervention": job["intervention"],
            "time": times[job["time_index"]],
            "time_index": job["time_index"],
            "psi": est, "se": se,
            "ci_lower": est - z * se, "ci_upper": est + z * se,
            "survival": 1 - est,
            "survival_ci_lower": 1 - est - z * se, "survival_ci_upper": 1 - est + z * se,
            "n": len(ids),
        })
        ic[(job["intervention"], job["time_index"])] = job["ic"]
        jobs_diag.append({
            "intervention": job["intervention"],
            "time": times[job["time_index"]],
            "time_index": job["time_index"],
            "targeting": job["targeting"],
            "weights": job["weights"],
            "train_q": job["train_q"],
        })
    est_df = pd.DataFrame(rows)
    order = {nm: i for i, nm in enumerate(names)}
    est_df = (est_df.assign(_o=est_df["intervention"].map(order))
              .sort_values(["_o", "time_index"]).drop(columns="_o").reset_index(drop=True))
    diagnostics = {
        "warnings": out["warnings"],
        "train_g": out["train_g"],
        "propensity": out["propensity"],
        "jobs": jobs_diag,
    }
    return DLTMLEResult(
        estimates=est_df, ic=ic, ids=np.asarray(ids), times=np.asarray(times),
        diagnostics=diagnostics, hparams=out["hparams"], seed=int(seed),
        device=out["device"], deterministic=bool(deterministic), estimator=estimator,
        version=__version__, n_treatment=out["n_treatment"], n_censoring=out["n_censoring"],
        treatment_levels=list(treatment_levels), censoring_levels=list(censoring_levels),
        censoring_timing=censoring_timing, interventions=list(names),
        ci_level=float(ci_level), extra=extra or {},
    )


def _run(W, L, A, C, Y, gstars, *, ids, times, eval_times, hparams, seed, estimator,
         device, deterministic, num_threads, ci_level, verbose, targeting,
         n_treatment, n_censoring, treatment_levels, censoring_levels,
         censoring_timing, extra=None):
    # the high-level API initialises the g heads at the marginal log-odds by
    # default (legacy fit() keeps the original initialisation)
    hparams = {"init_bias": True, **dict(hparams or {})}
    old_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    try:
        out = run_estimation(
            W, L, A, C, Y, gstars, hparams,
            estimator=estimator, seed=seed, device=device, deterministic=deterministic,
            time_indices=_eval_indices(eval_times, times),
            n_treatment=n_treatment, n_censoring=n_censoring, verbose=verbose,
            targeting=targeting, censoring_timing=censoring_timing,
        )
    finally:
        if num_threads is not None:
            torch.set_num_threads(old_threads)
    return _build_result(
        out, ids=ids, times=times, names=list(gstars), estimator=estimator, seed=seed,
        deterministic=deterministic, ci_level=ci_level,
        treatment_levels=treatment_levels, censoring_levels=censoring_levels,
        censoring_timing=censoring_timing, extra=extra,
    )


# ---------------------------------------------------------------------------
# Array API
# ---------------------------------------------------------------------------

def estimate(
    W, L, A, C, Y,
    intervention,
    *,
    hparams: dict = None,
    seed: int = 0,
    estimator: str = "dltmle",
    eval_times=None,
    ids=None,
    times=None,
    n_treatment: int = None,
    n_censoring: int = None,
    treatment_levels: Sequence = None,
    censoring_levels: Sequence = None,
    censoring_timing: str = "after_treatment",
    device: str = "auto",
    deterministic: bool = False,
    num_threads: int = None,
    ci_level: float = 0.95,
    verbose: bool = False,
    targeting: dict = None,
) -> DLTMLEResult:
    """TMLE of ``psi(t) = P(Y_t^{g*} = 1)`` from panel arrays.

    ``A`` / ``C`` hold class codes ``0..J-1`` / ``0..K-1`` (``C = 0``: not
    censored).  ``intervention`` is a spec accepted by
    :func:`dltmle.intervention.resolve_intervention` (``Static``,
    ``Dynamic``, ``Stochastic``, int, arrays, JSON dict) or a dict of named
    specs.  ``treatment_levels`` optionally names the classes (then specs may
    use labels).
    """
    W, L, A, C, Y = _as_panel(W, L, A, C, Y)
    n, tau = L.shape[0], L.shape[1]
    if treatment_levels is not None and n_treatment is None:
        n_treatment = len(treatment_levels)
    if censoring_levels is not None and n_censoring is None:
        n_censoring = len(censoring_levels)
    J = _n_classes(A, n_treatment, "n_treatment")
    K = _n_classes(C, n_censoring, "n_censoring")
    treatment_levels = list(range(J)) if treatment_levels is None else list(treatment_levels)
    censoring_levels = list(range(K)) if censoring_levels is None else list(censoring_levels)
    ids = np.arange(n) if ids is None else np.asarray(ids)
    times = np.arange(tau) if times is None else np.asarray(times)

    gstars = {name: resolve_intervention(spec, W, L, A, J, treatment_levels)
              for name, spec in _named_interventions(intervention).items()}
    return _run(W, L, A, C, Y, gstars, ids=ids, times=times, eval_times=eval_times,
                hparams=hparams, seed=seed, estimator=estimator, device=device,
                deterministic=deterministic, num_threads=num_threads, ci_level=ci_level,
                verbose=verbose, targeting=targeting, n_treatment=J, n_censoring=K,
                treatment_levels=treatment_levels, censoring_levels=censoring_levels,
                censoring_timing=check_censoring_timing(censoring_timing))


# ---------------------------------------------------------------------------
# Long-format API
# ---------------------------------------------------------------------------

def resolve_long_intervention(spec, panel: PanelData) -> np.ndarray:
    """Resolve an intervention spec against long-format data → ``(n, tau, J)``.

    Accepted specs
    --------------
    * a treatment level (label or code) → static regime, e.g. ``"SGLT2i"``
    * ``{"type": "static", "level": x}``; ``{"type": "static", "levels":
      [x_t0, x_t1, ...]}`` (one level per time point) or ``{"type": "static",
      "levels": {time: level, ...}}``
    * ``{"type": "column", "column": "d"}`` (alias ``"dynamic"``) — per-row
      regime class in column ``d`` (static or dynamic regimes computed
      upstream, e.g. in R / stremr's ``intervened_TRT``).  A bare string that
      is a column name (and not a treatment level) is read the same way.
    * ``{"type": "stochastic", "columns": [p_level0, p_level1, ...]}`` —
      per-row class probabilities in treatment-level order (stremr-style
      stochastic intervention); ``{"type": "stochastic", "probs": [...]}``
      constant probabilities (``(J,)`` or per time ``(tau, J)``)
    * a list of J column names → stochastic columns
    * callable ``fn(frame) -> per-row levels (len(frame),) or probs (len(frame), J)``
      evaluated on the sorted long frame (rows after the end of follow-up
      removed)
    * :class:`~dltmle.intervention.Static` / ``Dynamic`` / ``Stochastic``
      objects (evaluated on the panel arrays)

    Per-row values are required on every used row, including censoring rows
    (the pseudo-outcome at a censoring row uses ``g*`` at that row).
    """
    J = len(panel.treatment_levels)
    levels = panel.treatment_levels
    frame = panel.frame
    tau = panel.tau

    def _rows_to_gstar(values):
        values = np.asarray(values)
        if values.ndim == 2 and values.shape[1] == J and values.dtype.kind in "fiu":
            return panel.to_cell_array(_check_probs(values, J))
        if values.ndim != 1 or len(values) != len(frame):
            raise ValueError(f"per-row regime must have {len(frame)} values, got shape {values.shape}")
        if pd.isna(values).any():
            raise ValueError("per-row regime contains missing values")
        codes = np.array([_level_index(v, levels) for v in values], dtype=np.int64)
        return panel.to_cell_array(_onehot(codes, J))

    if isinstance(spec, _SPEC_TYPES):
        return spec.resolve(panel.W, panel.L, panel.A, J, levels)
    if callable(spec):
        return _rows_to_gstar(spec(frame))
    if isinstance(spec, Mapping):
        kind = spec.get("type")
        if kind == "static":
            if "levels" in spec:
                lv = spec["levels"]
                if isinstance(lv, Mapping):
                    by_t = {str(k): v for k, v in lv.items()}
                    missing = [t for t in panel.times if str(t) not in by_t]
                    if missing:
                        raise ValueError(f"static regime has no level for times {missing}")
                    lv = [by_t[str(t)] for t in panel.times]
                return Static(list(lv)).resolve(panel.W, panel.L, panel.A, J, levels)
            return Static(spec["level"]).resolve(panel.W, panel.L, panel.A, J, levels)
        if kind in ("column", "dynamic"):
            col = spec["column"]
            if col not in frame.columns:
                raise KeyError(f"regime column {col!r} not found")
            return _rows_to_gstar(frame[col].to_numpy())
        if kind == "stochastic":
            if "columns" in spec:
                cols = list(spec["columns"])
                if len(cols) != J:
                    raise ValueError(f"stochastic regime needs {J} probability columns, got {len(cols)}")
                return _rows_to_gstar(frame[cols].to_numpy(dtype=float))
            p = np.asarray(spec["probs"], dtype=float)
            return Stochastic(p).resolve(panel.W, panel.L, panel.A, J, levels)
        raise ValueError(f"unsupported intervention spec {spec!r}")
    if isinstance(spec, (list, tuple)) and spec and all(isinstance(c, str) and c in frame.columns for c in spec):
        return resolve_long_intervention({"type": "stochastic", "columns": list(spec)}, panel)
    if isinstance(spec, str) and spec in frame.columns and spec not in map(str, levels):
        return _rows_to_gstar(frame[spec].to_numpy())
    if np.isscalar(spec):
        return Static(spec).resolve(panel.W, panel.L, panel.A, J, levels)
    raise ValueError(f"unsupported intervention spec {spec!r}")


def estimate_long(
    data: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    intervention,
    censoring_cols=None,
    static_cols: Sequence[str] = None,
    dynamic_cols: Sequence[str] = None,
    treatment_levels: Sequence = None,
    censoring_levels: Sequence = None,
    censoring_timing: str = "after_treatment",
    implicit_censoring: Any = None,
    standardize: bool = False,
    hparams: dict = None,
    seed: int = 0,
    estimator: str = "dltmle",
    eval_times=None,
    device: str = "auto",
    deterministic: bool = False,
    num_threads: int = None,
    ci_level: float = 0.95,
    verbose: bool = False,
    targeting: dict = None,
) -> DLTMLEResult:
    """TMLE from a long-format (person-time) DataFrame.

    Parameters
    ----------
    data : DataFrame with one row per (``id_col``, ``time_col``).
    outcome_col : binary event indicator for interval t (observed after A_t;
        stremr's ``Y.tplus1``).
    treatment_col : treatment class (labels or integer codes).
    censoring_cols : one categorical censoring column (``censoring_levels[0]``
        = not censored) or a list of binary columns, one per censoring reason.
    static_cols / dynamic_cols : baseline W / time-varying L covariates
        (numeric; one-hot encode categorical covariates first).
    censoring_timing : ``"after_treatment"`` (dltmle: L→A→C→Y) or
        ``"before_treatment"`` (stremr: L→C→A→Y).
    implicit_censoring : censoring level for subjects whose rows end early
        without an event / censoring code (default: error).
    standardize : z-score W and L using person-time means / SDs.
    Other parameters: see :mod:`dltmle.api`.

    Returns
    -------
    DLTMLEResult
    """
    check_censoring_timing(censoring_timing)
    panel = long_to_panel(
        data, id_col=id_col, time_col=time_col, outcome_col=outcome_col,
        treatment_col=treatment_col, censoring_cols=censoring_cols,
        static_cols=static_cols, dynamic_cols=dynamic_cols,
        treatment_levels=treatment_levels, censoring_levels=censoring_levels,
        censoring_timing=censoring_timing, implicit_censoring=implicit_censoring,
    )
    gstars = {name: resolve_long_intervention(spec, panel)
              for name, spec in _named_interventions(intervention).items()}

    W, L = panel.W, panel.L
    extra = {
        "columns": {
            "id": id_col, "time": time_col, "outcome": outcome_col,
            "treatment": treatment_col, "censoring": censoring_cols,
            "static": panel.static_cols, "dynamic": panel.dynamic_cols,
        },
        "standardize": bool(standardize),
    }
    if standardize:
        W, L, scale = _standardize(panel)
        extra["standardization"] = scale

    return _run(W, L, panel.A, panel.C, panel.Y, gstars,
                ids=panel.ids, times=panel.times, eval_times=eval_times,
                hparams=hparams, seed=seed, estimator=estimator, device=device,
                deterministic=deterministic, num_threads=num_threads, ci_level=ci_level,
                verbose=verbose, targeting=targeting,
                n_treatment=len(panel.treatment_levels),
                n_censoring=len(panel.censoring_levels),
                treatment_levels=panel.treatment_levels,
                censoring_levels=panel.censoring_levels,
                censoring_timing=censoring_timing, extra=extra)


def _standardize(panel: PanelData):
    scale = {}
    W = panel.W.copy()
    L = panel.L.copy()
    if panel.static_cols:
        m, s = W.mean(0), W.std(0)
        s[s == 0] = 1.0
        W = (W - m) / s
        scale["static"] = {c: [float(a), float(b)] for c, a, b in zip(panel.static_cols, m, s)}
    if panel.dynamic_cols:
        rows = panel.frame[panel.dynamic_cols].to_numpy(dtype=float)
        m, s = rows.mean(0), rows.std(0)
        s[s == 0] = 1.0
        L = (L - m) / s
        scale["dynamic"] = {c: [float(a), float(b)] for c, a, b in zip(panel.dynamic_cols, m, s)}
    return W, L, scale


def estimate_file(path: str, **kwargs) -> DLTMLEResult:
    """:func:`estimate_long` on a Parquet / CSV file (see :func:`dltmle.data.read_table`)."""
    return estimate_long(read_table(path), **kwargs)
