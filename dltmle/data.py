"""Long-format (person-time) data ↔ dltmle panel arrays.

Long format: one row per subject (``id_col``) and time point (``time_col``),
with outcome, treatment, censoring and covariate columns — the layout used by
stremr and most pooled-logistic software.

Within-interval ordering
------------------------
dltmle models each interval t as ``L_t → A_t → C_t → Y_t``
(``censoring_timing="after_treatment"``, default).  stremr assumes
``L_t → C_t → A_t → Y_t`` (``censoring_timing="before_treatment"``).  In both
cases the outcome column on row t is the event indicator observed *after*
treatment at t (stremr's ``Y.tplus1``), and ``C_t > 0`` means "censored in
interval t, Y_t not observed".  With ``"before_treatment"`` the treatment
value on a censoring row is not observed and may be missing; it is replaced
by the last observed treatment (it never enters the likelihood or the
clever covariate).

Rules
-----
* Each subject's rows must cover consecutive time points starting at the
  first time point of the grid (the sorted unique values of ``time_col``).
* A subject's follow-up ends at its first event (``Y = 1``) or censoring
  (``C > 0``) row; later rows are dropped with a warning.
* If a subject's rows end before the last time point without an event or a
  censoring code, ``implicit_censoring`` decides: ``None`` (default) raises,
  otherwise the given censoring level is assigned at the next time point
  (covariates and treatment carried forward).  With
  ``censoring_timing="after_treatment"`` the carried-forward treatment on that
  row enters the g^A likelihood, so coding censoring explicitly (or using
  ``"before_treatment"``) is preferable.
* After the end of follow-up the panel is filled by carrying the last row
  forward (absorbing Y / C), which is what the estimator expects.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .model.categorical import check_censoring_timing


@dataclass
class PanelData:
    """Panel arrays plus the metadata needed to map results back."""
    W: np.ndarray                 # (n, dim_static)
    L: np.ndarray                 # (n, tau, dim_dynamic)
    A: np.ndarray                 # (n, tau, 1) class codes
    C: np.ndarray                 # (n, tau, 1) censoring codes (0 = none)
    Y: np.ndarray                 # (n, tau, 1)
    ids: np.ndarray               # (n,)
    times: np.ndarray             # (tau,)
    treatment_levels: list
    censoring_levels: list
    static_cols: list = field(default_factory=list)
    dynamic_cols: list = field(default_factory=list)
    # row → panel cell mapping of the rows that were used (sorted frame)
    frame: pd.DataFrame = None
    row_i: np.ndarray = None
    row_t: np.ndarray = None
    last_t: np.ndarray = None      # (n,) index of each subject's last used row

    @property
    def n(self):
        return len(self.ids)

    @property
    def tau(self):
        return len(self.times)

    def to_cell_array(self, values: np.ndarray) -> np.ndarray:
        """Scatter per-row ``values`` (rows of ``frame``) onto the (n, tau, ...)
        grid, carrying the last row forward after the end of follow-up."""
        values = np.asarray(values)
        out = np.zeros((self.n, self.tau) + values.shape[1:], dtype=values.dtype)
        out[self.row_i, self.row_t] = values
        t_grid = np.arange(self.tau)[None, :]
        src_t = np.minimum(t_grid, self.last_t[:, None])
        return out[np.arange(self.n)[:, None], src_t]


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def read_table(path: str, **kwargs) -> pd.DataFrame:
    """Read a Parquet (``.parquet``/``.pq``) or CSV (``.csv``, ``.csv.gz``,
    ``.tsv``) file into a DataFrame."""
    p = str(path).lower()
    if p.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, **kwargs)
    if p.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t", **kwargs)
    if p.endswith((".csv", ".csv.gz", ".txt")):
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"unsupported file type: {os.path.basename(str(path))} (use .parquet or .csv)")


def _levels(values: pd.Series, levels, what: str):
    if levels is not None:
        levels = list(levels)
        if len(set(map(str, levels))) != len(levels):
            raise ValueError(f"{what}_levels contain duplicates: {levels}")
        return levels
    uniq = pd.unique(values.dropna())
    if pd.api.types.is_numeric_dtype(values):
        u = np.asarray(uniq, dtype=float)
        if np.all(np.equal(np.mod(u, 1), 0)) and (u.size == 0 or u.min() >= 0):
            return list(range(max(int(u.max()) + 1 if u.size else 0, 2)))
    return sorted(uniq, key=lambda v: str(v))


def _codes(values: pd.Series, levels: list, what: str) -> np.ndarray:
    lookup = {}
    for j, lv in enumerate(levels):
        lookup[lv] = j
        lookup[str(lv)] = j
        if isinstance(lv, (int, np.integer)):
            lookup[float(lv)] = j
    out = np.empty(len(values), dtype=np.int64)
    bad = []
    for r, v in enumerate(values.to_numpy()):
        key = v
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            key = int(v)
        j = lookup.get(key, lookup.get(str(v)))
        if j is None:
            bad.append(v)
            j = -1
        out[r] = j
    if bad:
        raise ValueError(f"unknown {what} value(s) {sorted(set(map(str, bad)))[:10]}; levels = {levels}")
    return out


def long_to_panel(
    data: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    censoring_cols=None,
    static_cols: Sequence[str] = None,
    dynamic_cols: Sequence[str] = None,
    treatment_levels: Sequence[Any] = None,
    censoring_levels: Sequence[Any] = None,
    censoring_timing: str = "after_treatment",
    implicit_censoring: Any = None,
) -> PanelData:
    """Convert a long-format DataFrame to panel arrays.

    Parameters
    ----------
    data : DataFrame, one row per (id, time).
    id_col, time_col, outcome_col, treatment_col : str
    censoring_cols : str or list of str, optional
        * one column holding the censoring category (``censoring_levels[0]``
          = not censored; default levels: ``0..max`` for non-negative integer
          codes, otherwise you must pass ``censoring_levels``), or
        * several binary columns, one per censoring reason (stremr-style
          ``CENS = c("C1", "C2")``): category k = the k-th column.
        If omitted, no censoring (K = 2 with all zeros).
    static_cols : baseline covariates W (taken from each subject's first row).
    dynamic_cols : time-varying covariates L.
    treatment_levels : class order for the treatment (default: ``0..max`` for
        integer codes, otherwise sorted unique values).
    censoring_timing : ``"after_treatment"`` (dltmle) or ``"before_treatment"`` (stremr).
    implicit_censoring : censoring level assigned when follow-up ends early
        without an event / censoring code (default: raise).
    """
    check_censoring_timing(censoring_timing)
    static_cols = _as_list(static_cols)
    dynamic_cols = _as_list(dynamic_cols)
    cens_cols = _as_list(censoring_cols)

    needed = [id_col, time_col, outcome_col, treatment_col] + cens_cols + static_cols + dynamic_cols
    missing = [c for c in dict.fromkeys(needed) if c not in data.columns]
    if missing:
        raise KeyError(f"columns not found in data: {missing}")

    df = data.sort_values([id_col, time_col], kind="mergesort").reset_index(drop=True)
    if df.duplicated([id_col, time_col]).any():
        dup = df.loc[df.duplicated([id_col, time_col]), [id_col, time_col]].head(5)
        raise ValueError(f"duplicate (id, time) rows, e.g.\n{dup}")

    times = np.sort(pd.unique(df[time_col]))
    t_idx = np.searchsorted(times, df[time_col].to_numpy())
    rank = df.groupby(id_col, sort=False).cumcount().to_numpy()
    if np.any(t_idx != rank):
        bad = df.loc[t_idx != rank, id_col].unique()[:5]
        raise ValueError(
            "each subject's rows must cover consecutive time points starting at "
            f"{times[0]!r}; offending ids: {list(bad)}"
        )

    # ---- censoring codes
    if not cens_cols:
        c_code = np.zeros(len(df), dtype=np.int64)
        censoring_levels = list(censoring_levels) if censoring_levels is not None else [0, 1]
    elif len(cens_cols) == 1:
        col = df[cens_cols[0]]
        if col.isna().any():
            raise ValueError(f"censoring column {cens_cols[0]!r} contains missing values")
        if censoring_levels is None and not pd.api.types.is_numeric_dtype(col):
            raise ValueError("non-numeric censoring column: pass censoring_levels "
                             "(first level = not censored)")
        censoring_levels = _levels(col, censoring_levels, "censoring")
        c_code = _codes(col, censoring_levels, "censoring")
    else:
        mat = df[cens_cols].to_numpy()
        if pd.isna(mat).any():
            raise ValueError("censoring indicator columns contain missing values")
        if not np.isin(mat, (0, 1)).all():
            raise ValueError("censoring indicator columns must be 0/1")
        if (mat.sum(axis=1) > 1).any():
            raise ValueError("more than one censoring indicator is 1 on the same row")
        c_code = np.where(mat.sum(axis=1) > 0, mat.argmax(axis=1) + 1, 0).astype(np.int64)
        censoring_levels = ["none"] + list(cens_cols) if censoring_levels is None else list(censoring_levels)
        if len(censoring_levels) != len(cens_cols) + 1:
            raise ValueError("censoring_levels must have len(censoring_cols) + 1 entries")

    # ---- outcome
    y = df[outcome_col]
    censored_row = c_code > 0
    if y[~censored_row].isna().any():
        raise ValueError(f"outcome column {outcome_col!r} has missing values on uncensored rows")
    y = y.fillna(0).to_numpy(dtype=float, copy=True)
    y[censored_row] = 0.0
    if not np.isin(y, (0, 1)).all():
        raise ValueError("outcome must be binary 0/1")

    # ---- end of follow-up: first event or censoring row
    ended = (y == 1) | censored_row
    ended_before = (
        pd.Series(ended).groupby(df[id_col].to_numpy(), sort=False).cumsum().to_numpy() - ended
    ) > 0
    if ended_before.any():
        warnings.warn(f"dropping {int(ended_before.sum())} rows after an event or censoring")
        keep = ~ended_before
        df = df.loc[keep].reset_index(drop=True)
        t_idx, c_code, y, censored_row = t_idx[keep], c_code[keep], y[keep], censored_row[keep]
        ended = ended[keep]

    # ---- treatment codes
    a_col = df[treatment_col]
    if censoring_timing == "before_treatment":
        # A is not observed on censoring rows: carry the last treatment forward
        a_col = a_col.where(~censored_row)
        a_col = a_col.groupby(df[id_col].to_numpy(), sort=False).ffill()
        first_level = treatment_levels[0] if treatment_levels is not None else None
        if a_col.isna().any():
            fill = first_level if first_level is not None else a_col.dropna().iloc[0]
            a_col = a_col.fillna(fill)
    if a_col.isna().any():
        raise ValueError(f"treatment column {treatment_col!r} has missing values")
    treatment_levels = _levels(a_col, treatment_levels, "treatment")
    a_code = _codes(a_col, treatment_levels, "treatment")

    # ---- covariates
    for c in static_cols + dynamic_cols:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            raise ValueError(f"covariate {c!r} is not numeric (one-hot encode it first)")
        if df[c].isna().any():
            raise ValueError(f"covariate {c!r} has missing values (impute first)")

    ids, row_i = np.unique(df[id_col].to_numpy(), return_inverse=True)
    # keep ids in order of first appearance in the sorted frame (= sorted ids)
    n, tau = len(ids), len(times)
    last_t = np.zeros(n, dtype=np.int64)
    np.maximum.at(last_t, row_i, t_idx)

    # implicit end of follow-up
    last_rows = np.zeros(n, dtype=np.int64)
    last_rows[row_i] = np.arange(len(df))          # last row index per subject (sorted)
    open_end = ~ended[last_rows] & (last_t < tau - 1)
    extra_c = None
    if open_end.any():
        if implicit_censoring is None:
            raise ValueError(
                f"{int(open_end.sum())} subject(s) leave follow-up before the last time point "
                f"without an event or censoring code (e.g. ids {list(ids[open_end][:5])}); "
                "code the censoring explicitly or pass implicit_censoring=<censoring level>"
            )
        extra_c = _codes(pd.Series([implicit_censoring]), censoring_levels, "censoring")[0]
        if extra_c == 0:
            raise ValueError("implicit_censoring must be a censoring level other than 'not censored'")

    W = (df.groupby(row_i, sort=True)[static_cols].first().to_numpy(dtype=float)
         if static_cols else np.zeros((n, 1)))

    panel = PanelData(
        W=W, L=None, A=None, C=None, Y=None, ids=ids, times=times,
        treatment_levels=treatment_levels, censoring_levels=censoring_levels,
        static_cols=static_cols, dynamic_cols=dynamic_cols,
        frame=df, row_i=row_i, row_t=t_idx, last_t=last_t,
    )
    L = (panel.to_cell_array(df[dynamic_cols].to_numpy(dtype=float))
         if dynamic_cols else np.zeros((n, tau, 1)))
    A = panel.to_cell_array(a_code).astype(float)
    C = panel.to_cell_array(c_code).astype(float)
    Y = panel.to_cell_array(y).astype(float)

    if extra_c is not None:
        after = np.arange(tau)[None, :] > last_t[:, None]
        C[open_end[:, None] & after] = extra_c

    panel.L, panel.A, panel.C, panel.Y = L, A[..., None], C[..., None], Y[..., None]
    return panel


def panel_to_long(
    W, L, A, C, Y, *,
    ids=None, times=None,
    static_names=None, dynamic_names=None,
    id_col="id", time_col="t", outcome_col="Y", treatment_col="A", censoring_col="C",
    drop_after_end: bool = True,
) -> pd.DataFrame:
    """Inverse of :func:`long_to_panel` for array data (useful for tests,
    examples and exporting simulated data to R).  With ``drop_after_end``
    the carried-forward rows after an event / censoring are omitted."""
    W = np.asarray(W); L = np.asarray(L)
    A = np.asarray(A).reshape(L.shape[0], L.shape[1])
    C = np.asarray(C).reshape(L.shape[0], L.shape[1])
    Y = np.asarray(Y).reshape(L.shape[0], L.shape[1])
    n, tau = L.shape[0], L.shape[1]
    ids = np.arange(n) if ids is None else np.asarray(ids)
    times = np.arange(tau) if times is None else np.asarray(times)
    static_names = static_names or [f"W{j}" for j in range(W.shape[1])]
    dynamic_names = dynamic_names or [f"L{j}" for j in range(L.shape[2])]

    cols = {
        id_col: np.repeat(ids, tau),
        time_col: np.tile(times, n),
        outcome_col: Y.ravel(),
        treatment_col: A.ravel().astype(np.int64),
        censoring_col: C.ravel().astype(np.int64),
    }
    for j, nm in enumerate(static_names):
        cols[nm] = np.repeat(W[:, j], tau)
    for j, nm in enumerate(dynamic_names):
        cols[nm] = L[:, :, j].ravel()
    df = pd.DataFrame(cols)
    if drop_after_end:
        ended = (Y == 1) | (C > 0)
        ended_before = np.zeros_like(ended)
        ended_before[:, 1:] = np.cumsum(ended, axis=1)[:, :-1] > 0
        df = df.loc[~ended_before.ravel()].reset_index(drop=True)
    return df
