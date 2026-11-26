"""Result container returned by :func:`dltmle.estimate` / :func:`dltmle.estimate_long`.

Designed to be easy to consume from R (reticulate or the CLI JSON):
every table is available both as a pandas DataFrame and as a
column-oriented ``dict`` of plain lists (``to_dict()``), which
``jsonlite::fromJSON`` / ``as.data.frame`` turn into an R data.frame.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .ral import RALEstimate

Z95 = 1.959963984540054


def z_value(ci_level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + ci_level / 2))


def _jsonable(x):
    """Recursively convert numpy / pandas objects to JSON-safe Python values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.DataFrame):
        return {c: _jsonable(x[c].to_numpy()) for c in x.columns}
    return x


@dataclass
class DLTMLEResult:
    """TMLE estimates of the counterfactual cumulative risk
    ``psi(t) = P(Y_t^{g*} = 1)`` (event by the end of interval t, with all
    censoring prevented) for each intervention and evaluation time.

    Attributes
    ----------
    estimates : DataFrame
        One row per (intervention, time): ``intervention, time, time_index,
        psi, se, ci_lower, ci_upper, survival, survival_ci_lower,
        survival_ci_upper, n``.
    ic : dict[(intervention, time_index)] -> (n,) array
        Influence curve per subject (aligned with ``ids``).
    ids, times : arrays
    diagnostics : dict
        Training / targeting / positivity diagnostics.
    hparams, seed, device, deterministic, estimator, version, ... : metadata
    """
    estimates: pd.DataFrame
    ic: dict
    ids: np.ndarray
    times: np.ndarray
    diagnostics: dict
    hparams: dict
    seed: int
    device: str
    deterministic: bool
    estimator: str
    version: str
    n_treatment: int
    n_censoring: int
    treatment_levels: list
    censoring_levels: list
    censoring_timing: str
    interventions: list
    ci_level: float = 0.95
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    def _key(self, intervention=None, time=None):
        if intervention is None:
            intervention = self.interventions[0]
        rows = self.estimates[self.estimates["intervention"] == intervention]
        if rows.empty:
            raise KeyError(f"unknown intervention {intervention!r}; available: {self.interventions}")
        if time is None:
            row = rows.iloc[-1]
        else:
            match = rows[(rows["time"] == time) | (rows["time"].astype(str) == str(time))]
            if match.empty:
                raise KeyError(f"no estimate at time {time!r}; available: {list(rows['time'])}")
            row = match.iloc[0]
        return intervention, int(row["time_index"])

    def ral(self, intervention=None, time=None) -> RALEstimate:
        """:class:`RALEstimate` (estimate + IC) for delta-method contrasts,
        e.g. ``res.ral("SGLT2i") - res.ral("SU")``.  Default: first
        intervention, last evaluation time."""
        key = self._key(intervention, time)
        row = self.estimates[(self.estimates["intervention"] == key[0])
                             & (self.estimates["time_index"] == key[1])].iloc[0]
        return RALEstimate(float(row["psi"]), self.ic[key])

    @property
    def psi(self) -> float:
        """Estimate for the first intervention at the last evaluation time."""
        return self.ral().est

    @property
    def se(self) -> float:
        return self.ral().se

    def estimates_frame(self) -> pd.DataFrame:
        return self.estimates.copy()

    def ic_frame(self, wide: bool = False) -> pd.DataFrame:
        """Influence curves.  Long format (default): columns ``id,
        intervention, time, time_index, ic``.  ``wide=True``: one row per id,
        one column per ``<intervention>@<time>``."""
        if wide:
            out = {"id": self.ids}
            for (iv, ti), v in self.ic.items():
                out[f"{iv}@{self.times[ti]}"] = v
            return pd.DataFrame(out)
        parts = []
        for (iv, ti), v in self.ic.items():
            parts.append(pd.DataFrame({
                "id": self.ids, "intervention": iv,
                "time": self.times[ti], "time_index": ti, "ic": v,
            }))
        return pd.concat(parts, ignore_index=True)

    def contrasts(self, reference=None, scale=("difference", "ratio")) -> pd.DataFrame:
        """Risk differences / ratios of every intervention vs ``reference``
        (default: first intervention) at each evaluation time, with
        delta-method SEs from the influence curves (ratio CI on log scale)."""
        reference = self.interventions[0] if reference is None else reference
        if isinstance(scale, str):
            scale = (scale,)
        z = z_value(self.ci_level)
        rows = []
        for iv in self.interventions:
            if iv == reference:
                continue
            for ti in sorted({k[1] for k in self.ic if k[0] == iv}):
                if (reference, ti) not in self.ic:
                    continue
                t = self.times[ti]
                a, b = self.ral(iv, t), self.ral(reference, t)
                if "difference" in scale:
                    d = a - b
                    rows.append({"intervention": iv, "reference": reference, "time": t,
                                 "time_index": ti, "scale": "difference", "estimate": d.est,
                                 "se": d.se, "ci_lower": d.est - z * d.se,
                                 "ci_upper": d.est + z * d.se})
                if "ratio" in scale and b.est > 0 and a.est > 0:
                    r = a / b
                    log_se = r.se / r.est
                    rows.append({"intervention": iv, "reference": reference, "time": t,
                                 "time_index": ti, "scale": "ratio", "estimate": r.est,
                                 "se": r.se, "ci_lower": r.est * math.exp(-z * log_se),
                                 "ci_upper": r.est * math.exp(z * log_se)})
        return pd.DataFrame(rows, columns=["intervention", "reference", "time", "time_index",
                                           "scale", "estimate", "se", "ci_lower", "ci_upper"])

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        return {
            "version": self.version,
            "estimator": self.estimator,
            "seed": self.seed,
            "device": self.device,
            "deterministic": self.deterministic,
            "hparams": self.hparams,
            "n": int(len(self.ids)),
            "tau": int(len(self.times)),
            "times": self.times,
            "interventions": self.interventions,
            "n_treatment": self.n_treatment,
            "n_censoring": self.n_censoring,
            "treatment_levels": self.treatment_levels,
            "censoring_levels": self.censoring_levels,
            "censoring_timing": self.censoring_timing,
            "ci_level": self.ci_level,
            **self.extra,
        }

    def to_dict(self, include_ic: bool = True, reference=None) -> dict:
        """JSON-safe nested dict.  Tables are column-oriented
        (``{"col": [values...]}``)."""
        out = {
            "metadata": self.metadata(),
            "estimates": self.estimates,
            "diagnostics": self.diagnostics,
        }
        if len(self.interventions) > 1:
            out["contrasts"] = self.contrasts(reference)
        if include_ic:
            out["ic"] = self.ic_frame()
        return _jsonable(out)

    def to_json(self, path=None, include_ic: bool = True, reference=None, indent=None):
        s = json.dumps(self.to_dict(include_ic=include_ic, reference=reference), indent=indent)
        if path is not None:
            with open(path, "w") as f:
                f.write(s)
        return s

    def __repr__(self):
        lines = [f"DLTMLEResult(estimator={self.estimator!r}, n={len(self.ids)}, "
                 f"tau={len(self.times)}, version={self.version!r})"]
        cols = ["intervention", "time", "psi", "se", "ci_lower", "ci_upper"]
        lines.append(self.estimates[cols].to_string(index=False))
        return "\n".join(lines)
