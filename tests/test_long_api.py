"""Long-format API: equivalence with the array API, column handling, results."""
import json

import numpy as np
import pandas as pd
import pytest

import dltmle
from dltmle import Stochastic

from .conftest import TINY_DEEP_HPARAMS

LEVELS = ["SU", "DPP4", "SGLT2i"]
CENS = ["none", "death", "disenroll"]
LIN = {"max_epochs": 20, "batch_size": 128}


@pytest.fixture(scope="module")
def panel():
    return dltmle.example_dgp_categorical(np.random.default_rng(3), n=600, tau=4)


@pytest.fixture(scope="module")
def long_df(panel):
    W, L, A, C, Y = panel
    df = dltmle.panel_to_long(W, L, A, C, Y, ids=np.arange(600) + 1000,
                              times=np.array([0, 30, 60, 90]),
                              static_names=["age"], dynamic_names=["hba1c"],
                              id_col="ID", time_col="day", outcome_col="event",
                              treatment_col="drug", censoring_col="cens")
    df["drug"] = np.array(LEVELS)[df["drug"]]
    df["cens"] = np.array(CENS)[df["cens"]]
    # shuffle rows: the converter must sort
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


COLS = dict(id_col="ID", time_col="day", outcome_col="event", treatment_col="drug",
            censoring_cols="cens", static_cols=["age"], dynamic_cols=["hba1c"],
            treatment_levels=LEVELS, censoring_levels=CENS)


def test_long_to_panel_roundtrip(panel, long_df):
    W, L, A, C, Y = panel
    p = dltmle.long_to_panel(long_df, **COLS)
    np.testing.assert_array_equal(p.ids, np.arange(600) + 1000)
    np.testing.assert_array_equal(p.times, [0, 30, 60, 90])
    np.testing.assert_allclose(p.W, W)
    for got, exp in ((p.L, L), (p.A, A), (p.C, C), (p.Y, Y)):
        np.testing.assert_allclose(got, exp)
    assert p.treatment_levels == LEVELS and p.censoring_levels == CENS
    assert len(long_df) < 600 * 4      # rows after events / censoring were absent


@pytest.mark.parametrize("estimator,hp", [("linear", LIN), ("dltmle", TINY_DEEP_HPARAMS)])
def test_long_equals_array(panel, long_df, estimator, hp):
    W, L, A, C, Y = panel
    kw = dict(estimator=estimator, hparams=hp, seed=11, device="cpu", eval_times="all")
    ra = dltmle.estimate(W, L, A, C, Y, {"SGLT2i": 2, "SU": 0}, **kw)
    rl = dltmle.estimate_long(long_df, intervention={"SGLT2i": "SGLT2i", "SU": "SU"}, **COLS, **kw)
    pd.testing.assert_frame_equal(
        ra.estimates.drop(columns="time"), rl.estimates.drop(columns="time"))
    for (iv, ti), ic in ra.ic.items():
        np.testing.assert_array_equal(ic, rl.ic[(iv, ti)])
    assert list(rl.estimates["time"].unique()) == [0, 30, 60, 90]


def test_binary_censoring_columns_and_column_regimes(panel, long_df):
    """stremr-style: one binary column per censoring reason; regimes given as
    per-row columns (static / dynamic computed upstream) or probability columns."""
    W, L, A, C, Y = panel
    df = long_df.copy()
    df["C_death"] = (df["cens"] == "death").astype(int)
    df["C_disenroll"] = (df["cens"] == "disenroll").astype(int)
    df["d_static"] = "DPP4"
    df["d_dyn"] = np.where(df["hba1c"] > 0, "SGLT2i", "SU")
    for j, lv in enumerate(LEVELS):
        df[f"p_{lv}"] = [0.2, 0.3, 0.5][j]
    cols = dict(COLS, censoring_cols=["C_death", "C_disenroll"], censoring_levels=None)
    kw = dict(estimator="linear", hparams=LIN, seed=0, device="cpu")
    rl = dltmle.estimate_long(df, intervention={
        "static_col": {"type": "column", "column": "d_static"},
        "bare_col": "d_static",
        "dyn": {"type": "dynamic", "column": "d_dyn"},
        "sto_cols": [f"p_{lv}" for lv in LEVELS],
        "sto_probs": {"type": "stochastic", "probs": [0.2, 0.3, 0.5]},
        "callable": lambda frame: np.where(frame["hba1c"] > 0, "SGLT2i", "SU"),
    }, **cols, **kw)
    ra = dltmle.estimate(W, L, A, C, Y, {
        "static_col": 1, "bare_col": 1,
        "dyn": lambda W, L, A: np.where(L[..., 0] > 0, 2, 0),
        "sto_cols": Stochastic([0.2, 0.3, 0.5]), "sto_probs": Stochastic([0.2, 0.3, 0.5]),
        "callable": lambda W, L, A: np.where(L[..., 0] > 0, 2, 0),
    }, **kw)
    assert rl.censoring_levels == ["none", "C_death", "C_disenroll"]
    np.testing.assert_array_equal(rl.estimates["psi"].to_numpy(), ra.estimates["psi"].to_numpy())


def test_eval_times_and_multiple_interventions_reuse_first_job(panel):
    W, L, A, C, Y = panel
    kw = dict(estimator="linear", hparams=LIN, seed=2, device="cpu")
    single = dltmle.estimate(W, L, A, C, Y, 2, **kw)
    multi = dltmle.estimate(W, L, A, C, Y, {"two": 2, "zero": 0}, eval_times="all", **kw)
    assert multi.ral("two").est == single.psi
    assert len(multi.estimates) == 8
    sub = dltmle.estimate(W, L, A, C, Y, 2, eval_times=[1, 3], **kw)
    assert list(sub.estimates["time"]) == [1, 3]


def test_result_serialisation(panel):
    W, L, A, C, Y = panel
    r = dltmle.estimate(W, L, A, C, Y, {"a": 2, "b": 0}, estimator="linear",
                        hparams=LIN, seed=5, device="cpu", eval_times="all")
    d = json.loads(r.to_json())
    meta = d["metadata"]
    assert meta["version"] == dltmle.__version__
    assert meta["seed"] == 5 and meta["device"] == "cpu" and meta["estimator"] == "linear"
    assert meta["hparams"]["max_epochs"] == 20 and meta["hparams"]["n_treatment"] == 3
    est = pd.DataFrame(d["estimates"])
    assert {"intervention", "time", "psi", "se", "ci_lower", "ci_upper", "survival"} <= set(est)
    np.testing.assert_allclose(est["ci_upper"] - est["psi"], 1.959963984540054 * est["se"])
    ic = pd.DataFrame(d["ic"])
    assert len(ic) == 600 * 8
    # the IC SE is reproducible from the exported IC table
    sub = ic[(ic["intervention"] == "a") & (ic["time"] == 3)]
    assert np.sqrt((sub["ic"] ** 2).mean() / 600) == pytest.approx(r.ral("a").se)
    assert "contrasts" in d and "warnings" in d["diagnostics"]
    assert r.ic_frame(wide=True).shape == (600, 1 + 8)


def test_deterministic_deep_repeatable(panel):
    W, L, A, C, Y = panel
    kw = dict(hparams=TINY_DEEP_HPARAMS, seed=4, device="cpu", deterministic=True)
    r1 = dltmle.estimate(W, L, A, C, Y, 1, **kw)
    r2 = dltmle.estimate(W, L, A, C, Y, 1, **kw)
    assert r1.psi == r2.psi
    np.testing.assert_array_equal(r1.ral().ic, r2.ral().ic)


def test_before_treatment_allows_missing_treatment_on_censoring_rows(long_df):
    df = long_df.copy()
    df.loc[df["cens"] != "none", "drug"] = np.nan
    df.loc[df["cens"] != "none", "event"] = np.nan
    with pytest.raises(ValueError):
        dltmle.estimate_long(df, intervention="SU", **COLS, estimator="linear",
                             hparams=LIN, device="cpu")
    r = dltmle.estimate_long(df, intervention="SU", **COLS, estimator="linear", hparams=LIN,
                             device="cpu", censoring_timing="before_treatment")
    assert r.censoring_timing == "before_treatment" and 0 < r.psi < 1


def test_input_validation(long_df):
    kw = dict(intervention="SU", estimator="linear", hparams=LIN, device="cpu")
    with pytest.raises(ValueError, match="duplicate"):
        dltmle.estimate_long(pd.concat([long_df, long_df.head(1)]), **COLS, **kw)
    gap = long_df[~((long_df["ID"] == 1000) & (long_df["day"] == 30))]
    with pytest.raises(ValueError, match="consecutive"):
        dltmle.estimate_long(gap, **COLS, **kw)
    bad = long_df.copy(); bad.loc[0, "drug"] = "INSULIN"
    with pytest.raises(ValueError, match="unknown treatment"):
        dltmle.estimate_long(bad, **COLS, **kw)
    with pytest.raises(KeyError):
        dltmle.estimate_long(long_df, **dict(COLS, dynamic_cols=["nope"]), **kw)

    # follow-up ending early without a code
    open_end = long_df[~((long_df["ID"] == 1001) & (long_df["day"] >= 60))]
    assert ((long_df["ID"] == 1001) & (long_df["day"] >= 60)).any()
    with pytest.raises(ValueError, match="implicit_censoring"):
        dltmle.estimate_long(open_end, **COLS, **kw)
    r = dltmle.estimate_long(open_end, **COLS, implicit_censoring="disenroll", **kw)
    assert np.isfinite(r.psi)
    p = dltmle.long_to_panel(open_end, **COLS, implicit_censoring="disenroll")
    i = list(p.ids).index(1001)
    np.testing.assert_array_equal(p.C[i, :, 0], [0, 0, 2, 2])


def test_rows_after_event_are_dropped(panel):
    W, L, A, C, Y = panel
    full = dltmle.panel_to_long(W, L, A, C, Y, drop_after_end=False)
    with pytest.warns(UserWarning, match="dropping"):
        p = dltmle.long_to_panel(full, id_col="id", time_col="t", outcome_col="Y",
                                 treatment_col="A", censoring_cols="C",
                                 static_cols=["W0"], dynamic_cols=["L0"])
    np.testing.assert_allclose(p.Y, Y)
    np.testing.assert_allclose(p.C, C)


def test_standardize_and_files(tmp_path, long_df):
    kw = dict(intervention="SU", estimator="linear", hparams=LIN, device="cpu", seed=1)
    r = dltmle.estimate_long(long_df, standardize=True, **COLS, **kw)
    assert set(r.metadata()["standardization"]) == {"static", "dynamic"}
    csv, pq = tmp_path / "d.csv", tmp_path / "d.parquet"
    long_df.to_csv(csv, index=False)
    long_df.to_parquet(pq, index=False)
    r_csv = dltmle.estimate_file(str(csv), **COLS, **kw)
    r_pq = dltmle.estimate_file(str(pq), **COLS, **kw)
    r_df = dltmle.estimate_long(long_df, **COLS, **kw)
    assert r_csv.psi == pytest.approx(r_df.psi, abs=1e-6)
    assert r_pq.psi == r_df.psi
