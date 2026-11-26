"""``python -m dltmle`` command-line interface."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import dltmle
from dltmle.cli import parse_intervention, main


@pytest.fixture(scope="module")
def data_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    W, L, A, C, Y = dltmle.example_dgp_categorical(np.random.default_rng(8), n=400, tau=3)
    df = dltmle.panel_to_long(W, L, A, C, Y, static_names=["age"], dynamic_names=["hba1c"])
    df["drug"] = np.array(["SU", "DPP4", "SGLT2i"])[df["A"]]
    df.to_csv(d / "data.csv", index=False)
    df.to_parquet(d / "data.parquet", index=False)
    return d, df


BASE = ["--id-col", "id", "--time-col", "t", "--outcome-col", "Y", "--treatment-col", "drug",
        "--censoring-cols", "C", "--static-cols", "age", "--dynamic-cols", "hba1c",
        "--treatment-levels", "SU,DPP4,SGLT2i", "--estimator", "linear",
        "--hparams", '{"max_epochs": 10}', "--seed", "3", "--device", "cpu"]


def test_version():
    out = subprocess.run([sys.executable, "-m", "dltmle", "version"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == dltmle.__version__


def test_parse_intervention():
    assert parse_intervention("SU") == ("SU", {"type": "static", "level": "SU"})
    assert parse_intervention("a=static:SU") == ("a", {"type": "static", "level": "SU"})
    assert parse_intervention("d=column:dcol") == ("d", {"type": "column", "column": "dcol"})
    assert parse_intervention("s=stochastic:p0,p1") == (
        "s", {"type": "stochastic", "columns": ["p0", "p1"]})
    assert parse_intervention("p=probs:0.2,0.8") == (
        "p", {"type": "stochastic", "probs": [0.2, 0.8]})
    assert parse_intervention('x={"type": "static", "level": 1}') == (
        "x", {"type": "static", "level": 1})


def test_cli_estimate_matches_python(data_files):
    d, df = data_files
    out_json, est_csv, ic_csv, con_csv = (d / "r.json", d / "est.csv", d / "ic.csv", d / "con.csv")
    cmd = [sys.executable, "-m", "dltmle", "estimate", "--input", str(d / "data.parquet"), *BASE,
           "--intervention", "SGLT2i=static:SGLT2i", "--intervention", "SU=static:SU",
           "--eval-times", "all", "--deterministic",
           "--output", str(out_json), "--estimates-csv", str(est_csv),
           "--ic-csv", str(ic_csv), "--contrasts-csv", str(con_csv), "--reference", "SU"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    res = json.loads(out_json.read_text())
    est = pd.read_csv(est_csv)
    assert len(est) == 6 and list(est["intervention"].unique()) == ["SGLT2i", "SU"]
    assert res["metadata"]["seed"] == 3 and res["metadata"]["deterministic"] is True
    assert res["metadata"]["version"] == dltmle.__version__
    assert len(pd.read_csv(ic_csv)) == 400 * 6
    con = pd.read_csv(con_csv)
    assert set(con["scale"]) == {"difference", "ratio"} and set(con["reference"]) == {"SU"}

    py = dltmle.estimate_long(
        df, id_col="id", time_col="t", outcome_col="Y", treatment_col="drug",
        censoring_cols="C", static_cols=["age"], dynamic_cols=["hba1c"],
        treatment_levels=["SU", "DPP4", "SGLT2i"], estimator="linear",
        hparams={"max_epochs": 10}, seed=3, device="cpu", eval_times="all",
        intervention={"SGLT2i": "SGLT2i", "SU": "SU"}, deterministic=True,
    )
    np.testing.assert_allclose(est["psi"].to_numpy(), py.estimates["psi"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["estimates"]["se"], py.estimates["se"].to_numpy(), atol=1e-12)


def test_cli_config_file_and_csv(data_files, capsys):
    d, _ = data_files
    cfg = {
        "input": str(d / "data.csv"), "id_col": "id", "time_col": "t", "outcome_col": "Y",
        "treatment_col": "drug", "censoring_cols": "C", "static_cols": ["age"],
        "dynamic_cols": ["hba1c"], "treatment_levels": ["SU", "DPP4", "SGLT2i"],
        "estimator": "linear", "hparams": {"max_epochs": 5}, "seed": 0, "device": "cpu",
        "intervention": {"sto": {"type": "stochastic", "probs": [0.2, 0.3, 0.5]}},
    }
    path = d / "cfg.json"
    path.write_text(json.dumps(cfg))
    assert main(["estimate", "--config", str(path), "--output", "-", "--no-ic-json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert "ic" not in res
    assert res["estimates"]["intervention"] == ["sto"]
    assert res["metadata"]["hparams"]["max_epochs"] == 5


def test_cli_errors(data_files, capsys):
    d, _ = data_files
    rc = main(["estimate", "--input", str(d / "data.csv"),
               *[x if x != "hba1c" else "nope" for x in BASE],
               "--intervention", "SU", "--output", str(d / "x.json")])
    assert rc == 2
    assert "nope" in capsys.readouterr().err
    assert main(["estimate", "--input", str(d / "data.csv"), "--output", "-"]) == 2
