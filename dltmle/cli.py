"""Command-line interface: ``python -m dltmle ...``.

Examples
--------
::

    python -m dltmle estimate \\
        --input data.parquet --id-col ID --time-col t --outcome-col Y \\
        --treatment-col drug --censoring-cols cens_reason \\
        --static-cols age,sex --dynamic-cols hba1c,egfr \\
        --intervention SGLT2i=static:SGLT2i --intervention SU=static:SU \\
        --estimator dltmle --hparams '{"max_epochs": 20}' --seed 1 \\
        --device cuda --deterministic --eval-times all \\
        --output result.json --estimates-csv estimates.csv --ic-csv ic.csv

    python -m dltmle estimate --config run.json     # same keys as estimate_long
    python -m dltmle version

Intervention specs (``--intervention [NAME=]SPEC``, repeatable):
``static:LEVEL``, ``column:COL`` (per-row regime class), ``stochastic:COL1,COL2,...``
(per-row class probabilities in level order), ``probs:P1,P2,...`` (constant
probabilities), or a JSON object (``{"type": "static", "level": "SU"}``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _csv_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def parse_intervention(text: str):
    """Parse ``[NAME=]SPEC`` → ``(name or None, spec)``."""
    name = None
    body = text.strip()
    if not body.startswith("{") and "=" in body and ":" not in body.split("=", 1)[0]:
        name, body = body.split("=", 1)
    body = body.strip()
    if body.startswith("{"):
        spec = json.loads(body)
    elif ":" in body:
        kind, arg = body.split(":", 1)
        kind = kind.strip().lower()
        if kind == "static":
            spec = {"type": "static", "level": arg}
        elif kind in ("column", "dynamic"):
            spec = {"type": "column", "column": arg}
        elif kind == "stochastic":
            spec = {"type": "stochastic", "columns": _csv_list(arg)}
        elif kind == "probs":
            spec = {"type": "stochastic", "probs": [float(x) for x in _csv_list(arg)]}
        else:
            raise ValueError(f"unknown intervention kind {kind!r} in {text!r}")
    else:
        spec = {"type": "static", "level": body}
    if name is None:
        name = spec.get("level") or spec.get("column") or "regime"
    return str(name), spec


def _load_json_arg(s):
    if s is None:
        return None
    if isinstance(s, dict):
        return s
    if os.path.exists(s):
        with open(s) as f:
            return json.load(f)
    return json.loads(s)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m dltmle", description="Deep LTMLE command-line interface")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print the package version")

    e = sub.add_parser("estimate", help="estimate counterfactual risks from a long-format file")
    e.add_argument("--config", help="JSON file with estimate_long keyword arguments (CLI flags override)")
    e.add_argument("--input", help="input .parquet or .csv (long format)")
    e.add_argument("--id-col")
    e.add_argument("--time-col")
    e.add_argument("--outcome-col")
    e.add_argument("--treatment-col")
    e.add_argument("--censoring-cols", help="one categorical column or comma-separated binary columns")
    e.add_argument("--static-cols", help="comma-separated baseline covariates")
    e.add_argument("--dynamic-cols", help="comma-separated time-varying covariates")
    e.add_argument("--treatment-levels", help="comma-separated treatment class order")
    e.add_argument("--censoring-levels", help="comma-separated censoring levels (first = not censored)")
    e.add_argument("--censoring-timing", choices=["after_treatment", "before_treatment"])
    e.add_argument("--implicit-censoring", help="censoring level for follow-up ending without a code")
    e.add_argument("--standardize", action="store_true", default=None)
    e.add_argument("--intervention", action="append", help="[NAME=]SPEC, repeatable")
    e.add_argument("--estimator", choices=["dltmle", "linear"])
    e.add_argument("--hparams", help="JSON string or path to a JSON file")
    e.add_argument("--seed", type=int)
    e.add_argument("--device", help="auto | cpu | cuda | cuda:N")
    e.add_argument("--deterministic", action="store_true", default=None)
    e.add_argument("--num-threads", type=int)
    e.add_argument("--eval-times", help="'all', 'last', or comma-separated time values")
    e.add_argument("--ci-level", type=float)
    e.add_argument("--targeting", help="JSON targeting options")
    e.add_argument("--output", help="output JSON path ('-' = stdout)")
    e.add_argument("--estimates-csv")
    e.add_argument("--ic-csv")
    e.add_argument("--contrasts-csv")
    e.add_argument("--reference", help="reference intervention for contrasts")
    e.add_argument("--no-ic-json", action="store_true", help="omit the IC table from the JSON output")
    e.add_argument("--verbose", action="store_true", default=None)
    return p


_KWARG_KEYS = {
    "id_col", "time_col", "outcome_col", "treatment_col", "censoring_cols",
    "static_cols", "dynamic_cols", "treatment_levels", "censoring_levels",
    "censoring_timing", "implicit_censoring", "standardize", "hparams", "seed",
    "estimator", "eval_times", "device", "deterministic", "num_threads",
    "ci_level", "verbose", "targeting", "intervention",
}


def _collect(args):
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    io = {k: cfg.pop(k, None) for k in
          ("input", "output", "estimates_csv", "ic_csv", "contrasts_csv", "reference", "no_ic_json")}

    cli = {
        "id_col": args.id_col, "time_col": args.time_col, "outcome_col": args.outcome_col,
        "treatment_col": args.treatment_col,
        "censoring_cols": _csv_list(args.censoring_cols),
        "static_cols": _csv_list(args.static_cols),
        "dynamic_cols": _csv_list(args.dynamic_cols),
        "treatment_levels": _csv_list(args.treatment_levels),
        "censoring_levels": _csv_list(args.censoring_levels),
        "censoring_timing": args.censoring_timing,
        "implicit_censoring": args.implicit_censoring,
        "standardize": args.standardize,
        "estimator": args.estimator,
        "hparams": _load_json_arg(args.hparams),
        "seed": args.seed, "device": args.device, "deterministic": args.deterministic,
        "num_threads": args.num_threads, "ci_level": args.ci_level,
        "verbose": args.verbose, "targeting": _load_json_arg(args.targeting),
    }
    if args.eval_times is not None:
        cli["eval_times"] = args.eval_times if args.eval_times in ("all", "last") else _csv_list(args.eval_times)
    if args.intervention:
        cli["intervention"] = dict(parse_intervention(s) for s in args.intervention)
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v
    for k, v in {"input": args.input, "output": args.output, "estimates_csv": args.estimates_csv,
                 "ic_csv": args.ic_csv, "contrasts_csv": args.contrasts_csv,
                 "reference": args.reference}.items():
        if v is not None:
            io[k] = v
    if args.no_ic_json:
        io["no_ic_json"] = True

    unknown = set(cfg) - _KWARG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    # config-file interventions may be given as a list of "NAME=SPEC" strings
    iv = cfg.get("intervention")
    if isinstance(iv, list):
        cfg["intervention"] = dict(parse_intervention(s) if isinstance(s, str) else (str(i), s)
                                   for i, s in enumerate(iv))
    if isinstance(cfg.get("censoring_cols"), list) and len(cfg["censoring_cols"]) == 1:
        cfg["censoring_cols"] = cfg["censoring_cols"][0]
    for req in ("id_col", "time_col", "outcome_col", "treatment_col", "intervention"):
        if cfg.get(req) is None:
            raise ValueError(f"missing required option: {req.replace('_', '-')}")
    if not io.get("input"):
        raise ValueError("missing required option: input")
    return cfg, io


def run_estimate(args) -> int:
    from .api import estimate_file

    cfg, io = _collect(args)
    res = estimate_file(io["input"], **cfg)

    payload = res.to_json(include_ic=not io.get("no_ic_json"), reference=io.get("reference"), indent=1)
    out = io.get("output")
    if out in (None, "-"):
        sys.stdout.write(payload + "\n")
    else:
        with open(out, "w") as f:
            f.write(payload)
    if io.get("estimates_csv"):
        res.estimates_frame().to_csv(io["estimates_csv"], index=False)
    if io.get("ic_csv"):
        res.ic_frame().to_csv(io["ic_csv"], index=False)
    if io.get("contrasts_csv"):
        res.contrasts(io.get("reference")).to_csv(io["contrasts_csv"], index=False)
    for w in res.diagnostics.get("warnings", []):
        print(f"warning: {w}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        from . import __version__
        print(__version__)
        return 0
    try:
        return run_estimate(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
