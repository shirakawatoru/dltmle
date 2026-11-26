import os
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm

from ..model.dltmle import DeepLTMLE
from ..model.q_model import QModel
from ..model.linear import LinearGModel, LinearQModel
from ..model.categorical import categorical_nll, mix_logits, check_censoring_timing
from ..intervention import resolve_intervention
from ..ral import RALEstimate
from ..utils import (
    seed_everything, deterministic_mode, resolve_device,
    get_cosine_schedule_with_warmup,
)


DEFAULT_HPARAMS = {
    "dim_model":     32,
    "num_layers":    2,
    "nhead":         4,
    "dim_feedforward": None,
    "dropout":       0.1,
    "learning_rate": 1e-3,
    "alpha":         1.0,
    "beta":          1.0,
    "ema_decay":     0.99,
    "max_epochs":    50,
    "batch_size":    256,
    "warmup_ratio":  0.05,
    "max_weight":    100.0,
    "init_bias":     False,
}

DEFAULT_LINEAR_HPARAMS = {
    "learning_rate": 1e-2,
    "alpha":         1.0,
    "beta":          1.0,
    "max_epochs":    200,
    "batch_size":    256,
    "warmup_ratio":  0.05,
    "max_weight":    100.0,
    "init_bias":     False,
}

ESTIMATORS = ("dltmle", "linear")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetWLACYa(Dataset):
    """Panel dataset.  ``gstar`` is the (n, tau, J) intervention probability
    array (see :mod:`dltmle.intervention`)."""

    def __init__(self, W, L, A, C, Y, gstar, dtype=torch.float32):
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.C = torch.tensor(C, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.gstar = torch.tensor(gstar, dtype=dtype)

        self.dim_static  = W.shape[1]
        self.dim_dynamic = L.shape[2]
        self.tau         = L.shape[1]

    def __len__(self):
        return self.W.shape[0]

    def __getitem__(self, index):
        return {
            "W": self.W[index],
            "L": self.L[index],
            "A": self.A[index],
            "C": self.C[index],
            "Y": self.Y[index],
            "gstar": self.gstar[index],
        }

    def view(self, h: int, gstar: torch.Tensor) -> "DatasetWLACYa":
        """Zero-copy view truncated to the first ``h`` time points with
        intervention ``gstar`` (n, tau, J) — avoids duplicating the panel
        for every (intervention, horizon) job."""
        v = DatasetWLACYa.__new__(DatasetWLACYa)
        v.W = self.W
        v.L, v.A, v.C, v.Y = (x[:, :h] for x in (self.L, self.A, self.C, self.Y))
        v.gstar = gstar[:, :h]
        v.dim_static, v.dim_dynamic, v.tau = self.dim_static, self.dim_dynamic, h
        return v


class BatchLoader:
    """Minimal in-memory batch loader over a :class:`DatasetWLACYa`.

    Slices whole batches from the stored tensors instead of collating
    individual samples (``torch.utils.data.DataLoader`` with per-item
    ``__getitem__`` dominates runtime for large n).  It consumes the global
    torch RNG exactly like ``DataLoader(shuffle=...)`` with the default
    ``RandomSampler`` (one base-seed draw per epoch, then one sampler-seed
    draw when shuffling), so results are identical to the DataLoader-based
    implementation for a fixed seed.
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self._keys = ("W", "L", "A", "C", "Y", "gstar")

    def __len__(self):
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        torch.empty((), dtype=torch.int64).random_()        # DataLoader base seed
        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            gen = torch.Generator()
            gen.manual_seed(seed)
            idx = torch.randperm(n, generator=gen)
        else:
            idx = None
        tensors = {k: getattr(self.dataset, k) for k in self._keys}
        for i in range(0, n, self.batch_size):
            if idx is None:
                yield {k: t[i:i + self.batch_size] for k, t in tensors.items()}
            else:
                b = idx[i:i + self.batch_size]
                yield {k: t[b] for k, t in tensors.items()}


# ---------------------------------------------------------------------------
# Loss helpers  (separate G / Q passes)
# ---------------------------------------------------------------------------

def _at_risk(C, Y):
    """Survival indicator R(t) = 1 - Y(t-1) and not-yet-censored R_C(t)."""
    R = torch.ones_like(Y)
    R[:, 1:] = 1 - Y[:, :-1]
    cens = (C > 0).to(C.dtype)
    R_C = torch.ones_like(C)
    R_C[:, 1:] = 1 - torch.cummax(cens, dim=1).values[:, :-1]
    return R, R_C


def _g_loss_parts(p_A, p_C, batch, censoring_timing: str = "after_treatment"):
    A, C, Y = batch["A"], batch["C"], batch["Y"]
    R, R_C = _at_risk(C, Y)
    # with C_t before A_t, A_t is not observed on the censoring row itself
    R_A = R_C if censoring_timing == "after_treatment" else R_C * (C == 0).to(C.dtype)
    loss_G_a = (R * R_A * categorical_nll(p_A, A)).sum(dim=1).mean()
    loss_G_c = (R * R_C * categorical_nll(p_C, C)).sum(dim=1).mean()
    return loss_G_a, loss_G_c


def _g_loss(p_A, p_C, batch, alpha: float = 1.0, beta: float = 1.0,
            censoring_timing: str = "after_treatment"):
    """Multinomial log-likelihood of treatment (J classes) and censoring
    (K categories) over at-risk, not-yet-censored person-time."""
    loss_G_a, loss_G_c = _g_loss_parts(p_A, p_C, batch, censoring_timing)
    return alpha * loss_G_a + beta * loss_G_c


def _q_loss(logit_Q, logit_V, batch):
    """Q-learning loss: Q(t) ≈ V_set(t+1).

    ``logit_V`` is the target-network value under the intervention,
    ``V(t) = sum_a g*(a|H_t) Q_t(a)`` (see :func:`mix_logits`).
    V_set(t) = 1 if Y(t-1)==1 (event absorption), V(t) otherwise.
    V_set(tau) = Y(tau-1) (terminal boundary).
    """
    C, Y = batch["C"], batch["Y"]
    bce = nn.BCELoss(reduction="none")

    R = torch.ones_like(Y);  R[:, 1:] = 1 - Y[:, :-1]   # survival indicator
    uncens = (C == 0).to(Y.dtype)

    Q = torch.sigmoid(logit_Q)                                # (B, tau, 1)
    V = torch.sigmoid(logit_V.detach())                       # (B, tau, 1)

    V_absorbed = torch.cat([
        V[:, :1],
        torch.where(Y[:, :-1] == 1, torch.ones_like(V[:, 1:]), V[:, 1:]),
    ], dim=1)   # (B, tau, 1)

    V_target = torch.cat([V_absorbed[:, 1:], Y[:, -1:]], dim=1)   # (B, tau, 1)

    return (R * uncens * bce(Q, V_target)).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Predict helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    results = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        results.append({k: v.cpu() for k, v in out.items()})
        results[-1]["Y"] = batch["Y"].cpu()
        results[-1]["A"] = batch["A"].cpu()
        results[-1]["C"] = batch["C"].cpu()
    return results


# ---------------------------------------------------------------------------
# Core training loop (shared by all estimators)
# ---------------------------------------------------------------------------

def _train_loop(
    g_model,
    q_model,
    loader,
    hparams: dict,
    device,
    log_dir: str = None,
    verbose: bool = True,
):
    """Train ``g_model`` and/or ``q_model`` (either may be ``None``).

    When both are given, the G step and the Q step alternate on the same
    batch (the original joint schedule).  Returns a per-epoch loss history
    (column-oriented dict).
    """
    models = [m for m in (g_model, q_model) if m is not None]
    for m in models:
        m.to(device)

    max_epochs  = hparams["max_epochs"]
    total_steps = len(loader) * max_epochs
    warmup      = int(total_steps * hparams.get("warmup_ratio", 0.05))

    if g_model is not None:
        opt_g = Adam(g_model.parameters(), lr=hparams["learning_rate"])
    if q_model is not None:
        opt_q = Adam(q_model.parameters(), lr=hparams["learning_rate"])
    if g_model is not None:
        sched_g = get_cosine_schedule_with_warmup(opt_g, warmup, total_steps)
    if q_model is not None:
        sched_q = get_cosine_schedule_with_warmup(opt_q, warmup, total_steps)

    alpha = hparams.get("alpha", 1.0)
    beta  = hparams.get("beta",  1.0)
    timing = getattr(g_model, "censoring_timing", "after_treatment")

    writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    history = {"epoch": [], "loss_G_a": [], "loss_G_c": [], "loss_Q": []}

    global_step = 0
    for epoch in tqdm(range(max_epochs), desc="Training", disable=not verbose):
        for m in models:
            m.train()

        sums = torch.zeros(3, device=device)
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- G step ---
            if g_model is not None:
                p_A, p_C = g_model(batch["W"], batch["L"], batch["A"])
                loss_G_a, loss_G_c = _g_loss_parts(p_A, p_C, batch, timing)
                loss_g = alpha * loss_G_a + beta * loss_G_c
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                sched_g.step()
                sums[0] += loss_G_a.detach()
                sums[1] += loss_G_c.detach()

            # --- Q step ---
            if q_model is not None:
                logit_Q, logit_V_cls = q_model(batch["W"], batch["L"], batch["A"])
                logit_V = mix_logits(logit_V_cls, batch["gstar"])
                loss_q = _q_loss(logit_Q, logit_V, batch)
                opt_q.zero_grad()
                loss_q.backward()
                opt_q.step()
                sched_q.step()
                sums[2] += loss_q.detach()

                # EMA update (only for transformer-based QModel; linear has no EMA)
                if hasattr(q_model, "update_ema"):
                    q_model.update_ema()

            if writer is not None:
                if g_model is not None:
                    writer.add_scalar("train/loss_G", loss_g.item(), global_step)
                if q_model is not None:
                    writer.add_scalar("train/loss_Q", loss_q.item(), global_step)
            global_step += 1
            n_batches += 1

        means = (sums / max(n_batches, 1)).tolist()
        history["epoch"].append(epoch)
        history["loss_G_a"].append(means[0] if g_model is not None else None)
        history["loss_G_c"].append(means[1] if g_model is not None else None)
        history["loss_Q"].append(means[2] if q_model is not None else None)

    if writer is not None:
        writer.close()

    return history


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _as_panel(W, L, A, C, Y):
    W = np.asarray(W, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if W.ndim == 1:
        W = W[:, None]
    if L.ndim == 2:
        L = L[:, :, None]
    A, C, Y = [x[:, :, None] if x.ndim == 2 else x for x in (A, C, Y)]
    n, tau = L.shape[0], L.shape[1]
    for name, x in (("W", W), ("A", A), ("C", C), ("Y", Y)):
        if x.shape[0] != n:
            raise ValueError(f"{name} has {x.shape[0]} rows, expected n = {n}")
    for name, x in (("A", A), ("C", C), ("Y", Y)):
        if x.shape != (n, tau, 1):
            raise ValueError(f"{name} must have shape (n, tau[, 1]) = {(n, tau, 1)}, got {x.shape}")
    for name, x in (("W", W), ("L", L), ("A", A), ("C", C), ("Y", Y)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} contains NaN / inf")
    for name, x in (("A", A), ("C", C)):
        if np.any(x < 0) or not np.all(np.equal(np.mod(x, 1), 0)):
            raise ValueError(f"{name} must contain non-negative integer class codes")
    if not np.all(np.isin(Y, (0, 1))):
        raise ValueError("Y must be binary (0/1)")
    return W, L, A, C, Y


def _n_classes(x, given, name):
    observed = int(x.max()) + 1 if x.size else 1
    if given is None:
        return max(observed, 2)
    given = int(given)
    if given < 2:
        raise ValueError(f"{name} must be >= 2")
    if observed > given:
        raise ValueError(f"{name}={given} but data contain class {observed - 1}")
    return given


def _resolve_hparams(hparams, estimator):
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    base = DEFAULT_HPARAMS if estimator == "dltmle" else DEFAULT_LINEAR_HPARAMS
    out = dict(base)
    out.update(dict(hparams or {}))
    for k in ("n_treatment", "n_censoring"):
        out.pop(k, None)
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _propensity_from_g(g_model, W, L, A, C, Y, hp, device, J, K, censoring_timing):
    """Fit diagnostics of g^A / g^C on the full panel."""
    g_model.eval()
    dtype = torch.float32
    bs = hp["batch_size"]
    pa, pc = [], []
    for s in range(0, W.shape[0], bs):
        sl = slice(s, s + bs)
        p_A, p_C = g_model(
            torch.tensor(W[sl], dtype=dtype, device=device),
            torch.tensor(L[sl], dtype=dtype, device=device),
            torch.tensor(A[sl], dtype=dtype, device=device),
        )
        pa.append(p_A.double().cpu().numpy())
        pc.append(p_C.double().cpu().numpy())
    return _propensity_diagnostics(
        np.concatenate(pa), np.concatenate(pc),
        A[..., 0].astype(int), C[..., 0].astype(int), Y[..., 0], J, K,
        censoring_timing,
    )


def _propensity_diagnostics(p_A, p_C, A, C, Y, J, K, censoring_timing="after_treatment"):

    n, tau = A.shape
    R = np.ones((n, tau)); R[:, 1:] = 1 - Y[:, :-1]
    cens = np.maximum.accumulate(C > 0, axis=1)
    R_C = np.ones((n, tau)); R_C[:, 1:] = 1 - cens[:, :-1]
    risk = (R * R_C) > 0
    risk_A = risk if censoring_timing == "after_treatment" else risk & (C == 0)

    pa_obs = np.take_along_axis(p_A, A[..., None], -1)[..., 0][risk_A]
    pc0 = p_C[..., 0][risk]

    def _q(x, qs=(0.0, 0.001, 0.01, 0.5)):
        return {f"q{q:g}": float(np.quantile(x, q)) for q in qs} if x.size else {}

    return {
        "n_person_time_at_risk": int(risk.sum()),
        "treatment_observed_freq": [float((A[risk_A] == j).mean()) for j in range(J)],
        "treatment_mean_predicted": [float(p_A[..., j][risk_A].mean()) for j in range(J)],
        "censoring_observed_freq": [float((C[risk] == k).mean()) for k in range(K)],
        "censoring_mean_predicted": [float(p_C[..., k][risk].mean()) for k in range(K)],
        "p_treatment_observed_quantiles": _q(pa_obs),
        "p_uncensored_quantiles": _q(pc0),
    }


def _weight_diagnostics(preds, max_weight):
    g = torch.cat([p["g"] for p in preds]).double().numpy()[:, -1, 0]   # cumulative at tau
    nz = g[g > 0]
    ess = float(g.sum() ** 2 / (g ** 2).sum()) if (g ** 2).sum() > 0 else 0.0
    return {
        "n_nonzero": int(nz.size),
        "mean": float(g.mean()),
        "max": float(g.max()) if g.size else 0.0,
        "q0.99_nonzero": float(np.quantile(nz, 0.99)) if nz.size else 0.0,
        "frac_clipped": float((g >= max_weight).mean()),
        "ess": ess,
    }


def _job_warnings(job, name, t_idx):
    out = []
    tg, wt = job["targeting"], job["weights"]
    where = f"intervention {name!r}, time index {t_idx}"
    if not tg["converged"]:
        out.append(f"{where}: targeting did not converge ({tg['n_iter']} iterations)")
    if abs(tg["eps_total"]) > 5:
        out.append(f"{where}: large fluctuation eps = {tg['eps_total']:.2f} "
                   "(sparse events among regime followers or poor initial Q); "
                   "the estimate / SE may be unreliable")
    if wt["frac_clipped"] > 0.01:
        out.append(f"{where}: {100 * wt['frac_clipped']:.1f}% of cumulative weights clipped "
                   "at max_weight (practical positivity violation)")
    if wt["n_nonzero"] < 20:
        out.append(f"{where}: only {wt['n_nonzero']} subjects follow the regime with nonzero weight")
    return out


def _propensity_warnings(prop):
    out = []
    q = prop.get("p_treatment_observed_quantiles", {})
    if q and q.get("q0.01", 1.0) < 0.01:
        out.append("1% of at-risk person-time has P(observed treatment) < 0.01 (positivity)")
    for what in ("treatment", "censoring"):
        obs = np.asarray(prop[f"{what}_observed_freq"])
        pred = np.asarray(prop[f"{what}_mean_predicted"])
        if np.any(np.abs(obs - pred) > 0.05 + 0.25 * obs):
            out.append(f"g^{what[0].upper()} looks miscalibrated (mean predicted "
                       f"{np.round(pred, 3).tolist()} vs observed {np.round(obs, 3).tolist()}); "
                       "consider more epochs / a larger learning rate")
    return out


@torch.no_grad()
def _init_g_bias(g_model, A, C, Y, censoring_timing):
    """Initialise the g^A / g^C head biases at the log marginal odds vs class 0
    (opt-in hparam ``init_bias``; speeds up fitting of rare classes)."""
    A_i, C_i, Y0 = A[..., 0].astype(int), C[..., 0].astype(int), Y[..., 0]
    n, tau = A_i.shape
    R = np.ones((n, tau)); R[:, 1:] = 1 - Y0[:, :-1]
    cens = np.maximum.accumulate(C_i > 0, axis=1)
    R_C = np.ones((n, tau)); R_C[:, 1:] = 1 - cens[:, :-1]
    risk = (R * R_C) > 0
    risk_A = risk if censoring_timing == "after_treatment" else risk & (C_i == 0)
    for head, x, m, mask in ((g_model.G_a, A_i, g_model.n_treatment, risk_A),
                             (g_model.G_c, C_i, g_model.n_censoring, risk)):
        freq = np.bincount(x[mask], minlength=m).astype(float) + 0.5
        head[0].bias.copy_(torch.tensor(np.log(freq[1:] / freq[0]), dtype=head[0].bias.dtype))


# ---------------------------------------------------------------------------
# Estimation engine
# ---------------------------------------------------------------------------

def run_estimation(
    W, L, A, C, Y,
    gstars: dict,
    hparams: dict,
    *,
    estimator: str = "dltmle",
    seed: int = 0,
    device=None,
    deterministic: bool = False,
    time_indices=None,
    n_treatment: int = None,
    n_censoring: int = None,
    verbose: bool = False,
    log_dir: str = None,
    targeting: dict = None,
    censoring_timing: str = "after_treatment",
):
    """Train nuisance models and compute TMLE estimates.

    One propensity model (g^A, g^C) is trained jointly with the Q model of
    the first job; it is shared by all jobs because its predictions at time t
    only depend on the history up to t.  A job is one (intervention, time)
    pair; each job has its own Q model (Q depends on both g* and the target
    horizon).  The first job is (first intervention, last requested time),
    which reproduces the single-estimate behaviour of :func:`fit` exactly.

    ``censoring_timing`` gives the within-interval order of A_t and C_t
    (``"after_treatment"``: L→A→C→Y, dltmle default; ``"before_treatment"``:
    L→C→A→Y, stremr convention).

    Returns a dict with keys ``jobs`` (list of per-job results), ``train_g``,
    ``propensity``, ``hparams``, ``n_treatment``, ``n_censoring``, ``device``.
    """
    W, L, A, C, Y = _as_panel(W, L, A, C, Y)
    n, tau = L.shape[0], L.shape[1]
    J = _n_classes(A, n_treatment, "n_treatment")
    K = _n_classes(C, n_censoring, "n_censoring")
    hp = _resolve_hparams(hparams, estimator)
    check_censoring_timing(censoring_timing)
    device = resolve_device(device)
    tgt = {"stop_pnic_se_ratio": True}
    tgt.update(targeting or {})

    if not gstars:
        raise ValueError("at least one intervention is required")
    for name, gs in gstars.items():
        if gs.shape != (n, tau, J):
            raise ValueError(f"intervention {name!r}: gstar shape {gs.shape} != {(n, tau, J)}")

    if time_indices is None:
        time_indices = [tau - 1]
    time_indices = sorted({int(t) for t in time_indices})
    if time_indices[0] < 0 or time_indices[-1] >= tau:
        raise ValueError(f"time indices must lie in 0..{tau - 1}")

    names = list(gstars)
    jobs = [(names[0], time_indices[-1])]
    jobs += [(nm, t) for nm in names for t in reversed(time_indices) if (nm, t) != jobs[0]]

    full = DatasetWLACYa(W, L, A, C, Y, gstars[names[0]])
    gstar_t = {nm: torch.tensor(gs, dtype=torch.float32) for nm, gs in gstars.items()}

    with deterministic_mode(deterministic):
        seed_everything(seed)

        g_model = None
        results = []
        warnings_out = []
        train_g = None
        propensity = None
        for k, (name, t_idx) in enumerate(jobs):
            h = t_idx + 1
            dataset = full.view(h, gstar_t[name])
            if k > 0:
                seed_everything(int(seed) + k)
            loader = BatchLoader(dataset, hp["batch_size"], shuffle=True)

            if k == 0:
                if estimator == "dltmle":
                    model = DeepLTMLE(
                        dataset.dim_static, dataset.dim_dynamic, dataset.tau,
                        n_treatment=J, n_censoring=K,
                        censoring_timing=censoring_timing, **hp,
                    )
                    g_model, q_model = model.g_model, model.q_model
                else:
                    g_model = LinearGModel(dataset.dim_static, dataset.dim_dynamic, J, K,
                                           censoring_timing)
                    q_model = LinearQModel(dataset.dim_static, dataset.dim_dynamic, J)
                if hp.get("init_bias", False):
                    _init_g_bias(g_model, A, C, Y, censoring_timing)
                hist = _train_loop(g_model, q_model, loader, hp, device, log_dir, verbose)
                train_g = {key: hist[key] for key in ("epoch", "loss_G_a", "loss_G_c")}
            else:
                if estimator == "dltmle":
                    q_model = QModel(
                        dataset.dim_static, dataset.dim_dynamic, dataset.tau,
                        hp["dim_model"], hp["nhead"], hp["num_layers"],
                        hp.get("dim_feedforward"), hp.get("dropout", 0.1),
                        n_treatment=J, ema_decay=hp.get("ema_decay", 0.99),
                    )
                else:
                    q_model = LinearQModel(dataset.dim_static, dataset.dim_dynamic, J)
                hist = _train_loop(None, q_model, loader, hp, device, None, verbose)

            tmle = DeepLTMLE.from_models(
                g_model, q_model, h,
                alpha=hp.get("alpha", 1.0), beta=hp.get("beta", 1.0),
                max_weight=hp.get("max_weight", 100.0),
            )
            tmle.to(device)
            pred_loader = BatchLoader(dataset, hp["batch_size"], shuffle=False)
            tdiag = tmle.solve_canonical_gradient_common_eps(pred_loader, device, verbose=verbose, **tgt)
            preds = _predict(tmle, pred_loader, device)
            est, se, ic = tmle.get_estimates_from_prediction(preds, verbose=verbose)
            tdiag["pnic"] = float(ic.mean())
            tdiag["pnic_over_se"] = float(abs(ic.mean()) / se) if se > 0 else 0.0

            if k == 0:
                propensity = _propensity_from_g(g_model, W, L, A, C, Y, hp, device, J, K,
                                                censoring_timing)

            results.append({
                "intervention": name,
                "time_index": t_idx,
                "est": est,
                "se": se,
                "ic": ic,
                "targeting": tdiag,
                "weights": _weight_diagnostics(preds, hp.get("max_weight", 100.0)),
                "train_q": {key: hist[key] for key in ("epoch", "loss_Q")},
            })
            warnings_out.extend(_job_warnings(results[-1], name, t_idx))

    hp_out = dict(hp)
    hp_out.update({"n_treatment": J, "n_censoring": K})
    return {
        "jobs": results,
        "train_g": train_g,
        "propensity": propensity,
        "warnings": _propensity_warnings(propensity) + warnings_out,
        "hparams": hp_out,
        "n_treatment": J,
        "n_censoring": K,
        "n": n,
        "tau": tau,
        "device": device,
        "censoring_timing": censoring_timing,
    }


# ---------------------------------------------------------------------------
# Public (legacy array) API
# ---------------------------------------------------------------------------

def _legacy_fit(estimator, rng_seed, hparams, W, L, A, C, Y, a,
                n_treatment, n_censoring, device, deterministic, verbose, log_dir):
    W, L, A, C, Y = _as_panel(W, L, A, C, Y)
    J = _n_classes(A, n_treatment, "n_treatment")
    gstar = resolve_intervention(a, W, L, A, J)
    out = run_estimation(
        W, L, A, C, Y, {"regime": gstar}, hparams,
        estimator=estimator, seed=rng_seed, device=device,
        deterministic=deterministic, n_treatment=J, n_censoring=n_censoring,
        verbose=verbose, log_dir=log_dir,
    )
    job = out["jobs"][0]
    res = RALEstimate(job["est"], job["ic"])
    res.diagnostics = {
        "targeting": job["targeting"], "weights": job["weights"],
        "propensity": out["propensity"], "train_g": out["train_g"],
        "train_q": job["train_q"],
    }
    return res


def fit(rng_seed: int, hparams: dict, W, L, A, C, Y, a, *,
        n_treatment: int = None, n_censoring: int = None,
        device=None, deterministic: bool = False, verbose: bool = True):
    """Train DeepLTMLE and return a TMLE estimate of ``P(Y_{tau-1} = 1)``
    under the intervention ``a``.

    Parameters
    ----------
    rng_seed : int
    hparams : dict
        Keys: dim_model, num_layers, nhead, dim_feedforward (optional),
        dropout, learning_rate, alpha, beta, ema_decay, max_epochs,
        batch_size, warmup_ratio (optional), max_weight (optional).
    W, L, A, C, Y : np.ndarray
        Longitudinal panel data.  ``A`` holds treatment class codes
        ``0..J-1`` (binary 0/1 as before); ``C`` holds censoring categories
        ``0..K-1`` with 0 = not censored (binary 0/1 as before; absorbing).
    a : intervention
        Counterfactual regime: array ``(n, tau, 1)`` of classes (legacy),
        ``(n, tau, J)`` probabilities, an int, or a
        :mod:`dltmle.intervention` object.  See
        :func:`dltmle.intervention.resolve_intervention`.
    n_treatment, n_censoring : int, optional
        Number of classes J / K (default: inferred, at least 2).
    device : str, optional
        ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or ``None``/``"auto"``.
    deterministic : bool
        Use deterministic algorithms.

    Returns
    -------
    RALEstimate  (with a ``diagnostics`` attribute)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir   = os.path.join("artifact", "fit", timestamp)
    return _legacy_fit("dltmle", rng_seed, hparams, W, L, A, C, Y, a,
                       n_treatment, n_censoring, device, deterministic, verbose, log_dir)


def fit_linear(rng_seed: int, hparams: dict, W, L, A, C, Y, a, *,
               n_treatment: int = None, n_censoring: int = None,
               device=None, deterministic: bool = False, verbose: bool = True):
    """Train a linear (pooled multinomial logistic regression) TMLE baseline.

    Same arguments as :func:`fit`; ``hparams`` keys: learning_rate,
    max_epochs, batch_size, alpha (optional), beta (optional),
    warmup_ratio (optional).

    Returns
    -------
    RALEstimate
    """
    return _legacy_fit("linear", rng_seed, hparams, W, L, A, C, Y, a,
                       n_treatment, n_censoring, device, deterministic, verbose, None)
