import os
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm

from ..model.dltmle import DeepLTMLE
from ..model.linear import LinearGModel, LinearQModel
from ..ral import RALEstimate
from ..utils import seed_everything, get_torch_device, get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetWLACYa(Dataset):
    def __init__(self, W, L, A, C, Y, a, dtype=torch.float32):
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.C = torch.tensor(C, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)
        self.a = torch.tensor(a, dtype=dtype)

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
            "a": self.a[index],
        }


# ---------------------------------------------------------------------------
# Loss helpers  (extracted from DeepLTMLE.loss for separate G / Q passes)
# ---------------------------------------------------------------------------

def _g_loss(G_a, G_c, batch, alpha: float = 1.0, beta: float = 1.0):
    A, C, Y = batch["A"], batch["C"], batch["Y"]
    bce = nn.BCELoss(reduction="none")

    R   = torch.ones_like(Y);   R[:, 1:]   = 1 - Y[:, :-1]
    R_C = torch.ones_like(C);   R_C[:, 1:] = 1 - C[:, :-1]

    loss_G_a = (R * R_C * bce(G_a, A)).sum(dim=1).mean()
    loss_G_c = (R * R_C * bce(G_c, C)).sum(dim=1).mean()
    return alpha * loss_G_a + beta * loss_G_c


def _q_loss(logit_Q, logit_V, batch):
    """Q-learning loss: Q(t) ≈ V_set(t+1).

    V_set(t) = 1 if Y(t-1)==1 (event absorption), V(t) otherwise.
    V_set(tau) = Y(tau-1) (terminal boundary).
    """
    C, Y = batch["C"], batch["Y"]
    bce = nn.BCELoss(reduction="none")

    R = torch.ones_like(Y);  R[:, 1:] = 1 - Y[:, :-1]   # survival indicator

    Q = torch.sigmoid(logit_Q)                                # (B, tau, 1)
    V = torch.sigmoid(logit_V.detach())                       # (B, tau, 1)

    # Apply event absorption: V(t) = 1 if Y(t-1) == 1 for t >= 1
    V_absorbed = torch.cat([
        V[:, :1],
        torch.where(Y[:, :-1] == 1, torch.ones_like(V[:, 1:]), V[:, 1:]),
    ], dim=1)   # (B, tau, 1)

    # Shift: target for Q(t) is V_set(t+1); last step uses terminal Y(tau-1)
    V_target = torch.cat([V_absorbed[:, 1:], Y[:, -1:]], dim=1)   # (B, tau, 1)

    return (R * (1 - C) * bce(Q, V_target)).sum(dim=1).mean()


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
    return results


# ---------------------------------------------------------------------------
# Core training loop (shared by fit and fit_linear)
# ---------------------------------------------------------------------------

def _train_loop(
    g_model,
    q_model,
    loader,
    hparams: dict,
    device,
    log_dir: str = None,
):
    g_model.to(device)
    q_model.to(device)

    opt_g = Adam(g_model.parameters(), lr=hparams["learning_rate"])
    opt_q = Adam(q_model.parameters(), lr=hparams["learning_rate"])

    max_epochs  = hparams["max_epochs"]
    total_steps = len(loader) * max_epochs
    warmup      = int(total_steps * hparams.get("warmup_ratio", 0.05))

    sched_g = get_cosine_schedule_with_warmup(opt_g, warmup, total_steps)
    sched_q = get_cosine_schedule_with_warmup(opt_q, warmup, total_steps)

    alpha = hparams.get("alpha", 1.0)
    beta  = hparams.get("beta",  1.0)

    writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

    global_step = 0
    for epoch in tqdm(range(max_epochs), desc="Training"):
        g_model.train()
        q_model.train()

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- G step ---
            G_a, G_c = g_model(batch["W"], batch["L"], batch["A"])
            loss_g = _g_loss(G_a, G_c, batch, alpha=alpha, beta=beta)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            sched_g.step()

            # --- Q step ---
            logit_Q, logit_V = q_model(batch["W"], batch["L"], batch["A"], batch["a"])
            loss_q = _q_loss(logit_Q, logit_V, batch)
            opt_q.zero_grad()
            loss_q.backward()
            opt_q.step()
            sched_q.step()

            # EMA update (only for transformer-based QModel; linear has no EMA)
            if hasattr(q_model, "update_ema"):
                q_model.update_ema()

            if writer is not None:
                writer.add_scalar("train/loss_G", loss_g.item(), global_step)
                writer.add_scalar("train/loss_Q", loss_q.item(), global_step)
            global_step += 1

    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit(rng_seed: int, hparams: dict, W, L, A, C, Y, a):
    """Train DeepLTMLE and return a TMLE estimate.

    Parameters
    ----------
    rng_seed : int
    hparams : dict
        Keys: dim_model, num_layers, nhead, dim_feedforward (optional),
        dropout, learning_rate, alpha, beta, ema_decay, max_epochs,
        batch_size, warmup_ratio (optional).
    W, L, A, C, Y : np.ndarray
        Longitudinal panel data.
    a : np.ndarray
        Counterfactual treatment regime.

    Returns
    -------
    RALEstimate
    """
    seed_everything(rng_seed)
    device = get_torch_device()

    dataset = DatasetWLACYa(W, L, A, C, Y, a)
    loader  = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=True)

    model = DeepLTMLE(
        dataset.dim_static, dataset.dim_dynamic, dataset.tau, **hparams
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir   = os.path.join("artifact", "fit", timestamp)

    _train_loop(model.g_model, model.q_model, loader, hparams, device, log_dir)

    # Canonical gradient debiasing
    pred_loader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=False)
    model.to(device)
    model.solve_canonical_gradient_common_eps(
        pred_loader, device, stop_pnic_se_ratio=True
    )

    preds = _predict(model, pred_loader, device)
    est, se, ic = model.get_estimates_from_prediction(preds)
    return RALEstimate(est, ic)


def fit_linear(rng_seed: int, hparams: dict, W, L, A, C, Y, a):
    """Train a linear (logistic regression) TMLE baseline and return an estimate.

    Parameters
    ----------
    rng_seed : int
    hparams : dict
        Keys: learning_rate, max_epochs, batch_size, alpha (optional),
        beta (optional), warmup_ratio (optional).
    W, L, A, C, Y : np.ndarray
    a : np.ndarray

    Returns
    -------
    RALEstimate
    """
    seed_everything(rng_seed)
    device = get_torch_device()

    dataset = DatasetWLACYa(W, L, A, C, Y, a)
    loader  = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=True)

    g_model = LinearGModel(dataset.dim_static, dataset.dim_dynamic)
    q_model = LinearQModel(dataset.dim_static, dataset.dim_dynamic)

    _train_loop(g_model, q_model, loader, hparams, device)

    # Build a DeepLTMLE shell to reuse forward / debiasing / estimation logic
    from ..model.dltmle import DeepLTMLE as _DLTMLE

    class _LinearTMLE(_DLTMLE):
        """Thin wrapper that swaps in pre-trained linear G/Q models."""
        def __init__(self, g, q, dim_static, dim_dynamic, tau, alpha, beta):
            nn.Module.__init__(self)
            self.tau     = tau
            self.alpha   = alpha
            self.beta    = beta
            self.g_model = g
            self.q_model = q
            self.eps     = nn.Parameter(torch.zeros(tau), requires_grad=False)

    alpha = hparams.get("alpha", 1.0)
    beta  = hparams.get("beta",  1.0)
    wrapper = _LinearTMLE(
        g_model, q_model,
        dataset.dim_static, dataset.dim_dynamic, dataset.tau,
        alpha=alpha, beta=beta,
    )

    pred_loader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=False)
    wrapper.to(device)
    wrapper.solve_canonical_gradient_common_eps(
        pred_loader, device, stop_pnic_se_ratio=True
    )

    preds = _predict(wrapper, pred_loader, device)
    est, se, ic = wrapper.get_estimates_from_prediction(preds)
    return RALEstimate(est, ic)
