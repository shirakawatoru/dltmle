import os
import datetime

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import optuna

from ..model.dltmle import DeepLTMLE
from ..utils import seed_everything, get_torch_device, get_cosine_schedule_with_warmup
from .fit import _g_loss, _q_loss


# ---------------------------------------------------------------------------
# Dataset  (without counterfactual treatment)
# ---------------------------------------------------------------------------

class DatasetWLACY(Dataset):
    def __init__(self, W, L, A, C, Y, dtype=torch.float32):
        self.W = torch.tensor(W, dtype=dtype)
        self.L = torch.tensor(L, dtype=dtype)
        self.A = torch.tensor(A, dtype=dtype)
        self.C = torch.tensor(C, dtype=dtype)
        self.Y = torch.tensor(Y, dtype=dtype)

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
            "a": self.A[index],   # use observed A as stand-in for tune
        }


# ---------------------------------------------------------------------------
# Validation loss helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_gq_loss(model, loader, device, alpha: float, beta: float) -> float:
    model.g_model.eval()
    model.q_model.eval()
    total, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        G_a, G_c = model.g_model(batch["W"], batch["L"], batch["A"])
        logit_Q, logit_V = model.q_model(
            batch["W"], batch["L"], batch["A"], batch["a"]
        )
        loss = (
            _g_loss(G_a, G_c, batch, alpha=alpha, beta=beta)
            + _q_loss(logit_Q, logit_V, batch)
        )
        total    += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune(
    rng_seed: int,
    hparam_candidates: dict,
    W, L, A, C, Y,
    n_trials: int = 100,
    log_hparams: bool = True,
):
    """Hyperparameter search via Optuna.

    Parameters
    ----------
    rng_seed : int
    hparam_candidates : dict
        Mapping from hparam name to list of candidate values.
    W, L, A, C, Y : np.ndarray
    n_trials : int
    log_hparams : bool
        If True, log hyperparameters to TensorBoard.

    Returns
    -------
    dict  Best hyperparameter values.
    """
    seed_everything(rng_seed)
    device = get_torch_device()

    dataset = DatasetWLACY(W, L, A, C, Y)
    train_set, val_set = random_split(dataset, [0.8, 0.2])

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def objective(trial: optuna.trial.Trial) -> float:
        hparams = {
            key: trial.suggest_categorical(key, value)
            for key, value in hparam_candidates.items()
        }

        model = DeepLTMLE(
            dataset.dim_static, dataset.dim_dynamic, dataset.tau, **hparams
        )
        model.to(device)

        train_loader = DataLoader(
            train_set, batch_size=hparams["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=hparams["batch_size"], shuffle=False
        )

        opt_g = Adam(model.g_model.parameters(), lr=hparams["learning_rate"])
        opt_q = Adam(model.q_model.parameters(), lr=hparams["learning_rate"])

        max_epochs  = hparams["max_epochs"]
        total_steps = len(train_loader) * max_epochs
        warmup      = int(total_steps * hparams.get("warmup_ratio", 0.05))

        sched_g = get_cosine_schedule_with_warmup(opt_g, warmup, total_steps)
        sched_q = get_cosine_schedule_with_warmup(opt_q, warmup, total_steps)

        alpha = hparams.get("alpha", 1.0)
        beta  = hparams.get("beta",  1.0)

        writer = None
        if log_hparams:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(
                    "artifact", "tune", timestamp, f"trial_{trial.number}"
                )
                writer = SummaryWriter(log_dir=log_dir)
                writer.add_hparams(hparams, {"val/GQ": 0})
            except ImportError:
                pass

        for _ in range(max_epochs):
            model.g_model.train()
            model.q_model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                G_a, G_c = model.g_model(batch["W"], batch["L"], batch["A"])
                loss_g = _g_loss(G_a, G_c, batch, alpha=alpha, beta=beta)
                opt_g.zero_grad(); loss_g.backward(); opt_g.step(); sched_g.step()

                logit_Q, logit_V = model.q_model(
                    batch["W"], batch["L"], batch["A"], batch["a"]
                )
                loss_q = _q_loss(logit_Q, logit_V, batch)
                opt_q.zero_grad(); loss_q.backward(); opt_q.step(); sched_q.step()

                model.q_model.update_ema()

        val_gq = _eval_gq_loss(model, val_loader, device, alpha, beta)

        if writer is not None:
            writer.add_hparams(hparams, {"val/GQ": val_gq})
            writer.close()

        return val_gq

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.NSGAIISampler(
            seed=np.random.randint(0, 2**32, dtype=np.int64)
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
