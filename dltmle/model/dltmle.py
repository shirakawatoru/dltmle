import torch
import torch.nn as nn
import numpy as np
from scipy.special import logit, expit

from ..utils import solve_one_dimensional_submodel
from .g_model import GModel
from .q_model import QModel


class DeepLTMLE(nn.Module):
    """Deep Longitudinal Targeted Minimum Loss-based Estimator.

    Composes a :class:`GModel` (propensity) and a :class:`QModel`
    (Q-function) that share no parameters.  Training is handled
    externally in :func:`dltmle.function.fit.fit`; this class provides
    ``forward``, ``loss``, canonical-gradient debiasing, and prediction
    helpers.

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    tau : int
    dim_model : int
    num_layers : int
    nhead : int
    dim_feedforward : int or None
    dropout : float
    alpha : float
        Weight on G_a loss.
    beta : float
        Weight on G_c loss.
    ema_decay : float
    **kwargs :
        Absorbs unused hparam keys (e.g. learning_rate, batch_size).
    """

    def __init__(
        self,
        dim_static: int,
        dim_dynamic: int,
        tau: int,
        dim_model: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
        ema_decay: float = 0.99,
        **_,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.g_model = GModel(
            dim_static, dim_dynamic, tau,
            dim_model, nhead, num_layers, dim_feedforward, dropout,
        )
        self.q_model = QModel(
            dim_static, dim_dynamic, tau,
            dim_model, nhead, num_layers, dim_feedforward, dropout,
            ema_decay=ema_decay,
        )
        self.eps = nn.Parameter(torch.zeros(tau), requires_grad=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        W, L, A, C, Y, a = (
            batch["W"], batch["L"], batch["A"],
            batch["C"], batch["Y"], batch["a"],
        )
        batch_size, tau = L.shape[0], L.shape[1]

        G_a, G_c = self.g_model(W, L, A)

        logit_Q_raw, logit_V_raw = self.q_model(W, L, A, a)

        # eps correction  (shape: 1, tau, 1)
        eps = self.eps.view(1, tau, 1)

        logit_Q_star_raw = logit_Q_raw + eps
        logit_V_star_raw = logit_V_raw + eps.detach()

        # Q: [0, Q(t=0), ..., Q(t=tau-1)]  — prepend zero (dummy initial slot)
        # V: [V(t=0), ..., V(t=tau-1), 0]  — append zero (overwritten to Y in boundary)
        def _prepend_zero(x):
            return torch.cat([torch.zeros(batch_size, 1, 1, device=x.device), x], dim=1)

        def _append_zero(x):
            return torch.cat([x, torch.zeros(batch_size, 1, 1, device=x.device)], dim=1)

        logit_Q      = _prepend_zero(logit_Q_raw)
        logit_Q_star = _prepend_zero(logit_Q_star_raw)
        logit_V      = _append_zero(logit_V_raw)
        logit_V_star = _append_zero(logit_V_star_raw)

        Q      = torch.sigmoid(logit_Q)
        Q_star = torch.sigmoid(logit_Q_star)
        V      = torch.sigmoid(logit_V)
        V_star = torch.sigmoid(logit_V_star)

        Q, Q_star, V, V_star = self._set_deterministic_Q(Q, Q_star, V, V_star, Y)

        # IPW weights
        J_a = (a * A + (1 - a) * (1 - A)) / (
            G_a * A + (1 - G_a) * (1 - A)
        ).detach()
        g_a = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_a[:, 1:] = J_a.cumprod(dim=1)

        J_c = (C == 0) / (G_c * C + (1 - G_c) * (1 - C)).detach()
        g_c = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_c[:, 1:] = J_c.cumprod(dim=1)

        g = torch.clip(g_a * g_c, 0, 100)

        IC = (g * (V_star - Q_star)).sum(dim=1)   # (B, 1)

        return {
            "Q": Q, "Q_star": Q_star,
            "V": V, "V_star": V_star,
            "G_a": G_a, "G_c": G_c,
            "g": g, "IC": IC,
        }

    # ------------------------------------------------------------------
    # Deterministic boundary conditions on Q / V
    # ------------------------------------------------------------------

    def _set_deterministic_Q(self, Q, Q_star, V, V_star, Y):
        n, tau, _ = Y.shape

        R  = torch.ones((n, tau + 1, 1), device=Y.device)
        R[:, 2:] = 1 - Y[:, :-1]                            # survival indicator

        T0 = torch.zeros((n, tau + 1, 1), device=Y.device)
        T0[:, 0] = 1

        # V boundary
        V[:, -1]    = Y[:, -1]
        V[:, 1:-1]  = torch.where(Y[:, :-1] == 1, torch.ones_like(V[:, 1:-1]), V[:, 1:-1])

        V_star[:, -1]   = Y[:, -1]
        V_star[:, 1:-1] = torch.where(Y[:, :-1] == 1, torch.ones_like(V_star[:, 1:-1]), V_star[:, 1:-1])

        # Q boundary
        Q      = torch.where(R == 1, Q,      torch.ones_like(Q))
        Q      = torch.where(T0 == 0, Q,     V[:, 0].mean() * torch.ones_like(Q))

        Q_star = torch.where(R == 1, Q_star, torch.ones_like(Q_star))
        Q_star = torch.where(T0 == 0, Q_star, V_star[:, 0].mean() * torch.ones_like(Q_star))

        return Q, Q_star, V, V_star

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, S_hat: dict, S: dict) -> dict:
        Q, Q_star, V, V_star, G_a, G_c, g, IC = S_hat.values()
        W, L, A, C, Y, a = S.values()

        bce = nn.BCELoss(reduction="none")

        R   = torch.ones_like(Y)
        R[:, 1:] = 1 - Y[:, :-1]          # survival

        R_C = torch.ones_like(C)
        R_C[:, 1:] = 1 - C[:, :-1]        # not-yet-censored

        loss_Q     = (R * (1 - C) * bce(Q[:, 1:],      V[:, 1:])).sum(dim=1).mean()
        loss_G_a   = (R * R_C     * bce(G_a, A)).sum(dim=1).mean()
        loss_G_c   = (R * R_C     * bce(G_c, C)).sum(dim=1).mean()

        loss_Q_star = (g[:, 1:] * R * bce(Q_star[:, 1:], V_star[:, 1:])).sum(dim=1).mean()

        return {
            "G":      loss_G_a + loss_G_c,
            "G_a":    loss_G_a,
            "G_c":    loss_G_c,
            "Q":      loss_Q,
            "GQ":     loss_Q + loss_G_a + loss_G_c,
            "Q_star": loss_Q_star,
            "PnIC":   IC.mean(),
            "PnIC2":  (IC ** 2).mean(),
        }

    # ------------------------------------------------------------------
    # Canonical gradient (post-training debiasing)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_all(self, loader, device):
        """Run predict loop; return list of per-batch output dicts."""
        self.eval()
        results = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self(batch)
            # detach and move to CPU for numpy operations
            results.append({k: v.cpu() for k, v in out.items()})
            results[-1]["Y"] = batch["Y"].cpu()
        return results

    def solve_canonical_gradient_common_eps(
        self,
        loader,
        device,
        max_iter: int = 1000,
        tol: float = 1e-6,
        stop_pnic_se_ratio: bool = False,
        max_delta_eps=None,
    ):
        n = len(loader.dataset)
        tau = self.tau

        preds = self._predict_all(loader, device)

        Y_all    = torch.cat([p["Y"]    for p in preds], dim=0).numpy()[:, :, 0]
        y_hat    = torch.cat([p["Q"]    for p in preds], dim=0).numpy()[:, 1:tau+1, 0]
        y        = torch.cat([p["V"]    for p in preds], dim=0).numpy()[:, 1:tau+1, 0]
        g        = torch.cat([p["g"]    for p in preds], dim=0).numpy()[:, 1:tau+1, 0]

        r        = np.ones((n, tau))
        r[:, 1:] = 1 - Y_all[:, :-1]
        H        = r * g

        eps_vals = np.zeros(max_iter)

        for i in range(max_iter):
            _eps = solve_one_dimensional_submodel(y_hat.ravel(), y.ravel(), H.ravel())

            if max_delta_eps is not None:
                _eps = np.clip(_eps, -max_delta_eps, max_delta_eps)

            eps_vals[i] = _eps

            y_hat        = expit(logit(y_hat) + _eps)
            y[:, :-1]    = expit(logit(y[:, :-1]) + _eps)

            if stop_pnic_se_ratio:
                ic = (g * (y - y_hat)).sum(axis=1)
                se = np.sqrt((ic ** 2).mean() / n)
                if np.abs(ic.mean() / se) < 1 / np.log(n):
                    break

            if np.abs(_eps) < tol:
                break

        total_eps = eps_vals[:i + 1].sum()
        print(f"eps: {eps_vals[:i+1]}")
        print(f"eps.sum = {total_eps}")

        self.eps = nn.Parameter(
            torch.full((tau,), total_eps, device=device), requires_grad=False
        )

    # ------------------------------------------------------------------
    # Estimate extraction
    # ------------------------------------------------------------------

    def get_estimates_from_prediction(self, preds: list, verbose: bool = True):
        Q_a_star = torch.cat([p["V_star"] for p in preds], dim=0).numpy().squeeze()
        ic       = torch.cat([p["IC"]     for p in preds], dim=0).numpy().squeeze()

        est = Q_a_star[:, 0].mean()
        se  = np.sqrt((ic ** 2).mean() / ic.shape[0])

        PnIC  = ic.mean()
        PnIC2 = (ic ** 2).mean()
        EIC   = np.abs(PnIC / PnIC2 ** 0.5)

        if verbose:
            print(f"est:           {est:.6f}")
            print(f"CI (95%):      [{est - 1.96*se:.6f}, {est + 1.96*se:.6f}]")
            print(f"se:            {se:.6f}")
            print(f"E_n[IC]:       {PnIC:.6f}")
            print(f"PnIC/√PnIC2:   {EIC:.6f}")

        return est, se, ic
