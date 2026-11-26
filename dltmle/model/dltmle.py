import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit

from ..utils import solve_one_dimensional_submodel_offset
from .categorical import mix_logits, mix_prob_np, categorical_nll
from .g_model import GModel
from .q_model import QModel


class DeepLTMLE(nn.Module):
    """Deep Longitudinal Targeted Minimum Loss-based Estimator.

    Composes a :class:`GModel` (propensity) and a :class:`QModel`
    (Q-function) that share no parameters.  Training is handled
    externally in :mod:`dltmle.function.fit`; this class provides
    ``forward``, ``loss``, canonical-gradient debiasing, and prediction
    helpers.

    Batches are dicts with keys ``W, L, A, C, Y, gstar``:

    * ``A`` (B, T, 1): treatment class index in ``0..J-1``
    * ``C`` (B, T, 1): censoring category in ``0..K-1`` (0 = not censored)
    * ``Y`` (B, T, 1): binary absorbing outcome
    * ``gstar`` (B, T, J): intervention probabilities ``g*(a | history)``
      (one-hot for static / deterministic dynamic regimes).

    Under the intervention every censoring category is eliminated, so the
    censoring part of the clever covariate is ``1(C_t = 0) / P(C_t = 0 | .)``.

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
    n_treatment : int
        Number of treatment classes J (2 = binary).
    n_censoring : int
        Number of censoring categories K incl. "not censored" (2 = binary).
    max_weight : float
        Upper clip of the cumulative inverse-probability weight. Default 100.
    censoring_timing : {"after_treatment", "before_treatment"}
        Within-interval order of A_t and C_t (see :class:`GModel`).
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
        n_treatment: int = 2,
        n_censoring: int = 2,
        max_weight: float = 100.0,
        censoring_timing: str = "after_treatment",
        **_,
    ):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.max_weight = max_weight

        self.g_model = GModel(
            dim_static, dim_dynamic, tau,
            dim_model, nhead, num_layers, dim_feedforward, dropout,
            n_treatment=n_treatment, n_censoring=n_censoring,
            censoring_timing=censoring_timing,
        )
        self.q_model = QModel(
            dim_static, dim_dynamic, tau,
            dim_model, nhead, num_layers, dim_feedforward, dropout,
            n_treatment=n_treatment, ema_decay=ema_decay,
        )
        self.eps = nn.Parameter(torch.zeros(tau), requires_grad=False)

    @classmethod
    def from_models(cls, g_model: nn.Module, q_model: nn.Module, tau: int,
                    alpha: float = 1.0, beta: float = 1.0,
                    max_weight: float = 100.0) -> "DeepLTMLE":
        """Wrap already-trained G / Q models (e.g. linear baselines, or a
        G model shared across horizons) for targeting and estimation."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.max_weight = max_weight
        self.g_model = g_model
        self.q_model = q_model
        self.eps = nn.Parameter(torch.zeros(tau), requires_grad=False)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        W, L, A, C, Y, gstar = (
            batch["W"], batch["L"], batch["A"],
            batch["C"], batch["Y"], batch["gstar"],
        )
        batch_size, tau = L.shape[0], L.shape[1]

        p_A, p_C = self.g_model(W, L, A)

        logit_Q_raw, logit_V_cls = self.q_model(W, L, A)

        # eps correction  (shape: 1, tau, 1)
        eps = self.eps.view(1, tau, 1)

        logit_V_raw      = mix_logits(logit_V_cls, gstar)
        logit_Q_star_raw = logit_Q_raw + eps
        logit_V_star_raw = mix_logits(logit_V_cls + eps.detach(), gstar)

        # Q: [0, Q(t=0), ..., Q(t=tau-1)]  — prepend zero (dummy initial slot)
        # V: [V(t=0), ..., V(t=tau-1), 0]  — append zero (overwritten to Y in boundary)
        def _prepend_zero(x):
            return torch.cat([torch.zeros(batch_size, 1, 1, device=x.device), x], dim=1)

        def _append_zero(x):
            return torch.cat([x, torch.zeros(batch_size, 1, 1, device=x.device)], dim=1)

        Q      = torch.sigmoid(_prepend_zero(logit_Q_raw))
        Q_star = torch.sigmoid(_prepend_zero(logit_Q_star_raw))
        V      = torch.sigmoid(_append_zero(logit_V_raw))
        V_star = torch.sigmoid(_append_zero(logit_V_star_raw))

        Q, Q_star, V, V_star = self._set_deterministic_Q(Q, Q_star, V, V_star, Y)

        g = self.clever_weights(p_A, p_C, A, C, gstar)

        # IC without the t=0 centring term; the full-sample centring
        # V*(0) - psi is added in get_estimates_from_prediction.
        IC_tail = (g[:, 1:] * (V_star[:, 1:] - Q_star[:, 1:])).sum(dim=1)   # (B, 1)
        IC = IC_tail + (V_star[:, 0] - Q_star[:, 0])

        return {
            "Q": Q, "Q_star": Q_star,
            "V": V, "V_star": V_star,
            "p_A": p_A, "p_C": p_C,
            "g": g, "IC": IC, "IC_tail": IC_tail,
            "logit_Q_raw": logit_Q_raw, "logit_V_cls": logit_V_cls,
        }

    def clever_weights(self, p_A, p_C, A, C, gstar):
        """Cumulative IPW weights ``g`` of shape (B, tau + 1, 1).

        ``g[:, t+1] = prod_{s<=t} g*(A_s|H_s) / g(A_s|H_s) * 1(C_s=0) / P(C_s=0|H_s)``
        (clipped at ``max_weight``).
        """
        batch_size, tau = A.shape[0], A.shape[1]
        A_idx = A.long()
        J_a = gstar.gather(-1, A_idx) / p_A.gather(-1, A_idx).detach()
        g_a = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_a[:, 1:] = J_a.cumprod(dim=1)

        J_c = torch.where(C == 0, 1.0 / p_C[..., :1].detach(), torch.zeros_like(C))
        g_c = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_c[:, 1:] = J_c.cumprod(dim=1)

        return torch.clip(g_a * g_c, 0, self.max_weight)

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
        Q, Q_star, V, V_star = S_hat["Q"], S_hat["Q_star"], S_hat["V"], S_hat["V_star"]
        p_A, p_C, g, IC = S_hat["p_A"], S_hat["p_C"], S_hat["g"], S_hat["IC"]
        A, C, Y = S["A"], S["C"], S["Y"]

        bce = nn.BCELoss(reduction="none")

        R   = torch.ones_like(Y)
        R[:, 1:] = 1 - Y[:, :-1]          # survival

        cens = (C > 0).float()
        R_C = torch.ones_like(C)
        R_C[:, 1:] = 1 - torch.cummax(cens, dim=1).values[:, :-1]   # not-yet-censored
        R_A = R_C if getattr(self.g_model, "censoring_timing", "after_treatment") == "after_treatment" \
            else R_C * (1 - cens)

        loss_Q     = (R * (1 - cens) * bce(Q[:, 1:],      V[:, 1:])).sum(dim=1).mean()
        loss_G_a   = (R * R_A * categorical_nll(p_A, A)).sum(dim=1).mean()
        loss_G_c   = (R * R_C * categorical_nll(p_C, C)).sum(dim=1).mean()

        loss_Q_star = (g[:, 1:] * R * bce(Q_star[:, 1:], V_star[:, 1:])).sum(dim=1).mean()

        return {
            "G":      self.alpha * loss_G_a + self.beta * loss_G_c,
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
        """Run predict loop; return list of per-batch output dicts (CPU)."""
        self.eval()
        results = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self(batch)
            results.append({k: v.cpu() for k, v in out.items()})
            results[-1]["Y"] = batch["Y"].cpu()
            results[-1]["gstar"] = batch["gstar"].cpu()
        return results

    def solve_canonical_gradient_common_eps(
        self,
        loader,
        device,
        max_iter: int = 1000,
        tol: float = 1e-6,
        stop_pnic_se_ratio: bool = False,
        max_delta_eps=None,
        verbose: bool = True,
    ) -> dict:
        """Iterative TMLE targeting with a single (time-common) fluctuation eps.

        Fluctuation submodel on the logit scale: ``Q_t(a) -> expit(logit Q_t(a) + eps)``
        for every treatment class ``a``; the pseudo-outcome ``V_{t+1} = sum_a
        g*(a|H) Q_{t+1}(a)`` is recomputed under the updated ``eps`` at each
        iteration (this matters for stochastic interventions).

        Returns a dict of targeting diagnostics.
        """
        n = len(loader.dataset)
        tau = self.tau
        eps0 = float(self.eps[0].item()) if tau > 0 else 0.0

        preds = self._predict_all(loader, device)

        def _cat(key):
            return torch.cat([p[key] for p in preds], dim=0).double().numpy()

        Y_all = _cat("Y")[:, :, 0]                 # (n, tau)
        lq    = _cat("logit_Q_raw")[:, :, 0]       # (n, tau)
        lv    = _cat("logit_V_cls")                # (n, tau, J)
        gstar = _cat("gstar")                      # (n, tau, J)
        g     = _cat("g")[:, 1:tau + 1, 0]         # (n, tau)

        r        = np.ones((n, tau))
        r[:, 1:] = 1 - Y_all[:, :-1]
        H        = r * g

        def _targets(e):
            # Q_t = 1 deterministically once the event has occurred
            y_hat = np.where(r == 0, 1.0, expit(lq + e))
            y = np.empty((n, tau))
            if tau > 1:
                v = mix_prob_np(lv[:, 1:] + e, gstar[:, 1:])
                y[:, :-1] = np.where(Y_all[:, :-1] == 1, 1.0, v)
            y[:, -1] = Y_all[:, -1]
            return y_hat, y

        eps_vals = []
        total = eps0
        converged = False
        stop_reason = "max_iter"
        ratio = np.nan
        for i in range(max_iter):
            y_hat, y = _targets(total)
            _eps = solve_one_dimensional_submodel_offset(
                (lq + total).ravel(), y.ravel(), H.ravel()
            )
            if max_delta_eps is not None:
                _eps = float(np.clip(_eps, -max_delta_eps, max_delta_eps))
            eps_vals.append(float(_eps))
            total += _eps

            if stop_pnic_se_ratio:
                y_hat, y = _targets(total)
                ic = (g * (y - y_hat)).sum(axis=1)
                se = np.sqrt((ic ** 2).mean() / n)
                ratio = float(np.abs(ic.mean() / se)) if se > 0 else 0.0
                if ratio < 1 / np.log(n):
                    converged, stop_reason = True, "pnic_se_ratio"
                    break

            if np.abs(_eps) < tol:
                converged, stop_reason = True, "tol"
                break

        if verbose:
            print(f"eps: {np.array(eps_vals)}")
            print(f"eps.sum = {total - eps0}")

        self.eps = nn.Parameter(
            torch.full((tau,), total, device=device, dtype=torch.float32),
            requires_grad=False,
        )

        return {
            "eps_total": float(total),
            "eps_steps": eps_vals,
            "n_iter": len(eps_vals),
            "converged": converged,
            "stop_reason": stop_reason,
            "pnic_se_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Estimate extraction
    # ------------------------------------------------------------------

    def get_estimates_from_prediction(self, preds: list, verbose: bool = True):
        """Return ``(est, se, ic)`` with ``ic`` the per-subject efficient
        influence curve ``sum_{t>=1} g_t (V*_t - Q*_t) + V*_0 - psi``."""
        V0      = torch.cat([p["V_star"][:, 0, 0] for p in preds], dim=0).double().numpy()
        ic_tail = torch.cat([p["IC_tail"][:, 0] for p in preds], dim=0).double().numpy()

        est = float(V0.mean())
        ic  = ic_tail + V0 - est
        se  = float(np.sqrt((ic ** 2).mean() / ic.shape[0]))

        PnIC  = ic.mean()
        PnIC2 = (ic ** 2).mean()
        EIC   = np.abs(PnIC / PnIC2 ** 0.5) if PnIC2 > 0 else 0.0

        if verbose:
            print(f"est:           {est:.6f}")
            print(f"CI (95%):      [{est - 1.96*se:.6f}, {est + 1.96*se:.6f}]")
            print(f"se:            {se:.6f}")
            print(f"E_n[IC]:       {PnIC:.6f}")
            print(f"PnIC/√PnIC2:   {EIC:.6f}")

        return est, se, ic
