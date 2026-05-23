import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearGModel(nn.Module):
    """Pooled logistic regression baseline for propensity estimation.

    Uses a single shared linear layer across all time steps (pooled over t).

    Features
    --------
    - G_a: [W, L_t, A_{t-1}]  →  P(A_t=1 | ...)
    - G_c: [W, L_t, A_t]      →  P(C_t=0 | ...)

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    """

    def __init__(self, dim_static: int, dim_dynamic: int):
        super().__init__()
        dim_in = dim_static + dim_dynamic + 1
        self.G_a = nn.Sequential(nn.Linear(dim_in, 1), nn.Sigmoid())
        self.G_c = nn.Sequential(nn.Linear(dim_in, 1), nn.Sigmoid())

    def forward(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        A: torch.Tensor,
    ):
        """
        Parameters
        ----------
        W : (B, dim_static)
        L : (B, T, dim_dynamic)
        A : (B, T, 1)

        Returns
        -------
        G_a : (B, T, 1)
        G_c : (B, T, 1)
        """
        tau = L.shape[1]
        W_exp = W.unsqueeze(1).expand(-1, tau, -1)       # (B, T, dim_static)
        A_lag = F.pad(A[:, :-1], (0, 0, 1, 0))           # A_{t-1}, zero-pad at t=0
        feat_a = torch.cat([W_exp, L, A_lag], dim=-1)
        feat_c = torch.cat([W_exp, L, A],     dim=-1)
        return self.G_a(feat_a), self.G_c(feat_c)


class LinearQModel(nn.Module):
    """Pooled logistic regression baseline for Q-function estimation.

    Uses a single shared linear layer across all time steps.

    Features
    --------
    - Q: [W, L_t, A_t]  →  E[Y | ...]

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    """

    def __init__(self, dim_static: int, dim_dynamic: int):
        super().__init__()
        dim_in = dim_static + dim_dynamic + 1
        self.head = nn.Sequential(nn.Linear(dim_in, 1), nn.Sigmoid())

    def forward(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        A: torch.Tensor,
        a: torch.Tensor,
    ):
        """
        Parameters
        ----------
        W : (B, dim_static)
        L : (B, T, dim_dynamic)
        A : (B, T, 1)  observed treatment
        a : (B, T, 1)  counterfactual treatment

        Returns
        -------
        logit_Q : (B, T, 1)  logit Q under observed A
        logit_V : (B, T, 1)  logit V under counterfactual a  (detached)
        """
        tau = L.shape[1]
        W_exp = W.unsqueeze(1).expand(-1, tau, -1)

        def _logit(feat):
            p = self.head(feat).clamp(1e-6, 1 - 1e-6)
            return torch.logit(p)

        logit_Q = _logit(torch.cat([W_exp, L, A], dim=-1))
        logit_a1 = _logit(torch.cat([W_exp, L, torch.ones_like(A)],  dim=-1))
        logit_a0 = _logit(torch.cat([W_exp, L, torch.zeros_like(A)], dim=-1))
        logit_V = (logit_a1 * a + logit_a0 * (1 - a)).detach()

        return logit_Q, logit_V
