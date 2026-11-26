import torch
import torch.nn as nn
import torch.nn.functional as F

from .categorical import reference_softmax, onehot_drop_first, check_censoring_timing


class LinearGModel(nn.Module):
    """Pooled (multinomial) logistic regression baseline for propensities.

    Uses a single shared linear layer across all time steps (pooled over t).
    Treatment enters the features as a one-hot vector without the reference
    class (for J=2 this is the scalar A, i.e. the previous binary model).

    Features
    --------
    - g^A: [W, L_t, onehot(A_{t-1})]  →  P(A_t = j | ...)   (J classes)
    - g^C: [W, L_t, onehot(A_t)]      →  P(C_t = k | ...)   (K categories)
      (``[W, L_t, onehot(A_{t-1})]`` with ``censoring_timing="before_treatment"``)

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    n_treatment : int
    n_censoring : int
    censoring_timing : {"after_treatment", "before_treatment"}
    """

    def __init__(self, dim_static: int, dim_dynamic: int,
                 n_treatment: int = 2, n_censoring: int = 2,
                 censoring_timing: str = "after_treatment"):
        super().__init__()
        self.censoring_timing = check_censoring_timing(censoring_timing)
        self.n_treatment = n_treatment
        self.n_censoring = n_censoring
        dim_in = dim_static + dim_dynamic + n_treatment - 1
        self.G_a = nn.Sequential(nn.Linear(dim_in, n_treatment - 1))
        self.G_c = nn.Sequential(nn.Linear(dim_in, n_censoring - 1))

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
        A : (B, T, 1)  class indices

        Returns
        -------
        p_A : (B, T, J)
        p_C : (B, T, K)
        """
        tau = L.shape[1]
        W_exp = W.unsqueeze(1).expand(-1, tau, -1)       # (B, T, dim_static)
        A_oh = onehot_drop_first(A, self.n_treatment).to(L.dtype)
        A_lag = F.pad(A_oh[:, :-1], (0, 0, 1, 0))         # A_{t-1}, zero-pad at t=0
        feat_a = torch.cat([W_exp, L, A_lag], dim=-1)
        if self.censoring_timing == "after_treatment":
            feat_c = torch.cat([W_exp, L, A_oh],  dim=-1)
        else:
            feat_c = feat_a
        return reference_softmax(self.G_a(feat_a)), reference_softmax(self.G_c(feat_c))


class LinearQModel(nn.Module):
    """Pooled logistic regression baseline for Q-function estimation.

    Uses a single shared linear layer across all time steps.

    Features
    --------
    - Q: [W, L_t, onehot(A_t)]  →  E[Y | ...]

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    n_treatment : int
    """

    def __init__(self, dim_static: int, dim_dynamic: int, n_treatment: int = 2):
        super().__init__()
        self.n_treatment = n_treatment
        dim_in = dim_static + dim_dynamic + n_treatment - 1
        self.head = nn.Sequential(nn.Linear(dim_in, 1), nn.Sigmoid())

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
        A : (B, T, 1)  observed treatment class index

        Returns
        -------
        logit_Q     : (B, T, 1)  logit Q under observed A
        logit_V_cls : (B, T, J)  logit Q under each class (detached)
        """
        B, tau = L.shape[0], L.shape[1]
        J = self.n_treatment
        W_exp = W.unsqueeze(1).expand(-1, tau, -1)

        def _logit(feat):
            p = self.head(feat).clamp(1e-6, 1 - 1e-6)
            return torch.logit(p)

        A_oh = onehot_drop_first(A, J).to(L.dtype)
        logit_Q = _logit(torch.cat([W_exp, L, A_oh], dim=-1))

        with torch.no_grad():
            cls = []
            for j in range(J):
                oh = torch.zeros(B, tau, J - 1, dtype=L.dtype, device=L.device)
                if j > 0:
                    oh[..., j - 1] = 1
                cls.append(_logit(torch.cat([W_exp, L, oh], dim=-1)))
            logit_V_cls = torch.cat(cls, dim=-1)       # (B, T, J)

        return logit_Q, logit_V_cls
