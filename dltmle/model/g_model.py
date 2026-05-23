import torch
import torch.nn as nn

from .htt import HeterogeneousTokenTransformer


class GModel(nn.Module):
    """Propensity model for treatment and censoring.

    Owns its own ``HeterogeneousTokenTransformer`` instance; parameters are
    fully independent of :class:`QModel`.

    Parameters
    ----------
    dim_static : int
        Dimension of static baseline covariates W.
    dim_dynamic : int
        Dimension of time-varying covariates L.
    tau : int
        Number of time steps (unused at construction but kept for API consistency).
    dim_model : int
        Transformer model dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int or None
        FFN hidden size. Defaults to 4 * dim_model.
    dropout : float
        Dropout rate.
    n_treatment : int
        Number of treatment values (2 for binary).
    """

    def __init__(
        self,
        dim_static: int,
        dim_dynamic: int,
        tau: int,
        dim_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        n_treatment: int = 2,
    ):
        super().__init__()
        self.htt = HeterogeneousTokenTransformer(
            dim_static, dim_dynamic, n_treatment, dim_model,
            num_heads, num_layers, dim_feedforward, dropout,
        )
        self.G_a = nn.Sequential(nn.Linear(dim_model, 1), nn.Sigmoid())
        self.G_c = nn.Sequential(nn.Linear(dim_model, 1), nn.Sigmoid())

    def forward(
        self,
        W: torch.Tensor,
        L: torch.Tensor,
        A: torch.Tensor,
    ):
        """
        Parameters
        ----------
        W : (batch, dim_static)
        L : (batch, tau, dim_dynamic)
        A : (batch, tau, 1)  float treatment (cast to long internally)

        Returns
        -------
        G_a : (batch, tau, 1)  P(A_t=1 | W, L_{0:t+1}, A_{0:t-1})
        G_c : (batch, tau, 1)  P(C_t=0 | W, L_{0:t+1}, A_{0:t})
        """
        A_int = A.long().squeeze(-1)          # (batch, tau)
        z_L, z_A = self.htt(W, L, A_int)
        return self.G_a(z_L), self.G_c(z_A)
