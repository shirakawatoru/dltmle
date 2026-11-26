import torch
import torch.nn as nn

from .htt import HeterogeneousTokenTransformer
from .categorical import reference_softmax, check_censoring_timing


class GModel(nn.Module):
    """Propensity model for (categorical) treatment and (categorical) censoring.

    Owns its own ``HeterogeneousTokenTransformer`` instance; parameters are
    fully independent of :class:`QModel`.

    * ``g^A``: multinomial softmax over J treatment classes, read from the
      L-position tokens (sees W, L_{0:t}, A_{0:t-1}).
    * ``g^C``: multinomial softmax over K censoring categories (0 = not
      censored, 1..K-1 = censoring reasons, i.e. cause-specific censoring
      hazards), read from the A-position tokens (sees W, L_{0:t}, A_{0:t}).

    Both heads use J-1 / K-1 logits against reference class 0, so J=K=2 is
    exactly the previous binary (sigmoid) model.

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
        Number of treatment classes J (2 for binary).
    n_censoring : int
        Number of censoring categories K including "not censored" (2 for binary).
    censoring_timing : {"after_treatment", "before_treatment"}
        Order of A_t and C_t within an interval.  ``"after_treatment"``
        (dltmle default: L_t → A_t → C_t → Y_t): g^C is read from the
        A_t token and conditions on A_t.  ``"before_treatment"`` (stremr
        convention: L_t → C_t → A_t → Y_t): g^C is read from the L_t token
        and does not condition on A_t.
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
        n_censoring: int = 2,
        censoring_timing: str = "after_treatment",
    ):
        super().__init__()
        if n_treatment < 2 or n_censoring < 2:
            raise ValueError("n_treatment and n_censoring must be >= 2")
        self.censoring_timing = check_censoring_timing(censoring_timing)
        self.n_treatment = n_treatment
        self.n_censoring = n_censoring
        self.htt = HeterogeneousTokenTransformer(
            dim_static, dim_dynamic, n_treatment, dim_model,
            num_heads, num_layers, dim_feedforward, dropout,
        )
        self.G_a = nn.Sequential(nn.Linear(dim_model, n_treatment - 1))
        self.G_c = nn.Sequential(nn.Linear(dim_model, n_censoring - 1))

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
        A : (batch, tau, 1)  treatment class index (float, cast to long)

        Returns
        -------
        p_A : (batch, tau, J)  P(A_t = j | W, L_{0:t}, A_{0:t-1})
        p_C : (batch, tau, K)  P(C_t = k | W, L_{0:t}, A_{0:t})  (or
              ``A_{0:t-1}`` with ``censoring_timing="before_treatment"``)
        """
        A_int = A.long().squeeze(-1)          # (batch, tau)
        z_L, z_A = self.htt(W, L, A_int)
        z_C = z_A if self.censoring_timing == "after_treatment" else z_L
        return reference_softmax(self.G_a(z_L)), reference_softmax(self.G_c(z_C))
