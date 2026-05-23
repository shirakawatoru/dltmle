import torch
import torch.nn as nn
import torch.nn.functional as F

from .htt import HeterogeneousTokenTransformer, FiLM


class QModel(nn.Module):
    """Q-function model with EMA target network.

    Owns its own ``HeterogeneousTokenTransformer`` instance; parameters are
    fully independent of :class:`GModel`.

    Architecture
    ------------
    - Backbone: HTT processes [W, L, A] → z_L (L-position outputs)
    - Critic:   FiLM(z_L) * gamma_critic + beta_critic → Linear → Q(observed A)
    - Target:   EMA copy of critic, no gradient → V(counterfactual a)

    Parameters
    ----------
    dim_static : int
    dim_dynamic : int
    tau : int
    dim_model : int
    num_heads : int
    num_layers : int
    dim_feedforward : int or None
    dropout : float
    n_treatment : int
    ema_decay : float
        EMA decay for the target network. Default 0.99.
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
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_treatment = n_treatment
        self.ema_decay = ema_decay

        self.htt = HeterogeneousTokenTransformer(
            dim_static, dim_dynamic, n_treatment, dim_model,
            num_heads, num_layers, dim_feedforward, dropout,
        )

        # Critic (trained by gradients)
        self.film_critic = FiLM(dim_model, n_treatment)
        self.q_critic = nn.Linear(dim_model, 1)

        # Target (EMA copy, no gradients)
        self.film_target = FiLM(dim_model, n_treatment)
        self.q_target = nn.Linear(dim_model, 1)
        self.film_target.requires_grad_(False)
        self.q_target.requires_grad_(False)
        self._sync_target()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_q(
        self,
        z_L: torch.Tensor,
        a_int: torch.Tensor,
        film: FiLM,
        head: nn.Linear,
    ) -> torch.Tensor:
        """Compute Q-values for given treatment indices.

        Parameters
        ----------
        z_L   : (B, T, D)
        a_int : (B, T)  long treatment indices
        film  : FiLM module
        head  : Linear(D, 1)

        Returns
        -------
        (B, T, 1)
        """
        z = film(z_L).transpose(-1, -2)           # (B, T, n_treatment, D)
        q = head(z).squeeze(-1)                    # (B, T, n_treatment)
        a_oh = F.one_hot(a_int, self.n_treatment).float()
        return (q * a_oh).sum(-1, keepdim=True)    # (B, T, 1)

    def _sync_target(self):
        """Initialise target network as an exact copy of the critic."""
        self.film_target.load_state_dict(self.film_critic.state_dict())
        self.q_target.load_state_dict(self.q_critic.state_dict())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        A : (B, T, 1)  float observed treatment
        a : (B, T, 1)  float counterfactual treatment

        Returns
        -------
        logit_Q : (B, T, 1)  logit Q under observed A  (critic, has gradient)
        logit_V : (B, T, 1)  logit V under counterfactual a  (target, detached)
        """
        A_int = A.long().squeeze(-1)   # (B, T)
        a_int = a.long().squeeze(-1)   # (B, T)

        z_L, _ = self.htt(W, L, A_int)

        logit_Q = self._eval_q(z_L, A_int, self.film_critic, self.q_critic)
        with torch.no_grad():
            logit_V = self._eval_q(z_L, a_int, self.film_target, self.q_target)

        return logit_Q, logit_V

    def update_ema(self):
        """EMA update: target ← (1-decay)*critic + decay*target."""
        with torch.no_grad():
            critic_params = (
                list(self.film_critic.parameters())
                + list(self.q_critic.parameters())
            )
            target_params = (
                list(self.film_target.parameters())
                + list(self.q_target.parameters())
            )
            for pc, pt in zip(critic_params, target_params):
                pt.lerp_(pc, 1 - self.ema_decay)
