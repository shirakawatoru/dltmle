import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class HeterogeneousTokenTransformer(nn.Module):
    """Transformer backbone for longitudinal [W, L, A] sequences.

    Token sequence: [W, L_0, A_0, L_1, A_1, ..., L_{tau-1}, A_{tau-1}]
    Causal masking ensures L_t sees (W, L_{0:t}, A_{0:t-1}) and
    A_t sees (W, L_{0:t}, A_{0:t}).

    Parameters
    ----------
    dim_static : int
        Dimension of static baseline covariates W.
    dim_dynamic : int
        Dimension of time-varying covariates L.
    n_treatment : int
        Number of treatment values (2 for binary).
    dim_model : int
        Transformer model dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int or None
        FFN hidden size. Defaults to 4 * dim_model.
    dropout : float
        Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        dim_static: int,
        dim_dynamic: int,
        n_treatment: int,
        dim_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim_model = dim_model

        self.emb_W = nn.Linear(dim_static, dim_model)
        self.emb_L = nn.Linear(dim_dynamic, dim_model)
        self.emb_A = nn.Embedding(n_treatment, dim_model)

        self.type_emb = nn.Parameter(torch.randn(3, dim_model))  # W, L, A
        self.positional_encoding = PositionalEncoding(dim_model)

        dim_ff = dim_feedforward if dim_feedforward is not None else 4 * dim_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, W: torch.Tensor, L: torch.Tensor, A: torch.Tensor):
        """
        Parameters
        ----------
        W : (batch, dim_static)
        L : (batch, tau, dim_dynamic)
        A : (batch, tau) — integer treatment indices

        Returns
        -------
        z_L : (batch, tau, dim_model)  transformer output at L-positions
        z_A : (batch, tau, dim_model)  transformer output at A-positions
        """
        batch_size, tau, _ = L.shape

        w = self.emb_W(W)          # (batch, dim_model)
        l = self.emb_L(L)          # (batch, tau, dim_model)
        a = self.emb_A(A)          # (batch, tau, dim_model)

        w = w + self.type_emb[0]
        l = l + self.type_emb[1]
        a = a + self.type_emb[2]

        w = self.positional_encoding(w.unsqueeze(1)).squeeze(1)
        l = self.positional_encoding(l)
        a = self.positional_encoding(a)

        x = torch.empty(
            (batch_size, 2 * tau + 1, self.dim_model),
            device=W.device, dtype=W.dtype,
        )
        x[:, 0] = w
        x[:, 1::2] = l
        x[:, 2::2] = a

        mask = _causal_mask(2 * tau + 1, device=W.device)
        z = self.transformer_encoder(x, mask=mask)  # (batch, 2*tau+1, dim_model)

        z_L = z[:, 1::2]  # (batch, tau, dim_model)
        z_A = z[:, 2::2]  # (batch, tau, dim_model)

        return z_L, z_A


def _causal_mask(seq_len: int, device=None) -> torch.Tensor:
    m = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    Applies treatment-specific scale and shift to z_L:
        output[a] = z_L * gamma[:, a] + beta[:, a]

    Parameters
    ----------
    dim_model : int
    n_treatment : int
    """

    def __init__(self, dim_model: int, n_treatment: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.randn(dim_model, n_treatment))
        self.beta = nn.Parameter(torch.randn(dim_model, n_treatment))

    def forward(self, z_L: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_L : (batch, tau, dim_model)

        Returns
        -------
        (batch, tau, dim_model, n_treatment)
        """
        return z_L.unsqueeze(-1) * self.gamma + self.beta
