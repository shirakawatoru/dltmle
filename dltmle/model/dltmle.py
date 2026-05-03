import torch
import torch.nn as nn
import lightning.pytorch as L

import numpy as np
from scipy.special import logit, expit
from transformers import get_cosine_schedule_with_warmup

from ..utils import SinusoidalEncoder, solve_one_dimensional_submodel

from functools import cache

@cache
def _get_attention_mask(tau, num_node_types, device):
    # input nodes = [W[0:1], node_type_0[0:tau], node_type_1[0:tau], ...]
    # DGP at t: L[t] > A[t]

    I = torch.triu(torch.full((tau, tau), True), diagonal=1) # attention <= t
    J = torch.triu(torch.full((tau, tau), True), diagonal=0) # attention < t

    mask = torch.full((tau * num_node_types + 1, tau * num_node_types + 1), True)
    mask[:, 0] = False

    for i in range(num_node_types):
        for j in range(num_node_types):
            mask[(i*tau+1):((i+1)*tau+1), (j*tau+1):((j+1)*tau+1)] = I if i >= j else J

    return mask.to(device)


class FiLMModulator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta = nn.Linear(d_model, d_model)

    def forward(self, h, z):
        return self.gamma(z) * h + self.beta(z)


class QBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.modulator = FiLMModulator(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, h, z_treatment):
        return self.head(self.modulator(h, z_treatment))


class DeepLTMLE(L.LightningModule):
    def __init__(self,
                 dim_static,
                 dim_dynamic,
                 tau,
                 dim_emb=16,
                 dim_emb_time=8,
                 dim_emb_type=8,
                 hidden_size=32,
                 num_layers=2,
                 nhead=8,
                 dropout=0.1,
                 learning_rate=1e-3,
                 alpha=1,
                 beta=1,
                 ema_decay=0.99,
                 warmup_ratio=0.05,
                 **kwargs):
        super().__init__()

        self.tau = tau
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.ema_decay = ema_decay
        self.warmup_ratio = warmup_ratio

        # embeddings: W, L, A only (C and Y not fed into transformer)
        self.emb_W = nn.Linear(dim_static, dim_emb)
        self.emb_L = nn.Linear(dim_dynamic, dim_emb)
        self.emb_A = nn.Linear(1, dim_emb)

        # temporal embeddings
        self.emb_time = nn.Sequential(
            SinusoidalEncoder(dim=dim_emb_time),
            nn.Linear(dim_emb_time, dim_emb_time)
        )

        # type embeddings: 3 types (W=0, L=1, A=2)
        self.emb_type = nn.Parameter(torch.randn(3, dim_emb_type), requires_grad=True)

        # transformer encoder
        d_model = dim_emb + dim_emb_time + dim_emb_type
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        # propensity heads
        self.G_a = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.G_c = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # Q branches: critic (trained) + target (EMA of critic)
        self.Q_critic_branch = QBranch(d_model)
        self.Q_target_branch = QBranch(d_model)
        self.Q_target_branch.requires_grad_(False)
        self.Q_target_branch.load_state_dict(self.Q_critic_branch.state_dict())

        self.eps = nn.Parameter(torch.zeros(tau, requires_grad=False))

    def forward(self, batch):
        W, L, A, C, Y, a = batch["W"], batch["L"], batch["A"], batch["C"], batch["Y"], batch["a"]

        batch_size, tau = L.shape[0], L.shape[1]

        # embeddings (W, L, A only)
        z_W = self.emb_W(W[:,None,:])
        z_L = self.emb_L(L)
        z_A = self.emb_A(A)

        a_ones = torch.ones_like(a)
        a_zeros = torch.zeros_like(a)
        z_a_ones = self.emb_A(a_ones)
        z_a_zeros = self.emb_A(a_zeros)
        # z_a = self.emb_A(a)

        # time embeddings
        T_W = self.emb_time(torch.tensor([-1])).repeat(batch_size, 1, 1)
        T = self.emb_time(torch.arange(tau)).repeat(batch_size, 1, 1)

        # type embeddings
        type_W = self.emb_type[0].repeat(batch_size, 1, 1)
        type_L = self.emb_type[1].repeat(batch_size, tau, 1)
        type_A = self.emb_type[2].repeat(batch_size, tau, 1)

        z_W = torch.cat([z_W, T_W, type_W], axis=-1)
        z_L = torch.cat([z_L, T,   type_L], axis=-1)
        z_A = torch.cat([z_A, T,   type_A], axis=-1)
        z_a_ones = torch.cat([z_a_ones, T, type_A], axis=-1)
        z_a_zeros = torch.cat([z_a_zeros, T, type_A], axis=-1)

        # transformer: [W, L, A], single pass
        # input:  W[0:1], L[0:tau], A[0:tau]
        # output: _[0:1], h[0:tau], z_A_out[0:tau]
        mask = _get_attention_mask(tau, 2, z_L.device)
        x = torch.cat([z_W, z_L, z_A], axis=1)
        x = self.transformer(x, mask=mask)

        h, z_A_out = x[:,1:,:].reshape(batch_size, 2, tau, -1).transpose(0, 1)
        # h       = L[t] output: history summary, context excludes A[t]
        # z_A_out = A[t] output: context includes A[t]

        G_a = self.G_a(h)        # P(A[t]=a | W, L[0:t+1], A[0:t])
        G_c = self.G_c(z_A_out)  # P(C[t]=0 | W, L[0:t+1], A[0:t+1])

        # Q: FiLM(h, z_A) via critic branch
        logit_Q = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q[:, 1:] = self.Q_critic_branch(h, z_A)

        eps = self.eps.view(1, tau, 1).repeat(batch_size, 1, 1)

        logit_Q_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_star[:, 1:] = logit_Q[:, 1:] + eps

        Q = torch.sigmoid(logit_Q)
        Q_star = torch.sigmoid(logit_Q_star)

        # V: FiLM(h, z_a) via target branch — no second transformer pass
        logit_V = torch.zeros(batch_size, tau + 1, 1, device=A.device)

        logit_V_ones = self.Q_target_branch(h, z_a_ones).detach()
        logit_V_zeros = self.Q_target_branch(h, z_a_zeros).detach()

        logit_V[:, :-1] = logit_V_ones * a + logit_V_zeros * (1 - a)

        logit_V_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_V_star[:, :-1] = logit_V[:, :-1] + eps.detach()

        V = torch.sigmoid(logit_V)
        V_star = torch.sigmoid(logit_V_star)

        # degeneration of Q
        Q, Q_star, V, V_star = self._set_deterministic_Q(Q, Q_star, V, V_star, Y)

        # IPW
        J_a = (a * A + (1 - a) * (1 - A)) / (G_a * A + (1 - G_a) * (1 - A)).detach()
        g_a = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_a[:, 1:] = J_a.cumprod(dim=1)

        J_c = (C == 0) / (G_c * C + (1 - G_c) * (1 - C)).detach()
        g_c = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g_c[:, 1:] = J_c.cumprod(dim=1)

        g = g_a * g_c
        g = torch.clip(g, 0, 100)

        # influence curve
        IC = (g * (V_star - Q_star)).sum(dim=1)

        return {
            "Q": Q,
            "Q_star": Q_star,
            "V": V,
            "V_star": V_star,
            "G_a": G_a,
            "G_c": G_c,
            "g": g,
            "IC": IC,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            for pc, pt in zip(self.Q_critic_branch.parameters(),
                              self.Q_target_branch.parameters()):
                pt.lerp_(pc, 1 - self.ema_decay)

    def _set_deterministic_Q(self, Q, Q_star, V, V_star, Y):
        n, tau, _ = Y.shape

        R = torch.ones((n, tau + 1, 1), device=Y.device) # suvrival indicator
        R[:, 2:] = 1 - Y[:, :-1]

        T0 = torch.zeros((n, tau + 1, 1), device=Y.device) # indicator of t == 0
        T0[:, 0] = 1

        V[:, -1] = Y[:, -1]
        V[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, V[:, 1:-1]) # V_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1

        Q = torch.where(R == 1, Q, 1) # Q_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
        Q = torch.where(T0 == 0, Q, V[:, 0].mean())

        V_star[:, -1] = Y[:, -1]
        V_star[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, V_star[:, 1:-1]) # Q_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1

        Q_star = torch.where(R == 1, Q_star, 1) # Q_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
        Q_star = torch.where(T0 == 0, Q_star, V_star[:, 0].mean())

        return Q, Q_star, V, V_star

    def loss(self, S_hat, S):
        Q, Q_star, V, V_star, G_a, G_c, g, IC = S_hat.values()
        W, L, A, C, Y, a = S.values()

        H = nn.BCELoss(reduction='none')

        R = torch.ones_like(Y)
        R[:, 1:] = (1 - Y[:, :-1]) # indicator of survival

        R_C = torch.ones_like(C)
        R_C[:, 1:] = (1 - C[:, :-1]) # indicator of survival

        loss_Q = (R * (1 - C) * H(Q[:, 1:], V[:, 1:])).sum(dim=1).mean()

        loss_G_a = (R * R_C * H(G_a, A)).sum(dim=1).mean()
        loss_G_c = (R * R_C * H(G_c, C)).sum(dim=1).mean()

        loss_Q_star = (g[:, 1:] * R * H(Q_star[:, 1:], V_star[:, 1:])).sum(dim=1).mean()
        loss_Q_last = (R[:,-1] * (1 - C[:, -1]) * H(Q_star[:, -1], V_star[:, -1])).mean()

        loss = loss_Q + self.alpha * loss_G_a + self.beta * loss_G_c

        return {
            "L": loss,
            "Q": loss_Q,
            "G": loss_G_a + loss_G_c,
            "G_a": loss_G_a,
            "G_c": loss_G_c,
            "GQ": loss_Q + loss_G_a + loss_G_c,
            "Q_star": loss_Q_star,
            "Q_last": loss_Q_last,
            "PnIC": IC.mean(),
            "PnIC2": (IC ** 2).mean()
        }

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch)

        for k, v in loss.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "L"), logger=True)

        return loss["L"]

    def validation_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch)

        for k, v in loss.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "L"), logger=True)

    def test_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch)

        for k, v in loss.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=(k == "L"), logger=True)

        return loss["L"]

    def predict_step(self, batch, batch_idx):
        x = self(batch)
        x["loss"] = self.loss(x, batch)

        x["Q_l"] = x["Q"]
        x["Q_l_star"] = x["Q_star"]
        x["Q_a"] = x["V"]
        x["Q_a_star"] = x["V_star"]

        return x

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def solve_canonical_gradient(self, trainer, loader, tau):
        eps = torch.zeros(tau+1)

        # survival indicator (needed for weight computation)
        r = np.ones((len(loader.dataset), tau))
        Y = np.concatenate([x["Y"] for x in loader], axis=0)[:,:,0]
        r[:, 1:] = 1 - Y[:,:-1]

        preds = trainer.predict(self, loader)
        y_hat = np.concatenate([x["Q"] for x in preds], axis=0)[:,:,0]
        y = np.concatenate([x["V"] for x in preds], axis=0)[:,:,0]
        g = np.concatenate([x["g"] for x in preds], axis=0)[:,:,0]

        for t in reversed(range(tau)):
            eps[t] = solve_one_dimensional_submodel(
                y_hat[:,t+1],
                expit(logit(y[:,t+1]) + float(eps[t+1])),
                r[:,t] * g[:,t+1]
                )

            print(t + 1, eps[t])

        self.eps = torch.nn.Parameter(eps[:-1], requires_grad=False)

    def solve_canonical_gradient_common_eps(
            self,
            trainer,
            loader,
            tau,
            max_iter=1000,
            tol=1e-6,
            stop_pnic_se_ratio=False,
            max_delta_eps=None,
            ):
        n = len(loader.dataset)
        r = np.ones((n, tau))
        Y = np.concatenate([x["Y"] for x in loader], axis=0)[:,:,0]
        r[:, 1:] = 1 - Y[:,:-1]

        preds = trainer.predict(self, loader)
        y_hat = np.concatenate([x["Q"] for x in preds], axis=0)[:,1:tau+1,0]
        y = np.concatenate([x["V"] for x in preds], axis=0)[:,1:tau+1,0]
        g = np.concatenate([x["g"] for x in preds], axis=0)[:,1:tau+1,0]
        H = r * g

        eps = np.zeros(max_iter)

        for i in range(max_iter):
            _eps = solve_one_dimensional_submodel(y_hat.ravel(), y.ravel(), H.ravel())

            if max_delta_eps is not None:
                _eps = np.clip(_eps, -max_delta_eps, max_delta_eps)

            eps[i] = _eps

            y_hat = expit(logit(y_hat) + eps[i])
            y[:,:-1] = expit(logit(y[:,:-1]) + eps[i])

            if stop_pnic_se_ratio:
                ic = (g * (y - y_hat)).sum(axis=1)
                se = np.sqrt((ic ** 2).mean() / n)
                if np.abs(ic.mean() / se) < 1 / np.log(n):
                    break

            if np.abs(eps[i]) < tol:
                break

        print('eps: ', eps[:i+1])
        print('eps.sum = ', eps.sum())

        eps = torch.full((tau,), eps.sum())
        self.eps = torch.nn.Parameter(eps, requires_grad=False)

    def get_estimates_from_prediction(self, pred, loader, verbose=True):
        Q_l_star = torch.cat([x["Q_l_star"] for x in pred], axis=0).detach().numpy().squeeze()
        Q_a_star = torch.cat([x["Q_a_star"] for x in pred], axis=0).detach().numpy().squeeze()
        ic = torch.cat([x["IC"] for x in pred], axis=0).detach().numpy().squeeze()

        PnIC = ic.mean()
        PnIC2 = (ic ** 2).mean()
        EIC = np.abs(PnIC / PnIC2 ** 0.5)

        # self.W[index], self.L[index], self.A[index], self.Y[index], self.A_cf_1[index]
        Y = torch.cat([x["Y"] for x in loader], axis=0).detach().numpy()[:,:,0]
        R = np.ones((Y.shape[0], Y.shape[1] + 1))
        R[:, 2:] = 1 - Y[:, :-1]

        est = Q_a_star[:, 0].mean()
        se = np.sqrt((ic ** 2).mean() / ic.shape[0])

        if verbose:
            print("est: ", est)
            print("CI: ", est - 1.96 * se, est + 1.96 * se)
            print("se: ", se)

            # print("lambda = {}".format(model.lam))
            print("E_n[IC] = {}".format(PnIC))
            print("PnIC/√PnIC2 = {}".format(EIC))

            print("Q_l_star", (R * Q_l_star).sum(axis=0) / R.sum(axis=0))
            print("Q_a_star", (R * Q_a_star).sum(axis=0) / R.sum(axis=0))

        return est, se, ic
