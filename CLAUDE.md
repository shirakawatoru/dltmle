# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync                    # Install all dependencies
uv add --dev pytest        # Install test dependencies
pip install -e .           # Install package in editable mode
```

## Running Tests

```bash
# Unit + integration tests (pytest)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_ral.py -v

# Legacy smoke scripts
python -m dltmle.test.fit
python -m dltmle.test.tune
```

## Architecture

### Public API (`dltmle/__init__.py`)

- `dltmle.fit(rng_seed, hparams, W, L, A, C, Y, a)` — train DeepLTMLE, return TMLE estimate
- `dltmle.fit_linear(rng_seed, hparams, W, L, A, C, Y, a)` — train linear (logistic regression) baseline
- `dltmle.tune(rng_seed, hparam_candidates, W, L, A, C, Y, n_trials)` — Optuna hyperparameter search
- `dltmle.example_dgp(rng, n, tau)` — generate synthetic longitudinal data

### Data Structure

Longitudinal panel data with absorbing events:
- `W` — static baseline covariates `(n, dim_static)`
- `L` — time-varying covariates `(n, tau, dim_dynamic)`
- `A` — binary treatment `(n, tau, 1)` float, values in {0, 1}
- `C` — binary censoring `(n, tau, 1)` (absorbing)
- `Y` — binary outcome `(n, tau, 1)` (absorbing)
- `a` — counterfactual treatment regime `(n, tau, 1)` passed to `fit()`

### Model Architecture (`dltmle/model/`)

**`htt.py`** — shared building blocks (imported by G and Q models):
- `HeterogeneousTokenTransformer` — backbone for both G and Q models. Token order: `[W, L_0, A_0, …, L_{τ-1}, A_{τ-1}]`. Uses `nn.Embedding` for A (requires integer input). Returns `z_L` (L-position outputs) and `z_A` (A-position outputs).
- `FiLM` — Feature-wise Linear Modulation: `z_L * gamma[a] + beta[a]`
- `PositionalEncoding` — sinusoidal positional encoding

**`g_model.py`** — `GModel`: own HTT instance, `G_a` head on `z_L`, `G_c` head on `z_A`

**`q_model.py`** — `QModel`: own HTT instance, FiLM critic + EMA target network (`film_target`, `q_target`). Call `update_ema()` each training step.

**`linear.py`** — `LinearGModel`, `LinearQModel`: pooled logistic regression baselines (no transformer)

**`dltmle.py`** — `DeepLTMLE(nn.Module)`: composes `GModel` + `QModel`, no Lightning. Key methods:
- `forward(batch)` — produces `{Q, Q_star, V, V_star, G_a, G_c, g, IC}`
- `loss(S_hat, S)` — returns dict of named losses; use `loss["G"]` and `loss["Q"]` for separate backward passes
- `solve_canonical_gradient_common_eps(loader, device)` — post-training TMLE debiasing
- `get_estimates_from_prediction(preds)` — extracts `(est, se, ic)` from predict loop output

### Token Structure and Loss Convention

Q and V use **opposite** boundary padding — critical for correct Q-learning:
- `Q = [0, Q(t=0), …, Q(t=τ-1)]` — prepended zero (slot 0 is set to `V[:,0].mean()` = the estimand)
- `V = [V(t=0), …, V(t=τ-1), Y(τ-1)]` — appended terminal outcome

Q-learning target: `Q(t) ≈ V_set(t+1)` where `V_set(t) = 1 if Y(t-1)==1`. The loss `H(Q[:,1:], V[:,1:])` aligns these.

### Training Loop (`dltmle/function/fit.py`)

Separate optimizers and backward passes for G and Q per batch:
```python
# G step
G_a, G_c = model.g_model(W, L, A)
loss_g = _g_loss(G_a, G_c, batch, alpha, beta)
opt_g.zero_grad(); loss_g.backward(); opt_g.step()

# Q step  
logit_Q, logit_V = model.q_model(W, L, A, a)
loss_q = _q_loss(logit_Q, logit_V, batch)   # Q(t) ≈ V_set(t+1)
opt_q.zero_grad(); loss_q.backward(); opt_q.step()
model.q_model.update_ema()
```

### Hyperparameters

New hparam names (changed from v0.0.12):

| New | Old (removed) | Notes |
|-----|---------------|-------|
| `dim_model` | `dim_emb`, `dim_emb_time`, `dim_emb_type` | Single embedding dim for HTT |
| `dim_feedforward` | `hidden_size` | Optional, defaults to 4×dim_model |

Unchanged: `num_layers`, `nhead`, `dropout`, `learning_rate`, `alpha`, `beta`, `ema_decay`, `max_epochs`, `batch_size`.

### Inference (`dltmle/ral.py`)

`RALEstimate` wraps `(est, ic)`. Supports delta-method arithmetic (`+`, `-`, `*`, `/`) for ATE, risk ratio, odds ratio. `se = sqrt(mean(ic²) / n)`.
