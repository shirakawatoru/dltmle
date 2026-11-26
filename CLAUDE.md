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

Stable high-level API (`dltmle/api.py`, used by the stremr / R wrapper), all returning `DLTMLEResult` (`dltmle/result.py`):
- `dltmle.estimate(W, L, A, C, Y, intervention, ...)` — array input
- `dltmle.estimate_long(df, id_col=..., time_col=..., outcome_col=..., treatment_col=..., censoring_cols=..., static_cols=..., dynamic_cols=..., intervention=..., ...)` — long-format DataFrame
- `dltmle.estimate_file(path, ...)` — Parquet / CSV
- CLI: `python -m dltmle estimate ...` (`dltmle/cli.py`)
- Interventions (`dltmle/intervention.py`): `Static`, `Dynamic`, `Stochastic` → resolved to `gstar` (n, tau, J)
- Long ↔ panel conversion: `dltmle/data.py` (`long_to_panel`, `panel_to_long`)
- Design notes: `docs/design-categorical-api.md`

Legacy array API (kept backward compatible):
- `dltmle.fit(rng_seed, hparams, W, L, A, C, Y, a)` — train DeepLTMLE, return TMLE estimate
- `dltmle.fit_linear(rng_seed, hparams, W, L, A, C, Y, a)` — train linear (logistic regression) baseline
- `dltmle.tune(rng_seed, hparam_candidates, W, L, A, C, Y, n_trials)` — Optuna hyperparameter search
- `dltmle.example_dgp(rng, n, tau)` — generate synthetic longitudinal data

### Data Structure

Longitudinal panel data with absorbing events:
- `W` — static baseline covariates `(n, dim_static)`
- `L` — time-varying covariates `(n, tau, dim_dynamic)`
- `A` — treatment class codes `(n, tau, 1)` float, values in {0..J-1} (binary = J=2)
- `C` — censoring category `(n, tau, 1)` in {0..K-1}, 0 = not censored (absorbing; binary = K=2)
- `Y` — binary outcome `(n, tau, 1)` (absorbing)
- `a` — counterfactual regime passed to `fit()`; resolved to `gstar` `(n, tau, J)` = g*(A_t=j | history)
- Within-interval order: `L_t → A_t → C_t → Y_t` (`censoring_timing="after_treatment"`) or `L_t → C_t → A_t → Y_t` (`"before_treatment"`, stremr)

### Model Architecture (`dltmle/model/`)

**`htt.py`** — shared building blocks (imported by G and Q models):
- `HeterogeneousTokenTransformer` — backbone for both G and Q models. Token order: `[W, L_0, A_0, …, L_{τ-1}, A_{τ-1}]`. Uses `nn.Embedding` for A (requires integer input). Returns `z_L` (L-position outputs) and `z_A` (A-position outputs).
- `FiLM` — Feature-wise Linear Modulation: `z_L * gamma[a] + beta[a]`
- `PositionalEncoding` — sinusoidal positional encoding

**`categorical.py`** — reference-coded softmax (J-1 logits vs class 0; J=2 ≡ sigmoid), categorical NLL (BCE for J=2), `mix_logits(logit_cls, gstar)`

**`g_model.py`** — `GModel`: own HTT instance, `G_a` head on `z_L` → `p_A (B,T,J)`, `G_c` head on `z_A` (or `z_L` for before_treatment) → `p_C (B,T,K)`

**`q_model.py`** — `QModel`: own HTT instance, FiLM critic + EMA target network (`film_target`, `q_target`). `forward(W, L, A) -> (logit_Q, logit_V_cls (B,T,J))`. Call `update_ema()` each training step.

**`linear.py`** — `LinearGModel`, `LinearQModel`: pooled (multinomial) logistic regression baselines (no transformer)

**`dltmle.py`** — `DeepLTMLE(nn.Module)`: composes `GModel` + `QModel`, no Lightning. Key methods:
- `forward(batch)` — batch keys `W, L, A, C, Y, gstar`; produces `{Q, Q_star, V, V_star, p_A, p_C, g, IC, IC_tail, logit_Q_raw, logit_V_cls}`
- `from_models(g_model, q_model, tau)` — wrap trained (e.g. linear or shared) G / Q models
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
p_A, p_C = model.g_model(W, L, A)
loss_g = _g_loss(p_A, p_C, batch, alpha, beta)
opt_g.zero_grad(); loss_g.backward(); opt_g.step()

# Q step
logit_Q, logit_V_cls = model.q_model(W, L, A)
logit_V = mix_logits(logit_V_cls, batch["gstar"])   # V = sum_a g*(a) Q(a)
loss_q = _q_loss(logit_Q, logit_V, batch)   # Q(t) ≈ V_set(t+1)
opt_q.zero_grad(); loss_q.backward(); opt_q.step()
model.q_model.update_ema()
```

`run_estimation()` trains one G model (jointly with the first job's Q) and one Q model per (intervention, eval time) job. `BatchLoader` replaces `DataLoader` with identical RNG consumption.

### Hyperparameters

New hparam names (changed from v0.0.12):

| New | Old (removed) | Notes |
|-----|---------------|-------|
| `dim_model` | `dim_emb`, `dim_emb_time`, `dim_emb_type` | Single embedding dim for HTT |
| `dim_feedforward` | `hidden_size` | Optional, defaults to 4×dim_model |

Unchanged: `num_layers`, `nhead`, `dropout`, `learning_rate`, `alpha`, `beta`, `ema_decay`, `max_epochs`, `batch_size`.

Added: `max_weight` (clip of cumulative IPW weight, default 100), `init_bias` (init g heads at marginal log-odds; default False in `fit()`, True in `estimate*()`).

### Inference (`dltmle/ral.py`)

`RALEstimate` wraps `(est, ic)`. Supports delta-method arithmetic (`+`, `-`, `*`, `/`) for ATE, risk ratio, odds ratio. `se = sqrt(mean(ic²) / n)`.
