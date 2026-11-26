<div align="center">

<img src="dltmle_logo.png" width="800"/>

[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

**Deep LTMLE** is a Python module for etstimating the mean countefactual outcomes under dynamic interventions identified through g-formula.

**Authors:** [Toru Shirakawa](https://github.com/shirakawatoru), [Sky Qiu](https://github.com/tq21), [Yulun Wu](https://github.com/yulun-rayn), [Yuxun Li](https://github.com/yuxuanliii), [Mark van der
Laan](https://vanderlaan-lab.org/)


## Install 

```bash
pip install git+https://github.com/shirakawatoru/dltmle.git
```

## Issues

If you encounter any bugs or have any specific feature requests, please [file an issue](https://github.com/shirakawatoru/dltmle/issues).


## Example

```python
import dltmle
import numpy as np

W, L, A, C, Y = dltmle.example_dgp(np.random.default_rng(0), 1000, 10)

hparams_candidates = {
    'dim_emb': [8, 16],
    'dim_emb_time': [4, 8],
    'dim_emb_type': [4, 8],
    'hidden_size': [8, 16, 32],
    'num_layers': [1, 2, 4],
    'nhead': [2, 4],
    'dropout': [0, 0.1, 0.2],
    'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5],
    'alpha': [0.05, 0.1, 0.5, 1],
    'beta': [0.05, 0.1, 0.5, 1],
    'ema_decay': [0.9, 0.95, 0.99],
    'max_epochs': [100],
    'batch_size': [64],
}

hparams = dltmle.tune(0, hparams_candidates, W, L, A, C, Y)

psi_0 = dltmle.fit(0, hparams, W, L, A, C, Y, np.zeros_like(A))
psi_1 = dltmle.fit(0, hparams, W, L, A, C, Y, np.ones_like(A))

print('mean counterfactual outcome under a = 0', psi_0)
print('mean counterfactual outcome under a = 1', psi_1)
print('ATE (risk difference)', psi_1 - psi_0)
print('risk ratio', psi_1 / psi_0)
print('odds ratio', (psi_1 / (1 - psi_1)) / (psi_0 / (1 - psi_0)))
```

## Categorical treatment / censoring and the long-format API

Treatments may have J classes and censoring K categories (0 = not censored,
1..K-1 = censoring reasons); all censoring is prevented under the intervention.
Interventions can be static, dynamic (a function of the observed history) or
stochastic (class probabilities).

```python
import dltmle
import numpy as np

W, L, A, C, Y = dltmle.example_dgp_categorical(np.random.default_rng(0), 2000, 4,
                                               n_treatment=3, n_censoring=3)
res = dltmle.estimate(
    W, L, A, C, Y,
    intervention={
        "always_0": 0,
        "always_2": dltmle.Static(2),
        "dynamic": dltmle.Dynamic(lambda W, L, A: np.where(L[..., 0] > 0, 2, 1)),
        "random": dltmle.Stochastic([0.2, 0.3, 0.5]),
    },
    eval_times="all", seed=0, device="cpu",
)
print(res)                          # psi(t) = P(Y_t = 1), SE, 95% CI per regime / time
print(res.contrasts("always_0"))    # risk differences / ratios
res.to_json("result.json")          # estimates, IC per id, diagnostics, hparams, seed, version
```

Long-format (person-time) data, e.g. from R / stremr:

```python
res = dltmle.estimate_long(
    df, id_col="ID", time_col="t", outcome_col="Y.tplus1", treatment_col="drug",
    censoring_cols=["C_death", "C_admin"], static_cols=["age"], dynamic_cols=["hba1c"],
    censoring_timing="before_treatment",   # stremr orders C(t) before A(t)
    intervention={"SU": "SU", "SGLT2i": "SGLT2i"},
)
```

Command line (Parquet / CSV in, JSON / CSV out):

```bash
python -m dltmle estimate --input data.parquet --id-col ID --time-col t \
    --outcome-col Y --treatment-col drug --censoring-cols C \
    --static-cols age --dynamic-cols hba1c \
    --intervention SU=static:SU --intervention SGLT2i=static:SGLT2i \
    --seed 1 --device cpu --deterministic --output result.json --estimates-csv est.csv
```

See [`docs/design-categorical-api.md`](docs/design-categorical-api.md) for the design notes.

## Citation

```bibtex
@inproceedings{shirakawa_2024_dltmle,
    title={Targeted Minimum Loss-Based Estimation with Temporal-Difference Heterogeneous Transformer},
    author={Shirakawa, Toru and Yi, Li and Yulun, Wu and Sky, Qiu and Yuxuan, Li and Mingduo, Zhao and Hiroyasu, Iso and Mark, van der Laan},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2024}
}
```

## License

© 2024 [Toru Shirakawa](https://github.com/shirakawatoru), [Sky Qiu](https://github.com/tq21), [Yulun Wu](https://github.com/yulun-rayn), [Yuxun Li](https://github.com/yuxuanliii), [Mark van der
Laan](https://vanderlaan-lab.org/)

The contents of this repository are distributed under the MIT license.
See file `LICENSE` for details.

## History
- 0.1.0
    - Categorical treatment (J classes, multinomial g^A) and categorical censoring (K categories, cause-specific g^C)
    - Static / dynamic / stochastic interventions for categorical treatments (`dltmle.Static`, `Dynamic`, `Stochastic`)
    - Stable high-level API `dltmle.estimate`, `estimate_long`, `estimate_file` returning `DLTMLEResult`
      (per-time psi, SE, CI, IC per id, diagnostics, hparams, seed, version)
    - CLI `python -m dltmle estimate ...`
    - `device`, `deterministic`, `censoring_timing` options; multiple regimes / evaluation times with a shared G model
    - IC is centred at the full-sample estimate (was: per-batch mean)
- 0.0.12
    - Implement stochastic intervension
    - Implement target Q head with a hyperparameter `ema_decay`
    - Change state representation for $G$ by FiLM: $Z(W,L_{1:t},A_{1:t})=f(W,L_{1:t},A_{1:t-1})\odot\sigma(A_t)+\mu(A_t)$ with a Transforer $f$ and linear modules $\sigma$ and $\mu$.
    - Implement cosine LR schduler

