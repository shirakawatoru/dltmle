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