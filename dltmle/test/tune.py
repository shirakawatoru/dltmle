import numpy as np
import dltmle

def main():
    W, L, A, C, Y = dltmle.example_dgp(np.random.default_rng(0), 1000, 10)

    hparams_candidates = {
        'dim_model': [16, 32, 64],
        'num_layers': [1, 2, 4],
        'nhead': [2, 4],
        'dropout': [0, 0.1, 0.2],
        'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5],
        'alpha': [0.05, 0.1, 0.5],
        'beta': [0.05, 0.1, 0.5],
        'ema_decay': [0.9, 0.95, 0.99],
        'max_epochs': [100],
        'batch_size': [64],
    }

    hparams = dltmle.tune(0, hparams_candidates, W, L, A, C, Y, n_trials=1)
    print(f'Best hyperparameters: {hparams}')

if __name__ == '__main__':
    main()
