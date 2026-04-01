import numpy as np
import dltmle

def main():
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

    hparams = dltmle.tune(0, hparams_candidates, W, L, A, C, Y, n_trials=1)

    print(f'Best hyperparameters: {hparams}')

if __name__ == '__main__':
    main()