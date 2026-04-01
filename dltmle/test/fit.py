import numpy as np
import dltmle

def main():
    W, L, A, C, Y = dltmle.example_dgp(np.random.default_rng(0), 1000, 10)

    hparams = {
        'dim_emb': 8,
        'dim_emb_time': 4,
        'dim_emb_type': 4,
        'hidden_size': 16,
        'num_layers': 2,
        'nhead': 4,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'alpha': 1,
        'beta': 1,
        'max_epochs': 100,
        'batch_size': 64,
    }

    psi_hat = dltmle.fit(0, hparams, W, L, A, C, Y, a=np.ones_like(A))
    print(f'psi_hat: {psi_hat}')

if __name__ == '__main__':
    main()
