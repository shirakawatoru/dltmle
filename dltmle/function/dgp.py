import numpy as np

def expit(x):
    '''expit function aka sigmoid function'''
    return (1 + np.exp(-x)) ** -1

def example_dgp(rng:np.random.Generator, n:int, tau:int, a_cf = None):
    W = rng.normal(0, 1, n)

    L = np.zeros((n, tau))
    A = np.zeros((n, tau))
    C = np.zeros((n, tau))
    Y = np.zeros((n, tau))

    t = 0
    L[:, t] = rng.normal(0.1 * W, 1)
    A[:, t] = rng.binomial(1, expit(-0.5 * W + L[:, t])) if a_cf is None else a_cf
    C[:, t] = rng.binomial(1, expit(-4 + 0.3 * W + 0.5 * L[:, t] - A[:, t])) if a_cf is None else 0
    Y[:, t] = rng.binomial(1, expit(-3 + 0.2 * W + 0.2 * L[:, t] - 2 * A[:, t]))

    Y[C[:, t] == 1, t] = 0

    for t in range(1, tau):
        L[:, t] = rng.normal(0.1 * W - 0.1 * L[:, t-1] - 0.1 * A[:, t-1], 1)
        A[:, t] = rng.binomial(1, expit(-0.5 + 0.3 * W + 0.3 * L[:, t] + 5 * A[:, t-1] - 1.5)) if a_cf is None else a_cf
        C[:, t] = rng.binomial(1, expit(-5 + 0.3 * W + 0.5 * L[:, t] - A[:, t])) if a_cf is None else 0
        Y[:, t] = rng.binomial(1, expit(-3 + 0.2 * W + 0.2 * L[:, t] - 2 * A[:, t]))

        C[C[:, t-1]==1, t] = 1
        C[Y[:, t-1]==1, t] = 0

        Y[C[:, t] == 1, t] = 0
        Y[Y[:, t-1]==1, t] = 1

        idx = (Y[:, t-1] == 1) | (C[:, t-1] == 1)
        L[idx, t] = L[idx, t-1]
        A[idx, t] = A[idx, t-1]
        Y[idx, t] = Y[idx, t-1]
        C[idx, t] = C[idx, t-1]

    return W[:,None], L[:,:,None], A[:,:,None], C[:,:,None], Y[:,:,None]
