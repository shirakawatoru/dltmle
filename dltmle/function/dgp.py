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


def _softmax_sample(rng, logits):
    """Sample one class per row from unnormalised logits (n, m)."""
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random((logits.shape[0], 1))
    return (u > p.cumsum(axis=1)).sum(axis=1).clip(max=logits.shape[1] - 1)


def example_dgp_categorical(
    rng: np.random.Generator,
    n: int,
    tau: int,
    n_treatment: int = 3,
    n_censoring: int = 3,
    regime=None,
):
    """Synthetic panel with a categorical treatment and categorical censoring.

    Treatment ``A_t ∈ {0..J-1}`` follows a multinomial logistic model in
    ``(W, L_t, onehot(A_{t-1}))``; censoring ``C_t ∈ {0..K-1}`` (0 = not
    censored, k >= 1 = reason k) follows a multinomial logistic model in
    ``(W, L_t, onehot(A_t))``; the outcome is an absorbing binary event.

    Parameters
    ----------
    regime : None, int, callable or array
        ``None`` → observational data.  Otherwise the data are generated under
        the intervention *without censoring* (for computing true values):
        an ``int`` class (static), a callable ``fn(t, W, L_hist, A_hist) ->
        class indices (n,)`` with ``L_hist = L[:, :t+1]`` and
        ``A_hist = A[:, :t]`` (dynamic), or a probability vector ``(J,)``
        (stochastic).

    Returns
    -------
    W (n, 1), L (n, tau, 1), A (n, tau, 1), C (n, tau, 1), Y (n, tau, 1)
    with A and C as integer-valued floats.
    """
    J, K = n_treatment, n_censoring
    W = rng.normal(0, 1, n)

    L = np.zeros((n, tau))
    A = np.zeros((n, tau), dtype=np.int64)
    C = np.zeros((n, tau), dtype=np.int64)
    Y = np.zeros((n, tau))

    bW = np.linspace(-0.4, 0.4, J)
    bL = np.linspace(0.5, -0.5, J)
    effY = np.linspace(0.0, -1.5, J)       # class J-1 most protective
    effL = np.linspace(0.0, -0.3, J)

    for t in range(tau):
        A_prev = A[:, t - 1] if t > 0 else None
        mean_L = 0.1 * W if t == 0 else 0.1 * W - 0.1 * L[:, t - 1] + effL[A_prev]
        L[:, t] = rng.normal(mean_L, 1)

        if regime is None:
            # A_{-1} := 0 (matches the zero lag padding of the linear model)
            lag = A_prev if t > 0 else np.zeros(n, dtype=np.int64)
            logits = (bW[None, :] * W[:, None] + bL[None, :] * L[:, t][:, None]
                      + 3.0 * np.eye(J)[lag])
            A[:, t] = _softmax_sample(rng, logits)
        elif callable(regime):
            A[:, t] = np.asarray(regime(t, W, L[:, :t + 1], A[:, :t]), dtype=np.int64)
        elif np.ndim(regime) == 0:
            A[:, t] = int(regime)
        else:
            A[:, t] = _softmax_sample(rng, np.log(np.clip(np.asarray(regime, float), 1e-300, None))[None, :].repeat(n, 0))

        if regime is None:
            c_logits = np.zeros((n, K))
            for k in range(1, K):
                c_logits[:, k] = (-4.0 - 0.3 * (k - 1) + 0.3 * W * (1 if k % 2 else -1)
                                  + 0.4 * L[:, t] - 0.4 * (A[:, t] == J - 1))
            C[:, t] = _softmax_sample(rng, c_logits)

        Y[:, t] = rng.binomial(1, expit(-3 + 0.2 * W + 0.3 * L[:, t] + effY[A[:, t]]))

        if t > 0:
            C[C[:, t - 1] > 0, t] = C[C[:, t - 1] > 0, t - 1]
            C[Y[:, t - 1] == 1, t] = 0
            Y[Y[:, t - 1] == 1, t] = 1
            idx = (Y[:, t - 1] == 1) | (C[:, t - 1] > 0)
            L[idx, t] = L[idx, t - 1]
            A[idx, t] = A[idx, t - 1]
            Y[idx, t] = Y[idx, t - 1]
        Y[C[:, t] > 0, t] = 0

    f = np.float64
    return W[:, None], L[:, :, None], A[:, :, None].astype(f), C[:, :, None].astype(f), Y[:, :, None]
