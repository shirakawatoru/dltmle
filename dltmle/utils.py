import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import numpy as np

from scipy.special import expit, logit
from scipy.optimize import minimize

def seed_everything(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def set_deterministic(flag: bool = True):
    '''Toggle deterministic execution (cuDNN / cuBLAS / torch algorithms).

    On CUDA this sets ``CUBLAS_WORKSPACE_CONFIG`` (effective only if CUDA has
    not been initialised yet in the process), ``torch.use_deterministic_algorithms``
    and disables the fused (flash / memory-efficient) attention kernels.  CPU runs with a fixed seed are
    already deterministic for a fixed number of threads.
    '''
    import os
    if flag:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = bool(flag)
    torch.backends.cudnn.benchmark = not flag
    torch.use_deterministic_algorithms(bool(flag), warn_only=True)
    # fused attention kernels have non-deterministic backward passes on CUDA
    torch.backends.cuda.enable_mem_efficient_sdp(not flag)
    torch.backends.cuda.enable_flash_sdp(not flag)


class deterministic_mode:
    '''Context manager: enable :func:`set_deterministic` inside the block when
    ``flag`` is true and restore the previous global torch settings on exit
    (no-op when ``flag`` is false).'''

    def __init__(self, flag: bool):
        self.flag = bool(flag)

    def __enter__(self):
        if self.flag:
            self._state = (
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.flash_sdp_enabled(),
            )
            set_deterministic(True)
        return self

    def __exit__(self, *exc):
        if self.flag:
            algo, warn_only, cudnn_det, bench, mem_eff, flash = self._state
            torch.use_deterministic_algorithms(algo, warn_only=warn_only)
            torch.backends.cudnn.deterministic = cudnn_det
            torch.backends.cudnn.benchmark = bench
            torch.backends.cuda.enable_mem_efficient_sdp(mem_eff)
            torch.backends.cuda.enable_flash_sdp(flash)
        return False


def resolve_device(device=None) -> str:
    '''Resolve a device spec: ``None``/``"auto"`` → cuda if available else cpu;
    ``"cpu"``, ``"cuda"``, ``"cuda:1"`` are validated and passed through.'''
    if device is None or str(device) == "auto":
        return get_torch_device()
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"device={device!r} requested but CUDA is not available")
    torch.device(device)   # validates the string
    return device

class SinusoidalEncoder(nn.Module):
    def __init__(self, data=None, dim=None, max_period=10000, dim_scale=0.25):
        super().__init__()
        assert data is not None or dim is not None

        if dim is None:
            dim = len(data)**dim_scale

        half = dim // 2
        self.freqs = nn.Parameter(
            torch.exp(-math.log(max_period) * 
                torch.arange(start=0, end=half, dtype=torch.float) / half
            ), requires_grad=False)

        self.res = dim % 2

    def forward(self, timesteps):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.FloatTensor(timesteps)
        timesteps = timesteps.to(self.freqs.device)

        args = (timesteps[..., None] * self.freqs).view(timesteps.shape[0], -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([embedding, torch.zeros_like(embedding[:, :self.res])], dim=-1)

def solve_one_dimensional_submodel_offset(offset, y, H):
    '''Weighted logistic regression ``y ~ offset + eps`` (intercept-only
    fluctuation with logit offset); returns the minimising ``eps``.

    Equivalent to :func:`solve_one_dimensional_submodel` with
    ``y_hat = expit(offset)`` but avoids the logit/expit round trip.
    '''
    offset = np.asarray(offset, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        s = expit(offset + eps)
        return -np.mean(H * (y * _safelog(s) + (1 - y) * _safelog(1 - s)))

    def _jac(eps):
        s = expit(offset + eps)
        return -np.mean(H * (y - s))

    return float(minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0])


def solve_one_dimensional_submodel(y_hat, y, H):
    '''fit univariate logistic regression: y ~ reg.predict(X) + eps with weight H
    
    Parameters
    ----------
    logit_y_hat: n dimensional vector
        logit of predicted probability of y = 1
    y: n dimensional vector
        labels for each sample
    H: n dimensional vector
        weight for logistic regression

    Returns
    -------
    eps: `float`
        translation which minimize the logistic loss    
    '''
    logit_y_hat = logit(y_hat)

    def _safelog(x, delta=1e-8):
        return np.log(np.clip(x, delta, np.inf))

    def _loss(eps):
        '''weighted logistic loss function (binary cross entropy loss)'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y * _safelog(s) + (1 - y) * _safelog(1 - s)))
    
    def _jac(eps):
        '''gradient of the loss function'''
        s = expit(logit_y_hat + eps)
        return -np.mean(H * (y - s))
    
    return minimize(_loss, 0, method='L-BFGS-B', jac=_jac, tol=1e-14).x[0]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Cosine LR schedule with linear warmup (no external dependencies)."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def get_torch_device(use_cpu=False):
    if use_cpu:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    # elif torch.backends.mps.is_available():
    #     return 'mps'
    
    return 'cpu'