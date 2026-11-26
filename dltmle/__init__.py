try:
    from importlib.metadata import version as _version, PackageNotFoundError
    try:
        __version__ = _version("dltmle")
    except PackageNotFoundError:          # running from a source tree
        __version__ = "unknown"
except ImportError:                        # pragma: no cover
    __version__ = "unknown"

from .function.fit import fit, fit_linear, DEFAULT_HPARAMS, DEFAULT_LINEAR_HPARAMS
from .function.tune import tune
from .function.dgp import example_dgp, example_dgp_categorical
from .intervention import Static, Dynamic, Stochastic
from .api import estimate, estimate_long, estimate_file
from .data import long_to_panel, panel_to_long, read_table
from .result import DLTMLEResult
from .ral import RALEstimate

__all__ = [
    "__version__",
    "fit", "fit_linear", "tune",
    "estimate", "estimate_long", "estimate_file",
    "Static", "Dynamic", "Stochastic",
    "DLTMLEResult", "RALEstimate",
    "long_to_panel", "panel_to_long", "read_table",
    "example_dgp", "example_dgp_categorical",
    "DEFAULT_HPARAMS", "DEFAULT_LINEAR_HPARAMS",
]
