# flake8: noqa
from importlib_metadata import distribution

from . import (
    bias_removal,
    bootstrap,
    comparisons,
    constants,
    exceptions,
    graphics,
    horizon,
    metrics,
    prediction,
    relative_entropy,
    smoothing,
    stats,
    testing,
    tutorial,
)
from .classes import HindcastEnsemble, PerfectModelEnsemble
from .options import set_options
from .preprocessing import mpi, shared
from .versioning.print_versions import show_versions

__version__ = distribution(__name__).version
