# flake8: noqa
from pkg_resources import DistributionNotFound, get_distribution

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

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover
