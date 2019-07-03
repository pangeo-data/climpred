from pkg_resources import DistributionNotFound, get_distribution

from . import (
    bootstrap,
    constants,
    comparisons,
    exceptions,
    graphics,
    metrics,
    prediction,
    relative_entropy,
    stats,
    tutorial,
)
from .classes import HindcastEnsemble, PerfectModelEnsemble
from .versioning.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
