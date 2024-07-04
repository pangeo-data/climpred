"""Verification of weather and climate forecasts and prediction."""

# flake8: noqa
from importlib.metadata import PackageNotFoundError, version as _get_version

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
    __version__ = _get_version("climpred")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass  # pragma: no cover
