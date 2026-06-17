"""Verification of weather and climate forecasts and prediction."""

# Breadcrumbs for humans and AI coding agents:
# - Docs (HTML):   https://climpred.readthedocs.io/en/stable/
# - Docs (LLM):    https://climpred.readthedocs.io/en/latest/llms.txt
# - Dev guide:     AGENTS.md (symlinked as CLAUDE.md) at the repository root
# - Agent skill:   .agents/skills/climpred-forecast-verification/SKILL.md
# Main entry points are the classes climpred.HindcastEnsemble and
# climpred.PerfectModelEnsemble in src/climpred/classes.py.

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
