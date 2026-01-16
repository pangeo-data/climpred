# AGENTS.md - climpred Development Guide

This file provides guidelines for AI agents working on the climpred codebase.

## Build, Lint, and Test Commands

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install climpred and sync dependencies from pyproject.toml
uv sync

# Add optional dependencies using uv add (updates pyproject.toml and uv.lock)
uv add --extra test  # add test dependencies
uv add --extra complete  # add all extras
uv add --extra regridding  # add xesmf for regridding

# Run pytest using uvx (runs in isolated environment with dependencies)
uvx pytest --doctest-modules climpred --ignore climpred/tests

# Or activate the environment and run pytest directly
source .venv/bin/activate
pytest --doctest-modules climpred --ignore climpred/tests

# For development, install in editable mode
uv pip install -e .

# Or via conda
conda env create -f ci/requirements/climpred-dev.yml
conda activate climpred-dev
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files src/climpred/*.py
```

### Testing

```bash
# Run all tests (using uvx to ensure dependencies are available)
uvx pytest climpred

# Run a single test file
uvx pytest climpred/tests/test_checks.py

# Run a specific test
uvx pytest climpred/tests/test_checks.py::test_has_dims_str

# Run tests matching a pattern
uvx pytest -k "test_has_dims"

# Run doctests
uvx pytest --doctest-modules climpred --ignore climpred/tests

# Run tests without slow marker
uvx pytest -m "not slow"

# Run with coverage
uvx pytest --cov=climpred --cov-report=term-missing

# Or activate the virtual environment first
source .venv/bin/activate
pytest climpred
```

### Linting and Formatting

```bash
# Black formatting
black src/climpred/

# isort imports
isort src/climpred/

# flake8 linting
flake8 src/climpred/ --max-line-length=93

# mypy type checking
mypy src/climpred/
```

### Documentation

```bash
# Build docs locally
cd docs
make html
# Output in docs/build/html/

# Generate API docs
cd docs/source
sphinx-autogen -o api api.rst
```

README.rst and docs/source/index.rst explain how to get started

## Code Style Guidelines

### General Principles

- Follow PEP-8 naming conventions
- Write numpydoc-style docstrings for all public functions
- Use type hints (mypy is enforced via pre-commit)
- Keep functions focused and reasonably sized (<150 lines when possible)
- Add docstrings to all public classes and functions

### Imports

```python
# Standard library first
import warnings
from typing import List, Optional, Union

# Third-party imports
import dask
import xarray as xr

# Local imports
from .constants import VALID_LEAD_UNITS
from .exceptions import DatasetError, DimensionError
from .options import OPTIONS

# Use isort with: known_first_party=climpred, multi_line_output=3
```

### Naming Conventions

```python
# Classes: PascalCase
class HindcastEnsemble:
class PerfectModelEnsemble:
class PredictionEnsemble:

# Functions and variables: snake_case
def has_valid_lead_units():
def match_initialized_dims():
valid_lead_units = ...

# Private methods: leading underscore
def _check_valid_reference():
def _internal_helper():

# Constants: UPPER_SNAKE_CASE
VALID_LEAD_UNITS = ["days", "months", "years"]
NCPU = dask.system.CPU_COUNT

# Type variables: PascalCase if generic
T = TypeVar("T")
ArrayLike = Union[np.ndarray, xr.DataArray]
```

### Type Hints

```python
# Use type hints throughout
def has_dims(xobj, dims, kind) -> bool:
    ...

def _check_valid_reference(reference: Optional[Union[List[str], str]]) -> List[str]:
    ...

# For xarray objects, use xr.DataArray | xr.Dataset where applicable
def match_initialized_dims(
    init: Union[xr.DataArray, xr.Dataset],
    verif: Union[xr.DataArray, xr.Dataset],
    uninitialized: bool = False
) -> bool:
    ...
```

### Docstrings

```python
def match_calendars(
    ds1, ds2, ds1_time="init", ds2_time="time", kind1="initialized", kind2="observation"
):
    """Check that calendars match between two xarray Datasets.

    This assumes that the two datasets coming in have cftime time axes.

    Args:
        ds1, ds2 (xarray.Dataset, xr.DataArrays): to compare calendars on.
            For classes, ds1 can be thought of Dataset already existing in
            the object, and ds2 the one being added.
        ds1_time, ds2_time (str, default 'time'): Name of time dimension to
            look for calendar in.
        kind1, kind2 (str): Puts `ds1` and `ds2` into context for custom
            error message. Defaults to `ds1` being "initialized" and
            `ds2` being "observation".

    Returns:
        True if calendars match, False if they do not match.
    """
```

### Error Handling

```python
# Use custom exceptions from climpred.exceptions
from climpred.exceptions import DatasetError, DimensionError, VariableError

# Raise with descriptive messages
if not all(dim in xobj.dims for dim in dims):
    raise DimensionError(
        f"Your {kind} object must contain the "
        f"following dimensions at the minimum: {dims}"
        f", found {list(xobj.dims)}."
    )

# Use warnings for non-critical issues
warnings.warn(
    f"Consider chunking input `ds` along other dimensions..."
)
```

### Formatting

- Line length: 88 characters (black default)
- Use parentheses for long function calls and conditionals
- Sort imports with isort
- Use trailing commas in multi-line calls

### Testing

```python
# Test file naming: test_<module>.py
# Test function naming: test_<function_name>

import pytest
from climpred.checks import has_dims
from climpred.exceptions import DimensionError

def test_has_dims_str(da1):
    """Test if check works for a string."""
    assert has_dims(da1, "x", "arbitrary")

def test_has_dims_fail(da1):
    """Test if check fails properly for a string."""
    with pytest.raises(DimensionError) as e:
        has_dims(da1, "z", "arbitrary")
    assert "Your arbitrary object must contain" in str(e.value)

# Use pytest.mark.parametrize for multiple test cases
@pytest.mark.parametrize("lead_units", VALID_LEAD_UNITS)
def test_valid_lead_units(da_lead, lead_units):
    ...
```

### xarray Patterns

```python
# Return xarray objects from functions
def attach_long_names(xobj):
    xobj2 = xobj.copy()
    for key, value in CF_LONG_NAMES.items():
        if key in xobj2.coords:
            xobj2.coords[key].attrs["long_name"] = value
    return xobj2

# Use .attrs for metadata
xobj["lead"].attrs["units"] = units

# Support both DataArray and Dataset
def check(xobj: Union[xr.DataArray, xr.Dataset]) -> bool:
    ...
```

### Performance Considerations

- Consider dask compatibility for large datasets
- Use `dask.is_dask_collection()` to check for chunking
- Warn about chunking in `warn_if_chunking_would_increase_performance()`
- Use `NCPU = dask.system.CPU_COUNT` for parallelization decisions

### Project Structure

```
climpred/
├── __init__.py           # Main package exports
├── classes.py            # HindcastEnsemble, PerfectModelEnsemble
├── checks.py             # Validation functions
├── comparisons.py        # Comparison classes
├── constants.py          # Constants (VALID_*, CF_*)
├── exceptions.py         # Custom exceptions
├── metrics.py            # Metric definitions
├── prediction.py         # Prediction functions
├── reference.py          # Reference forecast functions
├── bootstrap.py          # Bootstrap functions
├── bias_removal.py       # Bias correction
├── smoothing.py          # Smoothing functions
├── stats.py              # Statistics utilities
├── graphics.py           # Plotting functions
├── options.py            # Runtime options
├── preprocessing/        # Data preprocessing
│   ├── shared.py
│   └── mpi.py
├── tests/                # Test suite
│   ├── test_*.py
│   └── conftest.py       # Pytest fixtures
└── tutorial.py           # Tutorial data loading
```

### Key Configuration Files

- `pyproject.toml`: Black, isort, mypy, pytest configuration
- `.pre-commit-config.yaml`: Pre-commit hooks (black, isort, flake8, mypy, etc.)
- `ci/requirements/climpred-dev.yml`: Development environment
