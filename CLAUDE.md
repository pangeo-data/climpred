# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

climpred is a Python package for verification of weather and climate forecasts. It provides tools for evaluating initialized forecast quality, comparing predictions against observations and references, and applying various verification metrics.

The package is built on xarray and supports both hindcast (real-world) and perfect-model (climate model) ensemble verification workflows.

## Development Commands

### Installation

Install for development:
```bash
python -m pip install -e .[complete]
```

Install minimal dependencies:
```bash
python -m pip install -e .
```

### Testing

Run all tests:
```bash
pytest
```

Run tests in parallel (faster):
```bash
pytest -n 4
```

Run specific test file:
```bash
pytest src/climpred/tests/test_HindcastEnsemble_class.py
```

Run tests with coverage:
```bash
pytest -n 4 --cov=climpred --cov-report=xml
```

Run tests excluding slow tests:
```bash
pytest -m "not slow"
```

Run doctests:
```bash
python -m pytest --doctest-modules src/climpred --ignore src/climpred/tests
```

Cache tutorial datasets before testing (required for parallel tests):
```bash
python -c "import climpred; climpred.tutorial._cache_all()"
```

### Code Quality

Run all pre-commit hooks:
```bash
pre-commit run --all-files
```

Format code with black:
```bash
black src/climpred
```

Sort imports with isort:
```bash
isort src/climpred
```

Lint with flake8:
```bash
flake8 --max-line-length=93 --extend-ignore=W503 src/climpred
```

Type check with mypy:
```bash
mypy src/climpred
```

### Documentation

Build documentation:
```bash
cd docs
make html
```

The documentation uses Sphinx with Jupyter notebooks. Notebooks are pre-compiled and checked in CI.

## Architecture

### Core Classes

The package has two main prediction ensemble classes in `src/climpred/classes.py`:

1. **HindcastEnsemble**: For real-world initialized predictions verified against observations
   - Add initialized forecast: `.add_ensemble()`
   - Add observations: `.add_observations()`
   - Add uninitialized reference: `.add_uninitialized()`
   - Compute skill: `.verify()`
   - Bootstrap confidence intervals: `.bootstrap()`

2. **PerfectModelEnsemble**: For climate model experiments where ensemble members serve as verification
   - Add initialized data: `.add_ensemble()`
   - Add control simulation: `.add_control()`
   - Compute skill: `.verify()`
   - Bootstrap confidence intervals: `.bootstrap()`

Both classes inherit shared functionality from a base class and follow a similar API pattern.

### Verification Workflow Components

**Metrics** (`src/climpred/metrics.py`):
- Defines ~50+ verification metrics (RMSE, correlation, CRPS, etc.)
- Each metric is a `Metric` class with metadata (deterministic/probabilistic, positive/negative orientation)
- Metrics are applied via xskillscore library
- See `__ALL_METRICS__` for the complete list

**Comparisons** (`src/climpred/comparisons.py`):
- Defines how forecasts are paired with verification data
- Perfect-model comparisons: `m2m` (member-to-member), `m2e` (member-to-ensemble-mean), `m2c` (member-to-control), `e2c` (ensemble-mean-to-control)
- Hindcast comparisons: `m2o` (member-to-observations), `e2o` (ensemble-mean-to-observations)
- Each comparison is a `Comparison` class that determines whether it supports deterministic/probabilistic metrics

**Alignment** (`src/climpred/alignment.py`):
- Handles temporal alignment between forecasts and verification
- Supports different calendar types via cftime

**Bias Removal** (`src/climpred/bias_removal.py`):
- Methods for removing systematic forecast biases
- Integrates with xclim's bias correction methods
- Supports train/test splitting for cross-validation

**Bootstrap** (`src/climpred/bootstrap.py`):
- Resampling methods for computing confidence intervals
- Supports bootstrapping over init, member, and other dimensions
- Generates p-values and significance tests

**Reference Forecasts** (`src/climpred/reference.py`):
- Persistence, climatology, and uninitialized references
- Used for skill scores (e.g., comparing against persistence baseline)

### Preprocessing

**Preprocessing modules** (`src/climpred/preprocessing/`):
- `shared.py`: General preprocessing utilities
- `mpi.py`: MPI-specific preprocessing for HPC environments

### Utilities

- `utils.py`: General utility functions for coordinate handling, time conversions
- `checks.py`: Validation functions for dimensions, calendars, datasets
- `stats.py`: Statistical utilities
- `graphics.py`: Basic plotting helpers
- `tutorial.py`: Demo datasets for examples and testing

### Key Design Patterns

1. **Dimension naming**: climpred uses standardized dimension names defined in `constants.py`:
   - `init`: initialization time
   - `lead`: forecast lead time
   - `member`: ensemble member
   - Code checks and renames dimensions to ensure consistency

2. **Lazy evaluation**: Operations preserve dask arrays for out-of-core computation

3. **xarray-centric**: All data flows through xarray Datasets/DataArrays with rich metadata

4. **Composable verification**: Metrics, comparisons, and references can be mixed and matched

## Testing Notes

- Tests use pytest with lazy fixtures (`pytest-lazy-fixture`)
- Test data is loaded via `climpred.tutorial` module which uses pooch for caching
- Tests include CI variants: minimum dependencies, maximum dependencies, upstream dev versions
- Doctests are run separately from unit tests
- Some tests are marked as `slow` and can be skipped
- Tests for MPI functionality are marked with `mistral` (requires HPC environment)

## Python Compatibility

Supports Python 3.9, 3.10, 3.11, 3.12

## Code Style

- Black formatting (line length 88)
- isort for import sorting
- flake8 for linting (max line length 93, ignoring W503)
- Type hints checked with mypy (though many dependencies have missing stubs)
