# AGENTS.md

Guide for AI agents working on the climpred codebase.

## Project Overview

**climpred** is a Python package for verification of weather and climate forecasts. It provides tools for analyzing initialized climate forecast predictions against observations.

- **Purpose**: Verification of initialized prediction ensembles (hindcasts and perfect-model frameworks)
- **Main Classes**: `HindcastEnsemble`, `PerfectModelEnsemble`
- **Homepage**: https://climpred.readthedocs.io
- **License**: MIT

## Development Setup

```bash
# Create development environment (recommended)
conda env create -f ci/requirements/climpred-dev.yml
conda activate climpred-dev
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Commands

### Testing

```bash
# Run all tests locally (recommended)
pytest -n auto

# Run tests with coverage
pytest --cov=climpred --cov-report=xml

# Run specific test file
pytest src/climpred/tests/test_metrics.py

# Run doctests
pytest --doctest-modules src/climpred --ignore src/climpred/tests

# Skip slow tests
pytest -n auto -m "not slow"
```

### Linting and Formatting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Individual formatters
black src/climpred
isort src/climpred
flake8 src/climpred

# Type checking
ty check
```

### Documentation

```bash
cd docs
make html
```

## Code Style

- **Line length**: 88 characters (black default)
- **Formatting**: black, isort
- **Linting**: flake8 (max-line-length=93)
- **Docstrings**: NumPy/Google style
- **Type hints**: Gradually being added

### Import Order

1. Standard library
2. Third-party (numpy, xarray, etc.)
3. Local imports (from .module import ...)

## Project Structure

```
src/climpred/
├── classes.py          # Main classes: HindcastEnsemble, PerfectModelEnsemble
├── metrics.py          # Verification metrics
├── comparisons.py      # Comparison methods
├── bootstrap.py        # Bootstrap confidence intervals
├── bias_removal.py     # Bias correction
├── alignment.py        # Alignment strategies
├── graphics.py         # Plotting
├── prediction.py       # Core computation
├── tests/              # Test suite
│   └── conftest.py     # Pytest fixtures
└── ...

docs/
├── source/             # Documentation source (rst, ipynb)
└── Makefile            # Build commands

ci/requirements/        # Conda environment files
```

## Key Conventions

- Use `xr.Dataset` and `xr.DataArray` from xarray
- Support dask for lazy computation
- Handle both `HindcastEnsemble` and `PerfectModelEnsemble` when adding features
- Add tests for new functionality
- Update CHANGELOG.rst for significant changes

## Dependencies

Managed via `pyproject.toml`. Core dependencies include:
- xarray >=2023.4.0
- dask >=2023.4.0
- numpy >=1.25
- pandas >=2.0
- xskillscore >=0.0.28

## CI/CD

- **Tests**: GitHub Actions on ubuntu/macos/windows, Python 3.9-3.13
- **Coverage**: Target 90%, reported to codecov
- **Pre-commit**: Runs on every commit (black, isort, flake8, ty)

## Useful Debug Commands

```bash
# Show versions for bug reports
python -c "import climpred; climpred.show_versions()"

# Run benchmarks (if changing core code)
cd asv_bench && asv continuous -f 1.1 upstream/main HEAD
```
