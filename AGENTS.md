# AGENTS.md — climpred

## Project Overview

climpred is a Python package for verification of weather and climate forecasts and predictions. It is built on xarray, dask, and xskillscore.

- **Source layout**: `src/climpred/` (src-layout)
- **Tests**: `src/climpred/tests/` (pytest)
- **Docs**: `docs/` (Sphinx with myst-nb)
- **Docs for LLMs**: the [`sphinx-llm`](https://github.com/NVIDIA/sphinx-llm)
  extension (`sphinx_llm.txt` in `docs/source/conf.py`) auto-generates machine-readable
  documentation following the [llms.txt standard](https://llmstxt.org) during the HTML
  build: `llms.txt` (index), `llms-full.txt` (full concatenated docs) and a `.md` version
  of every page. These are published at the docs root, e.g.
  `https://climpred.readthedocs.io/en/latest/llms.txt`. No manual upkeep needed — the
  output tracks the docs automatically.
- **Agent skill**: `.agents/skills/climpred-forecast-verification/` holds a task-oriented
  skill for AI coding agents.
- **Python**: ≥3.10

## Key Classes

- `HindcastEnsemble` and `PerfectModelEnsemble` in `src/climpred/classes.py` are the main user-facing classes.
- Configuration via `src/climpred/options.py` (`set_options`).

## Development Commands

```bash
# (optional) create venv with uv
uv venv

# (optional) activate venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[complete]"

# Run tests
pytest src/climpred/tests/ -x

# Run tests in parallel
pytest src/climpred/tests/ -x -n auto

# Run a specific test file
pytest src/climpred/tests/test_HindcastEnsemble_class.py -x

# Pre-commit checks (formatting, linting, type checking)
pre-commit run --all-files

# Check CI status of a PR
gh pr checks <PR_NUMBER>

# View CI run logs for a failed job
gh run view <RUN_ID> --log-failed

# List recent CI workflow runs
gh run list --limit 5

# Watch a running workflow
gh run watch <RUN_ID>
```

## Continuous Integration

Runner minutes are treated as a scarce resource, so `ci.yml` sizes its matrices
by event:

- **Pull requests** run a lean matrix: Linux only, Python 3.10 and 3.13. macOS
  bills at 10x and Windows at 2x Linux runner minutes.
- **The weekly schedule and `workflow_dispatch`** run the full matrix: Linux,
  macOS and Windows, Python 3.10 through 3.13.
- **A pull request that needs the full matrix before merging** can opt in with
  the `full-ci` label. Labels are read from the event payload, so add the label
  first and then re-run the workflow (or push again).
- Coverage is measured and uploaded by the Python 3.13 `maximum-test-conda`
  entry only.
- `[skip-ci]` in a commit message skips the test jobs; `[test-upstream]` opts a
  pull request into `upstream-dev.yml`; the `run-benchmark` label opts it into
  `benchmarks.yml`.

## Code Style & Conventions

- **Formatter**: Black (line-length 88)
- **Import sorting**: isort (profile compatible with Black)
- **Linting**: flake8 (max-line-length 93, ignore W503)
- **Docstring formatting**: blackdoc
- **Type checking**: ty (Red Knot)
- **Pre-commit**: All of the above run via pre-commit hooks; run `pre-commit run --all-files` before submitting changes.

## Coding Guidelines

- Follow NumPy-style docstrings.
- Use xarray and dask idioms — avoid raw NumPy loops over Dataset dimensions.
- Keep imports at the top of files; use lazy imports for heavy optional dependencies (matplotlib, numba, etc.).
- New public API functions/methods must include type annotations, a docstring, and a test.
- Use `xr.testing.assert_allclose` or `xr.testing.assert_equal` in tests.
- Test fixtures are defined in `src/climpred/conftest.py`.

## Dependencies

- Core: xarray, dask, numpy, pandas, cftime, xskillscore, cf-xarray, pooch
- Optional groups: `accel`, `bias-correction`, `io`, `viz`, `vwmp`, `relative-entropy`
- Do not add new core dependencies without discussion. Use optional dependency groups for non-essential features.

## Documentation & Architectural Changes

When making architectural changes, adding/removing public API, or changing project structure:

- Update `README.rst` if the change affects project overview, installation, or usage examples.
- Update this `AGENTS.md` (and its symlink `CLAUDE.md`) if the change affects development commands, conventions, dependencies, or project layout.
- Update `docs/` if the change affects user-facing functionality (add/modify API docs, tutorials, or changelog entries in `CHANGELOG.rst`).

## Git Workflow

- Default branch: `main`
- Do not commit to `main` directly; use feature branches and pull requests.
- Run `pre-commit run --all-files` before committing.
