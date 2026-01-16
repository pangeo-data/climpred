# UV Migration Specification

Date: 2026-01-16

## Overview

This document describes the migration of climpred's development environment and CI/CD pipeline from pip/conda to uv (https://github.com/astral-sh/uv).

## Motivation

- **Speed**: uv is 10-100x faster than pip for package installation
- **Reliability**: Lock files ensure reproducible builds
- **Simplicity**: Single tool for dependency management instead of pip + conda
- **Modern**: Written in Rust, leveraging modern Python packaging standards

## Changes

### 1. GitHub Actions Workflows

All workflows now use `astral-sh/setup-uv@v5` action for uv installation:

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
```

Package installation changed from `pip install` to `uv pip install`:

```yaml
# Before
- name: Install dependencies
  run: |
    pip install -e .
    pip install -e .[complete]

# After
- name: Install uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
- name: Install dependencies
  run: |
    uv pip install -e .
    uv pip install -e .[complete]
```

#### Modified Workflows

1. **`.github/workflows/climpred_testing.yml`**
   - Replaced micromamba/conda setup with uv
   - Removed conda environment caching (uv handles caching via `enable-cache`)
   - Python version matrix preserved

2. **`.github/workflows/climpred_installs.yml`**
   - Simplified Python setup using actions/setup-python
   - Added uv action for fast dependency installation

3. **`.github/workflows/benchmarks.yml`**
   - Removed conda dependency
   - Direct uv installation of asv and dependencies

4. **`.github/workflows/upstream-dev-ci.yml`**
   - Replaced conda environment setup with uv
   - Updated upstream wheel installation to use uv

5. **`.github/workflows/publish-production-pypi.yml`**
   - Simplified build process using uv

### 2. Lock File

Generated `uv.lock` file for reproducible installs:

```bash
uv pip compile pyproject.toml -o uv.lock --generate-hashes
```

This ensures:
- Deterministic builds across all environments
- Cryptographic verification of package integrity
- Fast resolution using uv's resolver

### 3. Documentation Updates

Updated installation instructions in:

- `README.rst` - Added uv as recommended installation method
- `docs/source/index.rst` - Added uv installation instructions
- `AGENTS.md` - Updated development guide with uv commands

### 4. Local Development

Developers can now use:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Install with all extras
uv pip install -e .[complete]

# Install test dependencies
uv pip install -e .[test]
```

## Removed Components

- `ci/requirements/minimum-tests.yml` - No longer needed for CI
- `ci/requirements/maximum-tests.yml` - No longer needed for CI
- `ci/requirements/docs.yml` - No longer needed for CI
- `ci/requirements/climpred-dev.yml` - Still available for conda users but optional
- `requirements_upstream.txt` - Still used for upstream CI but installed via uv

## Backward Compatibility

- Conda installation still works: `conda install -c conda-forge climpred`
- Pip installation still works: `pip install climpred`
- Conda environment files remain available for users preferring conda

## Migration Checklist

- [x] Generate uv.lock file with hashes
- [x] Update climpred_testing.yml
- [x] Update climpred_installs.yml
- [x] Update benchmarks.yml
- [x] Update upstream-dev-ci.yml
- [x] Update publish-production-pypi.yml
- [x] Update README.rst
- [x] Update docs/source/index.rst
- [x] Update AGENTS.md
- [x] Verify local installation works
- [ ] (Optional) Remove unused conda requirement files from CI workflows

## Benefits

1. **Faster CI**: Reduced dependency installation time from minutes to seconds
2. **Reproducible**: Lock file ensures consistent package versions
3. **Security**: Package hash verification prevents supply chain attacks
4. **Simpler**: Single tool replaces pip + conda combination
5. **Modern**: Uses latest Python packaging standards

## Caveats

- uv requires Python 3.8+ (climpred requires Python 3.9+)
- Some legacy packages may not be available via uv's index
- CI caching configuration differs from conda's approach

## References

- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Authority](https://packaging.python.org/)
