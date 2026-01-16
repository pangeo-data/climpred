# Fix climpred for xarray v2025.x

## Problem Statement

Tests are failing in the `updates-2025` branch due to breaking changes in xarray v2025.x.
The regression appears between `xarray@v2024.11.0` and `v2025.1.2`.

## Root Cause Analysis

### Breaking Changes in xarray v2025.x

1. **v2025.09.1**: `Dataset.update()` now returns `None` instead of the dataset
   - PR: https://github.com/pydata/xarray/pull/10658
   - Affects code that chains `.update()` calls
   - Status: Checked - climpred uses dict.update(), not Dataset.update()

2. **v2025.11.0**: All operations now preserve attributes by default
   - PR: https://github.com/pydata/xarray/pull/10818
   - Changes doctest expected outputs significantly
   - Affects: mean(), sum(), std(), and many other operations
   - Status: May need to handle in doctests

3. **v2025.09.0**: pandas 3.0 changes to `Day` frequency for resampling with CFTimeIndex
   - May emit new warnings for non-default origin/offset values
   - Status: May need to add filterwarnings

## Changes Made (2025-01-16)

### Completed

1. **pyproject.toml**:
   - Removed xarray version pin (`<2025.3.0`) from dependencies
   - Fixed pytest-lazy-fixtures import conflict (was using wrong package name)
   - Removed duplicate `pytest-lazy-fixture>=0.6.3` from dependency-groups

2. **CI Requirements**:
   - Removed xarray version pin from `ci/requirements/minimum-tests.yml`
   - Removed xarray version pin from `ci/requirements/maximum-tests.yml`

3. **Test Files**:
   - Fixed import conflict between `pytest-lazy-fixtures` and `pytest-lazy-fixture`

### Testing Notes

- Local testing limited by netCDF4/Python 3.14 segfault on macOS
- Basic xarray 2025.1.2 operations verified working
- Tests pushed to GitHub Actions for full verification

## Plan

### Phase 1: Fix Breaking API Changes (In Progress)

#### Task 1.1: Fix `Dataset.update()` usage ✓
- Verified climpred uses dict.update(), not Dataset.update()
- No changes needed

#### Task 1.2: Handle attribute preservation changes
- Decision needed: Set `xr.set_options(keep_attrs=False)` globally or update doctest outputs
- Waiting for CI results to see if doctests fail

#### Task 1.3: Handle pandas 3.0 resampling changes
- May need to suppress new warnings or adjust resample code for CFTimeIndex
- Add to `filterwarnings` in pyproject.toml if needed
- Waiting for CI results

### Phase 2: Update CI/Workflow

#### Task 2.1: Update doctest workflow
- Remove `xarray<2024.2.0` pin if present
- Add `xr.set_options(keep_attrs=False)` handling if needed
- Waiting for CI results

#### Task 2.2: Verify test configurations
- Ensure tests pass with latest xarray without version pins
- CI is running on branch `fix-xarray-v2025`

### Phase 3: Local Testing

Due to netCDF4/Python 3.14 segfault on macOS, local testing limited:
```bash
# Tests that work locally (don't require loading netCDF datasets)
uv run pytest src/climpred/tests/test_utils.py::test_get_metric_class -v
uv run pytest --doctest-modules src/climpred/utils.py -v
```

### Phase 4: GitHub Actions Verification

Branch `fix-xarray-v2025` pushed and CI running:
- https://github.com/pangeo-data/climpred/pull/new/fix-xarray-v2025

### Phase 5: Documentation

- Update CHANGELOG with breaking changes and adaptations (pending CI results)

## Success Criteria

- All GitHub Actions checks pass
- No version pins on xarray (use latest compatible)
- Doctests pass without pinning to old xarray versions

## References

- Original PR: https://github.com/pangeo-data/climpred/pull/880
- xarray Changelog: https://docs.xarray.dev/en/stable/whats-new.html
- xskillscore PRs: https://github.com/xarray-contrib/xskillscore/pulls
