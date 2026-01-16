# Fix climpred for xarray v2025.x

## Problem Statement

Tests are failing in the `updates-2025` branch due to breaking changes in xarray v2025.x.
The regression appears between `xarray@v2024.11.0` and `v2025.1.2`.

## Root Cause Analysis

### Breaking Changes in xarray v2025.x

1. **v2025.09.1**: `Dataset.update()` now returns `None` instead of the dataset
   - PR: https://github.com/pydata/xarray/pull/10658
   - Affects code that chains `.update()` calls

2. **v2025.11.0**: All operations now preserve attributes by default
   - PR: https://github.com/pydata/xarray/pull/10818
   - Changes doctest expected outputs significantly
   - Affects: mean(), sum(), std(), and many other operations

3. **v2025.09.0**: pandas 3.0 changes to `Day` frequency for resampling with CFTimeIndex
   - May emit new warnings for non-default origin/offset values

## Plan

### Phase 1: Fix Breaking API Changes

#### Task 1.1: Fix `Dataset.update()` usage

Find all usages where `.update()` return value is used and fix them:

```python
# Before (breaks in v2025.09.1+):
result = ds.update(other)

# After:
ds.update(other)
result = ds  # or use ds.merge(other)
```

Files to check:
- `src/climpred/classes.py`
- `src/climpred/reference.py`
- `src/climpred/utils.py`

#### Task 1.2: Handle attribute preservation changes

Options:
A. Set `xr.set_options(keep_attrs=False)` globally in doctests
B. Update doctest expected outputs to include attributes
C. Use `keep_attrs=False` on specific operations

Decision: Option A is simplest and maintains doctest compatibility.

#### Task 1.3: Handle pandas 3.0 resampling changes

- May need to suppress new warnings or adjust resample code for CFTimeIndex
- Add to `filterwarnings` in pyproject.toml if needed

### Phase 2: Update CI/Workflow

#### Task 2.1: Update doctest workflow

- Remove `xarray<2024.2.0` pin from `.github/workflows/climpred_testing.yml`
- Add `xr.set_options(keep_attrs=False)` handling

#### Task 2.2: Verify test configurations

- Ensure tests pass with latest xarray without version pins
- Check if `xskillscore@main` installation is still needed

### Phase 3: Local Testing

```bash
# Run full test suite
uv run pytest -n 4 --durations=20

# Run doctests
uv run pytest --doctest-modules src/climpred --ignore src/climpred/tests

# Run with specific xarray version
uv run pip install xarray==2025.1.2
```

### Phase 4: GitHub Actions Verification

1. Push branch and observe CI runs
2. Fix any remaining failures
3. Ensure all checks pass

### Phase 5: Documentation

- Update CHANGELOG with breaking changes and adaptations

## Success Criteria

- All tests pass locally with latest xarray (v2025.x)
- All GitHub Actions checks pass
- No version pins on xarray (use latest compatible)
- Doctests pass without pinning to old xarray versions

## References

- Original PR: https://github.com/pangeo-data/climpred/pull/880
- xarray Changelog: https://docs.xarray.dev/en/stable/whats-new.html
- xskillscore PRs: https://github.com/xarray-contrib/xskillscore/pulls
