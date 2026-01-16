# UPDATE.md - climpred Modernization: NumPy 2.x, Python 3.10+

## Context

climpred has been stalled due to dependency incompatibility with NumPy 2.x and xskillscore. The upstream fixes are now available:

- **xskillscore PR #440**: Complete NumPy 2.x compatibility (merged to main)
- **xclim**: Now supports NumPy 2.x (confirmed via pyproject.toml)

## Goal

Update climpred main branch to:
- Support NumPy 2.x (remove `<2.0.0` pin)
- Support Python 3.10-3.14 (drop 3.9)
- Use xskillscore from git main with NumPy 2.x fixes
- Pass full CI suite

## Changes Required

### 1. pyproject.toml

```toml
# Change: requires-python
- requires-python = ">=3.9"
+ requires-python = ">=3.10"

# Change: Classifiers (remove 3.9, add 3.13, 3.14)
- "Programming Language :: Python :: 3.9",
- "Programming Language :: Python :: 3.10",
- "Programming Language :: Python :: 3.11",
- "Programming Language :: Python :: 3.12",
+ "Programming Language :: Python :: 3.10",
+ "Programming Language :: Python :: 3.11",
+ "Programming Language :: Python :: 3.12",
+ "Programming Language :: Python :: 3.13",
+ "Programming Language :: Python :: 3.14",

# Change: bias-correction extra (remove numpy pin)
- bias-correction = ["xclim >=0.46", "bias-correction >=0.4", "numpy >=1.25.0,<2.0.0"]
+ bias-correction = ["xclim >=0.46", "bias-correction >=0.4"]

# Note: xskillscore pin stays as ">=0.0.20" since CI uses git main
```

### 2. ci/requirements/minimum-tests.yml

```yaml
# Change: Python version
- python >=3.9,<3.13
+ python >=3.10,<3.14
```

### 3. ci/requirements/maximum-tests.yml

```yaml
# Change: Python version
- python >=3.9,<3.13
+ python >=3.10,<3.14

# Change: Remove numpy pin (xclim now supports NumPy 2.x)
- numpy >=1.25.0,<2.0.0  # Pin below v2.0.0 until xclim supports it.
# (delete this line)
```

### 4. ci/requirements/climpred-dev.yml

```yaml
# Change: Python version
- python >=3.9
+ python
```

###  >=3.105. ci/requirements/docs.yml

```yaml
# Change: Python version
- python >=3.9,<3.13
+ python >=3.10,<3.14
```

### 6. .github/workflows/climpred_testing.yml

**minimum-test job:**
```yaml
matrix:
  python-version: ["3.10", "3.14"]  # was ["3.9", "3.12"]
```

**maximum-test job:**
```yaml
include:
  - env: "climpred-maximum-tests"
    python-version: "3.10"
  - env: "climpred-maximum-tests"
    python-version: "3.14"
  - env: "climpred-maximum-tests-upstream"
    python-version: "3.14"  # was 3.11
```

### 7. requirements_upstream.txt

```txt
# Keep xskillscore from git main (points to PR #440 fixes)
xskillscore @ git+https://github.com/xarray-contrib/xskillscore
```

## Pre-Release Steps

### 1. Release xskillscore v0.0.28
```bash
# In xskillscore repo
git checkout main
git pull
bump2version patch  # or appropriate version bump
git push
git push --tags
# PyPI release will be automatic via GitHub Actions
```

### 2. Create climpred PR

Title: `Update dependencies: NumPy 2.x, Python 3.10-3.14, xskillscore main`

Contents:
- All file changes from above
- Update CHANGELOG.rst with deprecation notice
- Run pre-commit on changed files

## Testing Checklist

- [ ] Minimum tests pass (Python 3.10, 3.14)
- [ ] Maximum tests pass (Python 3.10, 3.14)
- [ ] Doctests pass (NumPy 2.x compatible values)
- [ ] Notebooks build
- [ ] Coverage maintained (>90%)
- [ ] mypy type checks pass
- [ ] pre-commit hooks pass

## Post-Merge Steps

After xskillscore v0.0.28 is released and climpred PR is merged:

1. Update `pyproject.toml` xskillscore pin:
   ```toml
   - "xskillscore >=0.0.20"
   + "xskillscore >=0.0.28"
   ```

2. Update `requirements_upstream.txt`:
   ```txt
   - xskillscore @ git+https://github.com/xarray-contrib/xskillscore
   + xskillscore >=0.0.28
   ```

3. Create a follow-up PR with these version bumps

## Timeline

| Task | Effort |
|------|--------|
| Release xskillscore v0.0.28 | 10 min |
| Create/update climpred PR | 1-2 hours |
| CI validation | 1-2 days |
| Review + merge | 1 day |

**Total: ~1 week**

## Files Changed Summary

```
pyproject.toml
ci/requirements/minimum-tests.yml
ci/requirements/maximum-tests.yml
ci/requirements/climpred-dev.yml
ci/requirements/docs.yml
.github/workflows/climpred_testing.yml
requirements_upstream.txt
CHANGELOG.rst (update)
```

## References

- [xskillscore PR #440](https://github.com/xarray-contrib/xskillscore/pull/440)
- [xskillscore PR #437](https://github.com/xarray-contrib/xskillscore/pull/437)
- [climpred PR #870](https://github.com/pangeo-data/climpred/pull/870)
- [xclim pyproject.toml](https://github.com/Ouranosinc/xclim/blob/main/pyproject.toml)
