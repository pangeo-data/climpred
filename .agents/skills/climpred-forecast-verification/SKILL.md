---
name: climpred-forecast-verification
description: "Verify weather and climate forecasts using climpred. Use when computing forecast skill metrics (RMSE, ACC, CRPS, etc.), comparing hindcasts to observations, bootstrapping significance, removing bias, or working with HindcastEnsemble/PerfectModelEnsemble objects. Triggers on: forecast verification, prediction skill, hindcast, climate prediction, skill score, predictability."
license: MIT
---

# climpred — Forecast Verification

climpred verifies weather and climate forecasts. Built on xarray, dask, and xskillscore.

**Docs**: https://climpred.readthedocs.io | **API for LLMs**: https://context7.com/pangeo-data/climpred/llms.txt

## Install

```bash
pip install climpred[complete]
# or
conda install -c conda-forge climpred
```

## Core Concepts

- **`init`**: initialization time dimension (forecast reference time)
- **`lead`**: forecast lead time dimension (with units attribute: `"years"`, `"months"`, `"days"`, etc.)
- **`member`**: ensemble member dimension
- **`time`**: observation/verification time dimension

## Two Ensemble Classes

### HindcastEnsemble (real-world verification)

```python
import climpred
from climpred import HindcastEnsemble
from climpred.tutorial import load_dataset

# initialized must have dims (init, lead, [member])
init = load_dataset("CESM-DP-SST")  # dims: (init, lead, member)
obs = load_dataset("ERSST")  # dims: (time,)
uninit = load_dataset("CESM-LE")  # dims: (time, member)

hindcast = HindcastEnsemble(init).add_observations(obs).add_uninitialized(uninit)

# Verify
skill = hindcast.verify(
    metric="rmse",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    reference=["persistence", "climatology", "uninitialized"],
)
```

### PerfectModelEnsemble (model-internal verification)

```python
from climpred import PerfectModelEnsemble

init = load_dataset("MPI-PM-DP-1D")  # dims: (init, lead, member)
control = load_dataset("MPI-control-1D")  # dims: (time,)

pm = PerfectModelEnsemble(init).add_control(control)
skill = pm.verify(metric="acc", comparison="m2e", dim=["init", "member"])
```

## Key Methods

| Method | Description |
|--------|-------------|
| `.verify()` | Compute verification skill |
| `.bootstrap()` | Bootstrap significance testing |
| `.remove_bias()` | Bias-correct forecasts (HindcastEnsemble only) |
| `.smooth()` | Temporal/spatial smoothing |
| `.add_observations()` | Add verification data (HindcastEnsemble) |
| `.add_uninitialized()` | Add uninitialized ensemble |
| `.add_control()` | Add control run (PerfectModelEnsemble) |
| `.plot_alignment()` | Visualize init-verification alignment |

## verify() Parameters

All keyword-only:

- **`metric`** (required): See reference/metrics.md
- **`comparison`** (required): See reference/comparisons.md
- **`dim`**: Dimensions to reduce. `None` = all except `lead`. Use `[]` for per-init skill.
- **`alignment`** (HindcastEnsemble only): `"maximize"`, `"same_inits"`, or `"same_verifs"`
- **`reference`**: `["persistence", "climatology", "uninitialized"]` or subset
- **`groupby`**: Group init before verification (e.g., `"month"`, `"season"`)
- **`**metric_kwargs`**: Passed to metric (e.g., `category_edges` for `rps`)

## bootstrap() Parameters

Same as `verify()` plus:

- **`iterations`**: Number of bootstrap iterations (recommend ≥500)
- **`sig`**: Significance level in percent (default: 95)
- **`resample_dim`**: `"member"` (default) or `"init"`

Returns Dataset with `results` dim: `["verify skill", "p", "low_ci", "high_ci"]`

## Comparisons

**HindcastEnsemble**: `"e2o"` (ensemble mean vs obs), `"m2o"` (members vs obs)

**PerfectModelEnsemble**: `"m2m"`, `"m2e"`, `"m2c"`, `"e2c"`

Use `"e2o"` for deterministic metrics, `"m2o"` for probabilistic metrics (CRPS, RPS).

## Common Metrics Quick Reference

| Category | Metrics |
|----------|---------|
| Correlation | `pearson_r` (alias: `acc`), `spearman_r` |
| Distance | `mse`, `rmse`, `mae`, `me` (bias), `median_absolute_error` |
| Normalized | `nmse`, `nrmse`, `nmae`, `msess` (MSSS) |
| Percentage | `mape`, `smape` |
| Probabilistic | `crps`, `crpss`, `crpss_es`, `brier_score`, `rps`, `rank_histogram`, `reliability`, `discrimination`, `roc` |
| Bias decomposition | `unconditional_bias`, `conditional_bias`, `mul_bias`, `bias_slope`, `msess_murphy` |
| Other | `uacc`, `std_ratio`, `less` |

**Full metric details**: See reference/metrics.md

## Alignment (HindcastEnsemble only)

- `"maximize"`: Most degrees of freedom; different inits per lead
- `"same_inits"`: Same initializations across all leads
- `"same_verifs"`: Same verification dates across all leads

## Bias Removal

```python
hindcast_corrected = hindcast.remove_bias(
    alignment="maximize",
    how="additive_mean",  # or "modified_quantile", "DetrendedQuantileMapping", etc.
    train_test_split="unfair",  # or "fair" (leave-one-out cross-validation)
    cv="LOO",  # cross-validation strategy
)
```

## Common Workflows

### 1. Deterministic skill with references

```python
skill = hindcast.verify(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    reference=["persistence", "climatology"],
)
skill["SST"].plot(hue="skill")
```

### 2. Probabilistic skill (CRPS)

```python
skill = hindcast.verify(
    metric="crps",
    comparison="m2o",
    dim=["init", "member"],
    alignment="same_verifs",
)
```

### 3. Bootstrap significance

```python
bs = hindcast.bootstrap(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    iterations=500,
    reference=["persistence", "climatology"],
)
# p < 0.05 means initialized beats reference
```

### 4. Per-init skill (no dimension reduction)

```python
skill = hindcast.verify(
    metric="rmse",
    comparison="e2o",
    dim=[],
    alignment="same_verifs",
)
```

### 5. Seasonal groupby

```python
skill = hindcast.verify(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    groupby="month",
)
```

## Data Requirements

**Initialized forecast** (`init`):
- Must have `init` (initialization time) and `lead` dimensions
- `lead` must have a `units` attribute: `"years"`, `"months"`, `"days"`, etc.
- Optional: `member` dimension for ensemble

**Observations** (`obs`):
- Must have `time` dimension
- Variables must match initialized dataset

**Lead units attribute** — set before creating ensemble:
```python
ds["lead"].attrs["units"] = "years"
```

## Tutorial Datasets

```python
from climpred.tutorial import load_dataset

# call load_dataset() without args to list all available datasets

# Decadal (HindcastEnsemble)
init = load_dataset("CESM-DP-SST")
obs = load_dataset("ERSST")
uninit = load_dataset("CESM-LE")

# Perfect model
init_pm = load_dataset("MPI-PM-DP-1D")
control = load_dataset("MPI-control-1D")

# Subseasonal
s2s_init = load_dataset("ECMWF_S2S_Germany")
s2s_obs = load_dataset("Observations_Germany")
```

## Gotchas

1. **`dim` must match comparison**: Use `dim="init"` with `comparison="e2o"` (no member dim). Use `dim=["init", "member"]` with `comparison="m2o"`.
2. **Lead units**: Always set `ds["lead"].attrs["units"]` before creating the ensemble.
3. **Integer init coords**: Auto-converted to annual cftime. Use cftime or datetime init coords for sub-annual data.
4. **`alignment` is required** for `HindcastEnsemble.verify()` (not for PerfectModelEnsemble).
5. **`reference="uninitialized"`** requires `.add_uninitialized()` first.

## Advanced: reference files

- **Setting up data & toy examples**: See reference/setting_up_data.md
- **Bias removal guide**: See reference/bias_removal.md
- **Full metrics catalog**: See reference/metrics.md
- **Comparisons detail**: See reference/comparisons.md
