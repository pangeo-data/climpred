# Bias Removal in climpred

## Contents
- [Overview](#overview)
- [Method: remove_bias()](#method-remove_bias)
- [Bias Correction Methods (how)](#bias-correction-methods-how)
- [Train/Test Split Strategies](#traintest-split-strategies)
- [Seasonality Options](#seasonality-options)
- [Workflow Examples](#workflow-examples)
- [Common Pitfalls](#common-pitfalls)

## Overview

Climate models have systematic biases relative to observations. `HindcastEnsemble.remove_bias()` corrects these before verification. Bias is calculated per lead time and grouped by seasonality (default: `"month"`).

The method returns a new `HindcastEnsemble` with corrected forecasts — it does not modify in place.

```python
corrected = hindcast.remove_bias(alignment="same_verifs", how="additive_mean")
skill = corrected.verify(
    metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
)
```

## Method: remove_bias()

```python
HindcastEnsemble.remove_bias(
    alignment,  # required: "maximize", "same_inits", "same_verifs"
    how="additive_mean",  # bias correction method
    train_test_split="unfair",  # "unfair", "unfair-cv", "fair"
    train_init=None,  # slice or DataArray for fair split (with same_inits/maximize)
    train_time=None,  # slice or DataArray for fair split (with same_verifs)
    cv=False,  # "LOO" for leave-one-out (with unfair-cv)
    **metric_kwargs,  # passed to xclim.sdba or bias_correction
)
```

**Returns**: `HindcastEnsemble` with bias-corrected initialized forecasts.

## Bias Correction Methods (how)

### Built-in methods (no extra dependencies)

| Method | Description |
|--------|-------------|
| `"additive_mean"` | Subtract mean bias (forecast - obs). Default. |
| `"multiplicative_mean"` | Divide by multiplicative bias ratio. |
| `"multiplicative_std"` | Correct ensemble spread to match observed variability. |

### bias_correction package methods (requires `pip install bias-correction`)

| Method | Description |
|--------|-------------|
| `"modified_quantile"` | Modified quantile mapping |
| `"basic_quantile"` | Basic quantile mapping |
| `"gamma_mapping"` | Gamma distribution mapping |
| `"normal_mapping"` | Normal distribution mapping |

### xclim.sdba methods (requires `pip install xclim`)

| Method | Description |
|--------|-------------|
| `"EmpiricalQuantileMapping"` | Empirical quantile mapping |
| `"DetrendedQuantileMapping"` | Detrended quantile mapping |
| `"PrincipalComponents"` | Principal components adjustment |
| `"QuantileDeltaMapping"` | Quantile delta mapping |
| `"Scaling"` | Additive or multiplicative scaling |
| `"LOCI"` | Local intensity scaling |

**Note**: xclim methods require `pint`-compatible `units` attribute on the data variable:
```python
hindcast._datasets["initialized"]["sst"].attrs["units"] = "C"
hindcast._datasets["observations"]["sst"].attrs["units"] = "C"
```

When using xclim methods, pass `group` instead of relying on `set_options(seasonality=...)`:
```python
hindcast.remove_bias(
    alignment="same_inits", how="EmpiricalQuantileMapping", group="init"
)
```

## Train/Test Split Strategies

Based on Risbey et al. (2021, doi:10/gk8k7k):

### unfair (default)

Train and test periods completely overlap. Fast, easy, but optimistically biased — the model "sees" verification data during bias estimation.

```python
hindcast.remove_bias(
    alignment="same_verifs", how="additive_mean", train_test_split="unfair"
)
```

### unfair-cv

Train and test overlap except the current initialization is left out (leave-one-out cross-validation). Slightly more honest than unfair. Slower.

```python
hindcast.remove_bias(
    alignment="same_verifs",
    how="additive_mean",
    train_test_split="unfair-cv",
    cv="LOO",
)
```

### fair (recommended for publication)

No overlap between train and test. You must specify which period to use for training. Remaining initializations are the test set.

With `alignment="same_inits"` or `"maximize"`, provide `train_init`:
```python
hindcast.remove_bias(
    alignment="same_inits",
    how="additive_mean",
    train_test_split="fair",
    train_init=slice("1960", "1990"),  # train on 1960-1990, test on rest
)
```

With `alignment="same_verifs"`, provide `train_time`:
```python
hindcast.remove_bias(
    alignment="same_verifs",
    how="additive_mean",
    train_test_split="fair",
    train_time=slice("1982", "1998"),  # train on obs time 1982-1998
)
```

### Comparison

| Strategy | Speed | Fair? | Use case |
|----------|-------|-------|----------|
| `"unfair"` | Fast | No | Quick exploration |
| `"unfair-cv"` | Slow | Partially | Better than unfair, worse than fair |
| `"fair"` | Fast | Yes | Publication, operational comparison |

## Seasonality Options

Bias is grouped by seasonality before removal. Control via `climpred.set_options`:

```python
import climpred

# Default: monthly seasonality
climpred.set_options(seasonality="month")

# For daily data — finer grouping
climpred.set_options(seasonality="dayofyear")

# For seasonal data
climpred.set_options(seasonality="season")

# Available: "dayofyear", "weekofyear", "month", "season"
```

The seasonality option also affects `reference="climatology"` in `verify()`.

## Workflow Examples

### 1. Basic additive mean bias removal

```python
from climpred import HindcastEnsemble
from climpred.tutorial import load_dataset

init = load_dataset("CESM-DP-SST")
obs = load_dataset("ERSST")
hindcast = HindcastEnsemble(init).add_observations(obs)

# Remove bias and verify
corrected = hindcast.remove_bias(alignment="maximize", how="additive_mean")
skill = corrected.verify(
    metric="rmse",
    comparison="e2o",
    dim="init",
    alignment="maximize",
)
```

### 2. Compare raw vs bias-corrected skill

```python
metric_kwargs = dict(
    metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
)

raw_skill = hindcast.verify(**metric_kwargs)
corrected_skill = hindcast.remove_bias(
    alignment="same_verifs",
    how="additive_mean",
    train_test_split="unfair",
).verify(**metric_kwargs)

raw_skill["SST"].plot(label="raw")
corrected_skill["SST"].plot(label="bias corrected")
```

### 3. Fair train/test split for publication

```python
corrected = hindcast.remove_bias(
    alignment="same_inits",
    how="additive_mean",
    train_test_split="fair",
    train_init=slice("1954", "1980"),  # first half for training
)

# Only test-period initializations remain
skill = corrected.verify(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    reference=["persistence", "climatology"],
)
```

### 4. Quantile mapping with xclim

```python
# Requires: pip install xclim
# Variables need units attribute for xclim
hindcast._datasets["initialized"]["SST"].attrs["units"] = "K"
hindcast._datasets["observations"]["SST"].attrs["units"] = "K"

corrected = hindcast.remove_bias(
    alignment="same_inits",
    how="EmpiricalQuantileMapping",
    train_test_split="unfair",
    group="init",  # passed to xclim.sdba
)
```

### 5. Multiplicative std correction (ensemble calibration)

```python
corrected = hindcast.remove_bias(
    alignment="maximize",
    how="multiplicative_std",
    train_test_split="unfair",
)
# Ensemble spread now matches observed variability
```

### 6. Seasonal bias removal with NMME data

```python
init = load_dataset("NMME_hindcast_Nino34_sst")
obs = load_dataset("NMME_OIv2_Nino34_sst")

hindcast = HindcastEnsemble(init.sel(model="GFDL-CM2p5-FLOR-A06")).add_observations(obs)

# Monthly seasonality (default) is appropriate for seasonal forecasts
corrected = hindcast.remove_bias(
    alignment="same_verifs",
    how="additive_mean",
    train_test_split="fair",
    train_time=slice("1982", "1998"),
)
```

### 7. Diagnose the bias before removing it

```python
# Visualize bias as function of lead and init
bias = hindcast.verify(
    metric="additive_bias",
    comparison="e2o",
    dim=[],
    alignment="same_verifs",
)
bias["SST"].plot()  # QuadMesh: x=init, y=lead
```

## Common Pitfalls

1. **`train_init` / `train_time` required for fair split**: With `train_test_split="fair"`, you must provide `train_init` (for `alignment="same_inits"` or `"maximize"`) or `train_time` (for `alignment="same_verifs"`). Otherwise climpred raises `ValueError`.

2. **xclim requires units**: xclim.sdba methods need a `units` attribute on data variables. Set it manually if missing: `ds["var"].attrs["units"] = "K"`.

3. **Fair split reduces sample size**: The fair split drops training initializations from the test set. Skill may appear worse with fewer initializations — this is the honest cost of avoiding data leakage.

4. **cv="LOO" is discouraged**: Leave-one-out cross-validation is slow and has known issues. Use `train_test_split="fair"` instead for a clean separation.

5. **Seasonality must match data frequency**: Don't use `seasonality="dayofyear"` with monthly data — it will produce incorrect groupings. Match the seasonality to your data resolution.

6. **Bias removal is per-lead**: Bias is computed and removed independently at each lead time, which is the correct approach since model drift typically varies with lead.

7. **Chain with verify()**: `remove_bias()` returns a new `HindcastEnsemble`, so chain it: `hindcast.remove_bias(...).verify(...)`.
