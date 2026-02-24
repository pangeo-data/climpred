# Setting Up Data for climpred

## Contents
- [Dimension Requirements](#dimension-requirements)
- [Creating Toy HindcastEnsemble from Scratch](#creating-toy-hindcastensemble-from-scratch)
- [Creating Toy PerfectModelEnsemble from Scratch](#creating-toy-perfectmodelensemble-from-scratch)
- [CF Convention Auto-Renaming](#cf-convention-auto-renaming)
- [Converting Raw Model Output](#converting-raw-model-output)
- [Common Pitfalls](#common-pitfalls)

## Dimension Requirements

### Initialized forecast dataset

| Dimension | Type | Required | Notes |
|-----------|------|----------|-------|
| `init` | `pd.DatetimeIndex` or `xr.CFTimeIndex` | Yes | Initialization dates. `int` auto-converted to annual cftime. |
| `lead` | `int` or `float` | Yes | Lead time steps. **Must** have `units` attribute. |
| `member` | `int` or `str` | Only for probabilistic | Ensemble members |
| spatial dims | any | No | `lat`, `lon`, `depth`, etc. are broadcast |

**`lead` units attribute** — valid values:
`"years"`, `"seasons"`, `"months"`, `"weeks"`, `"pentads"`, `"days"`, `"hours"`, `"minutes"`, `"seconds"`

### Observations / verification dataset

| Dimension | Type | Required | Notes |
|-----------|------|----------|-------|
| `time` | `pd.DatetimeIndex` or `xr.CFTimeIndex` | Yes | Must cover the verification period |
| spatial dims | any | No | Must match initialized dims |

Variables must overlap with initialized dataset.

### Uninitialized ensemble (optional)

| Dimension | Type | Required | Notes |
|-----------|------|----------|-------|
| `time` | `pd.DatetimeIndex` or `xr.CFTimeIndex` | Yes | |
| `member` | `int` or `str` | No | |

## Creating Toy HindcastEnsemble from Scratch

### Minimal example (annual, deterministic)

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import HindcastEnsemble

# Dimensions
inits = pd.date_range("2000", periods=10, freq="YS")  # 10 initializations
leads = np.arange(1, 4)  # 3 lead years

# Create initialized forecast
init_data = np.random.randn(len(inits), len(leads))
initialized = xr.Dataset(
    {"temperature": (["init", "lead"], init_data)},
    coords={"init": inits, "lead": leads},
)
initialized["lead"].attrs["units"] = "years"

# Create observations
obs_time = pd.date_range("2000", periods=15, freq="YS")
obs = xr.Dataset(
    {"temperature": ("time", np.random.randn(len(obs_time)))},
    coords={"time": obs_time},
)

# Build ensemble and verify
hindcast = HindcastEnsemble(initialized).add_observations(obs)
skill = hindcast.verify(
    metric="rmse",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
)
```

### With ensemble members (probabilistic)

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import HindcastEnsemble

inits = pd.date_range("2000", periods=10, freq="YS")
leads = np.arange(1, 6)  # 5 lead years
members = np.arange(1, 11)  # 10 members

# Initialized forecast with member dimension
init_data = np.random.randn(len(inits), len(leads), len(members))
initialized = xr.Dataset(
    {"sst": (["init", "lead", "member"], init_data)},
    coords={"init": inits, "lead": leads, "member": members},
)
initialized["lead"].attrs["units"] = "years"

# Observations
obs_time = pd.date_range("2000", periods=15, freq="YS")
obs = xr.Dataset(
    {"sst": ("time", np.random.randn(len(obs_time)))},
    coords={"time": obs_time},
)

# Uninitialized ensemble (optional, for reference="uninitialized")
uninit = xr.Dataset(
    {"sst": (["time", "member"], np.random.randn(len(obs_time), len(members)))},
    coords={"time": obs_time, "member": members},
)

hindcast = HindcastEnsemble(initialized).add_observations(obs).add_uninitialized(uninit)

# Deterministic verification
skill = hindcast.verify(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
    reference=["persistence", "climatology", "uninitialized"],
)

# Probabilistic verification
crps = hindcast.verify(
    metric="crps",
    comparison="m2o",
    dim=["init", "member"],
    alignment="same_verifs",
)
```

### Monthly data

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import HindcastEnsemble

inits = pd.date_range("2000-01", periods=24, freq="MS")  # 24 monthly inits
leads = np.arange(1, 7)  # 6 lead months
members = np.arange(1, 6)

init_data = np.random.randn(len(inits), len(leads), len(members))
initialized = xr.Dataset(
    {"precip": (["init", "lead", "member"], init_data)},
    coords={"init": inits, "lead": leads, "member": members},
)
initialized["lead"].attrs["units"] = "months"  # monthly leads

obs_time = pd.date_range("2000-01", periods=36, freq="MS")
obs = xr.Dataset(
    {"precip": ("time", np.random.randn(len(obs_time)))},
    coords={"time": obs_time},
)

hindcast = HindcastEnsemble(initialized).add_observations(obs)
```

### Daily data (subseasonal / weather)

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import HindcastEnsemble

inits = pd.date_range("2020-01-01", periods=30, freq="7D")  # weekly inits
leads = np.arange(1, 15)  # 14 lead days
members = np.arange(1, 5)

init_data = np.random.randn(len(inits), len(leads), len(members))
initialized = xr.Dataset(
    {"t2m": (["init", "lead", "member"], init_data)},
    coords={"init": inits, "lead": leads, "member": members},
)
initialized["lead"].attrs["units"] = "days"

obs_time = pd.date_range("2020-01-01", periods=120, freq="D")
obs = xr.Dataset(
    {"t2m": ("time", np.random.randn(len(obs_time)))},
    coords={"time": obs_time},
)

hindcast = HindcastEnsemble(initialized).add_observations(obs)
```

### With spatial dimensions (lat/lon)

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import HindcastEnsemble

inits = pd.date_range("2000", periods=10, freq="YS")
leads = np.arange(1, 4)
members = np.arange(1, 4)
lats = np.arange(-90, 91, 30)
lons = np.arange(0, 360, 60)

init_data = np.random.randn(len(inits), len(leads), len(members), len(lats), len(lons))
initialized = xr.Dataset(
    {"sst": (["init", "lead", "member", "lat", "lon"], init_data)},
    coords={
        "init": inits,
        "lead": leads,
        "member": members,
        "lat": lats,
        "lon": lons,
    },
)
initialized["lead"].attrs["units"] = "years"

obs_time = pd.date_range("2000", periods=15, freq="YS")
obs_data = np.random.randn(len(obs_time), len(lats), len(lons))
obs = xr.Dataset(
    {"sst": (["time", "lat", "lon"], obs_data)},
    coords={"time": obs_time, "lat": lats, "lon": lons},
)

hindcast = HindcastEnsemble(initialized).add_observations(obs)

# Verify over init only — keeps lat/lon
skill = hindcast.verify(
    metric="acc",
    comparison="e2o",
    dim="init",
    alignment="same_inits",
)

# Verify over all dims — scalar result per lead
skill_all = hindcast.verify(
    metric="rmse",
    comparison="e2o",
    dim=None,
    alignment="same_inits",
)
```

## Creating Toy PerfectModelEnsemble from Scratch

```python
import numpy as np
import pandas as pd
import xarray as xr
from climpred import PerfectModelEnsemble

inits = pd.date_range("2000", periods=5, freq="YS")
leads = np.arange(1, 11)  # 10 lead years
members = np.arange(1, 4)

init_data = np.random.randn(len(inits), len(leads), len(members))
initialized = xr.Dataset(
    {"tos": (["init", "lead", "member"], init_data)},
    coords={"init": inits, "lead": leads, "member": members},
)
initialized["lead"].attrs["units"] = "years"

# Control run — long unforced simulation
control_time = pd.date_range("1850", periods=300, freq="YS")
control = xr.Dataset(
    {"tos": ("time", np.random.randn(len(control_time)))},
    coords={"time": control_time},
)

pm = PerfectModelEnsemble(initialized).add_control(control)
skill = pm.verify(metric="acc", comparison="m2e", dim=["init", "member"])
```

## CF Convention Auto-Renaming

climpred auto-renames dimensions if they have CF `standard_name` attributes:

| CF standard_name | Renamed to |
|-----------------|------------|
| `forecast_reference_time` | `init` |
| `forecast_period` | `lead` |
| `realization` | `member` |

```python
# This works — climpred renames "S" to "init", "L" to "lead", "M" to "member"
ds = xr.Dataset(
    {"temp": (["S", "L", "M"], np.random.randn(5, 3, 2))},
    coords={
        "S": pd.date_range("2000", periods=5, freq="YS"),
        "L": [1, 2, 3],
        "M": [0, 1],
    },
)
ds["S"].attrs["standard_name"] = "forecast_reference_time"
ds["L"].attrs["standard_name"] = "forecast_period"
ds["L"].attrs["units"] = "years"
ds["M"].attrs["standard_name"] = "realization"

hindcast = HindcastEnsemble(ds)  # auto-renames to init, lead, member
```

Also, `climpred.preprocessing.shared.rename_SLM_to_climpred_dims()` renames `S`→`init`, `L`→`lead`, `M`→`member` directly (for SubX/CESM conventions).

## Converting Raw Model Output

To convert multiple simulation files (each with a `time` dimension) into climpred format:

```python
import xarray as xr
import numpy as np

# Suppose you have files: init2000_member1.nc, init2000_member2.nc, etc.
init_list = []
for init_year in range(2000, 2010):
    member_list = []
    for member in range(1, 6):
        ds = xr.open_dataset(f"init{init_year}_member{member}.nc")
        # Convert time to integer lead steps
        ds = ds.rename({"time": "lead"})
        ds["lead"] = np.arange(1, len(ds.lead) + 1)
        member_list.append(ds)
    member_ds = xr.concat(member_list, dim="member")
    member_ds["member"] = range(1, 6)
    init_list.append(member_ds)

initialized = xr.concat(init_list, dim="init")
initialized["init"] = pd.date_range("2000", periods=10, freq="YS")
initialized["lead"].attrs["units"] = "years"
```

Or use the built-in helper:
```python
from climpred.preprocessing.shared import load_hindcast

ds = load_hindcast(inits=range(2000, 2010), members=range(1, 6))
ds["lead"].attrs["units"] = "years"
```

## Common Pitfalls

1. **Missing `lead` units**: Always set `ds["lead"].attrs["units"] = "years"` (or appropriate unit) before creating the ensemble. Without it, climpred raises an error.

2. **Integer `init` coordinates**: If `init` is integer (e.g., `[2000, 2001, 2002]`), climpred assumes annual data and converts to `cftime.DatetimeProlepticGregorian`. For sub-annual data, use `pd.DatetimeIndex` or `xr.CFTimeIndex`.

3. **`time` vs `init`**: Observations use `time`. Forecasts use `init` + `lead`. Do not use `time` in initialized datasets.

4. **Variable name mismatch**: Variables in initialized and observations must have the same names. Only shared variables are verified.

5. **Calendar mismatch**: initialized (`init`) and observations (`time`) should use the same calendar type. climpred will warn or error on mismatches.

6. **Lead starting at 0 vs 1**: `lead` can start at 0 or 1 — just be consistent. `lead=0` represents the initialization itself (analysis time).

7. **`valid_time`**: climpred computes `valid_time = init + lead` automatically. Do not set this manually.

8. **Observation time must cover verification period**: The `time` dimension in observations should span the entire period you want to verify against. climpred aligns `init + lead` to find matching observation times. You only need to provide observations with a `time` coordinate — do NOT include a `lead` dimension in observations.

9. **Unique time indices**: Observation time indices must be unique. Avoid duplicate timestamps in the `time` dimension.

10. **Avoid integer coordinates for annual data**: When creating toy data, prefer `pd.date_range` for both `init` and observation `time` rather than using integer coordinates. This avoids ambiguity and ensures proper datetime alignment.
