"""Annual DPP blocks should not require integer-valued year labels."""

import numpy as np
import pytest
import xarray as xr

from climpred.stats import dpp


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("calendar", ["standard", "noleap", "360_day", "datetime64"])
@pytest.mark.parametrize("lazy", [False, True])
def test_annual_dpp_datetime_labels_match_integer_years(dataset, calendar, lazy):
    data = xr.DataArray(
        np.sin(np.arange(24)[None, :] / 2)
        + np.array([0, 2])[:, None]
        + np.arange(24)[None, :] / 10,
        dims=("site", "time"),
        coords={"site": ["west", "east"], "time": np.arange(2000, 2024)},
        name="temperature",
    )
    if dataset:
        data = data.to_dataset()
        data["salinity"] = data.temperature * 3
    if lazy:
        data = data.chunk({"time": -1, "site": 1})
    expected = dpp(data, m=3).compute()
    time = xr.date_range(
        "2000",
        periods=24,
        freq="YS",
        calendar=calendar if calendar != "datetime64" else "standard",
        use_cftime=calendar != "datetime64",
    )
    actual = dpp(data.assign_coords(time=time), m=3).compute()
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("offset", [0, 1900])
def test_integer_annual_block_formula_is_unchanged(offset):
    values = np.arange(24, dtype=float) + np.sin(np.arange(24))
    data = xr.DataArray(values, dims="time", coords={"time": np.arange(24) + offset})
    means = values.reshape(-1, 3).mean(axis=1)
    s2v = means.var()
    s2e = (values.reshape(-1, 3) - means[:, None]).var()
    expected = (s2v - (s2v + s2e) / 3) / (s2v + s2e)
    np.testing.assert_allclose(dpp(data, m=3), expected)
