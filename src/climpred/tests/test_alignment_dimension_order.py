"""Alignment must not depend on valid_time coordinate dimension order."""

import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble
from climpred.alignment import _isin


@pytest.mark.parametrize("size", [99, 100])
@pytest.mark.parametrize("alignment", ["maximize", "same_inits", "same_verifs"])
def test_transposed_hindcast_verification(size, alignment):
    initialized = xr.Dataset(
        {"a": (("init", "lead", "member"), np.arange(size * 6).reshape(size, 2, 3))},
        coords={
            "init": xr.date_range("1900", periods=size, freq="YS", use_cftime=True),
            "lead": [1, 2],
            "member": [1, 2, 3],
        },
    )
    initialized.lead.attrs["units"] = "years"
    observations = xr.Dataset(
        {"a": ("time", np.arange(size + 3.0))},
        coords={"time": xr.date_range("1900", periods=size + 3, freq="YS", use_cftime=True)},
    ).isel(time=[i for i in range(size + 3) if i % 7])
    original = HindcastEnsemble(initialized)
    transposed = original.transpose("init", "lead", "member")
    kwargs = dict(metric="mse", comparison="e2o", dim="init", alignment=alignment)
    expected = original.add_observations(observations).verify(**kwargs)
    actual = transposed.add_observations(observations).verify(**kwargs)
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
def test_large_isin_preserves_axis_order(transpose, calendar):
    times = xr.date_range("1900", periods=101, freq="YS", use_cftime=True, calendar=calendar)
    values = np.array([times, times.shift(1, "YS")], dtype=object)
    matrix = xr.DataArray(values, dims=("lead", "time"), coords={"lead": [2, 1], "time": times})
    if transpose:
        matrix = matrix.transpose("time", "lead")
    verifs = xr.DataArray(times[::3], dims="time", coords={"time": times[::3]})
    xr.testing.assert_identical(_isin(matrix, verifs), matrix.isin(verifs))


def test_large_one_dimensional_isin_unchanged():
    times = xr.date_range("1900", periods=101, freq="YS", use_cftime=True)
    matrix = xr.DataArray(times, dims="time", coords={"time": times})
    xr.testing.assert_identical(_isin(matrix, times[::3]), matrix.isin(times[::3]))
