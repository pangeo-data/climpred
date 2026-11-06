"""Annual reference sampling must follow lead, not source dimension order."""

from itertools import permutations

import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble
from climpred.bootstrap import _bootstrap_by_stacking


@pytest.mark.parametrize("dims", list(permutations(["init", "member", "lead"])))
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_annual_control_windows_follow_lead(monkeypatch, dims, dataset, chunked):
    initialized = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("init", "member", "lead"),
        coords={
            "init": xr.date_range("2000", periods=2, freq="YS", use_cftime=True),
            "member": [1, 2],
            "lead": [0, 1, 2],
        },
        name="temperature",
    ).transpose(*dims)
    initialized.lead.attrs["units"] = "years"
    control = xr.DataArray(
        np.arange(8, dtype=float),
        dims="time",
        coords={"time": xr.date_range("1900", periods=8, freq="YS", use_cftime=True)},
        name="temperature",
    )
    if dataset:
        initialized = initialized.to_dataset()
        initialized["salinity"] = initialized.temperature + 100
        control = control.to_dataset()
        control["salinity"] = control.temperature + 100
    if chunked:
        initialized = initialized.chunk({"lead": 2, "init": 1})
        control = control.chunk({"time": 3})
    starts = np.array([0, 1, 2, 3])
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: starts.copy())
    actual = _bootstrap_by_stacking(initialized, control)
    expected_values = np.array([[[0, 1, 2], [1, 2, 3]], [[2, 3, 4], [3, 4, 5]]])
    expected = xr.DataArray(
        expected_values,
        dims=("init", "member", "lead"),
        coords={d: initialized[d] for d in ["init", "member", "lead"]},
        name="temperature",
    ).transpose(*dims)
    if dataset:
        expected = expected.to_dataset()
        expected["salinity"] = expected.temperature + 100
    xr.testing.assert_equal(actual.compute(), expected)
    assert actual.lead.attrs["units"] == "years"
    if chunked:
        assert (actual.temperature if dataset else actual).chunks is not None


def test_public_annual_control_layout(monkeypatch):
    initialized = xr.Dataset(
        {"a": (("lead", "member", "init"), np.zeros((3, 2, 2)))},
        coords={
            "lead": [0, 1, 2],
            "member": [1, 2],
            "init": xr.date_range("2000", periods=2, freq="YS", use_cftime=True),
        },
    )
    initialized.lead.attrs["units"] = "years"
    control = xr.Dataset(
        {"a": ("time", np.arange(8, dtype=float))},
        coords={"time": xr.date_range("1900", periods=8, freq="YS", use_cftime=True)},
    )
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.array([0, 1, 2, 3])
    )
    actual = (
        PerfectModelEnsemble(initialized)
        .add_control(control)
        .generate_uninitialized()
        .get_uninitialized()
    )
    np.testing.assert_array_equal(
        actual.a.transpose("init", "member", "lead"),
        [[[0, 1, 2], [1, 2, 3]], [[2, 3, 4], [3, 4, 5]]],
    )


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_annual_spatial_values_stay_with_their_coordinates(
    monkeypatch, dataset, chunked
):
    canonical = xr.DataArray(
        np.zeros((2, 2, 3, 2)),
        dims=("init", "member", "lead", "site"),
        coords={
            "init": [2000, 2001],
            "member": [1, 2],
            "lead": [0, 1, 2],
            "site": ["west", "east"],
        },
        name="temperature",
    )
    canonical.lead.attrs["units"] = "years"
    initialized = canonical.transpose("lead", "site", "member", "init")
    control = xr.DataArray(
        np.arange(8)[:, None] + np.array([0, 100])[None, :],
        dims=("time", "site"),
        coords={"time": range(8), "site": canonical.site},
        name="temperature",
    ).transpose("site", "time")
    expected = xr.DataArray(
        np.arange(4).reshape(2, 2, 1, 1)
        + np.arange(3).reshape(1, 1, 3, 1)
        + np.array([0, 100]).reshape(1, 1, 1, 2),
        dims=canonical.dims,
        coords=canonical.coords,
        name="temperature",
    ).transpose(*initialized.dims)
    if dataset:
        initialized = initialized.to_dataset()
        control = control.to_dataset()
        expected = expected.to_dataset()
        initialized["salinity"] = initialized.temperature + 10
        control["salinity"] = control.temperature + 10
        expected["salinity"] = expected.temperature + 10
    if chunked:
        initialized = initialized.chunk({"site": 1, "lead": 2})
        control = control.chunk({"time": 2, "site": 1})
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.array([0, 1, 2, 3])
    )
    actual = _bootstrap_by_stacking(initialized, control)
    xr.testing.assert_equal(actual.compute(), expected)
