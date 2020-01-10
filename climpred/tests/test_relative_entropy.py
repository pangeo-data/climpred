import numpy as np
import pytest
import xarray as xr

from climpred.graphics import plot_relative_entropy
from climpred.relative_entropy import (
    bootstrap_relative_entropy,
    compute_relative_entropy,
)


@pytest.fixture
def PM_da_ds3d():
    lead = np.arange(1, 4)
    lats = np.arange(4)
    lons = np.arange(3)
    member = np.arange(5)
    init = [3004, 3009, 3015, 3023]
    data = np.random.rand(len(lead), len(member), len(init), len(lats), len(lons))
    da = xr.DataArray(
        data,
        coords=[lead, member, init, lats, lons],
        dims=['lead', 'member', 'init', 'lat', 'lon'],
    )
    return da


@pytest.fixture
def PM_da_control3d():
    dates = np.arange(3000, 3050)
    lats = np.arange(4)
    lons = np.arange(3)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data, coords=[dates, lats, lons], dims=['time', 'lat', 'lon'])


def test_compute_relative_entropy(PM_da_ds3d, PM_da_control3d):
    """
    Checks that there are no NaNs.
    """
    actual = compute_relative_entropy(
        PM_da_ds3d, PM_da_control3d, nmember_control=5, neofs=2
    )
    actual_any_nan = actual.isnull().any()
    for var in actual_any_nan.data_vars:
        assert not actual_any_nan[var]


def test_bootstrap_relative_entropy(PM_da_ds3d, PM_da_control3d):
    """
    Checks that there are no NaNs.
    """
    actual = bootstrap_relative_entropy(
        PM_da_ds3d, PM_da_control3d, nmember_control=5, neofs=2, bootstrap=2
    )
    actual_any_nan = actual.isnull()
    for var in actual_any_nan.data_vars:
        assert not actual_any_nan[var]


def test_plot_relative_entropy(PM_da_ds3d, PM_da_control3d):
    res = compute_relative_entropy(
        PM_da_ds3d, PM_da_control3d, nmember_control=5, neofs=2
    )
    threshold = bootstrap_relative_entropy(
        PM_da_ds3d, PM_da_control3d, nmember_control=5, neofs=2, bootstrap=2
    )
    res_ax = plot_relative_entropy(res, threshold)
    assert res_ax is not None
