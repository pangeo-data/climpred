import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from climpred.comparisons import (
    _drop_members,
    _e2c,
    _m2c,
    _m2e,
    _m2m,
    _stack_to_supervector,
)
from climpred.tutorial import load_dataset


@pytest.fixture
def PM_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos']
    return da


def m2e(ds, supervector_dim='svd'):
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, 'init').rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list, 'init').rename({'init': supervector_dim})
    return forecast, reference


def test_e2c(PM_da_ds1d):
    """Test ensemble_mean-to-control (which can be any other one member) (e2c)
    comparison basic functionality.

    Clean comparison: Remove one control member from ensemble to use as reference.
    Take the remaining member mean as forecasts."""
    ds = PM_da_ds1d
    aforecast, areference = _e2c(ds)

    control_member = [0]
    supervector_dim = 'svd'
    reference = ds.isel(member=control_member).squeeze()
    if 'member' in reference.coords:
        del reference['member']
    reference = reference.rename({'init': supervector_dim})
    # drop the member being reference
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    forecast = ds.mean('member')
    forecast = forecast.rename({'init': supervector_dim})

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2c(PM_da_ds1d):
    """Test many-to-control (which can be any other one member) (m2c) comparison basic
    functionality.

    Clean comparison: Remove one control member from ensemble to use as reference.
    Take the remaining members as forecasts."""
    ds = PM_da_ds1d
    aforecast, areference = _m2c(ds)

    supervector_dim = 'svd'
    control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    forecast, reference = xr.broadcast(ds_dropped, reference)
    forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
    reference = _stack_to_supervector(reference, new_dim=supervector_dim)

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2e(PM_da_ds1d):
    """Test many-to-ensemble-mean (m2e) comparison basic functionality.

    Clean comparison: Remove one member from ensemble to use as reference.
    Take the remaining members as forecasts."""
    ds = PM_da_ds1d
    aforecast, areference = _m2e(ds)

    supervector_dim = 'svd'
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, 'init').rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list, 'init').rename({'init': supervector_dim})

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2m(PM_da_ds1d):
    """Test many-to-many (m2m) comparison basic functionality.

    Clean comparison: Remove one member from ensemble to use as reference. Take the
    remaining members as forecasts."""
    ds = PM_da_ds1d
    aforecast, areference = _m2m(ds)

    supervector_dim = 'svd'
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        # drop the member being reference
        ds_reduced = _drop_members(ds, rmd_member=[m])
        reference = ds.sel(member=m)
        for m2 in ds_reduced.member:
            for i in ds.init:
                reference_list.append(reference.sel(init=i))
                forecast_list.append(ds_reduced.sel(member=m2, init=i))
    reference = xr.concat(reference_list, supervector_dim)
    reference[supervector_dim] = np.arange(1, 1 + reference.svd.size)
    forecast = xr.concat(forecast_list, supervector_dim)
    forecast[supervector_dim] = np.arange(1, 1 + forecast.svd.size)
    eforecast, ereference = forecast, reference
    # very weak testing here
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size
