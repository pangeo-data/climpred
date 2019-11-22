import numpy as np
import pytest
import xarray as xr
from climpred.comparisons import _drop_members, _e2c, _m2c, _m2e, _m2m
from climpred.constants import PM_COMPARISONS, PROBABILISTIC_PM_COMPARISONS
from climpred.tutorial import load_dataset
from climpred.utils import get_comparison_function
from xarray.testing import assert_equal


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
    # .rename({'init': supervector_dim})
    reference = xr.concat(reference_list, 'init')
    # .rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list, 'init')
    return forecast, reference


def test_e2c(PM_da_ds1d):
    """Test ensemble_mean-to-control (which can be any other one member) (e2c)
    comparison basic functionality.

    Clean comparison: Remove one control member from ensemble to use as reference.
    Take the remaining member mean as forecasts."""
    ds = PM_da_ds1d
    aforecast, areference = _e2c(ds)

    control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    if 'member' in reference.coords:
        del reference['member']
    # reference = reference.rename({'init': supervector_dim})
    # drop the member being reference
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    forecast = ds.mean('member')
    # forecast = forecast.rename({'init': supervector_dim})

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

    control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    forecast, reference = xr.broadcast(ds_dropped, reference)
    # forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
    # reference = _stack_to_supervector(reference, new_dim=supervector_dim)

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

    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    # .rename({'init': supervector_dim})
    reference = xr.concat(reference_list, 'member')
    # .rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list, 'member')
    forecast['member'] = np.arange(forecast.member.size)
    reference['member'] = np.arange(reference.member.size)

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

    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m])
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        reference_list.append(reference)
        forecast_list.append(forecast)
    supervector_dim = 'forecast_member'
    reference = xr.concat(reference_list, supervector_dim)
    forecast = xr.concat(forecast_list, supervector_dim)
    reference[supervector_dim] = np.arange(reference[supervector_dim].size)
    forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
    eforecast, ereference = forecast, reference
    # very weak testing here
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size


@pytest.mark.parametrize('stack_dims', [True, False])
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_all(PM_da_ds1d, comparison, stack_dims):
    ds = PM_da_ds1d
    comparison = get_comparison_function(comparison, PM_COMPARISONS)
    forecast, reference = comparison(ds, stack_dims=stack_dims)
    if stack_dims is True:
        # same dimensions for deterministic metrics
        assert forecast.dims == reference.dims
    else:
        if comparison.__name__ in PROBABILISTIC_PM_COMPARISONS:
            # same but member dim for probabilistic
            assert set(forecast.dims) - set(['member']) == set(reference.dims)
