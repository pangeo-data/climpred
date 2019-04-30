import numpy as np
import pytest
import xarray as xr

from climpred.comparisons import _drop_members, _m2m
from climpred.loadutils import open_dataset


@pytest.fixture
def PM_da_ds1d():
    da = open_dataset('MPI-PM-DP-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_da_control1d():
    da = open_dataset('MPI-control-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_ds_ds1d():
    ds = open_dataset('MPI-PM-DP-1D')
    return ds


@pytest.fixture
def PM_ds_control1d():
    ds = open_dataset('MPI-control-1D')
    return ds


def test_m2m(PM_da_ds1d):
    "Test m2m basic functionality of many to many comparison"
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
                forecast_list.append(
                    ds_reduced.sel(member=m2, init=i))
    reference = xr.concat(
        reference_list, supervector_dim)
    reference[supervector_dim] = np.arange(1, 1+reference.svd.size)
    forecast = xr.concat(
        forecast_list, supervector_dim)
    forecast[supervector_dim] = np.arange(1, 1+forecast.svd.size)
    eforecast, ereference = forecast, reference

    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size
