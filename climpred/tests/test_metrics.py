import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from climpred.constants import PM_COMPARISONS
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from xarray.testing import assert_allclose


@pytest.fixture
def pm_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def ds_3d_NA():
    """ds North Atlantic"""
    ds = load_dataset(
        'MPI-PM-DP-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.fixture
def control_3d_NA():
    """control North Atlantic"""
    ds = load_dataset(
        'MPI-control-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.fixture
def initialized_da():
    da = load_dataset('CESM-DP-SST-3D')['SST']
    da = da - da.mean('init')
    return da


@pytest.fixture
def reconstruction_da():
    da = load_dataset('FOSI-SST-3D')['SST']
    da = da - da.mean('time')
    return da


def my_mse(forecast, reference, dim='svd', **metric_kwargs):
    return ((forecast - reference) ** 2).mean(dim)


@pytest.mark.skip(reason='not implemented yet')
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
@pytest.mark.parametrize('metric', [my_mse])
def test_new_metric_passed_to_compute(pm_da_ds1d, pm_da_control1d, metric, comparison):
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric=metric
    )

    expected = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric='mse'
    )

    assert_allclose(actual, expected)


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
def test_pm_metric_skipna(ds_3d_NA, control_3d_NA, metric):
    ds_3d_NA = ds_3d_NA.copy()
    # manipulating data
    # problem here: i dont get a nice example, somehow this doesnt mask
    ds_3d_NA.data[0, 0, 0, 80:100, 80:100] = np.nan

    base = compute_perfect_model(
        ds_3d_NA, control_3d_NA, metric=metric, skipna=False, dim='init', comparison='m2m',
    )  # .mean('member')
    skipping = compute_perfect_model(
        ds_3d_NA, control_3d_NA, metric=metric, skipna=True, dim='init', comparison='m2m'
    )  # .mean('member')
    print((base / skipping))  # .mean(['x', 'y']))
    assert ((base / skipping) != 1.).any()
    assert (xs.smape(base, skipping, ['x', 'y']) > 0.01).any()


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
@pytest.mark.parametrize('comparison', ('e2c', 'm2c'))
def test_pm_metric_weights(ds_3d_NA, control_3d_NA, comparison, metric):
    # distribute weights on initializations
    dim = 'init'
    base = compute_perfect_model(
        ds_3d_NA, control_3d_NA, dim=dim, metric=metric, comparison=comparison
    )
    weights = xr.DataArray(np.arange(1, 1 + ds_3d_NA[dim].size), dims=dim)
    weighted = compute_perfect_model(
        ds_3d_NA,
        control_3d_NA,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(['x', 'y']))
    # test for difference
    assert (xs.smape(base, weighted, ['x', 'y']) > 0.01).any()


@pytest.mark.skip(reason='comparisons dont work here')
@pytest.mark.parametrize('metric', ('rmse', 'mse'))
@pytest.mark.parametrize('comparison', ['m2e', 'm2m'])
def test_pm_metric_weights_m2x(ds_3d_NA, control_3d_NA, comparison, metric):
    # distribute weights on initializations
    dim = 'init'
    base = compute_perfect_model(
        ds_3d_NA, control_3d_NA, dim=dim, metric=metric, comparison=comparison
    )
    weights = xr.DataArray(np.arange(1, 1 + ds_3d_NA[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(1, 1 + ds_3d_NA[dim].size * ds_3d_NA['member'].size), dims='init'
    )

    weighted = compute_perfect_model(
        ds_3d_NA,
        control_3d_NA,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(['x', 'y']))
    # test for difference
    assert (xs.smape(base, weighted, ['x', 'y']) > 0.01).any()


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
def test_hindcast_metric_skipna(initialized_da, reconstruction_da, metric):
    initialized_da = initialized_da.copy()
    # manipulating data
    initialized_da.isel(init=0, lead=0, nlat=2, nlon=2).values = np.nan
    base = compute_hindcast(
        initialized_da, reconstruction_da, metric=metric, skipna=False, dim='init'
    )
    skipping = compute_hindcast(
        initialized_da, reconstruction_da, metric=metric, skipna=True, dim='init'
    )
    assert ((base / skipping) != 1).any()


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
@pytest.mark.parametrize('comparison', ['e2r'])
def test_hindcast_metric_weights(initialized_da, reconstruction_da, comparison, metric):
    # distribute weights on initializations
    dim = 'init'
    base = compute_hindcast(
        initialized_da, reconstruction_da, dim=dim, metric=metric, comparison=comparison
    )
    weights = xr.DataArray(
        np.arange(1, 1 + initialized_da[dim].size - initialized_da.lead.size),
        dims='time',
    )
    weighted = compute_hindcast(
        initialized_da,
        reconstruction_da,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    # test for difference
    assert ((base / weighted).mean(['nlon', 'nlat']) != 1).any()


@pytest.mark.skip(reason='comparisons dont work here')
@pytest.mark.parametrize('metric', ('rmse', 'mse'))
@pytest.mark.parametrize('comparison', ['e2r', 'm2r'])
def test_hindcast_metric_weights_x2r(
    initialized_da, reconstruction_da, comparison, metric
):
    # distribute weights on initializations
    dim = 'init'
    base = compute_hindcast(
        initialized_da, reconstruction_da, dim=dim, metric=metric, comparison=comparison
    )
    weights = xr.DataArray(
        np.arange(1, 1 + initialized_da[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(1, 1 + initialized_da[dim].size
                  * initialized_da['member'].size),
        dims='init',
    )

    weighted = compute_hindcast(
        initialized_da,
        reconstruction_da,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(['nlon', 'nlat']))
    # test for difference
    assert (xs.smape(base, weighted, ['nlat', 'nlon']) > 0.01).any()
