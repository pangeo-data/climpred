import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_perfect_model
from climpred.constants import PM_COMPARISONS
from climpred.metrics import __ALL_METRICS__ as all_metrics, Metric, __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset


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
    ds = load_dataset('MPI-PM-DP-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.fixture
def control_3d_NA():
    """control North Atlantic"""
    ds = load_dataset('MPI-control-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
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


def my_mse_function(forecast, reference, dim='svd', **metric_kwargs):
    return ((forecast - reference) ** 2).mean(dim)


my_mse = Metric(
    name='mse',
    function=my_mse_function,
    positive=True,
    probabilistic=False,
    unit_power=2,
    long_name='MSE',
    aliases=['mSe', '<<<SE'],
)


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_new_metric_passed_to_compute(pm_da_ds1d, pm_da_control1d, comparison):
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric=my_mse
    )

    expected = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric='mse'
    )

    assert_allclose(actual, expected)


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_new_metric_passed_to_bootstrap_compute(
    pm_da_ds1d, pm_da_control1d, comparison
):
    bootstrap = 3
    dim = 'init'
    np.random.seed(42)
    actual = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison=comparison,
        metric=my_mse,
        bootstrap=bootstrap,
        dim=dim,
    )

    expected = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison=comparison,
        metric='mse',
        bootstrap=bootstrap,
        dim=dim,
    )

    assert_allclose(actual, expected, rtol=0.1, atol=1)


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
def test_pm_metric_skipna(ds_3d_NA, control_3d_NA, metric):
    ds_3d_NA = ds_3d_NA.copy()
    # manipulating data
    ds_3d_NA.values[1:3, 1:4, 1:4, 4:6, 4:6] = np.nan

    base = compute_perfect_model(
        ds_3d_NA,
        control_3d_NA,
        metric=metric,
        skipna=False,
        dim='init',
        comparison='m2e',
    ).mean('member')
    skipping = compute_perfect_model(
        ds_3d_NA,
        control_3d_NA,
        metric=metric,
        skipna=True,
        dim='init',
        comparison='m2e',
    ).mean('member')
    assert ((base - skipping) != 0.0).any()
    assert base.isel(lead=2, x=5, y=5).isnull()
    assert not skipping.isel(lead=2, x=5, y=5).isnull()


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
@pytest.mark.parametrize('comparison', ['e2o'])
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
@pytest.mark.parametrize('comparison', ['e2o', 'm2o'])
def test_hindcast_metric_weights_x2r(
    initialized_da, reconstruction_da, comparison, metric
):
    # distribute weights on initializations
    dim = 'init'
    base = compute_hindcast(
        initialized_da, reconstruction_da, dim=dim, metric=metric, comparison=comparison
    )
    weights = xr.DataArray(np.arange(1, 1 + initialized_da[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(1, 1 + initialized_da[dim].size * initialized_da['member'].size),
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


def test_Metric_display():
    summary = __pearson_r.__repr__()
    assert 'Kind: deterministic' in summary.split('\n')[4]


def test_no_repeating_metric_aliases():
    """Tests that there are no repeating aliases for metrics, which would overwrite
    the earlier defined metric."""
    METRICS = []
    for m in all_metrics:
        if m.aliases is not None:
            for a in m.aliases:
                METRICS.append(a)
    duplicates = set([x for x in METRICS if METRICS.count(x) > 1])
    print(f'Duplicate metrics: {duplicates}')
    assert len(duplicates) == 0
