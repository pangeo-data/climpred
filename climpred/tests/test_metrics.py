import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import PM_COMPARISONS
from climpred.metrics import __ALL_METRICS__ as all_metrics, Metric, __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model


def my_mse_function(forecast, reference, dim=None, **metric_kwargs):
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
def test_new_metric_passed_to_compute(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    actual = compute_perfect_model(
        PM_da_initialized_1d, PM_da_control_1d, comparison=comparison, metric=my_mse,
    )

    expected = compute_perfect_model(
        PM_da_initialized_1d, PM_da_control_1d, comparison=comparison, metric='mse',
    )

    assert_allclose(actual, expected)


@pytest.mark.slow
def test_new_metric_passed_to_bootstrap_compute(PM_da_initialized_1d, PM_da_control_1d):
    comparison = 'e2c'
    BOOTSTRAP = 2
    dim = 'init'
    np.random.seed(42)
    actual = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=my_mse,
        bootstrap=BOOTSTRAP,
        dim=dim,
    )

    expected = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric='mse',
        bootstrap=BOOTSTRAP,
        dim=dim,
    )

    assert_allclose(actual, expected, rtol=0.1, atol=1)


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
def test_pm_metric_skipna(PM_da_initialized_3d, PM_da_control_3d, metric):
    PM_da_initialized_3d = PM_da_initialized_3d.copy()
    # manipulating data
    PM_da_initialized_3d.values[1:3, 1:4, 1:4, 4:6, 4:6] = np.nan

    base = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        metric=metric,
        skipna=False,
        dim='init',
        comparison='m2e',
    ).mean('member')
    skipping = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
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
def test_pm_metric_weights(PM_da_initialized_3d, PM_da_control_3d, comparison, metric):
    # distribute weights on initializations
    dim = 'init'
    base = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(np.arange(1, 1 + PM_da_initialized_3d[dim].size), dims=dim)
    weighted = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
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
def test_pm_metric_weights_m2x(
    PM_da_initialized_3d, PM_da_control_3d, comparison, metric
):
    # distribute weights on initializations
    dim = 'init'
    base = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(np.arange(1, 1 + PM_da_initialized_3d[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(
            1, 1 + PM_da_initialized_3d[dim].size * PM_da_initialized_3d['member'].size,
        ),
        dims='init',
    )

    weighted = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(['x', 'y']))
    # test for difference
    assert (xs.smape(base, weighted, ['x', 'y']) > 0.01).any()


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
def test_hindcast_metric_skipna(hind_da_initialized_3d, reconstruction_da_3d, metric):
    # manipulating data with nans
    hind_da_initialized_3d[0, 2, 0, 2] = np.nan
    base = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        metric=metric,
        skipna=False,
        dim='init',
    )
    skipping = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        metric=metric,
        skipna=True,
        dim='init',
    )

    div = base / skipping
    assert (div != 1).any()


@pytest.mark.parametrize('metric', ('rmse', 'mse'))
@pytest.mark.parametrize('comparison', ['e2o'])
def test_hindcast_metric_weights(
    hind_da_initialized_3d, reconstruction_da_3d, comparison, metric
):
    # distribute weights on initializations
    dim = 'init'
    base = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(
        np.arange(
            1, 1 + hind_da_initialized_3d[dim].size - hind_da_initialized_3d.lead.size,
        ),
        dims='time',
    )
    weighted = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
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
    hind_da_initialized_3d, reconstruction_da_3d, comparison, metric
):
    # distribute weights on initializations
    dim = 'init'
    base = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(np.arange(1, 1 + hind_da_initialized_3d[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(
            1,
            1
            + hind_da_initialized_3d[dim].size * hind_da_initialized_3d['member'].size,
        ),
        dims='init',
    )

    weighted = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
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
