import pytest
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
from climpred.constants import (
    PM_COMPARISONS,
    PM_METRICS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_METRICS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from climpred.utils import get_comparison_class, get_metric_class


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
def initialized_da():
    da = load_dataset('CESM-DP-SST')['SST']
    da = da - da.mean('init')
    return da


@pytest.fixture
def uninitialized_da():
    da = load_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = range(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.mark.parametrize('metric', ['crps', 'mse'])
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_pm_comparison_stack_dims_when_deterministic(pm_da_ds1d, comparison, metric):
    metric = get_metric_class(metric, PM_METRICS)
    comparison = get_comparison_class(comparison, PM_COMPARISONS)
    actual_f, actual_r = comparison.function(pm_da_ds1d, metric=metric)
    if not metric.probabilistic:
        assert 'member' in actual_f.dims
        assert 'member' in actual_r.dims
    else:
        assert 'member' in actual_f.dims
        assert 'member' not in actual_r.dims


# cannot work for e2c, m2e comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_compute_perfect_model_dim_over_member(pm_da_ds1d, pm_da_control1d, comparison):
    """Test deterministic metric calc skill over member dim."""
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric='rmse', dim='member'
    )
    assert 'init' in actual.dims
    assert not actual.isnull().any()


# cannot work for e2o comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
def test_compute_hindcast_dim_over_member(initialized_da, observations_da, comparison):
    """Test deterministic metric calc skill over member dim."""
    actual = compute_hindcast(
        initialized_da,
        observations_da,
        comparison=comparison,
        metric='rmse',
        dim='member',
    )
    assert 'init' in actual.dims
    # mean init because skill has still coords for init lead
    assert not actual.mean('init').isnull().any()


def test_compute_perfect_model_different_dims_quite_close(pm_da_ds1d, pm_da_control1d):
    """Test whether dim=['init','member'] and
    dim='member' results."""
    stack_dims_true = compute_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison='m2c',
        metric='rmse',
        dim=['init', 'member'],
    )
    stack_dims_false = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison='m2c', metric='rmse', dim='member'
    ).mean(['init'])
    # no more than 10% difference
    assert_allclose(stack_dims_true, stack_dims_false, rtol=0.1, atol=0.03)


def test_bootstrap_pm_dim(pm_da_ds1d, pm_da_control1d):
    """Test whether bootstrap_hindcast calcs skill over member dim and
    returns init dim."""
    actual = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        metric='rmse',
        dim='member',
        comparison='m2c',
        bootstrap=3,
    )
    assert 'init' in actual.dims
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


def test_bootstrap_hindcast_dim(initialized_da, uninitialized_da, observations_da):
    """Test whether bootstrap_hindcast calcs skill over member dim and
    returns init dim."""
    actual = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        observations_da,
        metric='rmse',
        dim='member',
        comparison='m2o',
        bootstrap=3,
    )
    assert 'init' in actual.dims
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


@pytest.mark.parametrize('metric', ('rmse', 'crpss'))
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('dim', ('init', 'member', ['init', 'member']))
def test_compute_pm_dims(pm_da_ds1d, pm_da_control1d, dim, comparison, metric):
    """Test whether compute_pm calcs skill over all possible dims
    and comparisons and just reduces the result by dim."""
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, metric=metric, dim=dim, comparison=comparison
    )
    # change dim as automatically in compute functions for probabilistic
    if dim in ['init', ['init', 'member']] and metric in PROBABILISTIC_METRICS:
        dim = ['member']
    elif isinstance(dim, str):
        dim = [dim]
    # check whether only dim got reduced from coords
    assert set(pm_da_ds1d.dims) - set(actual.dims) == set(dim)
    # check whether all nan
    print(actual.dims, actual.coords, actual)
    assert not actual.isnull().any()


@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
@pytest.mark.parametrize('metric', ('rmse', 'crpss', 'crpss_es'))
@pytest.mark.parametrize('dim', ('init', 'member'))
def test_compute_hindcast_dims(
    initialized_da, observations_da, dim, comparison, metric
):
    """Test whether compute_hindcast calcs skill over all possible dims
    and comparisons and just reduces the result by dim."""
    actual = compute_hindcast(
        initialized_da, observations_da, metric=metric, dim=dim, comparison=comparison
    )
    # change dim as automatically in compute functions for probabilistic
    if dim == 'init' and metric in PROBABILISTIC_METRICS:
        dim = 'member'
    # check whether only dim got reduced from coords
    assert set(initialized_da.dims) - set(actual.dims) == set([dim])
    # check whether all nan
    if 'init' in actual.dims:
        actual = actual.mean('init')
    assert not actual.isnull().any()
