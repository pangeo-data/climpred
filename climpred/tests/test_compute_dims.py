import pytest
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_hindcast
from climpred.comparisons import (
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.exceptions import DimensionError
from climpred.metrics import PM_METRICS, PROBABILISTIC_METRICS
from climpred.utils import get_comparison_class, get_metric_class

# TODO: move to conftest.py
ITERATIONS = 2

comparison_dim_PM = [
    ('m2m', 'init'),
    ('m2m', 'member'),
    ('m2m', ['init', 'member']),
    ('m2e', 'init'),
    ('m2e', 'member'),
    ('m2e', ['init', 'member']),
    ('m2c', 'init'),
    ('m2c', 'member'),
    ('m2c', ['init', 'member']),
    ('e2c', 'init'),
]


@pytest.mark.parametrize('metric', ['crps', 'mse'])
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_pm_comparison_stack_dims_when_deterministic(
    PM_da_initialized_1d, comparison, metric
):
    metric = get_metric_class(metric, PM_METRICS)
    comparison = get_comparison_class(comparison, PM_COMPARISONS)
    actual_f, actual_r = comparison.function(PM_da_initialized_1d, metric=metric)
    if not metric.probabilistic:
        assert 'member' in actual_f.dims
        assert 'member' in actual_r.dims
    else:
        assert 'member' in actual_f.dims
        assert 'member' not in actual_r.dims


# cannot work for e2c, m2e comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_compute_perfect_model_dim_over_member(
    perfectModelEnsemble_initialized_control, comparison
):
    """Test deterministic metric calc skill over member dim."""
    actual = perfectModelEnsemble_initialized_control.verify(
        comparison=comparison, metric='rmse', dim='member',
    )['tos']
    assert 'init' in actual.dims
    assert not actual.isnull().any()
    # check that init is cftime object
    assert 'cftime' in str(type(actual.init.values[0]))


# cannot work for e2o comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
def test_compute_hindcast_dim_over_member(hindcast_hist_obs_1d, comparison):
    """Test deterministic metric calc skill over member dim."""
    actual = hindcast_hist_obs_1d.verify(
        comparison=comparison, metric='rmse', dim='member',
    )['SST']
    print(actual.dims)
    assert 'init' in actual.dims
    # mean init because skill has still coords for init lead
    assert not actual.mean('init').isnull().any()


def test_compute_perfect_model_different_dims_quite_close(
    perfectModelEnsemble_initialized_control,
):
    """Tests nearly equal dim=['init','member'] and dim='member'."""
    stack_dims_true = perfectModelEnsemble_initialized_control.verify(
        comparison='m2c', metric='rmse', dim=['init', 'member'],
    )['tos']
    stack_dims_false = perfectModelEnsemble_initialized_control.verify(
        comparison='m2c', metric='rmse', dim='member',
    ).mean(['init'])['tos']
    # no more than 10% difference
    assert_allclose(stack_dims_true, stack_dims_false, rtol=0.1, atol=0.03)


def test_bootstrap_pm_dim(perfectModelEnsemble_initialized_control):
    """Test whether bootstrap_hindcast calcs skill over member dim and
    returns init dim."""
    actual = perfectModelEnsemble_initialized_control.bootstrap(
        metric='rmse',
        dim='member',
        comparison='m2c',
        iterations=ITERATIONS,
        resample_dim='member',
    )['tos']
    assert 'init' in actual.dims
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


def test_bootstrap_hindcast_dim(
    hind_da_initialized_1d, hist_da_uninitialized_1d, observations_da_1d
):
    """Test whether bootstrap_hindcast calcs skill over member dim and
    returns init dim."""
    actual = bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        metric='rmse',
        dim='member',
        comparison='m2o',
        iterations=ITERATIONS,
        resample_dim='member',
    )
    assert 'init' in actual.dims
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


@pytest.mark.parametrize('metric', ['rmse', 'crps'])
@pytest.mark.parametrize(
    'comparison,dim', comparison_dim_PM,
)
def test_compute_pm_dims(
    perfectModelEnsemble_initialized_control, dim, comparison, metric
):
    """Test whether compute_pm calcs skill over all possible dims
    and comparisons and just reduces the result by dim."""
    pm = perfectModelEnsemble_initialized_control
    actual = pm.verify(metric=metric, dim=dim, comparison=comparison)['tos']
    # change dim as automatically in compute functions for probabilistic
    if dim in ['init', ['init', 'member']] and metric in PROBABILISTIC_METRICS:
        dim = ['member']
    elif isinstance(dim, str):
        dim = [dim]
    # check whether only dim got reduced from coords
    assert set(pm.get_initialized().dims) - set(actual.dims) == set(dim)
    # check whether all nan
    assert not actual.isnull().any()


@pytest.mark.parametrize(
    'metric,dim', [('rmse', 'init'), ('rmse', 'member'), ('crps', 'member')]
)
def test_compute_hindcast_dims(hindcast_hist_obs_1d, dim, metric):
    """Test whether compute_hindcast calcs skill over all possible dims
    and comparisons and just reduces the result by dim."""
    actual = hindcast_hist_obs_1d.verify(metric=metric, dim=dim, comparison='m2o',)[
        'SST'
    ]
    # check whether only dim got reduced from coords
    assert set(hindcast_hist_obs_1d.get_initialized().dims) - set(actual.dims) == set(
        [dim]
    )
    # check whether all nan
    if 'init' in actual.dims:
        actual = actual.mean('init')
    assert not actual.isnull().any()


@pytest.mark.parametrize(
    'dim',
    ['init', 'member', None, ['init', 'member'], ['x', 'y'], ['x', 'y', 'member']],
)
def test_PM_multiple_dims(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic, dim
):
    """Test that PerfectModelEnsemble accepts dims as subset from initialized dims."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    assert pm.verify(metric='rmse', comparison='m2e', dim=dim).any()


def test_PM_multiple_dims_fail_if_not_in_initialized(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic,
):
    """Test that PerfectModelEnsemble.verify() for multiple dims fails when not subset
    from initialized dims."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic.isel(x=4, y=4)
    with pytest.raises(DimensionError) as excinfo:
        pm.verify(metric='rmse', comparison='m2e', dim=['init', 'member', 'x'])
    assert 'is expected to be a subset of `initialized.dims`' in str(excinfo.value)


def test_PM_fails_probabilistic_member_not_in_dim(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic,
):
    """Test that PerfectModelEnsemble.verify() raises ValueError for `member` not in
    dim if probabilistic metric."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    with pytest.raises(ValueError) as excinfo:
        pm.verify(metric='crps', comparison='m2c', dim=['init'])
    assert (
        'requires to be computed over dimension `member`, which is not found in'
        in str(excinfo.value)
    )


def test_pm_metric_weights(perfectModelEnsemble_initialized_control_3d_North_Atlantic):
    """Test PerfectModelEnsemble.verify() with weights yields different results."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    skipna = True
    metric = 'rmse'
    comparison = 'm2e'
    dim = ['x', 'y']
    weights = pm.get_initialized()['lat']
    s_no_weights = pm.verify(
        metric=metric, comparison=comparison, dim=dim, skipna=skipna
    )
    s_weights = pm.verify(
        metric=metric, comparison=comparison, dim=dim, skipna=skipna, weights=weights
    )
    # want to test for non equalness
    assert ((s_no_weights['tos'] - s_weights['tos']) != 0).all()


def test_hindcast_metric_weights(hindcast_recon_3d):
    """Test HindcastEnsemble.verify() with weights yields different results."""
    he = hindcast_recon_3d
    skipna = True
    metric = 'rmse'
    comparison = 'e2o'
    dim = ['nlat', 'nlon']
    alignment = 'same_verifs'
    weights = he.get_initialized()['TAREA']
    s_no_weights = he.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        skipna=skipna,
        alignment=alignment,
    )
    s_weights = he.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        skipna=skipna,
        weights=weights,
        alignment=alignment,
    )
    # want to test for non equalness
    assert ((s_no_weights['SST'] - s_weights['SST']) != 0).all()
