import pytest
from climpred.constants import PM_COMPARISONS, PROBABILISTIC_METRICS
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from climpred.utils import get_comparison_function
from xarray.testing import assert_allclose

# PM_COMPARISONS = {'m2c': '', 'e2c': ''}
PROBABILISTIC_METRICS.remove('brier_score')


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
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.mark.parametrize('comparison', ['m2c', 'm2m'])
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_compute_perfect_model_da1d_not_nan_probabilistic(
    pm_da_ds1d, pm_da_control1d, metric, comparison
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    if 'threshold' in metric:
        threshold = 10.5
    else:
        threshold = None

    actual = compute_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
    )
    actual = actual.isnull().any()
    assert not actual


@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
@pytest.mark.parametrize('comparison', ['m2r'])
def test_compute_hindcast_probabilistic(
    initialized_da, observations_da, metric, comparison
):
    """
    Checks that compute hindcast works without breaking.
    """
    if 'threshold' in metric:
        threshold = 10.5
    else:
        threshold = None
    res = compute_hindcast(
        initialized_da,
        observations_da,
        metric=metric,
        comparison=comparison,
        threshold=threshold,
    )
    # mean init because skill has still coords for init lead
    res = res.mean('init')
    res = res.isnull().any()
    assert not res


@pytest.mark.skip(reason='takes quite long')
def test_compute_perfect_model_da1d_not_nan_crpss_quadratic(
    pm_da_ds1d, pm_da_control1d
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            pm_da_ds1d,
            pm_da_control1d,
            comparison='m2c',
            metric='crpss',
            gaussian=False,
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.skip(reason='takes quite long')
def test_compute_hindcast_da1d_not_nan_crpss_quadratic(initialized_da, observations_da):
    """
    Checks that there are no NaNs on hindcast metrics of 1D time series.
    """
    actual = (
        compute_hindcast(
            initialized_da,
            observations_da,
            comparison='m2r',
            metric='crpss',
            gaussian=False,
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.skip(reason='brier_score doesnt work in current framework')
def test_compute_perfect_model_da1d_brier_score_not_nan(pm_da_ds1d, pm_da_control1d):
    """Check Brier Score PM. """
    actual = (
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison='m2c', metric='brier_score'
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.skip(reason='brier_score doesnt work in current framework')
def test_compute_hindcast_da1d_brier_score_not_nan(initialized_da, observations_da):
    """Check Brier Score hindcast. """
    th = observations_da.mean()
    actual = (
        compute_hindcast(
            (initialized_da > th).mean('member'),
            (observations_da > th),
            comparison='m2r',
            metric='brier_score',
        )
        .isnull()
        .any()
    )
    assert not actual


# Test apply_metric_to_member_dim


@pytest.mark.parametrize('stack', [True, False])
@pytest.mark.parametrize('comparison', ['m2m', 'm2c'])
def test_pm_comparison_stack(pm_da_ds1d, comparison, stack):
    comparison = get_comparison_function(comparison, PM_COMPARISONS)
    actual_f, actual_r = comparison(pm_da_ds1d, stack=stack)
    print(actual_f)
    print(actual_f.dims)
    if stack:
        assert 'svd' in actual_f.dims
        assert 'member' not in actual_f.dims
    else:
        assert 'member' in actual_f.dims
        assert 'svd' not in actual_f.dims


# cannot work for e2c, m2e comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', ['m2c', 'm2m'])
def test_compute_perfect_model_dim_over_member(pm_da_ds1d, pm_da_control1d, comparison):
    """Test whether stack=False and deterministic metric calc skill over member dim."""
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric='rmse', stack=False
    )
    print(actual)
    assert 'init' in actual.dims
    assert not actual.isnull().any()


# cannot work for e2r comparison because only 1:1 comparison
@pytest.mark.parametrize('comparison', ['m2r'])
def test_compute_hindcast_dim_over_member(initialized_da, observations_da, comparison):
    """Test whether stack=False and deterministic metric calc skill over member dim."""
    actual = compute_hindcast(
        initialized_da,
        observations_da,
        comparison=comparison,
        metric='rmse',
        stack=False,
    )
    print(actual)
    assert 'init' in actual.dims
    # mean init because skill has still coords for init lead
    assert not actual.mean('init').isnull().any()


def test_compute_perfect_model_stack_True_and_False_quite_close(
    pm_da_ds1d, pm_da_control1d
):
    """Test whether stack=False and stack=True give similar results."""
    stack_true = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison='m2m', metric='rmse', stack=True
    )
    stack_false = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison='m2m', metric='rmse', stack=False
    ).mean(['init', 'member'])
    print(stack_true - stack_false)
    # no more than 10% difference
    assert_allclose(stack_true, stack_false, rtol=0.1, atol=0.03)
