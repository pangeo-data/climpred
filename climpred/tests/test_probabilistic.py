import pytest
from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
from climpred.constants import (
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_METRICS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset

# PROBABILISTIC_METRICS.remove('less')


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


@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_compute_perfect_model_da1d_not_nan_probabilistic(
    pm_da_ds1d, pm_da_control1d, metric, comparison
):
    """
    Checks that there are no NaNs on perfect model probabilistic metrics of 1D
    time series.
    """
    if 'threshold' in metric:
        threshold = 10.5
    else:
        threshold = None

    if metric == 'brier_score':

        def func(x):
            return x > 0

    else:
        func = None

    actual = compute_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
        func=func,
    )
    actual = actual.isnull().any()
    assert not actual


@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
def test_compute_hindcast_probabilistic(
    initialized_da, observations_da, metric, comparison
):
    """
    Checks that compute hindcast works without breaking.
    """
    if 'threshold' in metric:
        threshold = 0.5  # initialized_da.mean()
    else:
        threshold = None
    if metric == 'brier_score':

        def func(x):
            return x > 0.5

    else:
        func = None
    res = compute_hindcast(
        initialized_da,
        observations_da,
        metric=metric,
        comparison=comparison,
        threshold=threshold,
        func=func,
    )
    # mean init because skill has still coords for init lead
    print(res)
    if 'init' in res.coords:
        res = res.mean('init')
    print(res)
    res = res.isnull().any()
    assert not res


@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_bootstrap_perfect_model_da1d_not_nan_probabilistic(
    pm_da_ds1d, pm_da_control1d, metric, comparison
):
    """
    Checks that there are no NaNs on perfect model probabilistic metrics of 1D
    time series.
    """
    if 'threshold' in metric:
        threshold = 10.5
    else:
        threshold = None

    if metric == 'brier_score':

        def func(x):
            return x > 0

    else:
        func = None

    actual = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
        func=func,
        bootstrap=3,
    )
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


# @pytest.mark.skip('reason=s')
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_bootstrap_hindcast_da1d_not_nan_probabilistic(
    initialized_da, uninitialized_da, observations_da, metric, comparison
):
    """
    Checks that there are no NaNs on hindcast probabilistic metrics of 1D
    time series.
    """
    if 'threshold' in metric:
        threshold = 10.5
    else:
        threshold = None

    if metric == 'brier_score':

        def func(x):
            return x > 0

    else:
        func = None

    actual = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        observations_da,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
        func=func,
        bootstrap=3,
    )
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


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


def test_hindcast_crpss_orientation(initialized_da, observations_da):
    """
    Checks that CRPSS hindcast as skill score > 0.
    """
    actual = compute_hindcast(
        initialized_da, observations_da, comparison='m2r', metric='crpss'
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()


def test_pm_crpss_orientation(pm_da_ds1d, pm_da_control1d):
    """
    Checks that CRPSS in PM as skill score > 0.
    """
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison='m2m', metric='crpss'
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()
