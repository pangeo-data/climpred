import pytest

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
from climpred.constants import (
    METRIC_ALIASES,
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_METRICS,
    PROBABILISTIC_PM_COMPARISONS,
)
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
        dim='member',
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
        dim='member',
    )
    # mean init because skill has still coords for init lead
    if 'init' in res.coords:
        res = res.mean('init')
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
        dim='member',
    )
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


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
        dim='member',
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
            dim='member',
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
            comparison='m2o',
            metric='crpss',
            gaussian=False,
            dim='member',
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
        initialized_da, observations_da, comparison='m2o', metric='crpss', dim='member'
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()


def test_pm_crpss_orientation(pm_da_ds1d, pm_da_control1d):
    """
    Checks that CRPSS in PM as skill score > 0.
    """
    actual = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison='m2m', metric='crpss', dim='member'
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()


# test api


NON_PROBABILISTIC_PM_COMPARISONS = list(
    set(PM_COMPARISONS) - set(PROBABILISTIC_PM_COMPARISONS)
)


# Probabilistic PM metrics dont work with non-prob. PM comparison m2e and e2c
@pytest.mark.parametrize('comparison', NON_PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_compute_pm_probabilistic_metric_non_probabilistic_comparison_fails(
    pm_da_ds1d, pm_da_control1d, metric, comparison
):
    with pytest.raises(ValueError) as excinfo:
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison=comparison, metric=metric
        )
    assert (
        f'Probabilistic metric {metric} cannot work with comparison {comparison}'
        in str(excinfo.value)
    )


@pytest.mark.parametrize('dim', ['init', ['init', 'member']])
@pytest.mark.parametrize('metric', ['crps'])
def test_compute_pm_probabilistic_metric_not_dim_member_warn(
    pm_da_ds1d, pm_da_control1d, metric, dim
):
    with pytest.warns(UserWarning) as record:
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison='m2c', metric=metric, dim=dim
        )
    expected = (
        f'Probabilistic metric {metric} requires to be '
        f'computed over dimension `dim="member"`. '
        f'Set automatically.'
    )
    assert record[0].message.args[0] == expected


@pytest.mark.parametrize('metric', ['crps'])
def test_compute_hindcast_probabilistic_metric_e2o_fails(
    initialized_da, observations_da, metric
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.raises(ValueError) as excinfo:
        compute_hindcast(
            initialized_da,
            observations_da,
            comparison='e2o',
            metric=metric,
            dim='member',
        )
    assert f'Probabilistic metric `{metric}` requires' in str(excinfo.value)


@pytest.mark.parametrize('dim', ['init'])
@pytest.mark.parametrize('metric', ['crps'])
def test_compute_hindcast_probabilistic_metric_not_dim_member_warn(
    initialized_da, observations_da, metric, dim
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.warns(UserWarning) as record:
        compute_hindcast(
            initialized_da, observations_da, comparison='m2o', metric=metric, dim=dim
        )
    expected = (
        f'Probabilistic metric {metric} requires to be '
        f'computed over dimension `dim="member"`. '
        f'Set automatically.'
    )
    # Set this to the third message since the first two are about converting the integer
    # time to annual `cftime`.
    assert record[2].message.args[0] == expected
