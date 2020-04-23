import pytest
from scipy.stats import norm

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
from climpred.comparisons import (
    NON_PROBABILISTIC_PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.metrics import METRIC_ALIASES, PROBABILISTIC_METRICS
from climpred.prediction import compute_hindcast, compute_perfect_model

ITERATIONS = 2


@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_compute_perfect_model_da1d_not_nan_probabilistic(
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
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
        PM_da_initialized_1d,
        PM_da_control_1d,
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
    hind_da_initialized_1d, observations_da_1d, metric, comparison
):
    """
    Checks that compute hindcast works without breaking.
    """
    if 'threshold' in metric:
        threshold = 0.5  # hind_da_initialized_1d.mean()
    else:
        threshold = None
    if metric == 'brier_score':

        def func(x):
            return x > 0.5

    else:
        func = None
    res = compute_hindcast(
        hind_da_initialized_1d,
        observations_da_1d,
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
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
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
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
        func=func,
        iterations=ITERATIONS,
        dim='member',
        resample_dim='member',
    )
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


@pytest.mark.slow
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_bootstrap_hindcast_da1d_not_nan_probabilistic(
    hind_da_initialized_1d,
    hist_da_uninitialized_1d,
    observations_da_1d,
    metric,
    comparison,
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
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        comparison=comparison,
        metric=metric,
        threshold=threshold,
        gaussian=True,
        func=func,
        iterations=ITERATIONS,
        dim='member',
        resample_dim='member',
    )
    for kind in ['init', 'uninit']:
        actualk = actual.sel(kind=kind, results='skill')
        if 'init' in actualk.coords:
            actualk = actualk.mean('init')
        actualk = actualk.isnull().any()
        assert not actualk


def test_compute_perfect_model_da1d_not_nan_crpss_quadratic(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            PM_da_initialized_1d.isel(lead=[0]),
            PM_da_control_1d,
            comparison='m2c',
            metric='crpss',
            gaussian=False,
            dim='member',
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.slow
def test_compute_perfect_model_da1d_not_nan_crpss_quadratic_kwargs(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            PM_da_initialized_1d.isel(lead=[0]),
            PM_da_control_1d,
            comparison='m2c',
            metric='crpss',
            gaussian=False,
            dim='member',
            tol=1e-6,
            xmin=None,
            xmax=None,
            cdf_or_dist=norm,
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.slow
@pytest.mark.skip(reason='takes quite long')
def test_compute_hindcast_da1d_not_nan_crpss_quadratic(
    hind_da_initialized_1d, observations_da_1d
):
    """
    Checks that there are no NaNs on hindcast metrics of 1D time series.
    """
    actual = (
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison='m2o',
            metric='crpss',
            gaussian=False,
            dim='member',
        )
        .isnull()
        .any()
    )
    assert not actual


def test_hindcast_crpss_orientation(hind_da_initialized_1d, observations_da_1d):
    """
    Checks that CRPSS hindcast as skill score > 0.
    """
    actual = compute_hindcast(
        hind_da_initialized_1d,
        observations_da_1d,
        comparison='m2o',
        metric='crpss',
        dim='member',
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()


def test_pm_crpss_orientation(PM_da_initialized_1d, PM_da_control_1d):
    """
    Checks that CRPSS in PM as skill score > 0.
    """
    actual = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison='m2m',
        metric='crpss',
        dim='member',
    )
    if 'init' in actual.coords:
        actual = actual.mean('init')
    assert not (actual.isel(lead=[0, 1]) < 0).any()


# test api
# Probabilistic PM metrics dont work with non-prob PM comparison m2e and e2c
@pytest.mark.parametrize('comparison', NON_PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize('metric', PROBABILISTIC_METRICS)
def test_compute_pm_probabilistic_metric_non_probabilistic_comparison_fails(
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
):
    with pytest.raises(ValueError) as excinfo:
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison=comparison,
            metric=metric,
        )
    assert f'Probabilistic metric `{metric}` requires comparison' in str(excinfo.value)


@pytest.mark.parametrize('dim', ['init', ['init', 'member']])
@pytest.mark.parametrize('metric', ['crps'])
def test_compute_pm_probabilistic_metric_not_dim_member_warn(
    PM_da_initialized_1d, PM_da_control_1d, metric, dim
):
    with pytest.warns(UserWarning) as record:
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison='m2c',
            metric=metric,
            dim=dim,
        )
    expected = (
        f'Probabilistic metric {metric} requires to be '
        f'computed over dimension `dim="member"`. '
        f'Set automatically.'
    )
    # get second warning here
    assert record[1].message.args[0] == expected


@pytest.mark.parametrize('metric', ['crps'])
def test_compute_hindcast_probabilistic_metric_e2o_fails(
    hind_da_initialized_1d, observations_da_1d, metric
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.raises(ValueError) as excinfo:
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison='e2o',
            metric=metric,
            dim='member',
        )
    assert f'Probabilistic metric `{metric}` requires' in str(excinfo.value)


@pytest.mark.parametrize('dim', ['init'])
@pytest.mark.parametrize('metric', ['crps'])
def test_compute_hindcast_probabilistic_metric_not_dim_member_warn(
    hind_da_initialized_1d, observations_da_1d, metric, dim
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.warns(UserWarning) as record:
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison='m2o',
            metric=metric,
            dim=dim,
        )
    expected = (
        f'Probabilistic metric {metric} requires to be '
        f'computed over dimension `dim="member"`. '
        f'Set automatically.'
    )
    # Set this to the third message since the first two are about converting the integer
    # time to annual `cftime`.
    assert record[0].message.args[0] == expected
