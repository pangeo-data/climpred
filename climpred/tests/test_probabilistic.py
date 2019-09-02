import pytest
from climpred.constants import PROBABILISTIC_METRICS
from climpred.prediction import compute_perfect_model
from climpred.tutorial import load_dataset

PM_COMPARISONS = {'m2c': '', 'e2c': ''}


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

    actual = (
        compute_perfect_model(
            pm_da_ds1d,
            pm_da_control1d,
            comparison=comparison,
            metric=metric,
            threshold=threshold,
            gaussian=True,
        )
        .isnull()
        .any()
    )
    print(actual)
    assert not actual


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
