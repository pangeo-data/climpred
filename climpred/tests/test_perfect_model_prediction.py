import pytest

from climpred.bootstrap import bootstrap_perfect_model
from climpred.metrics import ALL_PM_METRICS_DICT
from climpred.prediction import compute_perfect_model, compute_persistence
from climpred.tutorial import load_dataset

ALL_PM_COMPARISONS_DICT = {'m2c': '', 'e2c': ''}


@pytest.fixture
def PM_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def PM_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def PM_ds_ds1d():
    ds = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    return ds


@pytest.fixture
def PM_ds_control1d():
    ds = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    return ds


@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_compute_persistence_ds1d_not_nan(PM_ds_ds1d, PM_ds_control1d, metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = (
        compute_persistence(PM_ds_ds1d, PM_ds_control1d, metric=metric).isnull().any()
    )
    for var in actual.data_vars:
        assert not actual[var]


@pytest.mark.parametrize('comparison', ALL_PM_COMPARISONS_DICT.keys())
@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_compute_perfect_model_da1d_not_nan(
    PM_da_ds1d, PM_da_control1d, comparison, metric
):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = (
        compute_perfect_model(
            PM_da_ds1d, PM_da_control1d, comparison=comparison, metric=metric
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.parametrize('comparison', ALL_PM_COMPARISONS_DICT.keys())
@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_bootstrap_perfect_model_da1d_not_nan(
    PM_da_ds1d, PM_da_control1d, metric, comparison
):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D da.
    """
    actual = bootstrap_perfect_model(
        PM_da_ds1d,
        PM_da_control1d,
        metric=metric,
        comparison=comparison,
        sig=50,
        bootstrap=2,
    )
    actual_init_skill = actual.sel(kind='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(kind='uninit', results='p').isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize('comparison', ALL_PM_COMPARISONS_DICT.keys())
@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_bootstrap_perfect_model_ds1d_not_nan(
    PM_ds_ds1d, PM_ds_control1d, metric, comparison
):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D ds.
    """
    actual = bootstrap_perfect_model(
        PM_ds_ds1d,
        PM_ds_control1d,
        metric=metric,
        comparison=comparison,
        sig=50,
        bootstrap=2,
    )
    for var in actual.data_vars:
        actual_init_skill = actual[var].sel(kind='init', results='skill').isnull().any()
        assert not actual_init_skill
    for var in actual.data_vars:
        actual_uninit_p = actual[var].sel(kind='uninit', results='p').isnull().any()
        assert not actual_uninit_p
