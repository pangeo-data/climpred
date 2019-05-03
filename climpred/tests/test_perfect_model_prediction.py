import numpy as np
import pytest

from climpred.bootstrap import bootstrap_perfect_model
from climpred.loadutils import open_dataset
from climpred.prediction import compute_perfect_model, compute_persistence_pm

xskillscore_metrics = ('pearson_r', 'rmse', 'mse', 'mae')
xskillscore_distance_metrics = ('rmse', 'mse', 'mae')
PM_only_metrics = ('nrmse', 'nmse', 'nmae')
PM_comparisons = ('e2c', 'm2c', 'm2e', 'm2m')
all_metrics = xskillscore_metrics + PM_only_metrics


@pytest.fixture
def PM_da_ds3d():
    da = open_dataset('MPI-PM-DP-3D')
    # Box in South Atlantic with no NaNs.
    da = da.isel(x=slice(0, 5), y=slice(145, 150))
    return da['tos']


@pytest.fixture
def PM_da_control3d():
    da = open_dataset('MPI-control-3D')
    da = da.isel(x=slice(0, 5), y=slice(145, 150))
    # fix to span 300yr control
    t = list(np.arange(da.time.size))
    da = da.isel(time=t*6)
    da['time'] = np.arange(3000, 3000 + da.time.size)
    return da['tos']


@pytest.fixture
def PM_ds_ds3d():
    ds = open_dataset('MPI-PM-DP-3D')
    ds = ds.isel(x=slice(0, 5), y=slice(145, 150))
    return ds


@pytest.fixture
def PM_ds_control3d():
    ds = open_dataset('MPI-control-3D')
    ds = ds.isel(x=slice(0, 5), y=slice(145, 150))
    t = list(np.arange(ds.time.size))
    ds = ds.isel(time=t*6)
    ds['time'] = np.arange(3000, 3000 + ds.time.size)
    return ds


@pytest.fixture
def PM_da_ds1d():
    da = open_dataset('MPI-PM-DP-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_da_control1d():
    da = open_dataset('MPI-control-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_ds_ds1d():
    ds = open_dataset('MPI-PM-DP-1D')
    return ds


@pytest.fixture
def PM_ds_control1d():
    ds = open_dataset('MPI-control-1D')
    return ds


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_da_not_nan(PM_da_ds3d, PM_da_control3d, metric,
                                          comparison):
    """
    Checks that there are no NaNs on perfect model comparison for DataArray.
    """
    actual = compute_perfect_model(PM_da_ds3d, PM_da_control3d, metric=metric,
                                   comparison=comparison).isnull().any()
    # most pythonic way to assert value is False.
    # https://stackoverflow.com/questions/14733883/
    # best-practice-for-python-assert-command-false
    assert not actual


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_ds_not_nan(PM_ds_ds3d, PM_ds_control3d, metric,
                                          comparison):
    """
    Checks that there are no NaNs on perfect model comparison for Dataset.
    """
    actual = compute_perfect_model(PM_ds_ds3d, PM_ds_control3d, metric=metric,
                                   comparison=comparison).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]


@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_compute_persistence_pm_ds1d_not_nan(PM_ds_ds1d, PM_ds_control1d,
                                             metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = compute_persistence_pm(PM_ds_ds1d, PM_ds_control1d,
                                    metric=metric).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]


@pytest.mark.parametrize('metric', xskillscore_distance_metrics)
def test_compute_persistence_pm_ds3d_not_nan(PM_ds_ds3d, PM_ds_control3d,
                                             metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = compute_persistence_pm(PM_ds_ds3d, PM_ds_control3d,
                                    metric=metric).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]


# NOTE:
# Also, it breaks when any PM-only metrics are
# used, since `compute_persistence_pm` breaks on non-xskillscore metrics. [resolved]
@pytest.mark.parametrize('comparison', (PM_comparisons))
@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_bootstrap_perfect_model_da1d_not_nan(PM_da_ds1d, PM_da_control1d,
                                              metric, comparison):
    """
    Checks that there are no NaNs on bootstrap init skill or uninit p of 1D da time series.
    """
    actual = bootstrap_perfect_model(PM_da_ds1d, PM_da_control1d,
                                     metric=metric, comparison=comparison,
                                     sig=50,
                                     bootstrap=2)
    actual_init_skill = actual.sel(i='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(i='uninit', results='p').isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize('comparison', (PM_comparisons))
@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_bootstrap_perfect_model_ds1d_not_nan(PM_ds_ds1d, PM_ds_control1d,
                                              metric, comparison):
    """
    Checks that there are no NaNs on bootstrap init skill or uninit p of 1D ds time series.
    """
    actual = bootstrap_perfect_model(PM_ds_ds1d, PM_ds_control1d,
                                     metric=metric, comparison=comparison,
                                     sig=50,
                                     bootstrap=2)
    for var in actual.data_vars:
        actual_init_skill = actual[var].sel(
            i='init', results='skill').isnull().any()
        assert not actual_init_skill
    for var in actual.data_vars:
        actual_uninit_p = actual[var].sel(
            i='uninit', results='p').isnull().any()
        assert not actual_uninit_p


@pytest.mark.parametrize('comparison', (PM_comparisons))
@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_bootstrap_perfect_model_da3d_not_nan(PM_da_ds3d, PM_da_control3d,
                                              metric, comparison):
    """
    Checks that there are no NaNs on bootstrap init skill or uninit p of 3D da.
    """
    actual = bootstrap_perfect_model(PM_da_ds3d, PM_da_control3d,
                                     metric=metric, comparison=comparison,
                                     sig=50,
                                     bootstrap=2)
    actual_init_skill = actual.sel(i='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(i='uninit', results='p').isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize('comparison', (PM_comparisons))
@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_bootstrap_perfect_model_ds3d_not_nan(PM_ds_ds3d, PM_ds_control3d,
                                              metric, comparison):
    """
    Checks that there are no NaNs on bootstrap init skill or uninit p of 3D ds.
    """
    actual = bootstrap_perfect_model(PM_ds_ds3d, PM_ds_control3d,
                                     metric=metric, comparison=comparison,
                                     sig=50,
                                     bootstrap=2)
    for var in actual.data_vars:
        actual_init_skill = actual[var].sel(
            i='init', results='skill').isnull().any()
        assert not actual_init_skill
    for var in actual.data_vars:
        actual_uninit_p = actual[var].sel(
            i='uninit', results='p').isnull().any()
        assert not actual_uninit_p
