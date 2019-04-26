import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climpred.bootstrap import bootstrap_perfect_model
from climpred.prediction import compute_perfect_model, compute_persistence_pm

xskillscore_metrics = ('pearson_r', 'rmse', 'mse', 'mae')
xskillscore_distance_metrics = ('rmse', 'mse', 'mae')
PM_only_metrics = ('nrmse', 'nmse', 'nmae', 'less', 'lesss',
                   'crps', 'crpss')  # excl uacc because sqrt(neg)
PM_comparisons = ('e2c', 'm2c', 'm2e', 'm2m')
all_metrics = xskillscore_metrics + PM_only_metrics
all_metrics_wo_pearson_r = xskillscore_distance_metrics + PM_only_metrics


@pytest.fixture
def PM_da_ds():
    dates = pd.date_range('1/1/2000', '3/1/2000', freq='M')
    dates = range(3)
    lats = np.arange(4)
    lons = np.arange(3)
    member = np.arange(3)
    initialization = [2000, 2001]
    data = np.random.rand(len(dates), len(
        lats), len(lons), len(member), len(initialization))
    return xr.DataArray(data,
                        coords=[dates, lats, lons, member, initialization],
                        dims=['time', 'lat', 'lon', 'member',
                              'initialization'])


@pytest.fixture
def PM_da_control():
    dates = pd.date_range('1/1/2000', periods=30, freq='M')
    dates = range(30)
    lats = np.arange(4)
    lons = np.arange(3)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def PM_ds_ds():
    dates = pd.date_range('1/1/2000', '3/1/2000', freq='M')
    dates = range(3)
    lats = np.arange(4)
    lons = np.arange(3)
    member = np.arange(3)
    initialization = [2000, 2001]
    data = np.random.rand(len(dates), len(
        lats), len(lons), len(member), len(initialization))
    return xr.Dataset({'varname1': (['time', 'lat', 'lon', 'member',
                                     'initialization'], data),
                       'varname2': (['time', 'lat', 'lon', 'member',
                                     'initialization'], 2 * data)},
                      coords={'time': dates, 'lat': lats, 'lon': lons,
                              'member': member,
                              'initialization': initialization})


@pytest.fixture
def PM_ds_control():
    dates = pd.date_range('1/1/2000', periods=30, freq='M')
    dates = range(30)
    lats = np.arange(4)
    lons = np.arange(3)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.Dataset({'varname1': (['time', 'lat', 'lon'], data),
                       'varname2': (['time', 'lat', 'lon'], 2 * data)},
                      coords={'time': dates, 'lat': lats, 'lon': lons})


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_da_not_nan(PM_da_ds, PM_da_control, metric,
                                          comparison):
    actual = compute_perfect_model(PM_da_ds, PM_da_control, metric=metric,
                                   comparison=comparison).isnull().any()
    assert actual == False


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_ds_not_nan(PM_ds_ds, PM_ds_control, metric,
                                          comparison):
    actual = compute_perfect_model(PM_ds_ds, PM_ds_control, metric=metric,
                                   comparison=comparison).isnull().any()
    assert actual == False


# @pytest.mark.parametrize('comparison', ('e2c', 'm2c', 'm2e', 'm2m'))
# @pytest.mark.parametrize('metric', (all_metrics))
# def test_bootstrap_perfect_model_ds_not_nan(PM_ds_ds, PM_ds_control, metric, comparison):
#    actual = bootstrap_perfect_model(
#        PM_ds_ds, PM_ds_control, metric=metric, comparison=comparison, sig=50, bootstrap=2, compute_persistence_skill=False).isnull().any()
#    assert actual == False


# @pytest.mark.parametrize('comparison', ('e2c', 'm2c', 'm2e', 'm2m'))
# @pytest.mark.parametrize('metric', (all_metrics))
# def test_bootstrap_perfect_model_da_not_nan(PM_da_ds, PM_da_control, metric, comparison):
#    actual = bootstrap_perfect_model(
#        PM_da_ds, PM_da_control, metric=metric, comparison=comparison, sig=50, bootstrap=2, compute_persistence_skill=False).isnull().any()
#    assert actual == False


# @pytest.mark.parametrize('metric', xskillscore_distance_metrics)
# def test_compute_persistence_pm_da_not_nan(PM_da_ds, PM_da_control, metric):
#    actual = compute_persistence_pm(
#        PM_da_ds, PM_da_control, 2, metric=metric, dim='time').isnull().any()
#    assert actual == False


# @pytest.mark.parametrize('metric', xskillscore_distance_metrics)
# def test_compute_persistence_pm_ds_not_nan(PM_ds_ds, PM_ds_control, metric):
#    actual = compute_persistence_pm(
#        PM_ds_ds, PM_ds_control, 2, metric=metric, dim='time').isnull().any()
#    assert actual == False
