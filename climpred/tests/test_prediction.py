import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climpred.prediction import (bootstrap_perfect_model,
                                 compute_perfect_model, compute_persistence)

# from xarray.tests import assert_allclose


xskillscore_metrics = ('pearson_r', 'rmse', 'mse', 'mae')
PM_only_metrics = ('nrmse', 'nmse', 'nmae')  # excl uacc because sqrt(neg)
PM_comparisons = ('e2c', 'm2c', 'm2e', 'm2m')
all_metrics = xskillscore_metrics + PM_only_metrics


@pytest.fixture
def PM_da_ds():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    member = np.arange(3)
    ensemble = np.arange(3)
    data = np.random.rand(len(dates), len(
        lats), len(lons), len(member), len(ensemble))
    return xr.DataArray(data,
                        coords=[dates, lats, lons, member, ensemble],
                        dims=['time', 'lat', 'lon', 'member', 'ensemble'])


@pytest.fixture
def PM_da_control():
    dates = pd.date_range('1/1/2000', '1/30/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    data = np.random.rand(len(dates), len(lats), len(lons))
    return xr.DataArray(data,
                        coords=[dates, lats, lons],
                        dims=['time', 'lat', 'lon'])


@pytest.fixture
def PM_ds_ds():
    dates = pd.date_range('1/1/2000', '1/3/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
    member = np.arange(3)
    ensemble = np.arange(3)
    data = np.random.rand(len(dates), len(
        lats), len(lons), len(member), len(ensemble))
    return xr.Dataset({'varname1': (['time', 'lat', 'lon', 'member', 'ensemble'], data),
                       'varname2': (['time', 'lat', 'lon', 'member', 'ensemble'], 2 * data)},
                      coords={'time': dates, 'lat': lats, 'lon': lons,
                              'member': member, 'ensemble': ensemble})


@pytest.fixture
def PM_ds_control():
    dates = pd.date_range('1/1/2000', '1/30/2000', freq='D')
    lats = np.arange(4)
    lons = np.arange(5)
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


@pytest.mark.parametrize('comparison', ('e2c', 'm2c', 'm2e', 'm2m'))
@pytest.mark.parametrize('metric', (all_metrics))
def test_bootstrap_perfect_model_ds_not_nan(PM_ds_ds, PM_ds_control, metric,
                                            comparison):
    actual = bootstrap_perfect_model(PM_ds_ds, PM_ds_control, metric=metric,
                                     comparison=comparison, sig=50,
                                     bootstrap=5).isnull().any()
    assert actual == False


@pytest.mark.parametrize('comparison', ('e2c', 'm2c', 'm2e', 'm2m'))
@pytest.mark.parametrize('metric', (all_metrics))
def test_bootstrap_perfect_model_da_not_nan(PM_da_ds, PM_da_control, metric,
                                            comparison):
    actual = bootstrap_perfect_model(PM_da_ds, PM_da_control, metric=metric,
                                     comparison=comparison, sig=50,
                                     bootstrap=5).isnull().any()
    assert actual == False


@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_compute_persistence_da_not_nan(PM_da_control, metric):
    actual = compute_persistence(
        PM_da_control, nlags=3, metric=metric, dim='time').isnull().any()
    assert actual == False


@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_compute_persistence_ds_not_nan(PM_ds_control, metric):
    actual = compute_persistence(
        PM_ds_control, nlags=3, metric=metric, dim='time').isnull().any()
    assert actual == False
