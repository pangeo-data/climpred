import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climpred.prediction import compute_hindcast, compute_perfect_model


@pytest.fixture()
def daily_initialized():
    init = pd.date_range('1990-01', '1990-03', freq='D')
    lead = np.arange(5)
    member = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(init), len(lead), len(member)),
        dims=['init', 'lead', 'member'],
        coords=[init, lead, member],
    )


@pytest.fixture()
def daily_obs():
    time = pd.date_range('1990-01', '1990-03', freq='D')
    return xr.DataArray(np.random.rand(len(time)), dims=['time'], coords=[time])


@pytest.fixture()
def monthly_initialized():
    init = pd.date_range('1990-01', '1996-01', freq='MS')
    lead = np.arange(20)
    member = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(init), len(lead), len(member)),
        dims=['init', 'lead', 'member'],
        coords=[init, lead, member],
    )


@pytest.fixture()
def monthly_obs():
    time = pd.date_range('1990-01', '1996-01', freq='MS')
    return xr.DataArray(np.random.rand(len(time)), dims=['time'], coords=[time])


def test_daily_resolution_hindcast(daily_initialized, daily_obs):
    """Tests that daily resolution hindcast predictions work."""
    daily_initialized.lead.attrs['units'] = 'days'
    assert compute_hindcast(daily_initialized, daily_obs).all()


def test_daily_resolution_perfect_model(daily_initialized, daily_obs):
    """Tests that daily resolution perfect model predictions work."""
    daily_initialized.lead.attrs['units'] = 'days'
    assert compute_perfect_model(daily_initialized, daily_obs).all()


def test_pentadal_resolution_hindcast(daily_initialized, daily_obs):
    """Tests that pentadal resolution hindcast predictions work."""
    pentadal_hindcast = daily_initialized.resample(init='5D').mean()
    pentadal_obs = daily_obs.resample(time='5D').mean()
    pentadal_hindcast.lead.attrs['units'] = 'pentads'
    assert compute_hindcast(pentadal_hindcast, pentadal_obs).all()


def test_pentadal_resolution_perfect_model(daily_initialized, daily_obs):
    """Tests that pentadal resolution perfect model predictions work."""
    pentadal_pm = daily_initialized.resample(init='5D').mean()
    pentadal_obs = daily_obs.resample(time='5D').mean()
    pentadal_pm.lead.attrs['units'] = 'pentads'
    assert compute_hindcast(pentadal_pm, pentadal_obs).all()


def test_weekly_resolution_hindcast(daily_initialized, daily_obs):
    """Tests that weekly resolution hindcast predictions work."""
    weekly_hindcast = daily_initialized.resample(init='W').mean()
    weekly_obs = daily_obs.resample(time='W').mean()
    weekly_hindcast.lead.attrs['units'] = 'weeks'
    assert compute_hindcast(weekly_hindcast, weekly_obs).all()


def test_weekly_resolution_perfect_model(daily_initialized, daily_obs):
    """Tests that weekly resolution perfect model predictions work."""
    weekly_pm = daily_initialized.resample(init='W').mean()
    weekly_obs = daily_obs.resample(time='W').mean()
    weekly_pm.lead.attrs['units'] = 'weeks'
    assert compute_hindcast(weekly_pm, weekly_obs).all()


def test_monthly_resolution_hindcast(monthly_initialized, monthly_obs):
    """Tests that monthly resolution hindcast predictions work."""
    monthly_initialized.lead.attrs['units'] = 'months'
    assert compute_hindcast(monthly_initialized, monthly_obs).all()


def test_monthly_resolution_perfect_model(monthly_initialized, monthly_obs):
    """Tests that monthly resolution perfect model predictions work."""
    monthly_initialized.lead.attrs['units'] = 'months'
    assert compute_perfect_model(monthly_initialized, monthly_obs).all()


def test_seasonal_resolution_hindcast(monthly_initialized, monthly_obs):
    """Tests that seasonal resolution hindcast predictions work."""
    seasonal_hindcast = (
        monthly_initialized.rolling(lead=3, center=True).mean().dropna(dim='lead')
    )
    seasonal_hindcast = seasonal_hindcast.isel(init=slice(0, None, 3))
    seasonal_obs = monthly_obs.rolling(time=3, center=True).mean().dropna(dim='time')
    seasonal_hindcast.lead.attrs['units'] = 'seasons'
    assert compute_hindcast(seasonal_hindcast, seasonal_obs).all()


def test_seasonal_resolution_perfect_model(monthly_initialized, monthly_obs):
    """Tests that seasonal resolution perfect model predictions work."""
    seasonal_pm = (
        monthly_initialized.rolling(lead=3, center=True).mean().dropna(dim='lead')
    )
    seasonal_pm = seasonal_pm.isel(init=slice(0, None, 3))
    seasonal_obs = monthly_obs.rolling(time=3, center=True).mean().dropna(dim='time')
    assert compute_perfect_model(seasonal_pm, seasonal_obs).all()


def test_yearly_resolution_hindcast(monthly_initialized, monthly_obs):
    """Tests that yearly resolution hindcast predictions work."""
    yearly_hindcast = monthly_initialized.resample(init='YS').mean()
    yearly_obs = monthly_obs.resample(time='YS').mean()
    yearly_hindcast.lead.attrs['units'] = 'years'
    assert compute_hindcast(yearly_hindcast, yearly_obs).all()


def test_yearly_resolution_perfect_model(monthly_initialized, monthly_obs):
    """Tests that yearly resolution perfect model predictions work."""
    yearly_pm = monthly_initialized.resample(init='YS').mean()
    yearly_obs = monthly_obs.resample(time='YS').mean()
    yearly_pm.lead.attrs['units'] = 'years'
    assert compute_hindcast(yearly_pm, yearly_obs).all()
