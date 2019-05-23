import numpy as np
import pytest

from climpred.bootstrap import bootstrap_hindcast
# testing less separately: use ALL_PM_METRICS_DICT
from climpred.comparisons import ALL_HINDCAST_COMPARISONS_DICT
from climpred.loadutils import open_dataset
from climpred.metrics import ALL_PM_METRICS_DICT
from climpred.prediction import (compute_hindcast, compute_persistence,
                                 compute_uninitialized)


@pytest.fixture
def initialized_ds():
    da = open_dataset('CESM-DP-SST')
    da = da.sel(init=slice(1955, 2015))
    da = da - da.mean('init')
    return da


@pytest.fixture
def initialized_da():
    da = open_dataset('CESM-DP-SST')['SST']
    da = da.sel(init=slice(1955, 2015))
    da = da - da.mean('init')
    return da


@pytest.fixture
def observations_ds():
    da = open_dataset('ERSST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = open_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_ds():
    da = open_dataset('FOSI-SST')
    # same timeframe as DPLE
    da = da.sel(time=slice(1955, 2015))
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_da():
    da = open_dataset('FOSI-SST')['SST']
    # same timeframe as DPLE
    da = da.sel(time=slice(1955, 2015))
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_ds():
    da = open_dataset('CESM-LE')
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_da():
    da = open_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


def test_compute_hindcast_less_e2r(initialized_da, reconstruction_da):
    """Test raise KeyError for LESS e2r, because needs member."""
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(initialized_da,
                         reconstruction_da,
                         metric='less',
                         comparison='e2r')
    assert "LESS requires member dimension" in str(excinfo.value)


def test_compute_hindcast_less_m2r(initialized_da, reconstruction_da):
    """Test LESS m2r runs through."""
    actual = compute_hindcast(initialized_da,
                              reconstruction_da,
                              metric='less',
                              comparison='m2r').isnull().any()
    assert not actual


@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
@pytest.mark.parametrize('comparison', ALL_HINDCAST_COMPARISONS_DICT.keys())
def test_compute_hindcast(initialized_ds, reconstruction_ds, metric, comparison):
    """
    Checks that compute reference works without breaking.
    """
    res = compute_hindcast(initialized_ds,
                           reconstruction_ds,
                           metric=metric,
                           comparison=comparison).isnull().any()
    for var in res.data_vars:
        assert not res[var]


@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_persistence(initialized_da, reconstruction_da, metric):
    """
    Checks that compute persistence works without breaking.
    """
    res = compute_persistence(initialized_da, reconstruction_da,
                              metric=metric).isnull().any()
    assert not res


@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
@pytest.mark.parametrize('comparison', ALL_HINDCAST_COMPARISONS_DICT.keys())
def test_uninitialized(uninitialized_da, reconstruction_da, metric, comparison):
    """
    Checks that compute uninitialized works without breaking.
    """
    res = compute_uninitialized(uninitialized_da,
                                reconstruction_da,
                                metric=metric,
                                comparison=comparison).isnull().any()
    assert not res


@pytest.mark.parametrize('comparison', ALL_HINDCAST_COMPARISONS_DICT.keys())
@pytest.mark.parametrize('metric', ALL_PM_METRICS_DICT.keys())
def test_bootstrap_hindcast_da1d_not_nan(initialized_da, uninitialized_da, reconstruction_da, metric,
                                         comparison):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D da.
    """
    actual = bootstrap_hindcast(initialized_da,
                                uninitialized_da,
                                reconstruction_da,
                                metric=metric,
                                comparison=comparison,
                                sig=50,
                                bootstrap=2)
    actual_init_skill = actual.sel(kind='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(kind='uninit', results='p').isnull().any()
    assert not actual_uninit_p
