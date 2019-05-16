import numpy as np
import pytest

from climpred.loadutils import open_dataset
from climpred.prediction import (all_metric_strings, compute_persistence,
                                 compute_reference, compute_uninitialized)

metrics = tuple(all_metric_strings)
comparisons = ('e2r', 'm2r')


@pytest.fixture
def initialized():
    da = open_dataset('CESM-DP-SST')
    da = da.sel(init=slice(1955, 2015))
    da = da - da.mean('init')
    return da


@pytest.fixture
def observations():
    da = open_dataset('ERSST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction():
    da = open_dataset('FOSI-SST')
    # same timeframe as DPLE
    da = da.sel(time=slice(1955, 2015))
    da = da['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized():
    da = open_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = np.arange(len(da.member))
    da = da - da.mean('time')
    return da


@pytest.mark.parametrize('metric', metrics)
@pytest.mark.parametrize('comparison', comparisons)
def test_compute_reference(initialized, reconstruction, metric, comparison):
    """
    Checks that compute reference works without breaking.
    """
    res = compute_reference(initialized, reconstruction, metric=metric,
                            comparison=comparison).isnull().any()
    for var in res.data_vars:
        assert not res[var]


@pytest.mark.parametrize('metric', metrics)
def test_persistence(initialized, reconstruction, metric):
    """
    Checks that compute persistence works without breaking.
    """
    res = compute_persistence(initialized, reconstruction,
                              metric=metric).isnull().any()
    assert not res


@pytest.mark.parametrize('metric', metrics)
@pytest.mark.parametrize('comparison', comparisons)
def test_uninitialized(uninitialized, reconstruction, metric, comparison):
    """
    Checks that compute uninitialized works without breaking.
    """
    res = compute_uninitialized(uninitialized, reconstruction, metric=metric,
                                comparison=comparison).isnull().any()
    assert not res
