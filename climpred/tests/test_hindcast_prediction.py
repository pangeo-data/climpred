import numpy as np
import pytest

from climpred.bootstrap import bootstrap_hindcast

# testing less separately: use PM_METRICS
from climpred.constants import HINDCAST_COMPARISONS, PM_METRICS
from climpred.prediction import (
    compute_hindcast,
    compute_persistence,
    compute_uninitialized,
)
from climpred.tutorial import load_dataset


@pytest.fixture
def initialized_ds():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    return da


@pytest.fixture
def initialized_ds_lead0():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    # Change to a lead-0 framework
    da['init'] += 1
    da['lead'] -= 1
    return da


@pytest.fixture
def initialized_da():
    da = load_dataset('CESM-DP-SST')['SST']
    da = da - da.mean('init')
    return da


@pytest.fixture
def observations_ds():
    da = load_dataset('ERSST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_ds():
    da = load_dataset('FOSI-SST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_da():
    da = load_dataset('FOSI-SST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_ds():
    da = load_dataset('CESM-LE')
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_da():
    da = load_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


def test_compute_hindcast_less_e2r(initialized_da, reconstruction_da):
    """Test raise KeyError for LESS e2r, because needs member."""
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            initialized_da, reconstruction_da, metric='less', comparison='e2r'
        )
    assert 'LESS requires member dimension' in str(excinfo.value)


def test_compute_hindcast_less_m2r(initialized_da, reconstruction_da):
    """Test LESS m2r runs through."""
    actual = (
        compute_hindcast(
            initialized_da, reconstruction_da, metric='less', comparison='m2r'
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.parametrize('metric', PM_METRICS)
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_compute_hindcast(initialized_ds, reconstruction_ds, metric, comparison):
    """
    Checks that compute hindcast works without breaking.
    """
    res = (
        compute_hindcast(
            initialized_ds, reconstruction_ds, metric=metric, comparison=comparison
        )
        .isnull()
        .any()
    )
    for var in res.data_vars:
        assert not res[var]


@pytest.mark.parametrize('metric', PM_METRICS)
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_compute_hindcast_lead0_lead1(
    initialized_ds, initialized_ds_lead0, reconstruction_ds, metric, comparison
):
    """
    Checks that compute hindcast returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_hindcast(
        initialized_ds, reconstruction_ds, metric=metric, comparison=comparison
    )
    res2 = compute_hindcast(
        initialized_ds_lead0, reconstruction_ds, metric=metric, comparison=comparison
    )
    assert (res1.SST.values == res2.SST.values).all()


@pytest.mark.parametrize('metric', PM_METRICS)
def test_persistence(initialized_da, reconstruction_da, metric):
    """
    Checks that compute persistence works without breaking.
    """
    res = (
        compute_persistence(initialized_da, reconstruction_da, metric=metric)
        .isnull()
        .any()
    )
    assert not res


@pytest.mark.parametrize('metric', PM_METRICS)
def test_persistence_lead0_lead1(
    initialized_ds, initialized_ds_lead0, reconstruction_ds, metric
):
    """
    Checks that compute persistence returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_persistence(initialized_ds, reconstruction_ds, metric=metric)
    res2 = compute_persistence(initialized_ds_lead0, reconstruction_ds, metric=metric)
    assert (res1.SST.values == res2.SST.values).all()


@pytest.mark.parametrize('metric', PM_METRICS)
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_uninitialized(uninitialized_da, reconstruction_da, metric, comparison):
    """
    Checks that compute uninitialized works without breaking.
    """
    res = (
        compute_uninitialized(
            uninitialized_da, reconstruction_da, metric=metric, comparison=comparison
        )
        .isnull()
        .any()
    )
    assert not res


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
@pytest.mark.parametrize('metric', PM_METRICS)
def test_bootstrap_hindcast_da1d_not_nan(
    initialized_da, uninitialized_da, reconstruction_da, metric, comparison
):
    """
    Checks that there are no NaNs on bootstrap hindcast of 1D da.
    """
    actual = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        reconstruction_da,
        metric=metric,
        comparison=comparison,
        sig=50,
        bootstrap=2,
    )
    actual_init_skill = actual.sel(kind='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(kind='uninit', results='p').isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize('metric', ('AnomCorr', 'test', 'None'))
def test_compute_hindcast_metric_keyerrors(initialized_ds, reconstruction_ds, metric):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            initialized_ds, reconstruction_ds, comparison='e2r', metric=metric
        )
    assert 'Specify metric from' in str(excinfo.value)


@pytest.mark.parametrize('comparison', ('ensemblemean', 'test', 'None'))
def test_compute_hindcast_comparison_keyerrors(
    initialized_ds, reconstruction_ds, comparison
):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            initialized_ds, reconstruction_ds, comparison=comparison, metric='mse'
        )
    assert 'Specify comparison from' in str(excinfo.value)
