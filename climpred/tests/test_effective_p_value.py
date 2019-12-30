import numpy as np
import pytest
import scipy

from climpred.prediction import compute_hindcast
from climpred.tutorial import load_dataset


@pytest.fixture
def hindcast():
    da = load_dataset('CESM-DP-SST')
    da = da['SST']
    return da


@pytest.fixture
def reconstruction():
    da = load_dataset('FOSI-SST')
    da = da['SST']
    return da


@pytest.fixture
def PM_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da.sel(area='global', period='ym')['tos']
    return da


@pytest.fixture
def PM_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da.sel(area='global', period='ym')['tos']
    return da


def test_eff_sample_size_smaller_than_n(hindcast, reconstruction):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hindcast['init'].size
    eff_N = compute_hindcast(hindcast, reconstruction, metric='eff_n')
    assert (eff_N <= N).all()


def test_eff_pearson_p_same_as_p_with_full_n(hindcast, reconstruction):
    """Tests that the Pearson effective p value returns the same value as normal p value
    if the full sample size N is used."""
    # follow procedure for compute effective p value, but replace effective sample
    # size with true sample size.
    r = compute_hindcast(hindcast, reconstruction, metric='pr')
    n = hindcast['init'].size
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    direct_p = scipy.stats.t.sf(np.abs(t), n - 2) * 2
    normal_p = compute_hindcast(hindcast, reconstruction, metric='pearson_r_p_value')
    assert np.allclose(direct_p, normal_p)


def test_eff_pearson_p_greater_or_equal_to_normal_p(hindcast, reconstruction):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_hindcast(hindcast, reconstruction, metric='pearson_r_p_value')
    eff_p = compute_hindcast(hindcast, reconstruction, metric='pearson_r_eff_p_value')
    assert (normal_p <= eff_p).all()


def test_eff_spearman_p_same_as_p_with_full_n(hindcast, reconstruction):
    """Tests that the Spearman's effective p value returns the same value as normal p
    value if the full sample size N is used."""
    # follow procedure for compute effective p value, but replace effective sample
    # size with true sample size.
    r = compute_hindcast(hindcast, reconstruction, metric='sr')
    n = hindcast['init'].size
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    direct_p = scipy.stats.t.sf(np.abs(t), n - 2) * 2
    normal_p = compute_hindcast(hindcast, reconstruction, metric='spearman_r_p_value')
    assert np.allclose(direct_p, normal_p)


def test_eff_spearman_p_greater_or_equal_to_normal_p(hindcast, reconstruction):
    """Tests that the Spearman's effective p value (more conservative) is greater than
    or equal to the standard p value."""
    normal_p = compute_hindcast(hindcast, reconstruction, metric='spearman_r_p_value')
    eff_p = compute_hindcast(hindcast, reconstruction, metric='spearman_r_eff_p_value')
    assert (normal_p <= eff_p).all()
