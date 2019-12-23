import numpy as np
import pytest
import scipy
import xarray as xr

from climpred.prediction import compute_hindcast


@pytest.fixture
def hindcast():
    init = np.arange(1990, 2100)
    lead = np.arange(1, 2)
    hind = xr.DataArray(
        np.random.rand(len(init), len(lead)), dims=['init', 'lead'], coords=[init, lead]
    )
    return hind


@pytest.fixture
def obs():
    time = np.arange(1990, 2101)
    obs = xr.DataArray(np.random.rand(len(time)), dims=['time'], coords=[time])
    return obs


def test_eff_sample_size_smaller_than_n(hindcast, obs):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hindcast['init'].size
    eff_N = compute_hindcast(hindcast, obs, metric='eff_n')
    assert eff_N <= N


def test_eff_pearson_p_same_as_p_with_full_n(hindcast, obs):
    """Tests that the Pearson effective p value returns the same value as normal p value
    if the full sample size N is used."""
    # follow procedure for compute effective p-value, but replace effective sample
    # size with true sample size.
    r = compute_hindcast(hindcast, obs, metric='pr')
    n = hindcast['init'].size
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    direct_p = scipy.stats.t.sf(np.abs(t), n - 2) * 2
    normal_p = compute_hindcast(hindcast, obs, metric='pearson_r_p_value')
    assert np.allclose(direct_p, normal_p)


def test_eff_pearson_p_greater_or_equal_to_normal_p(hindcast, obs):
    """Tests that the Pearson effective p-value (more conservative) is greater than or
    equal to the standard p-value."""
    normal_p = compute_hindcast(hindcast, obs, metric='pr')
    eff_p = compute_hindcast(hindcast, obs, metric='pearson_r_eff_p_value')
    assert eff_p >= normal_p


def test_eff_spearman_p_same_as_p_with_full_n(hindcast, obs):
    """Tests that the Spearman's effective p value returns the same value as normal p
    value if the full sample size N is used."""
    # follow procedure for compute effective p-value, but replace effective sample
    # size with true sample size.
    r = compute_hindcast(hindcast, obs, metric='sr')
    n = hindcast['init'].size
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    direct_p = scipy.stats.t.sf(np.abs(t), n - 2) * 2
    normal_p = compute_hindcast(hindcast, obs, metric='spearman_r_p_value')
    assert np.allclose(direct_p, normal_p)


def test_eff_spearman_p_greater_or_equal_to_normal_p(hindcast, obs):
    """Tests that the Spearman's effective p-value (more conservative) is greater than
    or equal to the standard p-value."""
    normal_p = compute_hindcast(hindcast, obs, metric='sr')
    eff_p = compute_hindcast(hindcast, obs, metric='spearman_r_eff_p_value')
    assert eff_p >= normal_p
