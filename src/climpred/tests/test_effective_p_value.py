import pytest

from climpred.comparisons import HINDCAST_COMPARISONS
from climpred.metrics import PM_METRICS

kw = dict(dim="init", alignment="same_inits")


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_sample_size_smaller_than_n_hindcast_hist_obs_1d(
    hindcast_hist_obs_1d, comparison
):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hindcast_hist_obs_1d.get_initialized().mean("member").count("init")
    eff_N = hindcast_hist_obs_1d.verify(
        comparison=comparison,
        **kw,
        metric="eff_n",
    )
    assert (eff_N <= N).to_array().all()


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_hindcast_hist_obs_1d(
    hindcast_hist_obs_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = hindcast_hist_obs_1d.verify(
        comparison=comparison,
        **kw,
        metric="pearson_r_p_value",
    )
    eff_p = hindcast_hist_obs_1d.verify(
        comparison=comparison,
        **kw,
        metric="pearson_r_eff_p_value",
    )
    assert (normal_p <= eff_p).to_array().all()


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_hindcast_hist_obs_1d(
    hindcast_hist_obs_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = hindcast_hist_obs_1d.verify(
        comparison=comparison,
        **kw,
        metric="spearman_r_p_value",
    )
    eff_p = hindcast_hist_obs_1d.verify(
        comparison=comparison,
        **kw,
        metric="spearman_r_eff_p_value",
    )
    assert (normal_p <= eff_p).to_array().all()


def test_effective_metrics_not_in_PM():
    """Checks that the effective p value metrics are not included in PM."""
    assert "effective_sample_size" not in PM_METRICS
    assert "pearson_r_eff_p_value" not in PM_METRICS
    assert "spearman_r_eff_p_value" not in PM_METRICS
