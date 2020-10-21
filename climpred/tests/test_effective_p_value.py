import pytest

from climpred.comparisons import HINDCAST_COMPARISONS
from climpred.metrics import PM_METRICS
from climpred.prediction import compute_hindcast


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_sample_size_smaller_than_n_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hind_da_initialized_1d.mean("member").count("init")
    eff_N = compute_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric="eff_n",
    )
    assert (eff_N <= N).all()


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric="pearson_r_p_value",
        comparison=comparison,
    )
    eff_p = compute_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric="pearson_r_eff_p_value",
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric="spearman_r_p_value",
        comparison=comparison,
    )
    eff_p = compute_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric="spearman_r_eff_p_value",
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


def test_effective_metrics_not_in_PM():
    """Checks that the effective p value metrics are not included in PM."""
    assert "effective_sample_size" not in PM_METRICS
    assert "pearson_r_eff_p_value" not in PM_METRICS
    assert "spearman_r_eff_p_value" not in PM_METRICS
