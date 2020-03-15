import pytest

from climpred.comparisons import HINDCAST_COMPARISONS, PM_COMPARISONS
from climpred.prediction import verify_hindcast, verify_perfect_model


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_sample_size_smaller_than_n_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hind_da_initialized_1d.mean('member').count('init')
    eff_N = verify_hindcast(
        hind_da_initialized_1d, reconstruction_da_1d, metric='eff_n',
    )
    assert (eff_N <= N).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_sample_size_smaller_than_n_PM_da_initialized_1d(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    if comparison == 'e2c':
        N = PM_da_initialized_1d.mean('member').count('init')
    else:
        N = PM_da_initialized_1d.stack(stack_dims=['init', 'member']).count(
            'stack_dims'
        )
    eff_N = verify_perfect_model(
        PM_da_initialized_1d, PM_da_control_1d, metric='eff_n', comparison=comparison,
    )
    assert (eff_N <= N).all()


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = verify_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric='pearson_r_p_value',
        comparison=comparison,
    )
    eff_p = verify_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric='pearson_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_pm(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = verify_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='pearson_r_p_value',
        comparison=comparison,
    )
    eff_p = verify_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='pearson_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_hind_da_initialized_1d(
    hind_da_initialized_1d, reconstruction_da_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = verify_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric='spearman_r_p_value',
        comparison=comparison,
    )
    eff_p = verify_hindcast(
        hind_da_initialized_1d,
        reconstruction_da_1d,
        metric='spearman_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_pm(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = verify_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='spearman_r_p_value',
        comparison=comparison,
    )
    eff_p = verify_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='spearman_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()
