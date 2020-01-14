import pytest

from climpred.constants import HINDCAST_COMPARISONS, PM_COMPARISONS
from climpred.prediction import compute_hindcast, compute_perfect_model
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
def perfect_model():
    da = load_dataset('MPI-PM-DP-1D')
    da = da.sel(area='global', period='ym')['tos']
    return da


@pytest.fixture
def perfect_model_control():
    da = load_dataset('MPI-control-1D')
    da = da.sel(area='global', period='ym')['tos']
    return da


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_sample_size_smaller_than_n_hindcast(hindcast, reconstruction, comparison):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    N = hindcast.mean('member').count('init')
    eff_N = compute_hindcast(hindcast, reconstruction, metric='eff_n', max_dof=True)
    assert (eff_N <= N).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_sample_size_smaller_than_n_perfect_model(
    perfect_model, perfect_model_control, comparison
):
    """Tests that effective sample size is less than or equal to the actual sample size
    of the data."""
    if comparison == 'e2c':
        N = perfect_model.mean('member').count('init')
    else:
        N = perfect_model.stack(stack_dims=['init', 'member']).count('stack_dims')
    eff_N = compute_perfect_model(
        perfect_model, perfect_model_control, metric='eff_n', comparison=comparison,
    )
    assert (eff_N <= N).all()


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_hindcast(
    hindcast, reconstruction, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_hindcast(
        hindcast,
        reconstruction,
        metric='pearson_r_p_value',
        max_dof=True,
        comparison=comparison,
    )
    eff_p = compute_hindcast(
        hindcast,
        reconstruction,
        metric='pearson_r_eff_p_value',
        max_dof=True,
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_pearson_p_greater_or_equal_to_normal_p_pm(
    perfect_model, perfect_model_control, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_perfect_model(
        perfect_model,
        perfect_model_control,
        metric='pearson_r_p_value',
        comparison=comparison,
    )
    eff_p = compute_perfect_model(
        perfect_model,
        perfect_model_control,
        metric='pearson_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_hindcast(
    hindcast, reconstruction, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_hindcast(
        hindcast,
        reconstruction,
        metric='spearman_r_p_value',
        max_dof=True,
        comparison=comparison,
    )
    eff_p = compute_hindcast(
        hindcast,
        reconstruction,
        metric='spearman_r_eff_p_value',
        max_dof=True,
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_eff_spearman_p_greater_or_equal_to_normal_p_pm(
    perfect_model, perfect_model_control, comparison
):
    """Tests that the Pearson effective p value (more conservative) is greater than or
    equal to the standard p value."""
    normal_p = compute_perfect_model(
        perfect_model,
        perfect_model_control,
        metric='spearman_r_p_value',
        comparison=comparison,
    )
    eff_p = compute_perfect_model(
        perfect_model,
        perfect_model_control,
        metric='spearman_r_eff_p_value',
        comparison=comparison,
    )
    assert (normal_p <= eff_p).all()
