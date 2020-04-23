import numpy as np

from climpred.bootstrap import bootstrap_perfect_model
from climpred.graphics import plot_bootstrapped_skill_over_leadyear

ITERATIONS = 3


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear_da(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.DataArray.
    """
    res = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='pearson_r',
        iterations=ITERATIONS,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear_single_uninit_lead(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.DataArray.
    """
    res = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric='pearson_r',
        iterations=ITERATIONS,
    )
    # set all but first uninit lead to nan
    res[:, 2, 1:] = [np.nan] * (res.lead.size - 1)
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear_ds(
    PM_ds_initialized_1d, PM_ds_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.Dataset with one variable.
    """
    res = bootstrap_perfect_model(
        PM_ds_initialized_1d,
        PM_ds_control_1d,
        metric='pearson_r',
        iterations=ITERATIONS,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None
