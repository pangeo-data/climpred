from climpred.bootstrap import bootstrap_perfect_model
from climpred.graphics import plot_bootstrapped_skill_over_leadyear
from climpred.tutorial import load_dataset


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear_da():
    """
    Checks plots from bootstrap_perfect_model works for xr.DataArray.
    """
    da = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    PM_da_ds1d = da['tos']

    da = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    PM_da_control1d = da['tos']

    bootstrap = 3
    res = bootstrap_perfect_model(
        PM_da_ds1d, PM_da_control1d, metric='pearson_r', bootstrap=bootstrap,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear_ds():
    """
    Checks plots from bootstrap_perfect_model works for xr.Dataset with one variable.
    """
    da = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    PM_da_ds1d = da['tos'].to_dataset(name='tos')

    da = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    PM_da_control1d = da['tos'].to_dataset(name='tos')

    bootstrap = 3
    res = bootstrap_perfect_model(
        PM_da_ds1d, PM_da_control1d, metric='pearson_r', bootstrap=bootstrap,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None
