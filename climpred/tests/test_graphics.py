import numpy as np
import pytest

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.bootstrap import bootstrap_perfect_model
from climpred.checks import DimensionError
from climpred.graphics import plot_bootstrapped_skill_over_leadyear

ITERATIONS = 3


def test_mpi_he_plot_bootstrapped_skill_over_leadyear_da(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.DataArray.
    """
    res = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric="pearson_r",
        iterations=ITERATIONS,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


def test_mpi_he_plot_bootstrapped_skill_over_leadyear_single_uninit_lead(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.DataArray.
    """
    res = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric="pearson_r",
        iterations=ITERATIONS,
    )
    # set all but first uninit lead to nan
    res[:, 2, 1:] = [np.nan] * (res.lead.size - 1)
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


def test_mpi_he_plot_bootstrapped_skill_over_leadyear_ds(
    PM_ds_initialized_1d, PM_ds_control_1d
):
    """
    Checks plots from bootstrap_perfect_model works for xr.Dataset with one variable.
    """
    res = bootstrap_perfect_model(
        PM_ds_initialized_1d,
        PM_ds_control_1d,
        metric="pearson_r",
        iterations=ITERATIONS,
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


@pytest.mark.parametrize("cmap", ["tab10", "jet"])
@pytest.mark.parametrize("show_members", [True, False])
@pytest.mark.parametrize("variable", ["tos", None])
def test_PerfectModelEnsemble_plot(
    PM_ds_initialized_1d, PM_ds_control_1d, variable, show_members, cmap
):
    """Test PredictionEnsemble.plot()."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    kws = {"cmap": cmap, "show_members": show_members, "variable": variable}
    pm.plot(**kws)
    pm = pm.add_control(PM_ds_control_1d)
    pm.plot(**kws)
    pm = pm.generate_uninitialized()
    pm.plot(**kws)


def test_PerfectModelEnsemble_plot_fails_3d(PM_ds_initialized_3d):
    """Test PredictionEnsemble.plot()."""
    pm = PerfectModelEnsemble(PM_ds_initialized_3d)
    with pytest.raises(DimensionError) as excinfo:
        pm.plot()
    assert "does not allow dimensions other" in str(excinfo.value)


@pytest.mark.parametrize("cmap", ["tab10", "jet"])
@pytest.mark.parametrize("show_members", [True, False])
@pytest.mark.parametrize("variable", ["SST", None])
def test_HindcastEnsemble_plot(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    reconstruction_ds_1d,
    observations_ds_1d,
    variable,
    show_members,
    cmap,
):
    """Test PredictionEnsemble.plot()."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    kws = {"cmap": cmap, "show_members": show_members, "variable": variable}
    he.plot(**kws)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he.plot(**kws)
    he = he.add_observations(reconstruction_ds_1d)
    he.plot(**kws)
    he = he.add_observations(observations_ds_1d)
    he.plot(**kws)
