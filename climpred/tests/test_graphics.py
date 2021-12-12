"""Test graphics.py and PredictionEnsemble.plot()."""
import pytest

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.checks import DimensionError
from climpred.graphics import plot_bootstrapped_skill_over_leadyear

from . import requires_matplotlib, requires_nc_time_axis

ITERATIONS = 3


@requires_matplotlib
def test_PerfectModelEnsemble_plot_bootstrapped_skill_over_leadyear(
    perfectModelEnsemble_initialized_control,
):
    """
    Checks plots from PerfectModelEnsemble.bootstrap().
    """
    res = perfectModelEnsemble_initialized_control.bootstrap(
        metric="pearson_r",
        iterations=ITERATIONS,
        reference=["uninitialized", "persistence"],
        comparison="m2e",
        dim=["init", "member"],
    )
    res_ax = plot_bootstrapped_skill_over_leadyear(res)
    assert res_ax is not None


@requires_matplotlib
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


@requires_matplotlib
def test_PerfectModelEnsemble_plot_fails_3d(PM_ds_initialized_3d):
    """Test PredictionEnsemble.plot()."""
    pm = PerfectModelEnsemble(PM_ds_initialized_3d)
    with pytest.raises(DimensionError) as excinfo:
        pm.plot()
    assert "does not allow dimensions other" in str(excinfo.value)


@requires_matplotlib
@pytest.mark.parametrize("x", ["time", "init"])
@pytest.mark.parametrize("show_members", [True, False])
@pytest.mark.parametrize("variable", ["SST", None])
def test_PredictionEnsemble_plot(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    reconstruction_ds_1d,
    observations_ds_1d,
    variable,
    show_members,
    x,
):
    """Test PredictionEnsemble.plot()."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    kws = {"show_members": show_members, "variable": variable, "x": x}
    he.plot(**kws)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he.plot(**kws)
    he = he.add_observations(reconstruction_ds_1d)
    he.plot(**kws)
    he = he.add_observations(observations_ds_1d)
    he.plot(**kws)

    if x == "time":
        pm = PerfectModelEnsemble(hind_ds_initialized_1d)
        pm.plot(**kws)
        pm = pm.add_control(hist_ds_uninitialized_1d.isel(member=0, drop=True))
        pm.plot(**kws)


@requires_matplotlib
@requires_nc_time_axis
@pytest.mark.parametrize("alignment", ["same_inits", None])
@pytest.mark.parametrize("return_xr", [False, True])
def test_HindcastEnsemble_plot_alignment(hindcast_hist_obs_1d, alignment, return_xr):
    """Test HindcastEnsemble.plot_alignment()"""
    import matplotlib
    import xarray as xr

    if return_xr:
        assert isinstance(
            hindcast_hist_obs_1d.plot_alignment(
                alignment=alignment, return_xr=return_xr
            ),
            xr.DataArray,
        )
    else:
        assert isinstance(
            hindcast_hist_obs_1d.plot_alignment(
                alignment=alignment, return_xr=return_xr
            ),
            (xr.plot.facetgrid.FacetGrid, matplotlib.collections.QuadMesh),
        )
