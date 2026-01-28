"""Test graphics.py and PredictionEnsemble.plot()."""

import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.checks import DimensionError
from climpred.graphics import plot_bootstrapped_skill_over_leadyear

from . import requires_matplotlib, requires_nc_time_axis

# Set matplotlib to non-interactive backend for testing to avoid resource leaks
import matplotlib

matplotlib.use("Agg")

ITERATIONS = 3


@pytest.fixture(autouse=True)
def cleanup_matplotlib_figures():
    """Automatically clean up matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@requires_matplotlib
def test_PerfectModelEnsemble_plot_bootstrapped_skill_over_leadyear(
    synthetic_pm_1d_small,
):
    """
    Checks plots from PerfectModelEnsemble.bootstrap().
    """
    res = synthetic_pm_1d_small.bootstrap(
        metric="pearson_r",
        iterations=ITERATIONS * 100,
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
def test_PerfectModelEnsemble_plot(synthetic_pm_1d_small, variable, show_members, cmap):
    """Test PredictionEnsemble.plot()."""
    pm = synthetic_pm_1d_small
    kws = {"cmap": cmap, "show_members": show_members, "variable": variable}
    pm.plot(**kws)
    pm = pm.generate_uninitialized()
    pm.plot(**kws)


@requires_matplotlib
def test_PerfectModelEnsemble_plot_fails_3d(synthetic_pm_3d_small):
    """Test PredictionEnsemble.plot()."""
    pm = synthetic_pm_3d_small
    with pytest.raises(DimensionError) as excinfo:
        pm.plot()
    assert "does not allow dimensions other" in str(excinfo.value)


@requires_matplotlib
@pytest.mark.parametrize("x", ["time", "init"])
@pytest.mark.parametrize("show_members", [True, False])
@pytest.mark.parametrize("variable", ["SST", None])
def test_PredictionEnsemble_plot(
    synthetic_hindcast_1d_small, synthetic_uninitialized_1d_small, variable, show_members, x
):
    """Test PredictionEnsemble.plot()."""
    he = synthetic_hindcast_1d_small
    kws = {"show_members": show_members, "variable": variable, "x": x}
    he.plot(**kws)
    # Ensure synthetic_uninitialized_1d_small has the same frequencies as initialized
    # for plotting to work smoothly if needed, but synthetic_uninitialized_1d_small is YS
    # and he is MS. Let's make it MS for this test.
    uninit = synthetic_uninitialized_1d_small.rename("SST")
    he = he.add_uninitialized(uninit)
    he.plot(**kws)

    if x == "time":
        pm = PerfectModelEnsemble(he.get_initialized())
        pm.plot(**kws)
        pm = pm.add_control(he.get_observations())
        pm.plot(**kws)


@requires_matplotlib
@requires_nc_time_axis
@pytest.mark.parametrize("alignment", ["same_inits", None])
@pytest.mark.parametrize("return_xr", [False, True])
def test_HindcastEnsemble_plot_alignment(
    synthetic_hindcast_1d_small, alignment, return_xr
):
    """Test HindcastEnsemble.plot_alignment()"""
    import matplotlib
    import xarray as xr

    if return_xr:
        assert isinstance(
            synthetic_hindcast_1d_small.plot_alignment(
                alignment=alignment, return_xr=return_xr
            ),
            xr.DataArray,
        )
    else:
        assert isinstance(
            synthetic_hindcast_1d_small.plot_alignment(
                alignment=alignment, return_xr=return_xr
            ),
            (xr.plot.facetgrid.FacetGrid, matplotlib.collections.QuadMesh),
        )
