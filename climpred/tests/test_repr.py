import pytest
import xarray as xr
from IPython.display import display

from climpred.classes import HindcastEnsemble, PerfectModelEnsemble


@pytest.mark.parametrize("display_style", ("html", "text"))
def test_repr_PM(PM_da_initialized_1d, PM_da_control_1d, display_style):
    """Test html and text repr."""
    with xr.set_options(display_style=display_style):
        pm = PerfectModelEnsemble(PM_da_initialized_1d)
        display(pm)
        pm = pm.add_control(PM_da_control_1d)
        display(pm)
        pm = pm.generate_uninitialized()
        display(pm)


@pytest.mark.parametrize("display_style", ("html", "text"))
def test_repr_HC(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    observations_ds_1d,
    display_style,
):
    """Test html repr."""
    with xr.set_options(display_style=display_style):
        he = HindcastEnsemble(hind_ds_initialized_1d)
        display(he)
        he = he.add_uninitialized(hist_ds_uninitialized_1d)
        display(he)
        he = he.add_observations(observations_ds_1d)
        display(he)
        # no uninit
        he = HindcastEnsemble(hind_ds_initialized_1d)
        he = he.add_observations(observations_ds_1d)
        display(he)
