import pytest
import xarray as xr

from climpred.classes import HindcastEnsemble, PerfectModelEnsemble


def assert_repr(pe, display_style):
    if display_style == "text":
        repr_str = pe.__repr__()
    elif display_style == "html":
        repr_str = pe._repr_html_()
    assert "Ensemble" in repr_str
    if display_style == "text":
        assert "</pre>" not in repr_str
    elif display_style == "html":
        assert "icon" in repr_str


@pytest.mark.parametrize("display_style", ("html", "text"))
def test_repr_PerfectModelEnsemble(
    PM_ds_initialized_1d, PM_ds_control_1d, display_style
):
    """Test html and text repr."""
    with xr.set_options(display_style=display_style):
        pm = PerfectModelEnsemble(PM_ds_initialized_1d)
        assert_repr(pm, display_style)
        pm = pm.add_control(PM_ds_control_1d)
        assert_repr(pm, display_style)
        pm = pm.generate_uninitialized()
        assert_repr(pm, display_style)


@pytest.mark.parametrize("display_style", ("html", "text"))
def test_repr_HindcastEnsemble(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    observations_ds_1d,
    display_style,
):
    """Test html repr."""
    with xr.set_options(display_style=display_style):
        he = HindcastEnsemble(hind_ds_initialized_1d)
        assert_repr(he, display_style)
        he = he.add_uninitialized(hist_ds_uninitialized_1d)
        assert_repr(he, display_style)
        he = he.add_observations(observations_ds_1d)
        assert_repr(he, display_style)
        # no uninit
        he = HindcastEnsemble(hind_ds_initialized_1d)
        he = he.add_observations(observations_ds_1d)
        assert_repr(he, display_style)
