import pytest
import xarray as xr

from climpred.stats import rm_poly
from climpred.testing import assert_PredictionEnsemble

xr.set_options(display_style="text")


def test_PredictionEnsemble_raises_error(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble raises error."""
    with pytest.raises(TypeError):
        hindcast_hist_obs_1d.smooth({"lead": 2.5})  # expects int


def test_PredictionEnsemble_raises_warning(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble raises warning."""
    with pytest.warns(UserWarning):
        hindcast_hist_obs_1d.map(rm_poly, dim="init", deg=2)


def test_PredictionEnsemble_xr_calls(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble passes through xarray calls."""
    assert (
        hindcast_hist_obs_1d.isel(lead=[1, 2])
        .get_initialized()
        .identical(hindcast_hist_obs_1d.get_initialized().isel(lead=[1, 2]))
    )


def test_PredictionEnsemble_map_dim_or(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble allows dim0_or_dim1 as kwargs without UserWarning."""
    with pytest.warns(None):  # no warnings
        he_or = hindcast_hist_obs_1d.map(rm_poly, dim="init_or_time", deg=2)
    assert he_or != hindcast_hist_obs_1d

    with pytest.warns(UserWarning) as record:  # triggers warnings
        he_chained = hindcast_hist_obs_1d.map(rm_poly, dim="init", deg=2).map(
            rm_poly, dim="time", deg=2
        )
    assert he_chained != hindcast_hist_obs_1d

    assert len(record) == 3  # for init, uninit and obs
    for r in record:
        assert "Error due to " in str(r.message)

    assert_PredictionEnsemble(he_or, he_chained)


def test_PredictionEnsemble_map_dim_or_fails_if_both_dims_in_dataset(
    hindcast_hist_obs_1d,
):
    """Tests PredictionEnsemble dim0_or_dim1 in kwargs fails if both in all dataset."""
    with pytest.raises(ValueError, match="cannot be both in"):
        hindcast_hist_obs_1d.map(rm_poly, dim="init_or_lead", deg=2)
