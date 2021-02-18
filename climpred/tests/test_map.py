import pytest
from esmtools.stats import rm_poly


def test_PredictionEnsemble_raises_error(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble raises error."""
    with pytest.raises(TypeError):
        hindcast_hist_obs_1d.smooth({"lead": 2.5})  # expects int


def test_PredictionEnsemble_raises_warning(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble raises warning."""
    with pytest.warns(UserWarning):
        hindcast_hist_obs_1d.map(rm_poly, dim="init", order=2)


def test_PredictionEnsemble_xr_calls(hindcast_hist_obs_1d):
    """Tests that PredictionEnsemble passes through xarray calls."""
    assert (
        hindcast_hist_obs_1d.isel(lead=[1, 2])
        .get_initialized()
        .identical(hindcast_hist_obs_1d.get_initialized().isel(lead=[1, 2]))
    )
