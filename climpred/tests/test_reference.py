import pytest

from climpred import set_options


@pytest.mark.parametrize("seasonality", ["month", "season", "dayofyear", "weekofyear"])
@pytest.mark.parametrize("reference", ["persistence", "climatology", "uninitialized"])
def test_HindcastEnsemble_verify_reference(
    hindcast_hist_obs_1d, seasonality, reference
):
    with set_options(seasonality=seasonality):
        hindcast_hist_obs_1d.verify(
            metric="mse",
            comparison="e2o",
            dim="init",
            alignment="same_verifs",
            reference=reference,
        )
