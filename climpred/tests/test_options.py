import pytest

import climpred


@pytest.mark.xfail(
    reason="not properly implemented see https://github.com/pangeo-data/climpred/issues/605"
)
@pytest.mark.parametrize(
    "cross_validate", [False, pytest.param(True, marks=pytest.mark.xfail)]
)
def test_seasonality_remove_bias(hindcast_recon_1d_dm, cross_validate):
    """Test the climpred.set_option(seasonality) changes bias reduction. Currently fails for cross_validate bias reduction."""
    hindcast = hindcast_recon_1d_dm
    hindcast._datasets["initialized"] = (
        hindcast.get_initialized().resample(init="1MS").interpolate("linear")
    )
    print(hindcast.get_initialized().init.to_index())
    print(hindcast.get_observations().time.to_index())
    alignment = "maximize"
    kw = {
        "metric": "mse",
        "comparison": "e2o",
        "dim": "init",
        "alignment": alignment,
        "reference": None,
    }
    print(hindcast.get_initialized().init.dt.dayofyear.values)
    with climpred.set_options(seasonality="dayofyear"):
        dayofyear_seasonality = hindcast.remove_bias(
            alignment=alignment, cross_validate=cross_validate
        )
    with climpred.set_options(seasonality="weekofyear"):
        weekofyear_seasonality = hindcast.remove_bias(
            alignment=alignment, cross_validate=cross_validate
        )
    assert not weekofyear_seasonality.get_initialized().identical(
        dayofyear_seasonality.get_initialized()
    )
    assert not weekofyear_seasonality.verify(**kw).identical(
        dayofyear_seasonality.verify(**kw)
    )


def test_seasonality_climatology(hindcast_recon_1d_dm):
    """Test the climpred.set_option(seasonality) changes climatology."""
    hindcast = hindcast_recon_1d_dm
    alignment = "maximize"
    kw = {
        "metric": "mse",
        "comparison": "e2o",
        "dim": "init",
        "alignment": alignment,
        "reference": "climatology",
    }
    with climpred.set_options(seasonality="dayofyear"):
        dayofyear_seasonality = hindcast.verify(**kw).sel(skill="climatology")
    with climpred.set_options(seasonality="month"):
        month_seasonality = hindcast.verify(**kw).sel(skill="climatology")
    assert not month_seasonality.identical(dayofyear_seasonality)
