import pytest

import climpred


# @pytest.mark.xfail(
#    reason="not properly implemented see https://github.com/pangeo-data/climpred/issues/605"
# )
@pytest.mark.parametrize(
    "cross_validate", [False, pytest.param(True, marks=pytest.mark.xfail)]
)
def test_seasonality_remove_bias(hindcast_recon_1d_dm, cross_validate):
    """
    Test the climpred.set_option(seasonality) changes bias reduction.

    Currently fails for cross_validate bias reduction.
    """
    hindcast = hindcast_recon_1d_dm
    hindcast._datasets["initialized"] = (
        hindcast.get_initialized().resample(init="1MS").interpolate("linear")
    )
    print(
        hindcast.get_initialized().coords["init"][:4],
        hindcast.coords["lead"],
        hindcast.get_observations().coords["time"][:4],
    )

    alignment = "maximize"
    kw = {
        "metric": "mse",
        "comparison": "e2o",
        "dim": "init",
        "alignment": alignment,
        "reference": None,
    }

    with climpred.set_options(seasonality="dayofyear"):
        dayofyear_seasonality = hindcast.remove_bias(
            alignment=alignment, cross_validate=cross_validate
        )
    print("\n" * 4)
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


@pytest.mark.parametrize("option_bool", [False, True])
def test_option_warn_for_failed_PredictionEnsemble_xr_call(
    hindcast_recon_1d_dm, option_bool
):
    with climpred.set_options(warn_for_failed_PredictionEnsemble_xr_call=option_bool):
        with pytest.warns(None if not option_bool else UserWarning) as record:
            hindcast_recon_1d_dm.sel(lead=[1, 2])
        if not option_bool:
            assert len(record) == 0, print(record[0])


@pytest.mark.parametrize("option_bool", [False, True])
def test_climpred_warnings(hindcast_recon_1d_dm, option_bool):
    with climpred.set_options(warn_for_failed_PredictionEnsemble_xr_call=True):
        with climpred.set_options(climpred_warnings=option_bool):
            with pytest.warns(UserWarning if option_bool else None) as record:
                hindcast_recon_1d_dm.sel(lead=[1, 2])
            if not option_bool:
                assert len(record) == 0, print(record[0])
