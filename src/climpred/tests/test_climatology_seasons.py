"""Perfect-model seasonal climatology uses categorical season labels."""

import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble, set_options
from climpred.comparisons import __e2c, __m2c, __m2e, __m2m
from climpred.metrics import __mse


@pytest.mark.parametrize("periods", [3, 24])
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("comparison", ["m2e", "m2c", "m2m", "e2c"])
@pytest.mark.parametrize("seasonality", ["season", "month", "dayofyear"])
def test_perfect_climatology_group_labels(comparison, seasonality, dataset, periods):
    init = xr.date_range(
        "2000-01-01", periods=periods, freq="MS", calendar="noleap", use_cftime=True
    )
    data = xr.DataArray(
        np.random.default_rng(1).normal(size=(periods, 3, 2)),
        dims=("init", "member", "lead"),
        coords={"init": init, "member": [1, 2, 3], "lead": [1, 2]},
        name="temperature",
    )
    data.lead.attrs["units"] = "months"
    if dataset:
        data = data.to_dataset()
    ensemble = PerfectModelEnsemble(data)
    dim = ["init"] if comparison == "e2c" else ["init", "member"]
    with set_options(seasonality=seasonality):
        actual = ensemble.verify(
            metric="mse", comparison=comparison, dim=dim, reference="climatology"
        ).sel(skill="climatology", drop=True)
    comparison_class = {"e2c": __e2c, "m2c": __m2c, "m2e": __m2e, "m2m": __m2m}[
        comparison
    ]
    forecast, verif = comparison_class.function(
        ensemble.get_initialized(), metric=__mse
    )
    # Independent exact group lookup, not nearest on categorical strings.
    means = verif.groupby(f"init.{seasonality}").mean()
    expected_forecast = means.sel(
        {seasonality: getattr(forecast.init.dt, seasonality)}
    ).drop_vars(seasonality)
    expected = ((expected_forecast - verif) ** 2).mean(dim)
    if "forecast_member" in expected.dims:
        expected = expected.mean("forecast_member")
    xr.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("comparison", ["m2e", "m2c", "m2m", "e2c"])
def test_seasonal_bootstrap_climatology(comparison):
    init = xr.date_range(
        "2000-01-01", periods=24, freq="MS", calendar="noleap", use_cftime=True
    )
    data = xr.DataArray(
        np.random.default_rng(2).normal(size=(24, 3, 2)),
        dims=("init", "member", "lead"),
        coords={"init": init, "member": [1, 2, 3], "lead": [1, 2]},
        name="temperature",
    )
    data.lead.attrs["units"] = "months"
    ensemble = PerfectModelEnsemble(data)
    dim = ["init"] if comparison == "e2c" else ["init", "member"]
    with set_options(seasonality="season"):
        result = ensemble.bootstrap(
            metric="mse",
            comparison=comparison,
            dim=dim,
            reference="climatology",
            resample_dim="member",
            iterations=3,
        )
        expected = ensemble.verify(
            metric="mse", comparison=comparison, dim=dim, reference="climatology"
        ).sel(skill="climatology", drop=True)
    actual = result.sel(skill="climatology", results="verify skill", drop=True)
    xr.testing.assert_allclose(actual, expected)
    assert np.isfinite(result.sel(skill="climatology").temperature).all()
