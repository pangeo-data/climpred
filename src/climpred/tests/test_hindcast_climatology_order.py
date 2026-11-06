import numpy as np
import pytest
import xarray as xr
from climpred import HindcastEnsemble, set_options
from climpred.reference import climatology


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("seasonality", ["season", "month", "dayofyear"])
@pytest.mark.parametrize("periods", [3, 24])
def test_direct_climatology_group_lookup(dataset, seasonality, periods):
    time = xr.date_range(
        "1999-01-01", periods=48, freq="MS", calendar="noleap", use_cftime=True
    )
    dates = time[12 : 12 + periods]
    obs = xr.DataArray(
        np.arange(48.0) ** 2, dims="time", coords={"time": time}, name="a"
    )
    if dataset:
        obs = obs.to_dataset()
    init = xr.DataArray(dates, dims="time", coords={"time": dates})
    with set_options(seasonality=seasonality):
        actual, verif = climatology(obs, {1: init}, {1: dates}, 1)
    means = obs.groupby("time." + seasonality).mean()
    expected = means.sel({seasonality: getattr(init.time.dt, seasonality)}).drop_vars(
        seasonality
    )
    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_equal(verif, obs.sel(time=dates))


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("alignment", ["same_inits", "same_verifs", "maximize"])
@pytest.mark.parametrize("metric", ["mse", "mae", "pearson_r"])
def test_eager_hindcast_seasonal_reference(alignment, metric, bootstrap):
    rng = np.random.default_rng(4)
    init = xr.date_range(
        "2000", periods=24, freq="MS", calendar="noleap", use_cftime=True
    )
    time = xr.date_range(
        "1999", periods=60, freq="MS", calendar="noleap", use_cftime=True
    )
    f = xr.DataArray(
        rng.normal(size=(24, 4, 2)),
        dims=("init", "member", "lead"),
        coords={"init": init, "member": range(4), "lead": [1, 2]},
        name="a",
    )
    f.lead.attrs["units"] = "months"
    obs = xr.DataArray(
        rng.normal(size=60), dims="time", coords={"time": time}, name="a"
    )
    with set_options(seasonality="season"):
        ensemble = HindcastEnsemble(f).add_observations(obs)
        kwargs = dict(
            metric=metric,
            comparison="e2o",
            dim="init",
            alignment=alignment,
            reference="climatology",
        )
        result = (
            ensemble.bootstrap(iterations=3, resample_dim="member", **kwargs)
            if bootstrap
            else ensemble.verify(**kwargs)
        )
    assert np.isfinite(result.sel(skill="climatology").a).all()


@pytest.mark.parametrize("months", [[1, 4, 7, 10], [1, 7], [4, 7, 10], [10, 7, 4, 1]])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_seasons_preserve_nearest_integer_policy(months, reverse):
    times = xr.CFTimeIndex(
        [
            xr.date_range(
                f"2000-{m:02d}-01", periods=1, calendar="noleap", use_cftime=True
            )[0]
            for m in months
        ]
    )
    obs = xr.DataArray(
        np.asarray(months, dtype=float), dims="time", coords={"time": times}, name="a"
    )
    dates = xr.date_range(
        "2001-01-01", periods=12, freq="MS", calendar="noleap", use_cftime=True
    )
    if reverse:
        dates = dates[::-1]
    init = xr.DataArray(dates, dims="time", coords={"time": dates})
    codes = {"DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
    grouped = obs.groupby("time.season").mean()
    values = {
        codes[str(k)]: float(grouped.sel(season=k)) for k in grouped.season.values
    }
    # pandas nearest integer policy picks larger code in a tie, not circular distance.
    expected = [
        values[min(values, key=lambda k: (abs(k - codes[str(s)]), -k))]
        for s in init.time.dt.season.values
    ]
    with set_options(seasonality="season"):
        actual, _ = climatology(obs, {1: init}, {1: times}, 1)
    xr.testing.assert_allclose(
        actual,
        xr.DataArray(expected, dims="time", coords={"time": dates}, name="a").sortby(
            "time"
        ),
    )
