"""Threshold wrappers must pass their requested time dimension to statistics."""

import numpy as np
import pytest
import xarray as xr

from climpred import set_options
from climpred.bootstrap import dpp_threshold, varweighted_mean_period_threshold


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("dpp_chunk", [False, True])
def test_dpp_threshold_custom_dimension_matches_time(dataset, chunk, dpp_chunk):
    control = xr.DataArray(
        np.sin(np.arange(30) / 2) + np.arange(30) / 15,
        dims="time",
        coords={"time": np.arange(30)},
        name="temperature",
    )
    if dataset:
        control = control.to_dataset()
    if chunk:
        control = control.expand_dims(site=["west", "east"]).chunk(
            {"time": -1, "site": 1}
        )
    # The vectorized idx backend has an independent eager-axis bug for small
    # Dask spatial arrays. Exercise the supported iterative backend explicitly.
    with set_options(resample_iterations_func="resample_iterations"):
        np.random.seed(23)
        expected = dpp_threshold(
            control, sig=[5, 95], iterations=3, m=3, chunk=dpp_chunk
        )
        np.random.seed(23)
        actual = dpp_threshold(
            control.rename(time="year"),
            sig=[5, 95],
            iterations=3,
            dim="year",
            m=3,
            chunk=dpp_chunk,
        )
    xr.testing.assert_allclose(actual.compute(), expected.compute())


@pytest.mark.parametrize(
    "wrapper,argument",
    [(dpp_threshold, "dim"), (varweighted_mean_period_threshold, "time_dim")],
)
def test_threshold_wrapper_forwards_dimension_without_changing_sampling(
    monkeypatch, wrapper, argument
):
    import climpred.bootstrap as bootstrap

    control = xr.DataArray(np.arange(30), dims="year")
    calls = []
    sentinel = object()

    def observe(func, ds, resample_dim, **kwargs):
        calls.append((ds, resample_dim, kwargs))
        return sentinel

    monkeypatch.setattr(bootstrap, "_bootstrap_func", observe)
    # A statistic sentinel makes forwarding test independent of optional xrft.
    monkeypatch.setattr(bootstrap, "varweighted_mean_period", lambda ds: ds)
    assert wrapper(control, sig=80, iterations=7, **{argument: "year"}) is sentinel
    assert len(calls) == 1
    assert calls[0][0] is control
    assert calls[0][1] == "year"
    assert calls[0][2] == {"sig": 80, "iterations": 7, "dim": "year"}
