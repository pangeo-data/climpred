"""Threshold resampling follows the documented with-replacement contract."""

import numpy as np
import pytest
import xarray as xr
from xskillscore.core.resampling import resample_iterations, resample_iterations_idx

from climpred import set_options
from climpred.bootstrap import _bootstrap_func, dpp_threshold
from climpred.stats import dpp


@pytest.mark.parametrize(
    "resampler", ["resample_iterations", "resample_iterations_idx"]
)
@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("chunked", [False, True])
def test_public_dpp_threshold_matches_replacement_oracle(resampler, kind, chunked):
    ds = xr.DataArray(
        np.sin(np.arange(40) / 3) + np.arange(40) / 20,
        dims="time",
        coords={"time": np.arange(40)},
        name="v",
    )
    if kind == "dataset":
        ds = ds.to_dataset()
    if chunked:
        ds = ds.chunk({"time": -1})
    sampler = {
        "resample_iterations": resample_iterations,
        "resample_iterations_idx": resample_iterations_idx,
    }[resampler]
    state = np.random.get_state()
    try:
        np.random.seed(42)
        sampled = sampler(ds, 30, dim="time", replace=True)
        expected = dpp(sampled, m=5).quantile([0.05, 0.95], dim="iteration")
        np.random.seed(42)
        with set_options(resample_iterations_func=resampler):
            actual = dpp_threshold(ds, sig=[5, 95], iterations=30, m=5)
        xr.testing.assert_allclose(actual.compute(), expected.compute())
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize(
    "resampler", ["resample_iterations", "resample_iterations_idx"]
)
def test_documented_custom_statistic_has_non_degenerate_interval(resampler):
    ds = xr.DataArray([0.0, 1.0, 5.0, 9.0], dims="time")
    state = np.random.get_state()
    try:
        np.random.seed(42)
        with set_options(resample_iterations_func=resampler):
            actual = _bootstrap_func(
                lambda x: x.mean("time"), ds, "time", sig=[5, 95], iterations=100
            )
        assert actual.sel(quantile=0.05) < ds.mean()
        assert actual.sel(quantile=0.95) > ds.mean()
    finally:
        np.random.set_state(state)
