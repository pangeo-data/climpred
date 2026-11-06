"""Default resampling must work when non-resampled Dask chunks have size one."""

import numpy as np
import pytest
import xarray as xr

from climpred import set_options
from climpred.bootstrap import _get_resample_func, _resample_iterations, dpp_threshold


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("site_chunks", [1, 2])
def test_default_threshold_computes_with_singleton_spatial_chunks(dataset, site_chunks):
    control = xr.DataArray(
        np.sin(np.arange(30)[None, :] / 2) + np.array([0, 2])[:, None],
        dims=("site", "time"),
        coords={"site": ["west", "east"], "time": range(30)},
        name="temperature",
    ).chunk({"time": -1, "site": site_chunks})
    if dataset:
        control = control.to_dataset()
    with set_options(resample_iterations_func="resample_iterations"):
        np.random.seed(23)
        expected = dpp_threshold(control, iterations=3, m=3).compute()
    with set_options(resample_iterations_func="default"):
        np.random.seed(23)
        actual = dpp_threshold(control, iterations=3, m=3).compute()
    xr.testing.assert_allclose(actual, expected)


def test_default_route_avoids_singleton_chunk_squeeze():
    data = xr.DataArray(np.ones((2, 30)), dims=("site", "time")).chunk(
        {"site": 1, "time": -1}
    )
    with set_options(resample_iterations_func="default"):
        assert _get_resample_func(data) is _resample_iterations


def test_explicit_idx_choice_is_not_overridden():
    from climpred.bootstrap import _resample_iterations_idx

    data = xr.DataArray(np.ones((2, 30)), dims=("site", "time")).chunk(
        {"site": 1, "time": -1}
    )
    with set_options(resample_iterations_func="resample_iterations_idx"):
        assert _get_resample_func(data) is _resample_iterations_idx


def test_default_idx_route_retained_without_singleton_chunks():
    from climpred.bootstrap import _resample_iterations_idx

    data = xr.DataArray(np.ones((4, 30)), dims=("site", "time")).chunk(
        {"site": 2, "time": -1}
    )
    with set_options(resample_iterations_func="default"):
        assert _get_resample_func(data) is _resample_iterations_idx


def test_uneven_final_spatial_chunk_also_uses_safe_default():
    data = xr.DataArray(np.arange(90).reshape(3, 30), dims=("site", "time")).chunk(
        {"site": 2, "time": -1}
    )
    with set_options(resample_iterations_func="default"):
        assert _get_resample_func(data) is _resample_iterations
        assert dpp_threshold(data, iterations=3, m=3).compute().sizes == {"site": 3}
