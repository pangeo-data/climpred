import dask
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.bootstrap import dpp_threshold, varweighted_mean_period_threshold
from climpred.stats import decorrelation_time, dpp, varweighted_mean_period

ITERATIONS = 5


@pytest.mark.parametrize("chunk", (True, False))
def test_dpp(PM_da_control_3d, chunk):
    """Check for positive diagnostic potential predictability in NA SST."""
    res = dpp(PM_da_control_3d, chunk=chunk)
    assert res.mean() > 0


@pytest.mark.parametrize("func", (varweighted_mean_period, decorrelation_time))
def test_potential_predictability_likely(PM_da_control_3d, func):
    """Check for positive diagnostic potential predictability in NA SST."""
    res = func(PM_da_control_3d)
    assert res.mean() > 0


def test_bootstrap_dpp_sig50_similar_dpp(PM_da_control_3d):
    sig = 50
    actual = dpp_threshold(PM_da_control_3d, iterations=ITERATIONS, sig=sig).drop_vars(
        "quantile"
    )
    expected = dpp(PM_da_control_3d)
    xr.testing.assert_allclose(actual, expected, atol=0.5, rtol=0.5)


def test_bootstrap_vwmp_sig50_similar_vwmp(PM_da_control_3d):
    sig = 50
    actual = varweighted_mean_period_threshold(
        PM_da_control_3d, iterations=ITERATIONS, sig=sig
    ).drop_vars("quantile")
    expected = varweighted_mean_period(PM_da_control_3d)
    xr.testing.assert_allclose(actual, expected, atol=2, rtol=0.5)


def test_bootstrap_func_multiple_sig_levels(PM_da_control_3d):
    sig = [5, 95]
    actual = dpp_threshold(PM_da_control_3d, iterations=ITERATIONS, sig=sig)
    assert actual["quantile"].size == len(sig)
    assert (actual.isel(quantile=0).values <= actual.isel(quantile=1)).all()


@pytest.mark.parametrize(
    "func",
    (
        dpp,
        varweighted_mean_period,
        pytest.param(decorrelation_time, marks=pytest.mark.xfail(reason="some bug")),
    ),
)
def test_stats_functions_dask_single_chunk(PM_da_control_3d, func):
    """Test stats functions when single chunk not along dim."""
    step = -1  # single chunk
    for chunk_dim in PM_da_control_3d.dims:
        control_chunked = PM_da_control_3d.chunk({chunk_dim: step})
        for dim in PM_da_control_3d.dims:
            if dim != chunk_dim:
                res_chunked = func(control_chunked, dim=dim)
                res = func(PM_da_control_3d, dim=dim)
                # check for chunks
                assert dask.is_dask_collection(res_chunked)
                assert res_chunked.chunks is not None
                # check for no chunks
                assert not dask.is_dask_collection(res)
                assert res.chunks is None
                # check for identical result
                assert_allclose(res, res_chunked.compute())


@pytest.mark.parametrize(
    "func",
    [
        dpp,
        varweighted_mean_period,
        pytest.param(
            decorrelation_time, marks=pytest.mark.xfail(reason="some chunking bug"),
        ),
    ],
)
def test_stats_functions_dask_many_chunks(PM_da_control_3d, func):
    """Check whether selected stats functions be chunked in multiple chunks and
     computed along other dim."""
    step = 1
    for chunk_dim in PM_da_control_3d.dims:
        control_chunked = PM_da_control_3d.chunk({chunk_dim: step})
        for dim in PM_da_control_3d.dims:
            if dim != chunk_dim and dim in control_chunked.dims:
                res_chunked = func(control_chunked, dim=dim)
                res = func(PM_da_control_3d, dim=dim)
                # check for chunks
                assert dask.is_dask_collection(res_chunked)
                assert res_chunked.chunks is not None
                # check for no chunks
                assert not dask.is_dask_collection(res)
                assert res.chunks is None
                # check for identical result
                assert_allclose(res, res_chunked.compute())
