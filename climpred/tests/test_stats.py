import dask
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.bootstrap import dpp_threshold
from climpred.stats import decorrelation_time, dpp

try:
    from climpred.bootstrap import varweighted_mean_period_threshold
    from climpred.stats import varweighted_mean_period
except ImportError:
    pass

from . import requires_xrft

ITERATIONS = 2


@pytest.mark.parametrize("chunk", (True, False))
def test_dpp(PM_da_control_3d, chunk):
    """Check for positive diagnostic potential predictability in NA SST."""
    res = dpp(PM_da_control_3d, chunk=chunk)
    assert res.mean() > 0


@requires_xrft
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


@requires_xrft
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


@requires_xrft
@pytest.mark.parametrize("step", [1, 2, -1])
@pytest.mark.parametrize(
    "func",
    [dpp, varweighted_mean_period, decorrelation_time],
)
def test_stats_functions_dask_chunks(PM_da_control_3d, func, step):
    """Check whether selected stats functions be chunked and computed along other
    dim."""
    dim = "time"
    for chunk_dim in PM_da_control_3d.isel({dim: 0}).dims:
        control_chunked = PM_da_control_3d.chunk({chunk_dim: step})
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
