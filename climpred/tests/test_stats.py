import dask
import numpy as np
import pytest
import xarray as xr
from scipy.signal import correlate
from xarray.testing import assert_allclose

from climpred.bootstrap import dpp_threshold, varweighted_mean_period_threshold
from climpred.exceptions import DimensionError
from climpred.stats import (
    autocorr,
    corr,
    decorrelation_time,
    dpp,
    rm_trend,
    varweighted_mean_period,
)

ITERATIONS = 5


def test_rm_trend_missing_dim():
    with pytest.raises(DimensionError) as excinfo:
        rm_trend(xr.DataArray([0, 1, 2]), dim='non_existent')
        assert "Input dim, 'non_existent'" in excinfo.value.message


def test_rm_trend_1d_dataarray(two_dim_da):
    one_dim_da = two_dim_da.isel(row=0)
    one_dim_da_dt = rm_trend(one_dim_da, 'col')

    assert one_dim_da_dt.shape == (5,)
    # should have all values near 0 after detrending because it's linear
    # but because of floating point precision, may not be 0
    assert (one_dim_da_dt <= 1e-5).sum() == 5


def test_rm_trend_1d_dataarray_interp_nan(two_dim_da):
    one_dim_da = two_dim_da.isel(row=0)
    one_dim_da[:3] = np.nan
    one_dim_da_dt = rm_trend(one_dim_da, 'col')

    assert one_dim_da_dt.shape == (5,)
    assert np.isnan(one_dim_da_dt[:3]).all()  # should be replaced with nan
    # since it's bfill, the linear trend no longer holds
    assert (one_dim_da_dt <= 1e-5).sum() <= 20


def test_rm_trend_2d_dataarray(two_dim_da):
    two_dim_da_dt = rm_trend(two_dim_da, 'col')

    assert two_dim_da_dt.shape == (4, 5)
    # should have all values near 0 after detrending because it's linear
    # but because of floating point precision, may not be 0
    assert (two_dim_da_dt <= 1e-5).sum() == 20


def test_rm_trend_2d_dataarray_interp_nan(two_dim_da):
    two_dim_da[2, :] = np.nan
    two_dim_da_dt = rm_trend(two_dim_da, 'col')

    assert two_dim_da_dt.shape == (4, 5)
    # should have 15 values (5 NaNs) near 0 after detrending because it's
    # linear. But because of floating point precision, it may not be 0.
    assert (two_dim_da_dt <= 1e-5).sum() == 15
    assert np.isnan(two_dim_da_dt[2]).all()  # should be replaced with nan


def test_rm_trend_3d_dataset(multi_dim_ds):
    multi_dim_ds_dt = rm_trend(multi_dim_ds)

    # originally values were ~270 K, after detrending between -50 and 50
    assert multi_dim_ds_dt['air'].shape == (2920, 25, 53)
    assert multi_dim_ds_dt['airx2'].shape == (2920, 25, 53)
    assert float(multi_dim_ds_dt['air'].min()) > -50
    assert float(multi_dim_ds_dt['air'].max()) < 50


def test_rm_trend_3d_dataset_dim_order(multi_dim_ds):
    multi_dim_ds = multi_dim_ds.transpose('lon', 'time', 'lat')
    multi_dim_ds_dt = rm_trend(multi_dim_ds)

    # ensure the dims are back in its original state
    assert list(multi_dim_ds_dt['air'].dims) == ['lon', 'time', 'lat']
    assert list(multi_dim_ds_dt['airx2'].dims) == ['lon', 'time', 'lat']


@pytest.mark.parametrize('chunk', (True, False))
def test_dpp(PM_da_control_3d, chunk):
    """Check for positive diagnostic potential predictability in NA SST."""
    res = dpp(PM_da_control_3d, chunk=chunk)
    assert res.mean() > 0


@pytest.mark.parametrize(
    'func', (varweighted_mean_period, decorrelation_time, autocorr)
)
def test_potential_predictability_likely(PM_da_control_3d, func):
    """Check for positive diagnostic potential predictability in NA SST."""
    res = func(PM_da_control_3d)
    assert res.mean() > 0


def test_autocorr(PM_da_control_3d):
    """Check autocorr results with scipy."""
    ds = PM_da_control_3d.isel(x=5, y=5)
    actual = autocorr(ds)
    expected = correlate(ds, ds)
    np.allclose(actual, expected)


def test_corr(PM_da_control_3d):
    """Check autocorr results with scipy."""
    ds = PM_da_control_3d.isel(x=5, y=5)
    lag = 1
    actual = corr(ds, ds, lag=lag)
    expected = correlate(ds[:-lag], ds[lag:])
    np.allclose(actual, expected)


def test_bootstrap_dpp_sig50_similar_dpp(PM_da_control_3d):
    sig = 50
    actual = dpp_threshold(PM_da_control_3d, iterations=ITERATIONS, sig=sig).drop_vars(
        'quantile'
    )
    expected = dpp(PM_da_control_3d)
    xr.testing.assert_allclose(actual, expected, atol=0.5, rtol=0.5)


def test_bootstrap_vwmp_sig50_similar_vwmp(PM_da_control_3d):
    sig = 50
    actual = varweighted_mean_period_threshold(
        PM_da_control_3d, iterations=ITERATIONS, sig=sig
    ).drop_vars('quantile')
    expected = varweighted_mean_period(PM_da_control_3d)
    xr.testing.assert_allclose(actual, expected, atol=2, rtol=0.5)


def test_bootstrap_func_multiple_sig_levels(PM_da_control_3d):
    sig = [5, 95]
    actual = dpp_threshold(PM_da_control_3d, iterations=ITERATIONS, sig=sig)
    assert actual['quantile'].size == len(sig)
    assert (actual.isel(quantile=0).values <= actual.isel(quantile=1)).all()


@pytest.mark.parametrize(
    'func',
    (
        dpp,
        autocorr,
        varweighted_mean_period,
        pytest.param(decorrelation_time, marks=pytest.mark.xfail(reason='some bug')),
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
    'func',
    [
        dpp,
        autocorr,
        varweighted_mean_period,
        pytest.param(
            decorrelation_time, marks=pytest.mark.xfail(reason='some chunking bug'),
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


def test_varweighted_mean_period_dim(PM_da_control_3d):
    """Test varweighted_mean_period for different dims."""
    for d in PM_da_control_3d.dims:
        # single dim
        varweighted_mean_period(PM_da_control_3d, dim=d)
        # all but one dim
        di = [di for di in PM_da_control_3d.dims if di != d]
        varweighted_mean_period(PM_da_control_3d, dim=di)


@pytest.mark.xfail(reason='p value not aligned in the two functions.')
def test_corr_autocorr(PM_da_control_3d):
    res1 = corr(PM_da_control_3d, PM_da_control_3d, lag=1, return_p=True)
    res2 = autocorr(PM_da_control_3d, return_p=True)
    for i in [0, 1]:
        assert_allclose(res1[i], res2[i])
