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
from climpred.tutorial import load_dataset


@pytest.fixture
def two_dim_da():
    da = xr.DataArray(
        np.vstack(
            [
                np.arange(0, 5, 1.0),
                np.arange(0, 10, 2.0),
                np.arange(0, 40, 8.0),
                np.arange(0, 20, 4.0),
            ]
        ),
        dims=['row', 'col'],
    )
    return da


@pytest.fixture
def multi_dim_ds():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds = ds.assign(**{'airx2': ds['air'] * 2})
    return ds


@pytest.fixture
def control_3d_NA():
    """North Atlantic"""
    ds = load_dataset('MPI-control-3D')['tos'].isel(x=slice(110, 120), y=slice(50, 60))
    return ds


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
def test_dpp(control_3d_NA, chunk):
    """Check for positive diagnostic potential predictability in NA SST."""
    control = control_3d_NA
    res = dpp(control, chunk=chunk)
    assert res.mean() > 0


@pytest.mark.parametrize(
    'func', (varweighted_mean_period, decorrelation_time, autocorr)
)
def test_potential_predictability_likely(control_3d_NA, func):
    """Check for positive diagnostic potential predictability in NA SST."""
    control = control_3d_NA
    print(control.dims)
    res = func(control)
    assert res.mean() > 0


def test_autocorr(control_3d_NA):
    """Check autocorr results with scipy."""
    ds = control_3d_NA.isel(x=5, y=5)
    actual = autocorr(ds)
    expected = correlate(ds, ds)
    np.allclose(actual, expected)


def test_corr(control_3d_NA):
    """Check autocorr results with scipy."""
    ds = control_3d_NA.isel(x=5, y=5)
    lag = 1
    actual = corr(ds, ds, lag=lag)
    expected = correlate(ds[:-lag], ds[lag:])
    np.allclose(actual, expected)


def test_bootstrap_dpp_sig50_similar_dpp(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = 50
    actual = dpp_threshold(ds, bootstrap=bootstrap, sig=sig).drop_vars('quantile')
    expected = dpp(ds)
    xr.testing.assert_allclose(actual, expected, atol=0.5, rtol=0.5)


def test_bootstrap_vwmp_sig50_similar_vwmp(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = 50
    actual = varweighted_mean_period_threshold(
        ds, bootstrap=bootstrap, sig=sig
    ).drop_vars('quantile')
    expected = varweighted_mean_period(ds)
    xr.testing.assert_allclose(actual, expected, atol=2, rtol=0.5)


def test_bootstrap_func_multiple_sig_levels(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = [5, 95]
    actual = dpp_threshold(ds, bootstrap=bootstrap, sig=sig)
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
def test_stats_functions_dask_single_chunk(control_3d_NA, func):
    """Test stats functions when single chunk not along dim."""
    step = -1  # single chunk
    for chunk_dim in control_3d_NA.dims:
        control_chunked = control_3d_NA.chunk({chunk_dim: step})
        for dim in control_3d_NA.dims:
            if dim != chunk_dim:
                res_chunked = func(control_chunked, dim=dim)
                res = func(control_3d_NA, dim=dim)
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
            decorrelation_time, marks=pytest.mark.xfail(reason='some chunking bug')
        ),
    ],
)
def test_stats_functions_dask_many_chunks(control_3d_NA, func):
    """Check whether selected stats functions be chunked in multiple chunks and
     computed along other dim."""
    step = 1
    for chunk_dim in control_3d_NA.dims:
        control_chunked = control_3d_NA.chunk({chunk_dim: step})
        for dim in control_3d_NA.dims:
            if dim != chunk_dim and dim in control_chunked.dims:
                res_chunked = func(control_chunked, dim=dim)
                res = func(control_3d_NA, dim=dim)
                # check for chunks
                assert dask.is_dask_collection(res_chunked)
                assert res_chunked.chunks is not None
                # check for no chunks
                assert not dask.is_dask_collection(res)
                assert res.chunks is None
                # check for identical result
                assert_allclose(res, res_chunked.compute())


def test_varweighted_mean_period_dim(control_3d_NA):
    """Test varweighted_mean_period for different dims."""
    for d in control_3d_NA.dims:
        # single dim
        varweighted_mean_period(control_3d_NA, dim=d)
        # all but one dim
        di = [di for di in control_3d_NA.dims if di != d]
        varweighted_mean_period(control_3d_NA, dim=di)


@pytest.mark.xfail(reason='p value not aligned in the two functions.')
def test_corr_autocorr(control_3d_NA):
    res1 = corr(control_3d_NA, control_3d_NA, lag=1, return_p=True)
    res2 = autocorr(control_3d_NA, return_p=True)
    for i in [0, 1]:
        print(res1[i] - res2[i])
        assert_allclose(res1[i], res2[i])
