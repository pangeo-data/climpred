import numpy as np
import pytest
import xarray as xr
from scipy.signal import correlate

from climpred.bootstrap import DPP_threshold, varweighted_mean_period_threshold
from climpred.exceptions import DimensionError
from climpred.stats import (
    DPP,
    autocorr,
    corr,
    decorrelation_time,
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
    ds = load_dataset('MPI-control-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
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
def test_DPP(control_3d_NA, chunk):
    """Check for positive diagnostic potential predictability in NA SST."""
    control = control_3d_NA
    res = DPP(control, chunk=chunk)
    assert res.mean() > 0


@pytest.mark.parametrize(
    'func', (varweighted_mean_period, decorrelation_time, autocorr)
)
def test_potential_predictability_likely(control_3d_NA, func):
    """Check for positive diagnostic potential predictability in NA SST."""
    control = control_3d_NA
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


def test_bootstrap_DPP_sig50_similar_DPP(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = 50
    actual = DPP_threshold(ds, bootstrap=bootstrap, sig=sig).drop('quantile')
    expected = DPP(ds)
    xr.testing.assert_allclose(actual, expected, atol=0.5, rtol=0.5)


def test_bootstrap_vwmp_sig50_similar_vwmp(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = 50
    actual = varweighted_mean_period_threshold(ds, bootstrap=bootstrap, sig=sig).drop(
        'quantile'
    )
    expected = varweighted_mean_period(ds)
    xr.testing.assert_allclose(actual, expected, atol=2, rtol=0.5)


def test_bootstrap_func_multiple_sig_levels(control_3d_NA):
    ds = control_3d_NA
    bootstrap = 5
    sig = [5, 95]
    actual = DPP_threshold(ds, bootstrap=bootstrap, sig=sig)
    print(actual)
    assert actual['quantile'].size == len(sig)
    assert (actual.isel(quantile=0).values <= actual.isel(quantile=1)).all()
