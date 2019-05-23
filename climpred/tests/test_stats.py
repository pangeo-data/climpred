import numpy as np
import pytest
import xarray as xr

from climpred.stats import rm_trend


@pytest.fixture
def two_dim_da():
    da = xr.DataArray(
        np.vstack([np.arange(0, 5, 1.),
                   np.arange(0, 10, 2.),
                   np.arange(0, 40, 8.),
                   np.arange(0, 20, 4.)]),
        dims=['row', 'col']
    )
    return da


@pytest.fixture
def multi_dim_ds():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds = ds.assign(**{'airx2': ds['air'] * 2})
    return ds


def test_rm_trend_missing_dim():
    with pytest.raises(KeyError) as excinfo:
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
