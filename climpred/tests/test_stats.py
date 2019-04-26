import pytest
import numpy as np
import xarray as xr

from climpred.stats import xr_rm_trend


def test_xr_rm_trend_2d_dataarray():
    da = xr.DataArray(
        np.vstack([np.arange(0, 5, 1),
                   np.arange(0, 10, 2),
                   np.arange(0, 40, 8),
                   np.arange(0, 20, 4)]),
        dims=['row', 'col']
    )
    da_dt = xr_rm_trend(da)
    assert da_dt.shape == (4, 5)
    # should have all values equal to 0 after detrending because it's linear
    assert (xr_rm_trend(da, 'col') == 0).sum() == 20


def test_xr_rm_trend_3d_dataset():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds = ds.assign(**{'airx2': ds['air'] * 2})
    ds_dt = xr_rm_trend(ds)

    assert ds_dt['air'].shape == (2920, 25, 53)
    assert ds_dt['airx2'].shape == (2920, 25, 53)


def test_xr_rm_trend_3d_dataset_dim_order():
    ds = xr.tutorial.open_dataset('air_temperature')
    ds = ds.assign(**{'airx2': ds['air'] * 2})
    ds = ds.transpose('lon', 'time', 'lat')
    ds_dt = xr_rm_trend(ds)

    assert list(ds_dt['air'].dims) == ['lon', 'time', 'lat']
    assert list(ds_dt['airx2'].dims) == ['lon', 'time', 'lat']
