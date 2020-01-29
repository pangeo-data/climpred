import pytest

from climpred.prediction import compute_perfect_model
from climpred.smoothing import (
    _reset_temporal_axis,
    smooth_goddard_2013,
    spatial_smoothing_xrcoarsen,
    temporal_smoothing,
)

try:
    from climpred.smoothing import spatial_smoothing_xesmf

    xesmf_loaded = True
except ImportError:
    xesmf_loaded = False


def test_reset_temporal_axis(PM_da_control_3d_full):
    """Test whether correct new labels are set."""
    smooth = 10
    smooth_kws = {'time': smooth}
    first_ori = PM_da_control_3d_full.time[0].values
    first_actual = _reset_temporal_axis(
        PM_da_control_3d_full, smooth_kws=smooth_kws
    ).time.values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    assert first_actual == first_expected


def test_reset_temporal_axis_lead(PM_da_initialized_3d_full):
    """Test whether correct new labels are set."""
    smooth = 10
    dim = 'lead'
    smooth_kws = {dim: smooth}
    first_ori = PM_da_initialized_3d_full.lead[0].values
    first_actual = _reset_temporal_axis(
        PM_da_initialized_3d_full, smooth_kws=smooth_kws
    )[dim].values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    assert first_actual == first_expected


def test_temporal_smoothing_reduce_length(PM_da_control_3d_full):
    """Test whether dimsize is reduced properly."""
    smooth = 10
    smooth_kws = {'time': smooth}
    actual = temporal_smoothing(PM_da_control_3d_full, smooth_kws=smooth_kws).time.size
    expected = PM_da_control_3d_full.time.size - smooth + 1
    assert actual == expected


def test_spatial_smoothing_xrcoarsen_reduce_spatial_dims(PM_da_control_3d_full,):
    """Test whether spatial dimsizes are properly reduced."""
    da = PM_da_control_3d_full
    coarsen_kws = {'x': 4, 'y': 2}
    actual = spatial_smoothing_xrcoarsen(da, coarsen_kws)
    for dim in coarsen_kws:
        actual_x = actual[dim].size
        expected_x = PM_da_control_3d_full[dim].size // coarsen_kws[dim]
        assert actual_x == expected_x


def test_spatial_smoothing_xrcoarsen_reduce_spatial_dims_no_coarsen_kws(
    PM_da_control_3d_full,
):
    """Test whether spatial dimsizes are properly reduced if no coarsen_kws
    given."""
    da = PM_da_control_3d_full
    coarsen_kws = {'x': 2, 'y': 2}
    actual = spatial_smoothing_xrcoarsen(da, coarsen_kws=None)
    for dim in coarsen_kws:
        actual_dim_size = actual[dim].size
        expected_dim_size = PM_da_control_3d_full[dim].size // coarsen_kws[dim]
        assert actual_dim_size == expected_dim_size


def test_spatial_smoothing_xrcoarsen_reduce_spatial_dims_CESM(
    reconstruction_ds_3d_full,
):
    """Test whether spatial dimsizes are properly reduced."""
    da = reconstruction_ds_3d_full.isel(nlon=slice(0, 24), nlat=slice(0, 36))
    coarsen_kws = {'nlon': 4, 'nlat': 4}
    actual = spatial_smoothing_xrcoarsen(da, coarsen_kws)
    for dim in coarsen_kws:
        actual_x = actual[dim].size
        expected_x = da[dim].size // coarsen_kws[dim]
        assert actual_x == expected_x


@pytest.mark.skipif(not xesmf_loaded, reason='xesmf not installed')
def test_spatial_smoothing_xesmf_reduce_spatial_dims_MPI_curv(PM_da_control_3d_full,):
    """Test whether spatial dimsizes are properly reduced."""
    da = PM_da_control_3d_full
    step = 5
    actual = spatial_smoothing_xesmf(da, d_lon_lat_kws={'lon': step})
    expected_lat_size = 180 // step
    assert actual['lon'].size < da.lon.size
    assert actual['lat'].size == expected_lat_size


@pytest.mark.skipif(not xesmf_loaded, reason='xesmf not installed')
def test_spatial_smoothing_xesmf_reduce_spatial_dims_CESM(reconstruction_ds_3d_full,):
    """Test whether spatial dimsizes are properly reduced."""
    da = reconstruction_ds_3d_full
    step = 0.1
    actual = spatial_smoothing_xesmf(da, d_lon_lat_kws={'lat': step})
    # test whether upsampled
    assert actual['lon'].size >= da.nlon.size
    assert actual['lat'].size >= da.nlat.size


def test_smooth_goddard_2013(PM_da_control_3d_full):
    """Test whether Goddard 2013 recommendations are fulfilled by
    smooth_Goddard_2013."""
    da = PM_da_control_3d_full
    actual = smooth_goddard_2013(da)
    # test that x, y not in dims
    assert 'x' not in actual.dims
    assert 'y' not in actual.dims
    # tests whether nlat, nlon got reduced
    assert actual.time.size < da.time.size
    assert actual.lon.size < da.lon.size
    assert actual.lat.size < da.lat.size


def test_compute_after_smooth_goddard_2013(
    PM_da_initialized_3d_full, PM_da_control_3d_full
):
    """Test compute_perfect_model works after smoothings."""
    PM_da_control_3d_full = smooth_goddard_2013(PM_da_control_3d_full)
    PM_da_initialized_3d_full = smooth_goddard_2013(PM_da_initialized_3d_full)
    actual = compute_perfect_model(PM_da_initialized_3d_full, PM_da_control_3d_full)
    north_atlantic = actual.sel(lat=slice(40, 50), lon=slice(-30, -20))
    assert not north_atlantic.isnull().any()
