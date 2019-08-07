import pytest
from climpred.smoothing import (
    _reset_temporal_axis,
    smooth_goddard_2013,
    spatial_smoothing_xrcoarsen,
    temporal_smoothing,
)
from climpred.tutorial import load_dataset

try:
    from climpred.smoothing import spatial_smoothing_xesmf

    xesmf_loaded = True
except:
    xesmf_loaded = False


@pytest.fixture
def pm_da_control3d():
    da = load_dataset('MPI-control-3D')
    da = da['tos']
    return da


@pytest.fixture
def pm_da_ds3d():
    da = load_dataset('MPI-PM-DP-3D')
    da = da['tos']
    return da


@pytest.fixture
def fosi_3d():
    ds = load_dataset('FOSI-SST-3D')
    return ds


@pytest.fixture
def dple_3d():
    ds = load_dataset('CESM-DP-SST-3D')
    return ds


def test_reset_temporal_axis(pm_da_control3d):
    """Test whether correct new labels are set."""
    smooth = 10
    smooth_dict = {'time': smooth}
    first_ori = pm_da_control3d.time[0].values
    first_actual = _reset_temporal_axis(
        pm_da_control3d, smooth_dict=smooth_dict
    ).time.values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    assert first_actual == first_expected


def test_reset_temporal_axis_lead(pm_da_ds3d):
    """Test whether correct new labels are set."""
    smooth = 10
    dim = 'lead'
    smooth_dict = {dim: smooth}
    first_ori = pm_da_ds3d.lead[0].values
    first_actual = _reset_temporal_axis(pm_da_ds3d, smooth_dict=smooth_dict)[
        dim
    ].values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    print(first_actual, first_expected)
    assert first_actual == first_expected


def test_temporal_smoothing_reduce_length(pm_da_control3d):
    """Test whether dimsize is reduced properly."""
    smooth = 10
    smooth_dict = {'time': smooth}
    actual = temporal_smoothing(pm_da_control3d, smooth_dict=smooth_dict).time.size
    expected = pm_da_control3d.time.size - smooth + 1
    assert actual == expected


def test_spatial_smoothing_xrcoarsen_reduce_spatial_dims(pm_da_control3d):
    """Test whether spatial dimsizes are properly reduced."""
    da = pm_da_control3d
    coarsen_dict = {'x': 4, 'y': 2}
    actual = spatial_smoothing_xrcoarsen(da, coarsen_dict)
    for dim in coarsen_dict:
        actual_x = actual[dim].size
        expected_x = pm_da_control3d[dim].size // coarsen_dict[dim]
        assert actual_x == expected_x


def test_spatial_smoothing_xrcoarsen_reduce_spatial_dims_CESM(fosi_3d):
    """Test whether spatial dimsizes are properly reduced."""
    da = fosi_3d.isel(nlon=slice(0, 24), nlat=slice(0, 36))
    coarsen_dict = {'nlon': 4, 'nlat': 4}
    actual = spatial_smoothing_xrcoarsen(da, coarsen_dict)
    for dim in coarsen_dict:
        actual_x = actual[dim].size
        expected_x = da[dim].size // coarsen_dict[dim]
        assert actual_x == expected_x


if xesmf_loaded:

    def test_spatial_smoothing_xesmf_reduce_spatial_dims_MPI_curv(pm_da_control3d):
        """Test whether spatial dimsizes are properly reduced."""
        da = pm_da_control3d
        step = 5
        actual = spatial_smoothing_xesmf(da, d_lon_lat_dict={'lon': step})
        # expected_lon_size = 360 // step
        expected_lat_size = 180 // step
        # assert actual['lon'].size == expected_lon_size
        assert actual['lon'].size < da.lon.size
        assert actual['lat'].size == expected_lat_size

    def test_spatial_smoothing_xesmf_reduce_spatial_dims_CESM(fosi_3d):
        """Test whether spatial dimsizes are properly reduced."""
        da = fosi_3d
        step = 5
        actual = spatial_smoothing_xesmf(da, d_lon_lat_dict={'lat': step})
        # expected_lon = (fosi.TLAT.max() - fosi.TLAT.min()) // step
        # expected_lon = (fosi.TLON.max() - fosi.TLON.min()) // step
        # assert actual['lon'].size == expected_lon
        # assert actual['lat'].size == expected_lat
        assert actual['lon'].size < da.nlon.size
        assert actual['lat'].size < da.nlat.size


def test_smooth_goddard_2013(pm_da_control3d):
    """Test whether Goddard 2013 recommendations are fulfilled by smooth_Goddard_2013."""
    da = pm_da_control3d
    actual = smooth_goddard_2013(da)
    # tests whether nlat, nlon got reduced
    assert actual.time.size <= da.time.size
    assert actual.lon.size <= da.lon.size
    assert actual.lat.size <= da.lat.size
