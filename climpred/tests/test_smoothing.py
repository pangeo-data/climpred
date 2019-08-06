import pytest
from climpred.smoothing import (
    _reset_temporal_axis,
    smooth_Goddard_2013,
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


def test_reset_temporal_axis(pm_da_control3d):
    """Test whether correct new labels are set."""
    smooth = 10
    first_ori = pm_da_control3d.time[0].values
    first_actual = _reset_temporal_axis(pm_da_control3d, smooth=smooth).time.values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    assert first_actual == first_expected


def test_reset_temporal_axis_lead(pm_da_ds3d):
    """Test whether correct new labels are set."""
    smooth = 10
    first_ori = pm_da_ds3d.lead[0].values
    first_actual = _reset_temporal_axis(
        pm_da_ds3d, smooth=smooth, dim='lead'
    ).lead.values[0]
    first_expected = f'{first_ori}-{first_ori+smooth*1-1}'
    print(first_actual, first_expected)
    assert first_actual == first_expected


def test_temporal_smoothing_reduce_length(pm_da_control3d):
    """Test whether dimsize is reduced properly."""
    smooth = 10
    actual = temporal_smoothing(pm_da_control3d, smooth=smooth, dim='time').time.size
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


if xesmf_loaded:

    def test_spatial_smoothing_xesmf_reduce_spatial_dims(pm_da_control3d):
        """Test whether spatial dimsizes are properly reduced."""
        da = pm_da_control3d
        boxsize = (5, 5)
        actual = spatial_smoothing_xesmf(da, boxsize=boxsize)
        expected_lon = 360 // 5
        expected_lat = 180 // 5
        assert actual['lon'].size == expected_lon
        assert actual['lat'].size == expected_lat


def test_smooth_Goddard_2013(pm_da_control3d):
    """Test whether Goddard 2013 recommendations are fulfilled by smooth_Goddard_2013."""
    da = pm_da_control3d
    actual = smooth_Goddard_2013(da)
    before = da
    assert actual.time.size <= da.time.size
    assert actual.lon.size <= da.lon.size
    assert actual.lat.size <= da.lat.size
