import pytest

from climpred import HindcastEnsemble
from climpred.prediction import compute_perfect_model
from climpred.smoothing import (
    _reset_temporal_axis,
    smooth_goddard_2013,
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


@pytest.fixture
def hindcast_recon_3d(hind_ds_initialized_3d, reconstruction_ds_3d):
    """HindcastEnsemble initialized with `initialized`, `uninitialzed` and `obs`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_3d)
    hindcast = hindcast.add_observations(reconstruction_ds_3d, 'recon')
    hindcast = hindcast - hindcast.sel(time=slice('1964', '2014')).mean('time').sel(
        init=slice('1964', '2014')
    ).mean('init')
    return hindcast


@pytest.mark.parametrize('smooth_kws', [{'lead': 2}, {'lead': 4}])
def test_HindcastEnsemble_temproal_smooth_leadrange(hindcast_recon_3d, smooth_kws):
    he = hindcast_recon_3d
    dim = list(smooth_kws.keys())[0]
    he = he.smooth(smooth_kws=smooth_kws)
    assert he._datasets['initialized'].lead.attrs['units']
    skill = he.verify(metric='acc')
    assert skill.lead[0] == f'1-{1+smooth_kws[dim]-1}'
