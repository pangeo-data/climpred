import numpy as np
import pytest
import xarray as xr

from climpred.tutorial import load_dataset

# ordering: PM MPI, CESM; xr.Dataset, xr.DataArray; 1D, 3D


@pytest.fixture
def PM_da_initialized_1d():
    """MPI Perfect-model-framework initialized timeseries xr.DataArray."""
    return load_dataset('MPI-PM-DP-1D')['tos'].isel(area=1, period=-1)


@pytest.fixture
def PM_da_initialized_3d():
    """MPI Perfect-model-framework initialized maps xr.DataArray of subselected North
    Atlantic."""
    return load_dataset('MPI-PM-DP-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def PM_da_control_1d():
    """To MPI Perfect-model-framework corresponding control timeseries xr.DataArray."""
    return load_dataset('MPI-control-1D')['tos'].isel(area=1, period=-1)


@pytest.fixture
def PM_da_control_3d():
    """To MPI Perfect-model-framework corresponding control maps xr.DataArray of
    subselected North Atlantic."""
    return load_dataset('MPI-control-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def hind_ds_initialized_1d():
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset."""
    da = load_dataset('CESM-DP-SST')
    return da - da.mean('init')


@pytest.fixture
def hind_ds_initialized_1d_lead0(hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset in lead-0
    framework."""
    da = hind_ds_initialized_1d
    # Change to a lead-0 framework
    da['init'] += 1
    da['lead'] -= 1
    return da


@pytest.fixture
def hind_da_initialized_1d(hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.DataArray."""
    return hind_ds_initialized_1d['SST']


@pytest.fixture
def hind_da_initialized_3d():
    """CESM-DPLE initialized hindcast maps mean removed xr.DataArray."""
    da = load_dataset('CESM-DP-SST-3D')['SST'].isel(
        nlon=slice(0, 10), nlat=slice(0, 12)
    )
    return da - da.mean('init')


@pytest.fixture
def hind_ds_uninitialized_1d():
    """CESM-LE uninitialized historical timeseries members mean removed xr.Dataset."""
    da = load_dataset('CESM-LE')
    # add member coordinate
    da['member'] = range(1, 1 + da.member.size)
    return da - da.mean('time')


@pytest.fixture
def hind_da_uninitialized_1d(hind_ds_uninitialized_1d):
    """CESM-LE uninitialized historical timeseries members mean removed xr.DataArray."""
    return hind_ds_uninitialized_1d['SST']


@pytest.fixture
def reconstruction_ds_1d():
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.Dataset."""
    da = load_dataset('FOSI-SST')
    return da - da.mean('time')


@pytest.fixture
def reconstruction_da_1d(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.DataArray."""
    return reconstruction_ds_1d['SST']


@pytest.fixture
def reconstruction_da_3d():
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.DataArray."""
    da = load_dataset('FOSI-SST-3D')['SST'].isel(nlon=slice(0, 10), nlat=slice(0, 12))
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_ds_1d():
    """Historical timeseries from observations matching `hind_da_initialized_1d` and
    `hind_da_uninitialized_1d` mean removed xr.Dataset."""
    da = load_dataset('ERSST')
    return da - da.mean('time')


@pytest.fixture
def observations_da_1d(observations_ds_1d):
    """Historical timeseries from observations matching `hind_da_initialized_1d` and
    `hind_da_uninitialized_1d` mean removed xr.DataArray."""
    return observations_ds_1d['SST']


@pytest.fixture
def ds1():
    """Small plain multi-dimensional coords xr.Dataset."""
    return xr.Dataset(
        {'air': (('lon', 'lat'), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={'lon': [1, 3, 4], 'lat': [5, 6, 7]},
    )


@pytest.fixture
def ds2():
    """Small plain multi-dimensional coords xr.Dataset identical values but with
    different coords compared to ds1."""
    return xr.Dataset(
        {'air': (('lon', 'lat'), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={'lon': [1, 3, 6], 'lat': [5, 6, 9]},
    )


@pytest.fixture
def da1():
    """Small plain two-dimensional xr.DataArray."""
    return xr.DataArray([[0, 1], [3, 4], [6, 7]], dims=('x', 'y'))


@pytest.fixture
def da2():
    """Small plain two-dimensional xr.DataArray with different values compared to
    da1."""
    return xr.DataArray([[0, 1], [5, 6], [6, 7]], dims=('x', 'y'))


@pytest.fixture
def da_lead():
    """Small xr.DataArray with coords `init` and `lead`."""
    lead = np.arange(5)
    init = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init)),
        dims=['init', 'lead'],
        coords=[init, lead],
    )
