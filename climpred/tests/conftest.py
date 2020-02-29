import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble
from climpred.tutorial import load_dataset


def set_cftime_to_int_dim(ds, dim, freq='YS'):
    ds[dim] = xr.cftime_range(
        start=str(ds[dim].min().values), freq=freq, periods=ds[dim].size,
    )
    return ds


@pytest.fixture
def PM_ds_initialized_1d():
    """MPI Perfect-model-framework initialized timeseries xr.Dataset."""
    return load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)


@pytest.fixture
def PM_da_initialized_1d(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.DataArray."""
    return PM_ds_initialized_1d['tos']


@pytest.fixture
def PM_da_initialized_1d_lead0(PM_da_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.DataArray in lead-0
    framework."""
    da = PM_da_initialized_1d
    # Convert to lead zero for testing
    da['lead'] -= 1
    da['init'] += 1
    return da


@pytest.fixture
def PM_ds_initialized_3d_full():
    """MPI Perfect-model-framework initialized global maps xr.Dataset."""
    return load_dataset('MPI-PM-DP-3D')


@pytest.fixture
def PM_da_initialized_3d_full(PM_ds_initialized_3d_full):
    """MPI Perfect-model-framework initialized global maps xr.Dataset."""
    return PM_ds_initialized_3d_full['tos']


@pytest.fixture
def PM_ds_initialized_3d(PM_ds_initialized_3d_full):
    """MPI Perfect-model-framework initialized maps xr.Dataset of subselected North
    Atlantic."""
    return PM_ds_initialized_3d_full.sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def PM_da_initialized_3d(PM_ds_initialized_3d):
    """MPI Perfect-model-framework initialized maps xr.DataArray of subselected North
    Atlantic."""
    return PM_ds_initialized_3d['tos']


@pytest.fixture
def PM_ds_control_1d():
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset."""
    return load_dataset('MPI-control-1D').isel(area=1, period=-1)


@pytest.fixture
def PM_da_control_1d(PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.DataArray."""
    return PM_ds_control_1d['tos']


@pytest.fixture
def PM_ds_control_3d_full():
    """To MPI Perfect-model-framework corresponding control global maps xr.Dataset."""
    return load_dataset('MPI-control-3D')


@pytest.fixture
def PM_da_control_3d_full(PM_ds_control_3d_full):
    """To MPI Perfect-model-framework corresponding control global maps xr.DataArray."""
    return PM_ds_control_3d_full['tos']


@pytest.fixture
def PM_ds_control_3d(PM_ds_control_3d_full):
    """To MPI Perfect-model-framework corresponding control maps xr.Dataset of
    subselected North Atlantic."""
    return PM_ds_control_3d_full.sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def PM_da_control_3d(PM_ds_control_3d):
    """To MPI Perfect-model-framework corresponding control maps xr.DataArray of
    subselected North Atlantic."""
    return PM_ds_control_3d['tos']


@pytest.fixture
def perfectModelEnsemble_initialized_control(PM_ds_initialized_1d, PM_ds_control_1d):
    """PerfectModelEnsemble initialized with `initialized` and `control` xr.Dataset."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    return pm


@pytest.fixture
def hind_ds_initialized_1d():
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset."""
    da = load_dataset('CESM-DP-SST')
    da['init'] = da.init.astype('int')
    return da - da.mean('init')


@pytest.fixture
def hind_ds_initialized_1d_cftime(hind_ds_initialized_1d):
    """CESM-DPLE initialzed hindcast timeseries with cftime initializations."""
    ds = hind_ds_initialized_1d
    ds = set_cftime_to_int_dim(ds, 'init')
    ds.lead.attrs['units'] = 'years'
    return ds


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
def hind_ds_initialized_3d_full():
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    da = load_dataset('CESM-DP-SST-3D')
    return da - da.mean('init')


@pytest.fixture
def hind_ds_initialized_3d(hind_ds_initialized_3d_full):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    return hind_ds_initialized_3d_full.isel(nlon=slice(0, 10), nlat=slice(0, 12))


@pytest.fixture
def hind_da_initialized_3d(hind_ds_initialized_3d):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.DataArray."""
    return hind_ds_initialized_3d['SST']


@pytest.fixture
def hist_ds_uninitialized_1d():
    """CESM-LE uninitialized historical timeseries members mean removed xr.Dataset."""
    da = load_dataset('CESM-LE')
    # add member coordinate
    da['member'] = range(1, 1 + da.member.size)
    return da - da.mean('time')


@pytest.fixture
def hist_da_uninitialized_1d(hist_ds_uninitialized_1d):
    """CESM-LE uninitialized historical timeseries members mean removed xr.DataArray."""
    return hist_ds_uninitialized_1d['SST']


@pytest.fixture
def reconstruction_ds_1d():
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.Dataset."""
    da = load_dataset('FOSI-SST')
    return da - da.mean('time')


@pytest.fixture
def reconstruction_ds_1d_cftime(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries with cftime time axis."""
    ds = reconstruction_ds_1d
    ds = set_cftime_to_int_dim(ds, 'time')
    return ds


@pytest.fixture
def reconstruction_da_1d(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.DataArray."""
    return reconstruction_ds_1d['SST']


@pytest.fixture
def reconstruction_ds_3d_full():
    """CESM-FOSI historical Pacific reconstruction maps members mean removed
    xr.Dataset."""
    ds = load_dataset('FOSI-SST-3D')
    return ds - ds.mean('time')


@pytest.fixture
def reconstruction_ds_3d(reconstruction_ds_3d_full):
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.Dataset."""
    return reconstruction_ds_3d_full.isel(nlon=slice(0, 10), nlat=slice(0, 12))


@pytest.fixture
def reconstruction_da_3d(reconstruction_ds_3d):
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.DataArray."""
    return reconstruction_ds_3d['SST']


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


@pytest.fixture
def two_dim_da():
    """xr.DataArray with two dims."""
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
    """xr.Dataset with multi-dimensional coords."""
    ds = xr.tutorial.open_dataset('air_temperature')
    ds = ds.assign(**{'airx2': ds['air'] * 2})
    return ds


@pytest.fixture
def da_SLM():
    """Small xr.DataArray with dims `S`, `M` and  `L` for `init`, `member` and
    `lead`.
    """
    lead = np.arange(5)
    init = np.arange(5)
    member = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init), len(member)),
        dims=['S', 'L', 'M'],
        coords=[init, lead, member],
    )


@pytest.fixture
def da_dcpp():
    """Small xr.DataArray with coords `dcpp_init_year`, `member_id` and `time` as from
    `intake-esm` `hindcastA-dcpp`."""
    lead = np.arange(5)
    init = np.arange(5)
    member = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init), len(member)),
        dims=['dcpp_init_year', 'time', 'member_id'],
        coords=[init, lead, member],
    )


@pytest.fixture
def small_initialized_da():
    """Very small simulation of an initialized forecasting system."""
    inits = [1990, 1991, 1992, 1993]
    lead = [1]
    return xr.DataArray(
        np.random.rand(len(inits), len(lead)),
        dims=['init', 'lead'],
        coords=[inits, lead],
    )


@pytest.fixture
def small_verif_da():
    """Very small simulation of a verification product."""
    time = [1990, 1991, 1992, 1993, 1994]
    return xr.DataArray(np.random.rand(len(time)), dims=['time'], coords=[time])
