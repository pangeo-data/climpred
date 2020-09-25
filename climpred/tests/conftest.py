import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.constants import HINDCAST_CALENDAR_STR, PM_CALENDAR_STR
from climpred.tutorial import load_dataset
from climpred.utils import convert_time_index

CALENDAR = PM_CALENDAR_STR.strip("Datetime").lower()


@pytest.fixture
def PM_ds3v_initialized_1d():
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with three
    variables."""
    return load_dataset("MPI-PM-DP-1D").isel(area=1, period=-1)


@pytest.fixture
def PM_ds_initialized_1d(PM_ds3v_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset."""
    return PM_ds3v_initialized_1d.drop_vars(["sos", "AMO"])


@pytest.fixture
def PM_da_initialized_1d(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.DataArray."""
    return PM_ds_initialized_1d["tos"]


@pytest.fixture
def PM_da_initialized_1d_lead0(PM_da_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.DataArray in lead-0
    framework."""
    da = PM_da_initialized_1d
    # Convert to lead zero for testing
    da["lead"] = da["lead"] - 1
    da["init"] = da["init"] + 1
    return da


@pytest.fixture
def PM_ds_initialized_3d_full():
    """MPI Perfect-model-framework initialized global maps xr.Dataset."""
    return load_dataset("MPI-PM-DP-3D")


@pytest.fixture
def PM_da_initialized_3d_full(PM_ds_initialized_3d_full):
    """MPI Perfect-model-framework initialized global maps xr.Dataset."""
    return PM_ds_initialized_3d_full["tos"]


@pytest.fixture
def PM_ds_initialized_3d(PM_ds_initialized_3d_full):
    """MPI Perfect-model-framework initialized maps xr.Dataset of subselected North
    Atlantic."""
    return PM_ds_initialized_3d_full.sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def PM_da_initialized_3d(PM_ds_initialized_3d):
    """MPI Perfect-model-framework initialized maps xr.DataArray of subselected North
    Atlantic."""
    return PM_ds_initialized_3d["tos"]


@pytest.fixture
def PM_ds3v_control_1d():
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    three variables."""
    return load_dataset("MPI-control-1D").isel(area=1, period=-1)


@pytest.fixture
def PM_ds_control_1d(PM_ds3v_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset."""
    return PM_ds3v_control_1d.drop_vars(["sos", "AMO"])


@pytest.fixture
def PM_da_control_1d(PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.DataArray."""
    return PM_ds_control_1d["tos"]


@pytest.fixture
def PM_ds_control_3d_full():
    """To MPI Perfect-model-framework corresponding control global maps xr.Dataset."""
    return load_dataset("MPI-control-3D")


@pytest.fixture
def PM_da_control_3d_full(PM_ds_control_3d_full):
    """To MPI Perfect-model-framework corresponding control global maps xr.DataArray."""
    return PM_ds_control_3d_full["tos"]


@pytest.fixture
def PM_ds_control_3d(PM_ds_control_3d_full):
    """To MPI Perfect-model-framework corresponding control maps xr.Dataset of
    subselected North Atlantic."""
    return PM_ds_control_3d_full.sel(x=slice(120, 130), y=slice(50, 60))


@pytest.fixture
def PM_da_control_3d(PM_ds_control_3d):
    """To MPI Perfect-model-framework corresponding control maps xr.DataArray of
    subselected North Atlantic."""
    return PM_ds_control_3d["tos"]


@pytest.fixture
def perfectModelEnsemble_initialized_control_3d_North_Atlantic(
    PM_ds_initialized_3d, PM_ds_control_3d
):
    """PerfectModelEnsemble with `initialized` and `control` for the North Atlantic."""
    return PerfectModelEnsemble(PM_ds_initialized_3d).add_control(PM_ds_control_3d)


@pytest.fixture
def perfectModelEnsemble_initialized_control(PM_ds_initialized_1d, PM_ds_control_1d):
    """PerfectModelEnsemble initialized with `initialized` and `control` xr.Dataset."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    return pm


@pytest.fixture
def perfectModelEnsemble_3v_initialized_control_1d(
    PM_ds3v_initialized_1d, PM_ds3v_control_1d
):
    """PerfectModelEnsemble 1d initialized with `initialized` and `control` xr.Dataset
    with three variables."""
    return PerfectModelEnsemble(PM_ds3v_initialized_1d).add_control(PM_ds3v_control_1d)


@pytest.fixture
def hind_ds_initialized_1d():
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset."""
    ds = load_dataset("CESM-DP-SST")
    ds["init"] = ds.init.astype("int")
    return ds


@pytest.fixture
def hind_ds_initialized_1d_cftime(hind_ds_initialized_1d):
    """CESM-DPLE initialzed hindcast timeseries with cftime initializations."""
    ds = hind_ds_initialized_1d
    ds = convert_time_index(ds, "init", "ds.init", calendar=HINDCAST_CALENDAR_STR)
    ds.lead.attrs["units"] = "years"
    return ds


@pytest.fixture
def hind_ds_initialized_1d_lead0(hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset in lead-0
    framework."""
    da = hind_ds_initialized_1d
    # Change to a lead-0 framework
    da["init"] = da["init"] + 1
    da["lead"] = da["lead"] - 1
    return da


@pytest.fixture
def hind_da_initialized_1d(hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.DataArray."""
    return hind_ds_initialized_1d["SST"]


@pytest.fixture
def hind_ds_initialized_3d_full():
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    da = load_dataset("CESM-DP-SST-3D")
    return da - da.mean("init")


@pytest.fixture
def hind_ds_initialized_3d(hind_ds_initialized_3d_full):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    return hind_ds_initialized_3d_full.isel(nlon=slice(0, 10), nlat=slice(0, 12))


@pytest.fixture
def hind_da_initialized_3d(hind_ds_initialized_3d):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.DataArray."""
    return hind_ds_initialized_3d["SST"]


@pytest.fixture
def hist_ds_uninitialized_1d():
    """CESM-LE uninitialized historical timeseries members mean removed xr.Dataset."""
    da = load_dataset("CESM-LE")
    # add member coordinate
    da["member"] = range(1, 1 + da.member.size)
    return da - da.mean("time")


@pytest.fixture
def hist_da_uninitialized_1d(hist_ds_uninitialized_1d):
    """CESM-LE uninitialized historical timeseries members mean removed xr.DataArray."""
    return hist_ds_uninitialized_1d["SST"]


@pytest.fixture
def reconstruction_ds_1d():
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.Dataset."""
    da = load_dataset("FOSI-SST")
    return da - da.mean("time")


@pytest.fixture
def reconstruction_ds_1d_cftime(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries with cftime time axis."""
    ds = reconstruction_ds_1d
    ds = convert_time_index(ds, "time", "ds.init", calendar=HINDCAST_CALENDAR_STR)
    return ds


@pytest.fixture
def reconstruction_da_1d(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.DataArray."""
    return reconstruction_ds_1d["SST"]


@pytest.fixture
def reconstruction_ds_3d_full():
    """CESM-FOSI historical Pacific reconstruction maps members mean removed
    xr.Dataset."""
    ds = load_dataset("FOSI-SST-3D")
    return ds - ds.mean("time")


@pytest.fixture
def reconstruction_ds_3d(reconstruction_ds_3d_full):
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.Dataset."""
    return reconstruction_ds_3d_full.isel(nlon=slice(0, 10), nlat=slice(0, 12))


@pytest.fixture
def reconstruction_da_3d(reconstruction_ds_3d):
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.DataArray."""
    return reconstruction_ds_3d["SST"]


@pytest.fixture
def observations_ds_1d():
    """Historical timeseries from observations matching `hind_da_initialized_1d` and
    `hind_da_uninitialized_1d` mean removed xr.Dataset."""
    da = load_dataset("ERSST")
    return da - da.mean("time")


@pytest.fixture
def observations_da_1d(observations_ds_1d):
    """Historical timeseries from observations matching `hind_da_initialized_1d` and
    `hind_da_uninitialized_1d` mean removed xr.DataArray."""
    return observations_ds_1d["SST"]


@pytest.fixture
def hindcast_recon_3d(hind_ds_initialized_3d, reconstruction_ds_3d):
    """HindcastEnsemble initialized with `initialized`, `reconstruction`(`recon`)."""
    # fix to align coords
    for c in ["TLAT", "TLONG", "TAREA"]:
        reconstruction_ds_3d[c] = hind_ds_initialized_3d[c]
    hindcast = HindcastEnsemble(hind_ds_initialized_3d)
    hindcast = hindcast.add_observations(reconstruction_ds_3d)
    hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
        init=slice("1964", "2014")
    ).mean("init")
    return hindcast


@pytest.fixture
def hindcast_recon_1d_ym(hind_ds_initialized_1d, reconstruction_ds_1d):
    """HindcastEnsemble initialized with `initialized`, `uninitialzed` and `recon`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d)
    hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
        init=slice("1964", "2014")
    ).mean("init")
    return hindcast


@pytest.fixture
def hindcast_hist_obs_1d(
    hind_ds_initialized_1d, hist_ds_uninitialized_1d, observations_ds_1d
):
    """HindcastEnsemble initialized with `initialized`, `uninitialzed` and `obs`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    hindcast = hindcast.add_observations(observations_ds_1d)
    hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
        init=slice("1964", "2014")
    ).mean("init")
    return hindcast


@pytest.fixture
def hindcast_recon_1d_mm(hindcast_recon_1d_ym):
    """HindcastEnsemble with initialized and reconstruction (observations) as a monthly
    time series (no grid)."""
    hindcast = hindcast_recon_1d_ym.sel(time=slice("1964", "1970"))
    hindcast._datasets["initialized"].lead.attrs["units"] = "months"
    hindcast._datasets["observations"] = (
        hindcast._datasets["observations"].resample(time="1MS").interpolate("linear")
    )
    return hindcast


@pytest.fixture
def hindcast_recon_1d_dm(hindcast_recon_1d_ym):
    """HindcastEnsemble with initialized and reconstruction (observations) as a daily
    time series (no grid)."""
    hindcast = hindcast_recon_1d_ym.sel(time=slice("1964", "1970"))
    hindcast._datasets["initialized"].lead.attrs["units"] = "days"
    hindcast._datasets["observations"] = (
        hindcast._datasets["observations"].resample(time="1D").interpolate("linear")
    )
    return hindcast


@pytest.fixture
def ds1():
    """Small plain multi-dimensional coords xr.Dataset."""
    return xr.Dataset(
        {"air": (("lon", "lat"), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={"lon": [1, 3, 4], "lat": [5, 6, 7]},
    )


@pytest.fixture
def ds2():
    """Small plain multi-dimensional coords xr.Dataset identical values but with
    different coords compared to ds1."""
    return xr.Dataset(
        {"air": (("lon", "lat"), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={"lon": [1, 3, 6], "lat": [5, 6, 9]},
    )


@pytest.fixture
def da1():
    """Small plain two-dimensional xr.DataArray."""
    return xr.DataArray([[0, 1], [3, 4], [6, 7]], dims=("x", "y"))


@pytest.fixture
def da2():
    """Small plain two-dimensional xr.DataArray with different values compared to
    da1."""
    return xr.DataArray([[0, 1], [5, 6], [6, 7]], dims=("x", "y"))


@pytest.fixture
def da_lead():
    """Small xr.DataArray with coords `init` and `lead`."""
    lead = np.arange(5)
    init = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init)),
        dims=["init", "lead"],
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
        dims=["row", "col"],
    )
    return da


@pytest.fixture
def multi_dim_ds():
    """xr.Dataset with multi-dimensional coords."""
    ds = xr.tutorial.open_dataset("air_temperature")
    ds = ds.assign(**{"airx2": ds["air"] * 2})
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
        dims=["S", "L", "M"],
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
        dims=["dcpp_init_year", "time", "member_id"],
        coords=[init, lead, member],
    )


@pytest.fixture
def PM_ds_initialized_1d_ym_cftime(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    cftime."""
    PM_ds_initialized_1d = convert_time_index(
        PM_ds_initialized_1d,
        "init",
        "PM_ds_initialized_1d.init",
        calendar=PM_CALENDAR_STR,
    )
    PM_ds_initialized_1d["lead"].attrs["units"] = "years"
    return PM_ds_initialized_1d


@pytest.fixture
def PM_ds_control_1d_ym_cftime(PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime."""
    PM_ds_control_1d = convert_time_index(
        PM_ds_control_1d, "time", "PM_ds_control_1d.time", calendar=PM_CALENDAR_STR
    )
    return PM_ds_control_1d


@pytest.fixture
def perfectModelEnsemble_initialized_control_1d_ym_cftime(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries annual mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_ym_cftime)
    pm = pm.add_control(PM_ds_control_1d_ym_cftime)
    return pm


@pytest.fixture
def PM_ds_initialized_1d_mm_cftime(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    cftime faking all inits with monthly separation in one year and lead units to
    monthly."""
    PM_ds_initialized_1d["init"] = xr.cftime_range(
        start="3004",
        periods=PM_ds_initialized_1d.init.size,
        freq="MS",
        calendar=CALENDAR,
    )
    PM_ds_initialized_1d["lead"].attrs["units"] = "months"
    return PM_ds_initialized_1d


@pytest.fixture
def PM_ds_control_1d_mm_cftime(PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime faking the time resolution to monthly means."""
    PM_ds_control_1d["time"] = xr.cftime_range(
        start="3000", periods=PM_ds_control_1d.time.size, freq="MS", calendar=CALENDAR
    )
    return PM_ds_control_1d


@pytest.fixture
def perfectModelEnsemble_initialized_control_1d_mm_cftime(
    PM_ds_initialized_1d_mm_cftime, PM_ds_control_1d_mm_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries monthly mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_mm_cftime)
    pm = pm.add_control(PM_ds_control_1d_mm_cftime)
    return pm


@pytest.fixture
def PM_ds_initialized_1d_dm_cftime(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    cftime faking all inits with daily separation in one year and lead units to
    daily."""
    PM_ds_initialized_1d["init"] = xr.cftime_range(
        start="3004",
        periods=PM_ds_initialized_1d.init.size,
        freq="D",
        calendar=CALENDAR,
    )
    PM_ds_initialized_1d["lead"].attrs["units"] = "days"
    return PM_ds_initialized_1d


@pytest.fixture
def PM_ds_control_1d_dm_cftime(PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime faking the time resolution to daily means."""
    PM_ds_control_1d = PM_ds_control_1d.isel(
        time=np.random.randint(0, PM_ds_control_1d.time.size, 5000)
    )
    PM_ds_control_1d["time"] = xr.cftime_range(
        start="3000", periods=PM_ds_control_1d.time.size, freq="D", calendar=CALENDAR
    )
    return PM_ds_control_1d


@pytest.fixture
def perfectModelEnsemble_initialized_control_1d_dm_cftime(
    PM_ds_initialized_1d_dm_cftime, PM_ds_control_1d_dm_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries daily mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_dm_cftime)
    pm = pm.add_control(PM_ds_control_1d_dm_cftime)
    return pm


@pytest.fixture
def small_initialized_da():
    """Very small simulation of an initialized forecasting system."""
    inits = [1990, 1991, 1992, 1993]
    lead = [1]
    return xr.DataArray(
        np.random.rand(len(inits), len(lead)),
        dims=["init", "lead"],
        coords=[inits, lead],
    )


@pytest.fixture
def small_verif_da():
    """Very small simulation of a verification product."""
    time = [1990, 1991, 1992, 1993, 1994]
    return xr.DataArray(np.random.rand(len(time)), dims=["time"], coords=[time])
