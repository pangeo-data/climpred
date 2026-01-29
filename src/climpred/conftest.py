import warnings

import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

import climpred
from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.constants import HINDCAST_CALENDAR_STR, PM_CALENDAR_STR
from climpred.tutorial import load_dataset
from climpred.utils import convert_time_index

CALENDAR = PM_CALENDAR_STR.strip("Datetime").lower()

xr.set_options(display_style="text")


@pytest.fixture(autouse=True)
def add_standard_imports(
    doctest_namespace,
    hindcast_hist_obs_1d,
    hindcast_recon_3d,
    perfectModelEnsemble_initialized_control,
):
    """imports for doctest"""
    xr.set_options(display_style="text")
    doctest_namespace["np"] = np
    doctest_namespace["xr"] = xr
    doctest_namespace["climpred"] = climpred

    # always seed numpy.random to make the examples deterministic
    np.random.seed(42)

    # climpred data
    doctest_namespace["HindcastEnsemble"] = hindcast_hist_obs_1d
    doctest_namespace["HindcastEnsemble_3D"] = hindcast_recon_3d
    doctest_namespace["PerfectModelEnsemble"] = perfectModelEnsemble_initialized_control


# --- Session-scoped internal fixtures (prefixed with _) ---


@pytest.fixture(scope="session")
def _PM_ds3v_initialized_1d():
    return load_dataset("MPI-PM-DP-1D").isel(area=1, period=-1, drop=True).load()


@pytest.fixture(scope="session")
def _PM_ds_initialized_1d(_PM_ds3v_initialized_1d):
    return _PM_ds3v_initialized_1d.drop_vars(["sos", "AMO"]).load()


@pytest.fixture(scope="session")
def _PM_ds_initialized_3d_full():
    return load_dataset("MPI-PM-DP-3D").load()


@pytest.fixture(scope="session")
def _PM_ds_initialized_3d(_PM_ds_initialized_3d_full):
    return _PM_ds_initialized_3d_full.sel(x=slice(120, 130), y=slice(50, 60)).load()


@pytest.fixture(scope="session")
def _PM_ds3v_control_1d():
    return load_dataset("MPI-control-1D").isel(area=1, period=-1, drop=True).load()


@pytest.fixture(scope="session")
def _PM_ds_control_1d(_PM_ds3v_control_1d):
    return _PM_ds3v_control_1d.drop_vars(["sos", "AMO"]).load()


@pytest.fixture(scope="session")
def _PM_ds_control_3d_full():
    return load_dataset("MPI-control-3D").load()


@pytest.fixture(scope="session")
def _PM_ds_control_3d(_PM_ds_control_3d_full):
    return _PM_ds_control_3d_full.sel(x=slice(120, 130), y=slice(50, 60)).load()


@pytest.fixture(scope="session")
def _hind_ds_initialized_1d():
    ds = load_dataset("CESM-DP-SST").load()
    ds["SST"].attrs["units"] = "C"
    ds["init"] = ds.init.astype("int")
    return ds


@pytest.fixture(scope="session")
def _hind_ds_initialized_3d_full():
    ds = load_dataset("CESM-DP-SST-3D").load()
    return (ds - ds.mean("init")).load()


@pytest.fixture(scope="session")
def _hind_ds_initialized_3d(_hind_ds_initialized_3d_full):
    return _hind_ds_initialized_3d_full.isel(
        nlon=slice(0, 10), nlat=slice(0, 12)
    ).load()


@pytest.fixture(scope="session")
def _hist_ds_uninitialized_1d():
    ds = load_dataset("CESM-LE").load()
    ds["SST"].attrs["units"] = "C"
    ds["member"] = range(1, 1 + ds.member.size)
    ds = (ds - ds.mean("time")).load()
    ds["SST"].attrs["units"] = "C"
    return ds


@pytest.fixture(scope="session")
def _reconstruction_ds_1d():
    ds = load_dataset("FOSI-SST").load()
    ds = (ds - ds.mean("time")).load()
    ds["SST"].attrs["units"] = "C"
    return ds


@pytest.fixture(scope="session")
def _reconstruction_ds_3d_full():
    ds = load_dataset("FOSI-SST-3D").load()
    return (ds - ds.mean("time")).load()


@pytest.fixture(scope="session")
def _reconstruction_ds_3d(_reconstruction_ds_3d_full):
    return _reconstruction_ds_3d_full.isel(nlon=slice(0, 10), nlat=slice(0, 12)).load()


@pytest.fixture(scope="session")
def _observations_ds_1d():
    ds = load_dataset("ERSST").load()
    ds = (ds - ds.mean("time")).load()
    ds["SST"].attrs["units"] = "C"
    return ds


# --- Public function-scoped fixtures (return copies) ---


@pytest.fixture()
def PM_ds3v_initialized_1d(_PM_ds3v_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with three
    variables."""
    return _PM_ds3v_initialized_1d.copy(deep=True)


@pytest.fixture()
def PM_ds_initialized_1d(_PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset."""
    return _PM_ds_initialized_1d.copy(deep=True)


@pytest.fixture()
def PM_ds_initialized_1d_lead0(PM_ds_initialized_1d):
    """MPI Perfect-model-framework initialized timeseries in lead-0
    framework."""
    ds = PM_ds_initialized_1d
    # Convert to lead zero for testing
    ds["lead"] = ds["lead"] - 1
    ds["init"] = ds["init"] + 1
    return ds


@pytest.fixture()
def PM_ds_initialized_3d_full(_PM_ds_initialized_3d_full):
    """MPI Perfect-model-framework initialized global maps xr.Dataset."""
    return _PM_ds_initialized_3d_full.copy(deep=True)


@pytest.fixture()
def PM_ds_initialized_3d(_PM_ds_initialized_3d):
    """MPI Perfect-model-framework initialized maps xr.Dataset of subselected North
    Atlantic."""
    return _PM_ds_initialized_3d.copy(deep=True)


@pytest.fixture()
def PM_ds3v_control_1d(_PM_ds3v_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    three variables."""
    return _PM_ds3v_control_1d.copy(deep=True)


@pytest.fixture()
def PM_ds_control_1d(_PM_ds_control_1d):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset."""
    return _PM_ds_control_1d.copy(deep=True)


@pytest.fixture()
def PM_ds_control_3d_full(_PM_ds_control_3d_full):
    """To MPI Perfect-model-framework corresponding control global maps xr.Dataset."""
    return _PM_ds_control_3d_full.copy(deep=True)


@pytest.fixture()
def PM_ds_control_3d(_PM_ds_control_3d):
    """To MPI Perfect-model-framework corresponding control maps xr.Dataset of
    subselected North Atlantic."""
    return _PM_ds_control_3d.copy(deep=True)


@pytest.fixture()
def perfectModelEnsemble_initialized_control_3d_North_Atlantic(
    PM_ds_initialized_3d, PM_ds_control_3d
):
    """PerfectModelEnsemble with `initialized` and `control` for the North Atlantic."""
    return PerfectModelEnsemble(PM_ds_initialized_3d).add_control(PM_ds_control_3d)


@pytest.fixture()
def perfectModelEnsemble_initialized_control(PM_ds_initialized_1d, PM_ds_control_1d):
    """PerfectModelEnsemble initialized with `initialized` and `control` xr.Dataset."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d).add_control(PM_ds_control_1d)
    return pm.generate_uninitialized()


@pytest.fixture()
def perfectModelEnsemble_3v_initialized_control_1d(
    PM_ds3v_initialized_1d, PM_ds3v_control_1d
):
    """PerfectModelEnsemble 1d initialized with `initialized` and `control` xr.Dataset
    with three variables."""
    return PerfectModelEnsemble(PM_ds3v_initialized_1d).add_control(PM_ds3v_control_1d)


@pytest.fixture()
def hind_ds_initialized_1d(_hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset."""
    return _hind_ds_initialized_1d.copy(deep=True)


@pytest.fixture()
def hind_ds_initialized_1d_cftime(hind_ds_initialized_1d):
    """CESM-DPLE initialzed hindcast timeseries with cftime initializations."""
    ds = hind_ds_initialized_1d
    ds = convert_time_index(ds, "init", "ds.init", calendar=HINDCAST_CALENDAR_STR)
    ds.lead.attrs["units"] = "years"
    return ds


@pytest.fixture()
def hind_ds_initialized_1d_lead0(hind_ds_initialized_1d):
    """CESM-DPLE initialized hindcast timeseries mean removed xr.Dataset in lead-0
    framework."""
    ds = hind_ds_initialized_1d
    # Change to a lead-0 framework
    with xr.set_options(keep_attrs=True):
        ds["init"] = ds["init"] + 1
        ds["lead"] = ds["lead"] - 1
    return ds


@pytest.fixture()
def hind_ds_initialized_3d_full(_hind_ds_initialized_3d_full):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    return _hind_ds_initialized_3d_full.copy(deep=True)


@pytest.fixture()
def hind_ds_initialized_3d(_hind_ds_initialized_3d):
    """CESM-DPLE initialized hindcast Pacific maps mean removed xr.Dataset."""
    return _hind_ds_initialized_3d.copy(deep=True)


@pytest.fixture()
def hist_ds_uninitialized_1d(_hist_ds_uninitialized_1d):
    """CESM-LE uninitialized historical timeseries members mean removed xr.Dataset."""
    return _hist_ds_uninitialized_1d.copy(deep=True)


@pytest.fixture()
def reconstruction_ds_1d(_reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.Dataset."""
    return _reconstruction_ds_1d.copy(deep=True)


@pytest.fixture()
def reconstruction_ds_1d_cftime(reconstruction_ds_1d):
    """CESM-FOSI historical reconstruction timeseries with cftime time axis."""
    ds = reconstruction_ds_1d
    ds = convert_time_index(ds, "time", "ds.init", calendar=HINDCAST_CALENDAR_STR)
    return ds


@pytest.fixture()
def reconstruction_ds_3d_full(_reconstruction_ds_3d_full):
    """CESM-FOSI historical Pacific reconstruction maps members mean removed
    xr.Dataset."""
    return _reconstruction_ds_3d_full.copy(deep=True)


@pytest.fixture()
def reconstruction_ds_3d(_reconstruction_ds_3d):
    """CESM-FOSI historical reconstruction maps members mean removed
    xr.Dataset."""
    return _reconstruction_ds_3d.copy(deep=True)


@pytest.fixture()
def observations_ds_1d(_observations_ds_1d):
    """Historical timeseries from observations matching `hind_ds_initialized_1d` and
    `hind_ds_uninitialized_1d` mean removed."""
    return _observations_ds_1d.copy(deep=True)


@pytest.fixture()
def hindcast_recon_3d(hind_ds_initialized_3d, reconstruction_ds_3d):
    """HindcastEnsemble initialized with `initialized`, `reconstruction`(`recon`)."""
    hindcast = HindcastEnsemble(hind_ds_initialized_3d)
    hindcast = hindcast.add_observations(reconstruction_ds_3d)
    hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
        init=slice("1964", "2014")
    ).mean("init")
    return hindcast


@pytest.fixture()
def hindcast_recon_1d_ym(hind_ds_initialized_1d, reconstruction_ds_1d):
    """HindcastEnsemble initialized with `initialized` and `recon`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d).add_observations(
        reconstruction_ds_1d
    )
    hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
        init=slice("1964", "2014")
    ).mean("init")
    hindcast._datasets["initialized"]["SST"].attrs = hind_ds_initialized_1d["SST"].attrs
    hindcast._datasets["observations"]["SST"].attrs = reconstruction_ds_1d["SST"].attrs
    return hindcast


@pytest.fixture()
def hindcast_hist_obs_1d(
    hind_ds_initialized_1d, hist_ds_uninitialized_1d, observations_ds_1d
):
    """HindcastEnsemble initialized with `initialized`, `uninitialzed` and `obs`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    hindcast = hindcast.add_observations(observations_ds_1d)
    with xr.set_options(keep_attrs=True):
        hindcast = hindcast - hindcast.sel(time=slice("1964", "2014")).mean("time").sel(
            init=slice("1964", "2014")
        ).mean("init")
    return hindcast


def hindcast_obs_1d_for_alignment(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime
):
    """HindcastEnsemble initialized with `initialized`, `uninitialzed` and `obs`."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d_cftime)
    hindcast = hindcast.add_observations(reconstruction_ds_1d_cftime)
    return hindcast


@pytest.fixture()
def reconstruction_ds_1d_mm(reconstruction_ds_1d_cftime):
    """CESM-FOSI historical reconstruction timeseries members mean removed
    xr.Dataset in monthly interpolated."""
    return reconstruction_ds_1d_cftime.resample(time="1MS").interpolate("linear")


@pytest.fixture()
def hindcast_recon_1d_mm(hindcast_recon_1d_ym, reconstruction_ds_1d_mm):
    """HindcastEnsemble with initialized and reconstruction (observations) as a monthly
    observational and initialized time series (no grid)."""
    hind = hindcast_recon_1d_ym.get_initialized().sel(init=slice("1964", "1970"))
    del hind.coords["valid_time"]
    hind["lead"].attrs["units"] = "months"
    hindcast = HindcastEnsemble(hind)
    hindcast = hindcast.add_observations(reconstruction_ds_1d_mm)
    return hindcast


@pytest.fixture()
def hindcast_recon_1d_dm(hindcast_recon_1d_ym):
    """HindcastEnsemble with initialized and reconstruction (observations) as a daily
    time series (no grid)."""
    hindcast = hindcast_recon_1d_ym.sel(time=slice("1964", "1970"))
    hindcast._datasets["initialized"].lead.attrs["units"] = "days"
    hindcast._datasets["observations"] = (
        hindcast._datasets["observations"].resample(time="1D").interpolate("linear")
    )
    hindcast._datasets["observations"].attrs = hindcast_recon_1d_ym._datasets[
        "observations"
    ]

    if (
        "units" not in hindcast.get_initialized()["SST"].attrs
        or "units" not in hindcast_recon_1d_ym.get_observations()["SST"].attrs
        or "units" not in hindcast.get_observations()["SST"].attrs
    ):
        raise ValueError("Units should be present in hindcast. Verify testing data.")
    return hindcast


@pytest.fixture()
def hindcast_S2S_Germany():
    """S2S ECMWF on-the-fly hindcasts with daily leads and weekly inits and related
    observations from CPC (t2m, pr) and ERA5 (gh_500)."""
    init = load_dataset("ECMWF_S2S_Germany")
    obs = load_dataset("Observations_Germany")
    return HindcastEnsemble(init).add_observations(obs)


@pytest.fixture()
def hindcast_NMME_Nino34():
    """NMME hindcasts with monthly leads and monthly inits and related IOv2
    observations for SST of the Nino34 region."""
    if Version(np.__version__) >= Version("2.0.0") and Version(
        xr.__version__
    ) <= Version("2024.6.0"):
        warnings.warn("Skipping test due to incompatible numpy and xarray versions.")
        pytest.skip("Changes in numpy>=2.0.0 break xarray<=2024.6.0.")

    init = load_dataset("NMME_hindcast_Nino34_sst")
    obs = load_dataset("NMME_OIv2_Nino34_sst")
    init["sst"].attrs["units"] = "C"
    obs["sst"].attrs["units"] = "C"
    return HindcastEnsemble(init).add_observations(
        obs.broadcast_like(init, exclude=("L", "S", "M"))
    )


@pytest.fixture(scope="session")
def _da_lead():
    """Small xr.DataArray with coords `init` and `lead`."""
    lead = np.arange(5)
    init = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init)),
        dims=["init", "lead"],
        coords=[init, lead],
    )


@pytest.fixture()
def da_lead(_da_lead):
    return _da_lead.copy(deep=True)


@pytest.fixture(scope="session")
def _ds1():
    """Small plain multi-dimensional coords xr.Dataset."""
    return xr.Dataset(
        {"air": (("lon", "lat"), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={"lon": [1, 3, 4], "lat": [5, 6, 7]},
    )


@pytest.fixture()
def ds1(_ds1):
    return _ds1.copy(deep=True)


@pytest.fixture(scope="session")
def _ds2():
    """Small plain multi-dimensional coords xr.Dataset identical values but with
    different coords compared to ds1."""
    return xr.Dataset(
        {"air": (("lon", "lat"), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={"lon": [1, 3, 6], "lat": [5, 6, 9]},
    )


@pytest.fixture()
def ds2(_ds2):
    return _ds2.copy(deep=True)


@pytest.fixture(scope="session")
def _da1():
    """Small plain two-dimensional"""
    return xr.DataArray([[0, 1], [3, 4], [6, 7]], dims=("x", "y"))


@pytest.fixture()
def da1(_da1):
    return _da1.copy(deep=True)


@pytest.fixture(scope="session")
def _da2():
    """Small plain two-dimensional xr.DataArray with different values compared to
    da1."""
    return xr.DataArray([[0, 1], [5, 6], [6, 7]], dims=("x", "y"))


@pytest.fixture()
def da2(_da2):
    return _da2.copy(deep=True)


@pytest.fixture(scope="session")
def _multi_dim_ds():
    """xr.Dataset with multi-dimensional coords."""
    ds = xr.tutorial.open_dataset("air_temperature")
    ds = ds.assign(**{"airx2": ds["air"] * 2})
    return ds


@pytest.fixture()
def multi_dim_ds(_multi_dim_ds):
    return _multi_dim_ds.copy(deep=True)


@pytest.fixture(scope="session")
def _da_SLM():
    """Small xr.DataArray with dims `S`, `M` and  `L` for `init`, `member` and"""
    lead = np.arange(5)
    init = np.arange(5)
    member = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init), len(member)),
        dims=["S", "L", "M"],
        coords=[init, lead, member],
    )


@pytest.fixture()
def da_SLM(_da_SLM):
    return _da_SLM.copy(deep=True)


@pytest.fixture(scope="session")
def _da_dcpp():
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


@pytest.fixture()
def da_dcpp(_da_dcpp):
    return _da_dcpp.copy(deep=True)


@pytest.fixture(scope="session")
def _PM_ds_initialized_1d_ym_cftime(_PM_ds_initialized_1d):
    return convert_time_index(
        _PM_ds_initialized_1d,
        "init",
        "PM_ds_initialized_1d.init",
        calendar=PM_CALENDAR_STR,
    ).load()


@pytest.fixture()
def PM_ds_initialized_1d_ym_cftime(_PM_ds_initialized_1d_ym_cftime):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    cftime."""
    ds = _PM_ds_initialized_1d_ym_cftime.copy(deep=True)
    ds["lead"].attrs["units"] = "years"
    return ds


@pytest.fixture(scope="session")
def _PM_ds_control_1d_ym_cftime(_PM_ds_control_1d):
    return convert_time_index(
        _PM_ds_control_1d, "time", "PM_ds_control_1d.time", calendar=PM_CALENDAR_STR
    ).load()


@pytest.fixture()
def PM_ds_control_1d_ym_cftime(_PM_ds_control_1d_ym_cftime):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime."""
    return _PM_ds_control_1d_ym_cftime.copy(deep=True)


@pytest.fixture()
def perfectModelEnsemble_initialized_control_1d_ym_cftime(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries annual mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_ym_cftime)
    pm = pm.add_control(PM_ds_control_1d_ym_cftime)
    return pm


@pytest.fixture(scope="session")
def _PM_ds_initialized_1d_mm_cftime(_PM_ds_initialized_1d):
    ds = _PM_ds_initialized_1d.copy(deep=True)
    ds["init"] = xr.cftime_range(
        start="3004",
        periods=ds.init.size,
        freq="MS",
        calendar=CALENDAR,
    )
    return ds.load()


@pytest.fixture()
def PM_ds_initialized_1d_mm_cftime(_PM_ds_initialized_1d_mm_cftime):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    monthly."""
    ds = _PM_ds_initialized_1d_mm_cftime.copy(deep=True)
    ds["lead"].attrs["units"] = "months"
    return ds


@pytest.fixture(scope="session")
def _PM_ds_control_1d_mm_cftime(_PM_ds_control_1d):
    ds = _PM_ds_control_1d.copy(deep=True)
    ds["time"] = xr.cftime_range(
        start="3000", periods=ds.time.size, freq="MS", calendar=CALENDAR
    )
    return ds.load()


@pytest.fixture()
def PM_ds_control_1d_mm_cftime(_PM_ds_control_1d_mm_cftime):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime faking the time resolution to monthly means."""
    return _PM_ds_control_1d_mm_cftime.copy(deep=True)


@pytest.fixture()
def perfectModelEnsemble_initialized_control_1d_mm_cftime(
    PM_ds_initialized_1d_mm_cftime, PM_ds_control_1d_mm_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries monthly mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_mm_cftime)
    pm = pm.add_control(PM_ds_control_1d_mm_cftime)
    return pm


@pytest.fixture(scope="session")
def _PM_ds_initialized_1d_dm_cftime(_PM_ds_initialized_1d):
    ds = _PM_ds_initialized_1d.copy(deep=True)
    ds["init"] = xr.cftime_range(
        start="3004",
        periods=ds.init.size,
        freq="D",
        calendar=CALENDAR,
    )
    return ds.load()


@pytest.fixture()
def PM_ds_initialized_1d_dm_cftime(_PM_ds_initialized_1d_dm_cftime):
    """MPI Perfect-model-framework initialized timeseries xr.Dataset with init as
    daily."""
    ds = _PM_ds_initialized_1d_dm_cftime.copy(deep=True)
    ds["lead"].attrs["units"] = "days"
    return ds


@pytest.fixture(scope="session")
def _PM_ds_control_1d_dm_cftime(_PM_ds_control_1d):
    ds = _PM_ds_control_1d.copy(deep=True)
    # session scope randomization is fine here as it's just for testing coverage
    ds = ds.isel(time=np.random.randint(0, ds.time.size, 5000))
    ds["time"] = xr.cftime_range(
        start="3000", periods=ds.time.size, freq="D", calendar=CALENDAR
    )
    return ds.load()


@pytest.fixture()
def PM_ds_control_1d_dm_cftime(_PM_ds_control_1d_dm_cftime):
    """To MPI Perfect-model-framework corresponding control timeseries xr.Dataset with
    time as cftime faking the time resolution to daily means."""
    return _PM_ds_control_1d_dm_cftime.copy(deep=True)


@pytest.fixture()
def perfectModelEnsemble_initialized_control_1d_dm_cftime(
    PM_ds_initialized_1d_dm_cftime, PM_ds_control_1d_dm_cftime
):
    """PerfectModelEnsemble with MPI Perfect-model-framework initialized and control
    timeseries daily mean with cftime coords."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d_dm_cftime)
    pm = pm.add_control(PM_ds_control_1d_dm_cftime)
    return pm


@pytest.fixture(scope="session")
def _small_initialized_da():
    """Very small simulation of an initialized forecasting system."""
    inits = [1990, 1991, 1992, 1993]
    lead = [1]
    return xr.DataArray(
        np.random.rand(len(inits), len(lead)),
        dims=["init", "lead"],
        coords=[inits, lead],
        name="var",
    )


@pytest.fixture()
def small_initialized_da(_small_initialized_da):
    return _small_initialized_da.copy(deep=True)


@pytest.fixture(scope="session")
def _small_verif_da():
    """Very small simulation of a verification product."""
    time = [1990, 1991, 1992, 1993, 1994]
    return xr.DataArray(
        np.random.rand(len(time)), dims=["time"], coords=[time], name="var"
    )


@pytest.fixture()
def small_verif_da(_small_verif_da):
    return _small_verif_da.copy(deep=True)
