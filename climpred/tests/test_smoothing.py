import numpy as np
import pytest
import xarray as xr

from climpred.prediction import compute_perfect_model
from climpred.smoothing import (
    _reset_temporal_axis,
    _set_center_coord,
    smooth_goddard_2013,
    temporal_smoothing,
)
from climpred.testing import assert_PredictionEnsemble

try:
    from climpred.smoothing import spatial_smoothing_xesmf
except ImportError:
    pass

from . import requires_xesmf


def test_reset_temporal_axis(PM_da_control_3d_full):
    """Test whether correct new labels are set."""
    smooth = 10
    tsmooth_kws = {"time": smooth}
    first_ori = PM_da_control_3d_full.time[0].values
    first_actual = _reset_temporal_axis(
        PM_da_control_3d_full, tsmooth_kws=tsmooth_kws, dim="time"
    ).time.values[0]
    first_expected = f"{first_ori}-{first_ori+smooth*1-1}"
    assert first_actual == first_expected


def test_reset_temporal_axis_lead(PM_da_initialized_3d_full):
    """Test whether correct new labels are set."""
    smooth = 10
    dim = "lead"
    tsmooth_kws = {dim: smooth}
    first_ori = PM_da_initialized_3d_full.lead[0].values
    first_actual = _reset_temporal_axis(
        PM_da_initialized_3d_full, tsmooth_kws=tsmooth_kws
    )[dim].values[0]
    first_expected = f"{first_ori}-{first_ori+smooth*1-1}"
    assert first_actual == first_expected


def test_temporal_smoothing_reduce_length(PM_da_control_3d_full):
    """Test whether dimsize is reduced properly."""
    smooth = 10
    tsmooth_kws = {"time": smooth}
    actual = temporal_smoothing(
        PM_da_control_3d_full, tsmooth_kws=tsmooth_kws
    ).time.size
    expected = PM_da_control_3d_full.time.size - smooth + 1
    assert actual == expected


@requires_xesmf
def test_spatial_smoothing_xesmf_reduce_spatial_dims_MPI_curv(
    PM_da_control_3d_full,
):
    """Test whether spatial dimsizes are properly reduced."""
    da = PM_da_control_3d_full
    step = 5
    actual = spatial_smoothing_xesmf(
        da,
        d_lon_lat_kws={"lon": step},
    )
    expected_lat_size = 180 // step
    assert actual["lon"].size < da.lon.size
    assert actual["lat"].size == expected_lat_size


@requires_xesmf
def test_spatial_smoothing_xesmf_reduce_spatial_dims_CESM(
    reconstruction_ds_3d_full,
):
    """Test whether spatial dimsizes are properly reduced."""
    da = reconstruction_ds_3d_full
    step = 0.1
    actual = spatial_smoothing_xesmf(
        da,
        d_lon_lat_kws={"lat": step},
    )
    # test whether upsampled
    assert actual["lon"].size >= da.nlon.size
    assert actual["lat"].size >= da.nlat.size


@requires_xesmf
def test_smooth_goddard_2013(PM_da_control_3d_full):
    """Test whether Goddard 2013 recommendations are fulfilled by
    smooth_Goddard_2013."""
    da = PM_da_control_3d_full
    actual = smooth_goddard_2013(
        da,
    )
    # test that x, y not in dims
    assert "x" not in actual.dims
    assert "y" not in actual.dims
    # tests whether nlat, nlon got reduced
    assert actual.time.size < da.time.size
    assert actual.lon.size < da.lon.size
    assert actual.lat.size < da.lat.size


@requires_xesmf
def test_compute_after_smooth_goddard_2013(
    PM_da_initialized_3d_full, PM_da_control_3d_full
):
    """Test compute_perfect_model works after smoothings."""
    PM_da_control_3d_full = smooth_goddard_2013(
        PM_da_control_3d_full,
    )
    PM_da_initialized_3d_full = smooth_goddard_2013(
        PM_da_initialized_3d_full,
    )
    actual = compute_perfect_model(PM_da_initialized_3d_full, PM_da_control_3d_full)
    north_atlantic = actual.sel(lat=slice(40, 50), lon=slice(-30, -20))
    assert not north_atlantic.isnull().any()


@pytest.mark.parametrize("smooth", [2, 4])
@pytest.mark.parametrize(
    "pm",
    [
        pytest.lazy_fixture("perfectModelEnsemble_initialized_control_1d_ym_cftime"),
        pytest.lazy_fixture("perfectModelEnsemble_initialized_control_1d_mm_cftime"),
        pytest.lazy_fixture("perfectModelEnsemble_initialized_control_1d_dm_cftime"),
    ],
)
def test_PerfectModelEnsemble_temporal_smoothing_cftime_and_skill(pm, smooth):
    """Test that PredictionEnsemble.smooth({'lead': int}) aggregates lead."""
    pm = pm.isel(lead=range(6))
    pm_smoothed = pm.smooth({"lead": smooth})
    assert (
        pm_smoothed.get_initialized().lead.size
        == pm.get_initialized().lead.size - smooth + 1
    )
    assert pm_smoothed._temporally_smoothed
    skill = pm_smoothed.verify(metric="acc", comparison="m2e", dim=["member", "init"])
    assert skill.lead.size == pm.get_initialized().lead.size - smooth + 1
    assert skill.lead[0] == f"1-{1+smooth-1}"


@pytest.mark.parametrize("dim", ["time", "lead"])
@pytest.mark.parametrize("smooth", [2, 4])
@pytest.mark.parametrize(
    "he",
    [
        pytest.lazy_fixture("hindcast_recon_1d_ym"),
        pytest.lazy_fixture("hindcast_recon_1d_mm"),
        pytest.lazy_fixture("hindcast_recon_1d_dm"),
    ],
)
def test_HindcastEnsemble_temporal_smoothing_cftime_and_skill(he, smooth, dim):
    """Test that HindcastEnsemble.smooth({dim: int}) aggregates lead regardless whether
    time or lead is given as dim."""
    he_smoothed = he.smooth({dim: smooth})
    assert (
        he_smoothed.get_initialized().lead.size
        == he.get_initialized().lead.size - smooth + 1
    )
    skill = he_smoothed.verify(
        metric="acc", comparison="e2o", alignment="maximize", dim="init"
    )
    assert skill.lead.size == he.get_initialized().lead.size - smooth + 1
    assert skill.lead[0] == f"1-{1+smooth-1}"


@requires_xesmf
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("dim", [["lon"], ["lat"], ["lon", "lat"]])
def test_HindcastEnsemble_spatial_smoothing_dim_and_skill(hindcast_recon_3d, dim, step):
    """Test that HindcastEnsemble.smooth({dim: int}) aggregates dim."""
    he = hindcast_recon_3d
    smooth_kws = {key: step for key in dim}
    he_smoothed = he.smooth(smooth_kws)
    assert he_smoothed.get_initialized().lead.attrs is not None
    for d in dim:
        assert he_smoothed.get_initialized()[d].any()
        assert he_smoothed.get_observations()[d].any()
    assert he_smoothed.verify(
        metric="acc", comparison="e2o", alignment="same_verif", dim="init"
    ).any()


def test_temporal_smoothing_how(perfectModelEnsemble_initialized_control_1d_ym_cftime):
    """Test that PerfectModelEnsemble can smooth by mean and sum aggregation."""
    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    pm_smoothed_mean = pm.smooth({"lead": 4}, how="mean")
    pm_smoothed_sum = pm.smooth({"lead": 4}, how="sum")
    assert (
        pm_smoothed_sum.get_initialized().mean()
        > pm_smoothed_mean.get_initialized().mean() * 2
    )


@requires_xesmf
def test_spatial_smoothing_xesmf(hindcast_recon_3d):
    """Test different regridding methods from xesmf.regrid kwargs yield different
    results."""
    he = hindcast_recon_3d
    he_bil = he.smooth("goddard", method="bilinear")
    he_patch = he.smooth("goddard", method="patch")
    assert he_bil.get_initialized().mean() != he_patch.get_initialized().mean()


def test_set_center_coord():
    """Test that center coords are set to the middle of the lead range."""
    da = xr.DataArray(np.arange(2), dims="lead", coords={"lead": ["1-3", "2-4"]})
    actual = _set_center_coord(da).lead_center.values
    expected = [2.0, 3.0]
    assert (actual == expected).all()


def test_PerfectModelEnsemble_smooth_carries_lead_attrs(
    perfectModelEnsemble_initialized_control_1d_ym_cftime,
):
    """Test that PerfectModelEnsemble carries lead attrs after smooth  and verify."""
    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    pm_smooth = pm.smooth({"lead": 4}, how="mean")
    assert (
        pm_smooth.verify(metric="rmse", comparison="m2e", dim="init").lead.attrs[
            "units"
        ]
        == "years"
    )


def test_HindcastEnsemble_smooth_carries_lead_attrs(hindcast_recon_1d_ym):
    """Test that HindcastEnsemble carries lead attrs after smooth  and verify."""
    he = hindcast_recon_1d_ym
    he_smooth = he.smooth({"lead": 4}, how="mean")
    assert (
        he_smooth.verify(
            metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
        ).lead.attrs["units"]
        == "years"
    )


@requires_xesmf
@pytest.mark.parametrize(
    "smooth", [{"lead": 4, "lon": 5, "lat": 5}, "goddard", "goddard2013"]
)
def test_PredictionEnsemble_goddard(
    perfectModelEnsemble_initialized_control_1d_ym_cftime, smooth
):
    """Test that PredictionEnsemble.smooth() understands goodard keys and does multiple
    smoothings in one call."""
    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    assert pm.smooth(smooth)


def test_PredictionEnsemble_smooth_None(
    perfectModelEnsemble_initialized_control_1d_ym_cftime,
):
    """Test that PredictionEnsemble.smooth(None) does nothing."""
    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    pm_smoothed = pm.smooth(None)
    assert_PredictionEnsemble(pm, pm_smoothed)
