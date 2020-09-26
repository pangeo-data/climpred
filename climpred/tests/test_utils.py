import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import PM_COMPARISONS, __m2c
from climpred.metrics import DETERMINISTIC_PM_METRICS, __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from climpred.utils import (
    convert_time_index,
    copy_coords_from_to,
    find_start_dates_for_given_init,
    get_comparison_class,
    get_metric_class,
    intersect,
    lead_units_equal_control_time_stride,
    return_time_series_freq,
    shift_cftime_index,
    shift_cftime_singular,
)


def test_get_metric_class():
    """Test if passing in a string gets the right metric function."""
    actual = get_metric_class("pearson_r", DETERMINISTIC_PM_METRICS).name
    expected = __pearson_r.name
    assert actual == expected


def test_get_metric_class_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_metric_class("not_metric", DETERMINISTIC_PM_METRICS)
    assert "Specify metric from" in str(excinfo.value)


def test_get_comparison_class():
    """Test if passing in a string gets the right comparison function."""
    actual = get_comparison_class("m2c", PM_COMPARISONS).name
    expected = __m2c.name
    assert actual == expected


def test_get_comparison_class_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_comparison_class("not_comparison", PM_COMPARISONS)
    assert "Specify comparison from" in str(excinfo.value)


def test_intersect():
    """Test if the intersect (overlap) of two lists work."""
    x = [1, 5, 6]
    y = [1, 6, 7]
    actual = intersect(x, y)
    expected = np.array([1, 6])
    assert all(a == e for a, e in zip(actual, expected))


def test_da_assign_attrs(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test assigning attrs for compute_perfect_model and dataarrays."""
    metric = "pearson_r"
    comparison = "m2e"
    actual = compute_perfect_model(
        PM_ds_initialized_1d, PM_ds_control_1d, metric=metric, comparison=comparison,
    ).attrs
    assert actual["metric"] == metric
    assert actual["comparison"] == comparison
    if metric == "pearson_r":
        assert actual["units"] == "None"
    assert actual["skill_calculated_by_function"] == "compute_perfect_model"
    assert (
        actual["prediction_skill"]
        == "calculated by climpred https://climpred.readthedocs.io/"
    )


def test_ds_assign_attrs(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test assigning attrs for datasets."""
    metric = "mse"
    comparison = "m2e"
    dim = ["init", "member"]
    PM_ds_initialized_1d.attrs["units"] = "C"
    actual = compute_perfect_model(
        PM_ds_initialized_1d,
        PM_ds_control_1d,
        metric=metric,
        comparison=comparison,
        dim=dim,
    ).attrs
    assert actual["metric"] == metric
    assert actual["comparison"] == comparison
    if metric == "pearson_r":
        assert actual["units"] == "None"
    assert actual["skill_calculated_by_function"] == "compute_perfect_model"
    assert actual["units"] == "(C)^2"
    assert actual["dim"] == dim


def test_bootstrap_pm_assign_attrs():
    """Test assigning attrs for bootstrap_perfect_model."""
    v = "tos"
    metric = "pearson_r"
    comparison = "m2e"
    ITERATIONS = 3
    sig = 95
    da = load_dataset("MPI-PM-DP-1D")[v].isel(area=1, period=-1)
    control = load_dataset("MPI-control-1D")[v].isel(area=1, period=-1)
    actual = bootstrap_perfect_model(
        da,
        control,
        metric=metric,
        comparison=comparison,
        iterations=ITERATIONS,
        sig=sig,
    ).attrs
    assert actual["metric"] == metric
    assert actual["comparison"] == comparison
    assert actual["bootstrap_iterations"] == ITERATIONS
    assert str(round((1 - sig / 100) / 2, 3)) in actual["confidence_interval_levels"]
    if metric == "pearson_r":
        assert actual["units"] == "None"
    assert "bootstrap" in actual["skill_calculated_by_function"]


def test_hindcast_assign_attrs():
    """Test assigning attrs for compute_hindcast."""
    metric = "pearson_r"
    comparison = "e2o"
    da = load_dataset("CESM-DP-SST")
    control = load_dataset("ERSST")
    actual = compute_hindcast(da, control, metric=metric, comparison=comparison).attrs
    assert actual["metric"] == metric
    assert actual["comparison"] == comparison
    if metric == "pearson_r":
        assert actual["units"] == "None"
    assert actual["skill_calculated_by_function"] == "compute_hindcast"


def test_copy_coords_from_to_ds(PM_ds_control_3d):
    """Test whether coords are copied from one xr object to another."""
    xro = PM_ds_control_3d
    c_1time = xro.isel(time=4).drop_vars("time")
    assert "time" not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_da(PM_da_control_3d):
    """Test whether coords are copied from one xr object to another."""
    xro = PM_da_control_3d
    c_1time = xro.isel(time=4).drop_vars("time")
    assert "time" not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_ds_chunk(PM_ds_control_3d):
    """Test whether coords are copied from one xr object to another."""
    xro = PM_ds_control_3d.chunk({"time": 5})
    c_1time = xro.isel(time=4).drop_vars("time")
    assert "time" not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_da_different_xro(PM_ds_control_3d):
    xro = PM_ds_control_3d.chunk({"time": 5})
    c_1time = xro.isel(time=4).drop_vars("time")
    with pytest.raises(ValueError) as excinfo:
        copy_coords_from_to(xro.isel(time=2).tos, c_1time)
    assert "xro_from and xro_to must be both either" in str(excinfo.value)


def test_cftime_index_unchanged():
    """Tests that a CFTime index going through convert time is unchanged."""
    inits = xr.cftime_range("1990", "2000", freq="Y", calendar="noleap")
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert_allclose(new_inits.init, da.init)


def test_pandas_datetime_converted_to_cftime():
    """Tests that a pd.DatetimeIndex is converted to xr.CFTimeIndex."""
    inits = pd.date_range("1990", "2000", freq="YS")
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert isinstance(new_inits["init"].to_index(), xr.CFTimeIndex)


def test_int64_converted_to_cftime():
    """Tests the xr.Int64Index is converted to xr.CFTimeIndex."""
    inits = np.arange(1990, 2000)
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert isinstance(new_inits["init"].to_index(), xr.CFTimeIndex)


def test_float64_converted_to_cftime():
    """Tests the xr.Float64Index is converted to xr.CFTimeIndex."""
    inits = np.arange(1990, 2000) * 1.0
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert isinstance(new_inits["init"].to_index(), xr.CFTimeIndex)


def test_numeric_index_auto_appends_lead_attrs():
    """Tests that for numeric inits, lead units are automatically set to 'years'"""
    lead = np.arange(3)
    int_inits = np.arange(1990, 2000)
    float_inits = int_inits * 1.0
    int_da = xr.DataArray(
        np.random.rand(len(int_inits), len(lead)),
        dims=["init", "lead"],
        coords=[int_inits, lead],
    )
    float_da = xr.DataArray(
        np.random.rand(len(float_inits), len(lead)),
        dims=["init", "lead"],
        coords=[float_inits, lead],
    )
    new_int_da = convert_time_index(int_da, "init", "")
    new_float_da = convert_time_index(float_da, "init", "")
    assert new_int_da.lead.attrs["units"] == "years"
    assert new_float_da.lead.attrs["units"] == "years"


def test_convert_time_index_does_not_overwrite():
    """Tests that `convert_time_index` does not overwrite the original index."""
    inits = np.arange(1990, 2000)
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert isinstance(da.init.to_index(), pd.Int64Index)
    assert isinstance(new_inits.init.to_index(), xr.CFTimeIndex)


def test_irregular_initialization_dates():
    """Tests that irregularly spaced initializations convert properly."""
    inits = np.arange(1990, 2010)
    inits = np.delete(inits, [3, 5, 8, 12, 15])
    da = xr.DataArray(np.random.rand(len(inits)), dims="init", coords=[inits])
    new_inits = convert_time_index(da, "init", "")
    assert (new_inits["init"].to_index().year == inits).all()


def test_shift_cftime_singular():
    """Tests that a singular ``cftime`` is shifted the appropriate amount."""
    cftime_initial = cftime.DatetimeNoLeap(1990, 1, 1)
    cftime_expected = cftime.DatetimeNoLeap(1990, 3, 1)
    # Shift forward two months at month start.
    cftime_from_func = shift_cftime_singular(cftime_initial, 2, "MS")
    assert cftime_expected == cftime_from_func


def test_shift_cftime_index():
    """Tests that ``CFTimeIndex`` is shifted by the appropriate amount."""
    idx = xr.cftime_range("1990", "2000", freq="YS")
    da = xr.DataArray(np.random.rand(len(idx)), dims="time", coords=[idx])
    expected = idx.shift(3, "YS")
    res = shift_cftime_index(da, "time", 3, "YS")
    assert (expected == res).all()


@pytest.mark.parametrize(
    "init",
    [
        pytest.lazy_fixture("PM_ds_initialized_1d_ym_cftime"),
        pytest.lazy_fixture("PM_ds_initialized_1d_mm_cftime"),
        pytest.lazy_fixture("PM_ds_initialized_1d_dm_cftime"),
    ],
)
def test_return_time_series_freq_freq_init_pm(init):
    """Test that return_time_series_freq returns expected freq for different lead
    units."""
    actual = return_time_series_freq(init, "init")
    expected = init.lead.attrs["units"].strip("s")
    assert actual == expected


@pytest.mark.parametrize(
    "init, control",
    [
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_ym_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_ym_cftime"),
        ),
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_mm_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_mm_cftime"),
        ),
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_dm_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_dm_cftime"),
        ),
    ],
)
def test_lead_units_equal_control_time_stride_freq(init, control):
    """Test that init_pm and control are compatible when both same freq."""
    assert lead_units_equal_control_time_stride(init, control)


def test_lead_units_equal_control_time_stride_daily_fails(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
):
    """Test that init_pm annual and control daily is not compatible."""
    with pytest.raises(ValueError) as excinfo:
        lead_units_equal_control_time_stride(
            PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
        )
        assert "Please provide the same temporal resolution for control.time" in str(
            excinfo.value
        )


def test_find_start_dates_for_given_init(
    PM_ds_initialized_1d_mm_cftime, PM_ds_control_1d_mm_cftime
):
    """Test that start dates are one year apart."""
    for init in PM_ds_initialized_1d_mm_cftime.init:
        start_dates = find_start_dates_for_given_init(PM_ds_control_1d_mm_cftime, init)
        freq = return_time_series_freq(PM_ds_initialized_1d_mm_cftime, "init")
        assert return_time_series_freq(start_dates, "time") == "year"
        assert (getattr(start_dates.time.dt, freq) == getattr(init.dt, freq)).all()
        # same number of start dates are years or one less
        assert start_dates.time.size - len(
            np.unique(PM_ds_control_1d_mm_cftime.time.dt.year.values)
        ) in [0, 1]
