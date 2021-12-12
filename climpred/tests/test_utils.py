"""Test utils.py"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.comparisons import PM_COMPARISONS, __m2c
from climpred.metrics import DETERMINISTIC_PM_METRICS, __pearson_r
from climpred.utils import (
    convert_time_index,
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


@pytest.mark.parametrize("metric", ["mse", "pearson_r"])
def test_PerfectModelEnsemble_bootstrap_attrs(
    perfectModelEnsemble_initialized_control, metric
):
    """Test assigning attrs for PerfectModelEnsemble.bootstrap()."""
    comparison = "m2e"
    ITERATIONS = 2
    sig = 95
    v = "tos"
    perfectModelEnsemble_initialized_control._datasets["initialized"][v].attrs[
        "units"
    ] = "C"
    actual = perfectModelEnsemble_initialized_control.bootstrap(
        metric=metric,
        comparison=comparison,
        dim=["init"],
        iterations=ITERATIONS,
        sig=sig,
    )
    assert actual.attrs["metric"] == metric
    assert actual.attrs["comparison"] == comparison
    assert actual.attrs["iterations"] == ITERATIONS
    assert (
        str(round((1 - sig / 100) / 2, 3)) in actual.attrs["confidence_interval_levels"]
    )
    if metric == "pearson_r":
        assert actual[v].attrs["units"] == "None"
    else:
        assert actual[v].attrs["units"] == "(C)^2"


@pytest.mark.parametrize("metric", ["mse", "pearson_r"])
def test_HindcastEnsemble_bootstrap_attrs(hindcast_hist_obs_1d, metric):
    """Test assigning attrs for HindcastEnsemble.bootstrap()."""
    comparison = "e2o"
    alignment = "same_verif"
    v = "SST"
    iterations = 2
    sig = 95
    hindcast_hist_obs_1d._datasets["initialized"][v].attrs["units"] = "C"
    actual = hindcast_hist_obs_1d.bootstrap(
        metric=metric,
        comparison=comparison,
        iterations=iterations,
        dim="init",
        alignment=alignment,
        sig=sig,
    )
    assert actual.attrs["metric"] == metric
    assert actual.attrs["comparison"] == comparison
    assert actual.attrs["iterations"] == iterations
    assert actual.attrs["alignment"] == alignment
    assert (
        str(round((1 - sig / 100) / 2, 3)) in actual.attrs["confidence_interval_levels"]
    )
    if metric == "pearson_r":
        assert actual[v].attrs["units"] == "None"
    else:
        assert actual[v].attrs["units"] == "(C)^2"
    assert "description" in actual.results.attrs
    assert "description" in actual.skill.attrs


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


def test_add_time_from_init_lead(hindcast_recon_1d_mm):
    # todo improve
    assert (
        str(hindcast_recon_1d_mm.coords["valid_time"].isel(lead=0).to_index()[0])
        != "1965-01-01 00:00:00"
    ), print(hindcast_recon_1d_mm.coords)
