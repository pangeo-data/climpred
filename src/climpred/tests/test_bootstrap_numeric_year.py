"""Numeric year selection for synthetic control simulations."""

import cftime
import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble
from climpred.bootstrap import bootstrap_uninit_pm_ensemble_from_control_cftime


@pytest.fixture(autouse=True)
def add_standard_imports():
    """Replace the inherited doctest setup with isolated synthetic-test RNG state.

    The parent fixture loads downloaded ensemble data for doctest examples. These
    tests build their own inputs, so they need only repeatable random draws.
    """
    state = np.random.get_state()
    np.random.seed(11)
    yield
    np.random.set_state(state)


def make_inputs(year, calendar, units):
    date_type = {
        "noleap": cftime.DatetimeNoLeap,
        "360_day": cftime.Datetime360Day,
        "all_leap": cftime.DatetimeAllLeap,
    }[calendar]
    start = date_type(year, 1, 1, has_year_zero=True)
    inits = xr.date_range(start, periods=2, freq="YS", use_cftime=True)
    times = xr.date_range(
        start,
        periods=240 if units == "months" else 365 * 20,
        freq="MS" if units == "months" else "D",
        use_cftime=True,
    )
    initialized = xr.Dataset(
        {"v": (("init", "lead", "member"), np.zeros((2, 3, 2)))},
        coords={"init": inits, "lead": [1, 2, 3], "member": [1, 2]},
    )
    initialized.lead.attrs["units"] = units
    control = xr.Dataset(
        {"v": ("time", np.arange(times.size, dtype=float))},
        coords={"time": times},
    )
    return initialized, control


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "all_leap"])
@pytest.mark.parametrize("year", [300, 1000, 2000, 0, -30, 10000])
@pytest.mark.parametrize("units", ["months", "days"])
def test_numeric_year_bootstrap(year, calendar, units):
    initialized, control = make_inputs(year, calendar, units)
    result = bootstrap_uninit_pm_ensemble_from_control_cftime(initialized, control)
    assert dict(result.sizes) == dict(initialized.sizes)
    assert result.v.notnull().all()
    xr.testing.assert_identical(result.lead, initialized.lead)
    xr.testing.assert_identical(result.member, initialized.member)
    # Each pseudo-member must be a contiguous three-step control segment.
    np.testing.assert_array_equal(result.v.diff("lead"), 1)


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "all_leap"])
def test_public_generate_uninitialized_early_year(calendar):
    initialized, control = make_inputs(300, calendar, "months")
    result = (
        PerfectModelEnsemble(initialized).add_control(control).generate_uninitialized()
    )
    assert result.get_uninitialized().v.notnull().all()


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "all_leap"])
@pytest.mark.parametrize("units", ["months", "days"])
def test_calendar_year_shift_preserves_sampled_control_values(calendar, units):
    """Identical random draws must select the same positions after a year shift."""
    early, early_control = make_inputs(300, calendar, units)
    modern, modern_control = make_inputs(2000, calendar, units)
    expected = bootstrap_uninit_pm_ensemble_from_control_cftime(modern, modern_control)
    # Replay the modern-control draw for the otherwise identical early control.
    # Values are positional control samples, not verification skill scores.
    np.random.seed(11)
    actual = bootstrap_uninit_pm_ensemble_from_control_cftime(early, early_control)
    np.testing.assert_array_equal(actual.v.values, expected.v.values)
