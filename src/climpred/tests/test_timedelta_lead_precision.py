"""Timedelta lead conversion must not silently truncate forecast periods."""

from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble
from climpred.utils import convert_Timedelta_to_lead_units


@pytest.mark.parametrize(
    "leads", [[100, 200], [1100, 1900], [-100, 100], [-1100, 1900]]
)
@pytest.mark.parametrize("public", [False, True])
def test_subsecond_timedelta_leads_fail_without_truncation(leads, public):
    original = xr.DataArray(
        np.arange(8).reshape(2, 2, 2),
        dims=("init", "member", "lead"),
        coords={
            "init": xr.date_range("2000", periods=2, freq="YS", use_cftime=True),
            "member": [1, 2],
            "lead": np.array(leads, dtype="timedelta64[ms]"),
        },
        name="temperature",
    )
    unchanged = original.copy(deep=True)
    with pytest.raises(ValueError, match="whole seconds"):
        if public:
            HindcastEnsemble(original)
        else:
            convert_Timedelta_to_lead_units(original)
    xr.testing.assert_identical(original, unchanged)


@pytest.mark.parametrize(
    "leads,units",
    [
        ([1000, 2000], "seconds"),
        ([60000, 120000], "minutes"),
        ([3600000, 7200000], "hours"),
        ([-1000, 1000], "seconds"),
    ],
)
def test_whole_second_timedelta_conversion_unchanged(leads, units):
    original = xr.DataArray(
        np.zeros((2, 2)),
        dims=("init", "lead"),
        coords={
            "init": xr.date_range("2000", periods=2, freq="YS", use_cftime=True),
            "lead": np.array(leads, dtype="timedelta64[ms]"),
        },
        name="temperature",
    )
    converted = HindcastEnsemble(original).get_initialized()
    assert converted.lead.attrs["units"] == units
    scale = {"seconds": 1000, "minutes": 60000, "hours": 3600000}[units]
    np.testing.assert_array_equal(converted.lead, np.array(leads) / scale)
    for i, millis in enumerate(leads):
        assert converted.valid_time.isel(lead=i).values.tolist() == [
            v + timedelta(milliseconds=millis) for v in original.init.values
        ]


@pytest.mark.parametrize("leads", [[1, 2], [-1, 1]])
def test_nanosecond_timedelta_leads_are_not_silently_zeroed(leads):
    source = xr.Dataset(coords={"lead": np.array(leads, dtype="timedelta64[ns]")})
    with pytest.raises(ValueError, match="whole seconds"):
        convert_Timedelta_to_lead_units(source)


def test_missing_timedelta_leads_are_rejected_explicitly():
    source = xr.Dataset(coords={"lead": np.array(["NaT", 1], dtype="timedelta64[s]")})
    with pytest.raises(ValueError, match="must not contain missing"):
        convert_Timedelta_to_lead_units(source)
