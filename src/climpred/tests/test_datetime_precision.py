"""Calendar conversion must not silently collapse distinct timestamps."""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_identical

from climpred import HindcastEnsemble
from climpred.utils import convert_time_index


@pytest.mark.parametrize("kind", ["dataset", "dataarray"])
def test_datetime_conversion_preserves_microseconds(kind):
    dates = pd.to_datetime(["2000-01-01T00:00:00.123456", "2000-01-01T00:00:00.234567"])
    obj = xr.DataArray([1.0, 2.0], dims="init", coords={"init": dates}, name="a")
    if kind == "dataset":
        obj = obj.to_dataset()
    before = obj.copy(deep=True)
    actual = convert_time_index(obj, "init")
    expected = obj.assign_coords(
        init=[
            cftime.DatetimeProlepticGregorian(2000, 1, 1, microsecond=us)
            for us in [123456, 234567]
        ]
    )
    assert_identical(actual, expected)
    assert actual.init.to_index().is_unique
    assert_identical(obj, before)


def test_public_hindcast_constructor_preserves_distinct_initializations():
    obj = xr.Dataset(
        {"a": (("init", "lead", "member"), np.ones((2, 2, 2)))},
        coords={
            "init": pd.to_datetime(["2000-01-01T00:00:00.1", "2000-01-01T00:00:00.2"]),
            "lead": [0, 1],
            "member": [0, 1],
        },
    )
    obj.lead.attrs["units"] = "seconds"
    actual = HindcastEnsemble(obj).get_initialized()
    assert actual.init.to_index().is_unique
    assert [t.microsecond for t in actual.init.values] == [100000, 200000]
    assert [t.microsecond for t in actual.valid_time.isel(lead=1).values] == [
        100000,
        200000,
    ]


def test_submicrosecond_conversion_rejects_instead_of_truncating():
    obj = xr.Dataset(coords={"time": pd.to_datetime(["2000-01-01T00:00:00.000000001"])})
    with pytest.raises(ValueError, match="sub-microsecond precision"):
        convert_time_index(obj, "time")


def test_whole_second_conversion_is_unchanged():
    obj = xr.Dataset(coords={"time": pd.to_datetime(["2000-01-01T01:02:03"])})
    expected = obj.assign_coords(
        time=[cftime.DatetimeProlepticGregorian(2000, 1, 1, 1, 2, 3)]
    )
    expected.time.attrs.update(long_name="time", standard_name="time")
    assert_identical(convert_time_index(obj, "time"), expected)
