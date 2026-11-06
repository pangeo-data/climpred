"""Canonical dimensions count as found without tightening accepted inputs."""

import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble
from climpred.preprocessing.shared import rename_to_climpred_dims


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize(
    "dims",
    [
        ("init", "ensemble_member", "lead_time"),
        ("init", "member", "lead_time"),
    ],
)
def test_rename_accepts_mixed_canonical_and_alias_dimensions(kind, dims):
    da = xr.DataArray(np.zeros([2] * len(dims)), dims=dims, name="tas")
    obj = da.to_dataset() if kind == "dataset" else da
    actual = rename_to_climpred_dims(obj)
    expected = obj.rename(
        {d: {"ensemble_member": "member", "lead_time": "lead"}.get(d, d) for d in dims}
    )
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("dims", [("init", "lead")])
def test_memberless_hindcast_remains_supported(dims):
    obj = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=dims,
        coords={dims[0]: [2000, 2001], dims[1]: [1, 2, 3]},
        name="tas",
    )
    obj[dims[1]].attrs["units"] = "years"
    actual = HindcastEnsemble(obj).get_initialized()
    assert actual.sizes == {"init": 2, "lead": 3}
    assert "member" not in actual.dims
    np.testing.assert_array_equal(actual.tas, obj)


@pytest.mark.parametrize("dims", [("time",), ("lead",), ("init", "time", "lead_time")])
def test_rename_preserves_existing_permissive_inputs_and_time_precedence(dims):
    obj = xr.DataArray(np.zeros([2] * len(dims)), dims=dims, name="tas")
    actual = rename_to_climpred_dims(obj)
    expected = obj.rename({"time": "lead"}) if "time" in dims else obj
    xr.testing.assert_identical(actual, expected)
