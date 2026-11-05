"""Finite local NetCDF files exercise the public hindcast loading path."""

import numpy as np
import pytest
import xarray as xr

from climpred.preprocessing.shared import load_hindcast


@pytest.mark.parametrize("offset", [0, 2, -2, 1])
@pytest.mark.parametrize("ntime", [1, 3])
def test_local_hindcast_honors_offset_for_every_init_and_member(
    tmp_path, offset, ntime
):
    inits, members = [2000, 2001], [1, 3]
    for init in inits:
        for member in members:
            xr.Dataset(
                {
                    "temperature": (
                        "time",
                        init + member + np.arange(ntime, dtype=float),
                    )
                },
                coords={"time": np.arange(ntime) * 10 + 100},
            ).to_netcdf(tmp_path / f"{init}-{member}.nc", engine="scipy")

    def get_path(init, member):
        return str(tmp_path / f"{init}-{member}.nc")

    actual = load_hindcast(
        inits=inits,
        members=members,
        get_path=get_path,
        lead_offset=offset,
        parallel=False,
        engine="scipy",
    )
    try:
        np.testing.assert_array_equal(actual.lead, np.arange(offset, offset + ntime))
        assert dict(actual.sizes) == {"init": 2, "member": 2, "lead": ntime}
        np.testing.assert_array_equal(actual.init, inits)
        np.testing.assert_array_equal(actual.member, members)
        for init in inits:
            for member in members:
                np.testing.assert_array_equal(
                    actual.temperature.sel(init=init, member=member),
                    init + member + np.arange(ntime, dtype=float),
                )
    finally:
        actual.close()


@pytest.mark.parametrize("offset", [0, 4])
def test_offset_keeps_preprocess_and_path_options(tmp_path, offset):
    path = tmp_path / "single.nc"
    xr.Dataset(
        {"a": ("time", [1.0, 2.0, 3.0])}, coords={"time": [10, 20, 30]}
    ).to_netcdf(path, engine="scipy")
    calls = []

    def get_path(init, member, suffix):
        calls.append((init, member, suffix))
        return str(path)

    actual = load_hindcast(
        inits=[2005],
        members=[7],
        get_path=get_path,
        suffix="selected",
        preprocess=lambda data: data * 2,
        lead_offset=offset,
        parallel=False,
        engine="scipy",
    )
    try:
        assert calls == [(2005, 7, "selected")]
        np.testing.assert_array_equal(actual.lead, np.arange(offset, offset + 3))
        np.testing.assert_array_equal(actual.a.values.ravel(), [2.0, 4.0, 6.0])
    finally:
        actual.close()


def test_omitted_offset_keeps_default_one_based_leads(tmp_path):
    path = tmp_path / "single.nc"
    xr.Dataset({"a": ("time", [1.0, 2.0])}).to_netcdf(path, engine="scipy")
    actual = load_hindcast(
        inits=[2000],
        members=[1],
        get_path=lambda **kwargs: str(path),
        parallel=False,
        engine="scipy",
    )
    try:
        np.testing.assert_array_equal(actual.lead, [1, 2])
    finally:
        actual.close()
