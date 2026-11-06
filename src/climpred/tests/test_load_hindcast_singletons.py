"""Hindcast loading must retain meaningful singleton axes."""

import numpy as np
import pytest
import xarray as xr

from climpred.preprocessing.shared import load_hindcast


@pytest.mark.parametrize("nlead", [1, 3])
@pytest.mark.parametrize("source_ensemble_axes", [False, True])
@pytest.mark.parametrize(
    "spatial_dims",
    [
        {"lat": [42.0]},
        {"lat": [42.0], "lon": [-125.0]},
        {"lat": [41.0, 42.0], "lon": [-125.0, -124.0]},
    ],
)
def test_loading_preserves_singleton_lead_and_spatial_axes(
    tmp_path, nlead, spatial_dims, source_ensemble_axes
):
    dims = ["time", *spatial_dims]
    shape = (nlead, *(len(v) for v in spatial_dims.values()))
    source = xr.Dataset(
        {"temperature": (dims, np.arange(np.prod(shape)).reshape(shape))},
        coords={"time": np.arange(nlead), **spatial_dims},
    )
    source.temperature.attrs["units"] = "K"
    for init in [2000, 2001]:
        for member in [1, 2]:
            file_source = source + init * 10 + member
            if source_ensemble_axes:
                file_source = file_source.expand_dims(init=[init], member=[member])
            file_source.to_netcdf(tmp_path / f"{init}-{member}.nc", engine="scipy")
    result = load_hindcast(
        inits=[2000, 2001],
        members=[1, 2],
        parallel=False,
        engine="scipy",
        get_path=lambda init, member: str(tmp_path / f"{init}-{member}.nc"),
    )
    assert result.sizes == {
        "init": 2,
        "member": 2,
        "lead": nlead,
        **{k: len(v) for k, v in spatial_dims.items()},
    }
    for d in spatial_dims:
        xr.testing.assert_equal(result[d], source[d])
    for init in [2000, 2001]:
        for member in [1, 2]:
            expected = (
                (source + init * 10 + member)
                .rename({"time": "lead"})
                .assign_coords(lead=np.arange(1, nlead + 1))
            )
            xr.testing.assert_equal(
                result.sel(init=init, member=member, drop=True).compute(), expected
            )
    result.close()
