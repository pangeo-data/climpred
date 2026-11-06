"""Temporal aggregation must be consistent with and without spatial smoothing."""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_identical

from climpred import HindcastEnsemble, smoothing


@pytest.fixture
def spatial_identity(monkeypatch):
    # Keep these tests independent of ESMF: exercise the real temporal operation
    # and wrapper, while replacing only the separate spatial operation.
    seen = []

    def regrid(ds, **kwargs):
        seen.append(kwargs)
        return ds

    monkeypatch.setattr(smoothing, "spatial_smoothing_xesmf", regrid)
    return seen


@pytest.mark.parametrize("how", ["sum", "mean"])
def test_goddard_temporal_aggregation_matches_temporal_only(spatial_identity, how):
    ds = xr.Dataset({"rain": ("time", [1.0, 2.0, 3.0, 4.0])})
    ds = ds.assign_coords(time=[0, 1, 2, 3])
    expected = smoothing.temporal_smoothing(ds, {"time": 2}, how=how)
    actual = smoothing.smooth_goddard_2013(
        ds, tsmooth_kws={"time": 2}, d_lon_lat_kws={"lat": 1}, how=how
    )
    assert_identical(actual, expected)
    np.testing.assert_allclose(
        actual.rain, [3, 5, 7] if how == "sum" else [1.5, 2.5, 3.5]
    )
    assert spatial_identity == [{"d_lon_lat_kws": {"lat": 1}}]


def test_goddard_default_remains_mean(spatial_identity):
    ds = xr.Dataset({"rain": ("lead", [1.0, 2.0, 3.0, 4.0])})
    ds = ds.assign_coords(lead=[1, 2, 3, 4])
    assert_identical(
        smoothing.smooth_goddard_2013(ds, tsmooth_kws={"lead": 2}),
        smoothing.temporal_smoothing(ds, {"lead": 2}, how="mean"),
    )


@pytest.mark.parametrize("how", ["sum", "mean"])
def test_ensemble_spatial_temporal_aggregation_matches_temporal_only(
    spatial_identity, how
):
    values = np.broadcast_to(
        np.arange(1.0, 5.0)[None, :, None, None, None], (3, 4, 2, 2, 2)
    )
    ds = xr.Dataset(
        {"rain": (("init", "lead", "member", "lat", "lon"), values)},
        coords={
            "init": xr.date_range("2000", periods=3, freq="YS", use_cftime=True),
            "lead": [1, 2, 3, 4],
            "member": [0, 1],
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )
    ds.lead.attrs["units"] = "years"
    ensemble = HindcastEnsemble(ds)
    before = ensemble.get_initialized().copy(deep=True)
    expected = ensemble.smooth({"lead": 2}, how=how).get_initialized()
    actual = ensemble.smooth({"lead": 2, "lat": 1, "lon": 1}, how=how).get_initialized()
    assert_identical(actual, expected)
    assert_identical(ensemble.get_initialized(), before)
    assert len(spatial_identity) == 1
