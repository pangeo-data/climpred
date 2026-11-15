"""Dataset horizons reduce lead without dropping unrelated singleton axes."""

import numpy as np
import pytest
import xarray as xr

from climpred.horizon import horizon


@pytest.mark.parametrize("sizes", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("variables", [1, 2])
def test_horizon_preserves_dataset_spatial_dimensions(sizes, variables):
    lat, lon = sizes
    values = np.broadcast_to([True, True, False], (lat, lon, 3)).copy()
    cond = xr.DataArray(
        values,
        dims=["lat", "lon", "lead"],
        coords={
            "lat": np.arange(lat) + 52.0,
            "lon": np.arange(lon) + 8.0,
            "lead": [1, 2, 3],
        },
        name="skill",
    )
    cond.lead.attrs["units"] = "years"
    dataset = xr.Dataset({f"v{i}": cond for i in range(variables)})
    actual = horizon(dataset)
    expected = xr.Dataset(
        {name: horizon(da).drop_vars("lead") for name, da in dataset.data_vars.items()}
    )
    expected.attrs["units"] = "years"
    xr.testing.assert_identical(actual.compute(), expected.compute())
    assert actual.sizes == {"lat": lat, "lon": lon}
