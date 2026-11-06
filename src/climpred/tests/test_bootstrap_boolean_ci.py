"""Boolean bootstrap distributions remain quantile-compatible with NumPy 2."""

import numpy as np
import pytest
import xarray as xr

from climpred.bootstrap import _distribution_to_ci


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("dim", ["iteration", "draw"])
@pytest.mark.parametrize("dtype", [bool, np.float64, np.int64])
def test_distribution_quantiles_preserve_boolean_and_numeric_results(
    dataset, lazy, dim, dtype
):
    values = np.array([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=dtype)
    array = xr.DataArray(
        values,
        dims=(dim, "lead"),
        coords={dim: [2, 4, 6, 8], "lead": [1, 3]},
        name="event",
    )
    obj = array.to_dataset() if dataset else array
    if dataset:
        obj["numeric"] = array.astype(float) * 2 + 0.25
    if lazy:
        obj = obj.chunk({dim: 2, "lead": 1})
    actual = _distribution_to_ci(obj, 0.25, 0.75, dim=dim)
    expected = obj.astype(np.float64).quantile([0.25, 0.75], dim=dim, skipna=False)
    xr.testing.assert_identical(actual.compute(), expected.compute())
    if lazy:
        assert actual.chunks is not None


def test_custom_boolean_metric_bootstrap(perfectModelEnsemble_initialized_control):
    from climpred.metrics import Metric

    def positive_mean_error(forecast, verif, dim=None, **kwargs):
        return (forecast - verif).mean(dim) > 0

    metric = Metric(
        name="positive_mean_error",
        function=positive_mean_error,
        positive=False,
        probabilistic=False,
        unit_power=0,
    )
    result = perfectModelEnsemble_initialized_control.bootstrap(
        metric=metric,
        comparison="e2c",
        dim="init",
        resample_dim="init",
        iterations=5,
    )
    low = result.sel(results="low_ci", drop=True)
    high = result.sel(results="high_ci", drop=True)
    assert np.isfinite(low.to_array()).all()
    assert ((low >= 0) & (low <= high) & (high <= 1)).to_array().all()
