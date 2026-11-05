"""Detrending consumes fitted coefficients, not optional fit diagnostics."""

import numpy as np
import pytest
import xarray as xr

from climpred.stats import rm_poly, rm_trend


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize(
    "kwargs",
    [{"full": True}, {"cov": True}, {"cov": "unscaled"}, {"full": True, "cov": True}],
)
def test_optional_fit_diagnostics_do_not_change_detrending(kind, kwargs):
    data = xr.DataArray(
        2 * np.arange(10.0) ** 2 + np.arange(10.0) + np.sin(np.arange(10.0)),
        dims="time",
        coords={"time": np.arange(10)},
        name="temperature",
    )
    if kind == "dataset":
        data = data.to_dataset()
    before = data.copy(deep=True)
    expected = rm_poly(data, deg=2)
    actual = rm_poly(data, deg=2, **kwargs)
    xr.testing.assert_allclose(actual, expected)
    xr.testing.assert_identical(data, before)
    assert actual.sizes == expected.sizes


@pytest.mark.parametrize(
    "name",
    [
        "a_polyfit_coefficients_b",
        "a_polyfit_coefficients",
        "_polyfit_coefficients",
        "plain",
    ],
)
@pytest.mark.parametrize("kwargs", [{}, {"full": True}, {"cov": True}])
def test_dataset_variable_names_are_preserved_exactly(name, kwargs):
    t = np.arange(10.0)
    data = xr.Dataset(
        {name: ("time", 3 * t + np.sin(t)), "other": ("time", t**2)}, coords={"time": t}
    )
    actual = rm_poly(data, deg=2, **kwargs)
    assert set(actual.data_vars) == set(data.data_vars)
    for variable in data:
        expected = rm_poly(data[variable], deg=2)
        xr.testing.assert_allclose(actual[variable], expected)


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("kwargs", [{}, {"full": True}, {"cov": True}])
def test_rm_trend_wrapper_uses_only_linear_coefficients(kind, kwargs):
    t = np.arange(10.0)
    data = xr.DataArray(4 * t + 2, dims="time", coords={"time": t}, name="x")
    if kind == "dataset":
        data = data.to_dataset()
    actual = rm_trend(data, **kwargs)
    xr.testing.assert_allclose(actual, data * 0, atol=1e-12)


@pytest.mark.parametrize("dim", ["time", "init"])
def test_dask_dataset_diagnostics_preserve_lazy_result(dim):
    t = np.arange(12.0)
    data = xr.Dataset(
        {"temperature": (dim, t**2 + np.cos(t))}, coords={dim: t}
    ).chunk({dim: -1})
    actual = rm_poly(data, dim=dim, full=True)
    assert actual.temperature.chunks is not None
    xr.testing.assert_allclose(actual.compute(), rm_poly(data.compute(), dim=dim))
