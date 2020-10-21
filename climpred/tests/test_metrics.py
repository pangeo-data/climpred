import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import PM_COMPARISONS
from climpred.metrics import __ALL_METRICS__ as all_metrics, Metric, __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model


def my_mse_function(forecast, verif, dim=None, **metric_kwargs):
    # function
    return ((forecast - verif) ** 2).mean(dim)


my_mse = Metric(
    name="mse",
    function=my_mse_function,
    positive=True,
    probabilistic=False,
    unit_power=2,
    long_name="MSE",
    aliases=["mSe", "<<<SE"],
)

ITERATIONS = 2


@pytest.mark.parametrize("comparison", PM_COMPARISONS)
def test_custom_metric_passed_to_compute(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    """Test custom metric in compute_perfect_model."""
    actual = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=my_mse,
        dim="init",
    )

    expected = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric="mse",
        dim="init",
    )

    assert_allclose(actual, expected)


@pytest.mark.slow
def test_custom_metric_passed_to_bootstrap_compute(
    PM_da_initialized_1d, PM_da_control_1d
):
    """Test custom metric in bootstrap_perfect_model."""
    comparison = "e2c"
    dim = "init"
    np.random.seed(42)
    actual = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=my_mse,
        iterations=ITERATIONS,
        dim=dim,
    )

    expected = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric="mse",
        iterations=ITERATIONS,
        dim=dim,
    )

    assert_allclose(actual, expected, rtol=0.1, atol=1)


@pytest.mark.parametrize("metric", ("rmse", "mse"))
def test_pm_metric_skipna(PM_da_initialized_3d, PM_da_control_3d, metric):
    """Test skipna in compute_perfect_model."""
    PM_da_initialized_3d = PM_da_initialized_3d.copy()
    # manipulating data
    PM_da_initialized_3d.values[1:3, 1:4, 1:4, 4:6, 4:6] = np.nan

    base = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        metric=metric,
        skipna=False,
        dim="init",
        comparison="m2e",
    ).mean("member")
    skipping = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        metric=metric,
        skipna=True,
        dim="init",
        comparison="m2e",
    ).mean("member")
    assert ((base - skipping) != 0.0).any()
    assert base.isel(lead=2, x=5, y=5).isnull()
    assert not skipping.isel(lead=2, x=5, y=5).isnull()


@pytest.mark.skip(reason="comparisons dont work here")
@pytest.mark.parametrize("metric", ("rmse", "mse"))
@pytest.mark.parametrize("comparison", ["m2e", "m2m"])
def test_pm_metric_weights_m2x(
    PM_da_initialized_3d, PM_da_control_3d, comparison, metric
):
    """Test init weights in compute_perfect_model."""
    # distribute weights on initializations
    dim = "init"
    base = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(np.arange(1, 1 + PM_da_initialized_3d[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(
            1,
            1 + PM_da_initialized_3d[dim].size * PM_da_initialized_3d["member"].size,
        ),
        dims="init",
    )

    weighted = compute_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(["x", "y"]))
    # test for difference
    assert (xs.smape(base, weighted, ["x", "y"]) > 0.01).any()


@pytest.mark.parametrize("metric", ("rmse", "mse"))
def test_hindcast_metric_skipna(hind_da_initialized_3d, reconstruction_da_3d, metric):
    """Test skipna argument in hindcast_metric."""
    # manipulating data with nans
    hind_da_initialized_3d[0, 2, 0, 2] = np.nan
    base = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        metric=metric,
        skipna=False,
        dim="init",
        alignment="same_inits",
    )
    skipping = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        metric=metric,
        skipna=True,
        dim="init",
        alignment="same_inits",
    )
    div = base / skipping
    assert (div != 1).any()


@pytest.mark.skip(reason="comparisons dont work here")
@pytest.mark.parametrize("metric", ("rmse", "mse"))
@pytest.mark.parametrize("comparison", ["e2o", "m2o"])
def test_hindcast_metric_weights_x2r(
    hind_da_initialized_3d, reconstruction_da_3d, comparison, metric
):
    """Test init weights in compute_hindcast."""
    dim = "init"
    base = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        dim=dim,
        metric=metric,
        comparison=comparison,
    )
    weights = xr.DataArray(np.arange(1, 1 + hind_da_initialized_3d[dim].size), dims=dim)
    weights = xr.DataArray(
        np.arange(
            1,
            1
            + hind_da_initialized_3d[dim].size * hind_da_initialized_3d["member"].size,
        ),
        dims="init",
    )

    weighted = compute_hindcast(
        hind_da_initialized_3d,
        reconstruction_da_3d,
        dim=dim,
        comparison=comparison,
        metric=metric,
        weights=weights,
    )
    print((base / weighted).mean(["nlon", "nlat"]))
    # test for difference
    assert (xs.smape(base, weighted, ["nlat", "nlon"]) > 0.01).any()


def test_Metric_display():
    summary = __pearson_r.__repr__()
    assert "Kind: deterministic" in summary.split("\n")[4]


def test_no_repeating_metric_aliases():
    """Tests that there are no repeating aliases for metrics, which would overwrite
    the earlier defined metric."""
    METRICS = []
    for m in all_metrics:
        if m.aliases is not None:
            for a in m.aliases:
                METRICS.append(a)
    duplicates = set([x for x in METRICS if METRICS.count(x) > 1])
    print(f"Duplicate metrics: {duplicates}")
    assert len(duplicates) == 0
