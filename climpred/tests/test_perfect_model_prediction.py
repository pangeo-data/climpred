import dask
import numpy as np
import pytest
import xarray as xr

from climpred.bootstrap import bootstrap_perfect_model
from climpred.constants import CLIMPRED_DIMS
from climpred.metrics import DETERMINISTIC_PM_METRICS
from climpred.prediction import compute_perfect_model
from climpred.reference import compute_persistence

# uacc is sqrt(MSSS), fails when MSSS negative
DETERMINISTIC_PM_METRICS_LUACC = DETERMINISTIC_PM_METRICS.copy()
DETERMINISTIC_PM_METRICS_LUACC.remove("uacc")

comparison_dim_PM = [
    ("m2m", "init"),
    ("m2m", "member"),
    ("m2m", ["init", "member"]),
    ("m2e", "init"),
    ("m2e", "member"),
    ("m2e", ["init", "member"]),
    ("m2c", "init"),
    ("m2c", "member"),
    ("m2c", ["init", "member"]),
    ("e2c", "init"),
]

# run less tests
PM_COMPARISONS = {"m2c": "", "e2c": ""}

ITERATIONS = 2

xr.set_options(display_style="text")

category_edges = np.array([10.0, 10.5, 11.0])


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_pvalue_from_bootstrapping(PM_da_initialized_1d, PM_da_control_1d, metric):
    """Test that pvalue of initialized ensemble first lead is close to 0."""
    sig = 95
    actual = (
        bootstrap_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            metric=metric,
            iterations=ITERATIONS,
            comparison="e2c",
            sig=sig,
            dim="init",
        )
        .sel(skill="uninitialized", results="p")
        .isel(lead=0)
    )
    assert actual.values < 2 * (1 - sig / 100)


@pytest.mark.parametrize("metric", ["mse", "pearson_r"])
def test_compute_persistence_add_attrs(PM_ds_initialized_1d, PM_ds_control_1d, metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    attrs = (
        compute_persistence(
            PM_ds_initialized_1d,
            PM_ds_control_1d,
            metric=metric,
            alignment="same_inits",
        )
    ).attrs
    assert (
        attrs["prediction_skill"]
        == "calculated by climpred https://climpred.readthedocs.io/"
    )
    assert attrs["skill_calculated_by_function"] == "compute_persistence"
    assert "number of members" not in attrs
    assert attrs["metric"] == metric


@pytest.mark.parametrize("metric", DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_persistence_ds1d_not_nan(
    PM_ds_initialized_1d, PM_ds_control_1d, metric
):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    else:
        metric_kwargs = {}
    actual = (
        compute_persistence(
            PM_ds_initialized_1d,
            PM_ds_control_1d,
            metric=metric,
            alignment="same_inits",
            **metric_kwargs
        )
        # .isnull()
        # .any()
    )
    print(actual.tos)
    actual = actual.isnull().any()
    for var in actual.data_vars:
        assert not actual[var], actual[var]


@pytest.mark.parametrize("metric", ["mse", "pearson_r"])
def test_compute_persistence_lead0_lead1(
    PM_da_initialized_1d, PM_da_initialized_1d_lead0, PM_da_control_1d, metric
):
    """
    Checks that persistence forecast results are identical for a lead 0 and lead 1 setup
    """
    res1 = compute_persistence(
        PM_da_initialized_1d, PM_da_control_1d, metric=metric, alignment="same_inits"
    )
    res2 = compute_persistence(
        PM_da_initialized_1d_lead0,
        PM_da_control_1d,
        metric=metric,
        alignment="same_inits",
    )
    assert (res1.values == res2.values).all()


@pytest.mark.parametrize("comparison,dim", comparison_dim_PM)
@pytest.mark.parametrize("metric", DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_perfect_model_da1d_not_nan(
    PM_da_initialized_1d, PM_da_control_1d, comparison, metric, dim
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    else:
        metric_kwargs = {}
    # acc on dim member only is ill defined
    if dim == "member" and metric in [
        "pearson_r",
        "spearman_r",
        "pearson_r_p_value",
        "spearman_r_p_value",
        "msess_murphy",
        "bias_slope",
        "conditional_bias",
    ]:
        dim = ["init", "member"]
    actual = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=metric,
        dim=dim,
        **metric_kwargs
    )
    if metric == "contingency":
        assert not actual.isnull().all()
    else:
        assert not actual.isnull().any()


@pytest.mark.parametrize("comparison,dim", comparison_dim_PM)
@pytest.mark.parametrize("metric", ["rmse", "mae"])
def test_compute_perfect_model_lead0_lead1(
    PM_da_initialized_1d,
    PM_da_initialized_1d_lead0,
    PM_da_control_1d,
    comparison,
    metric,
    dim,
):
    """
    Checks that metric results are identical for a lead 0 and lead 1 setup.
    """
    res1 = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=comparison,
        metric=metric,
        dim=dim,
    )
    res2 = compute_perfect_model(
        PM_da_initialized_1d_lead0,
        PM_da_control_1d,
        comparison=comparison,
        metric=metric,
        dim=dim,
    )
    assert (res1.values == res2.values).all()


def test_bootstrap_perfect_model_da1d_not_nan(PM_da_initialized_1d, PM_da_control_1d):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D da.
    """
    actual = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric="rmse",
        comparison="e2c",
        dim="init",
        sig=50,
        iterations=ITERATIONS,
    )
    actual_init_skill = (
        actual.sel(skill="initialized", results="verify skill").isnull().any()
    )
    assert not actual_init_skill
    actual_uninit_p = actual.sel(skill="uninitialized", results="p").isnull().any()
    assert not actual_uninit_p


@pytest.mark.slow
def test_bootstrap_perfect_model_ds1d_not_nan(PM_ds_initialized_1d, PM_ds_control_1d):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D ds.
    """
    actual = bootstrap_perfect_model(
        PM_ds_initialized_1d,
        PM_ds_control_1d,
        metric="rmse",
        comparison="e2c",
        dim="init",
        sig=50,
        iterations=ITERATIONS,
    )
    for var in actual.data_vars:
        actual_init_skill = (
            actual[var].sel(skill="initialized", results="verify skill").isnull().any()
        )
        assert not actual_init_skill
    for var in actual.data_vars:
        actual_uninit_p = (
            actual[var].sel(skill="uninitialized", results="p").isnull().any()
        )
        assert not actual_uninit_p


@pytest.mark.parametrize("metric", ("AnomCorr", "test", "None"))
def test_compute_perfect_model_metric_keyerrors(
    PM_da_initialized_1d, PM_da_control_1d, metric
):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison="e2c",
            metric=metric,
        )
    assert "Specify metric from" in str(excinfo.value)


@pytest.mark.parametrize("comparison", ("ensemblemean", "test", "None"))
def test_compute_perfect_model_comparison_keyerrors(
    PM_da_initialized_1d, PM_da_control_1d, comparison
):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison=comparison,
            metric="mse",
        )
    assert "Specify comparison from" in str(excinfo.value)


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
@pytest.mark.parametrize("comparison", PM_COMPARISONS)
def test_compute_pm_dask_spatial(
    PM_ds_initialized_3d, PM_ds_control_3d, comparison, metric
):
    """Chunking along spatial dims."""
    # chunk over dims in both
    for dim in PM_ds_initialized_3d.dims:
        if dim in PM_ds_control_3d.dims:
            step = 5
            res_chunked = compute_perfect_model(
                PM_ds_initialized_3d.chunk({dim: step}),
                PM_ds_control_3d.chunk({dim: step}),
                comparison=comparison,
                metric=metric,
                dim="init",
            )
            # check for chunks
            assert dask.is_dask_collection(res_chunked)
            assert res_chunked.chunks is not None


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
@pytest.mark.parametrize("comparison", PM_COMPARISONS)
def test_compute_pm_dask_climpred_dims(
    PM_ds_initialized_3d, PM_ds_control_3d, comparison, metric
):
    """Chunking along climpred dims if available."""
    step = 5
    for dim in CLIMPRED_DIMS:
        if dim in PM_ds_initialized_3d.dims:
            PM_ds_initialized_3d = PM_ds_initialized_3d.chunk({dim: step})
        if dim in PM_ds_control_3d.dims:
            PM_ds_control_3d = PM_ds_control_3d.chunk({dim: step})
        res_chunked = compute_perfect_model(
            PM_ds_initialized_3d,
            PM_ds_control_3d,
            comparison=comparison,
            metric=metric,
            dim="init",
        )
        # check for chunks
        assert dask.is_dask_collection(res_chunked)
        assert res_chunked.chunks is not None


def test_bootstrap_perfect_model_keeps_lead_units(
    PM_da_initialized_1d, PM_da_control_1d
):
    """Test that lead units is kept in compute."""
    sig = 95
    units = "years"
    PM_da_initialized_1d.lead.attrs["units"] = "years"
    actual = bootstrap_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        metric="mse",
        iterations=ITERATIONS,
        comparison="e2c",
        sig=sig,
        dim="init",
    )
    assert actual.lead.attrs["units"] == units
