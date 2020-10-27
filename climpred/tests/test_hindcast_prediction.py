import dask
import numpy as np
import pytest

from climpred.bootstrap import bootstrap_hindcast
from climpred.comparisons import HINDCAST_COMPARISONS
from climpred.constants import CLIMPRED_DIMS
from climpred.metrics import DETERMINISTIC_HINDCAST_METRICS
from climpred.prediction import compute_hindcast
from climpred.reference import compute_persistence

# uacc is sqrt(MSSS), fails when MSSS negative
DETERMINISTIC_HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()
DETERMINISTIC_HINDCAST_METRICS.remove("uacc")

ITERATIONS = 2

category_edges = np.array([0, 0.5, 1])


@pytest.mark.skip(reason="less not properly implemented")
def test_compute_hindcast_less_m2o(hind_da_initialized_1d, reconstruction_da_1d):
    """Test LESS m2o runs through"""
    actual = (
        compute_hindcast(
            hind_da_initialized_1d,
            reconstruction_da_1d,
            metric="less",
            comparison="m2o",
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.parametrize("metric", DETERMINISTIC_HINDCAST_METRICS)
@pytest.mark.parametrize("comparison", HINDCAST_COMPARISONS)
def test_compute_hindcast(
    hind_ds_initialized_1d, reconstruction_ds_1d, metric, comparison
):
    """
    Checks that compute hindcast works without breaking.
    """
    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    else:
        metric_kwargs = {}
    res = (
        compute_hindcast(
            hind_ds_initialized_1d,
            reconstruction_ds_1d,
            metric=metric,
            comparison=comparison,
            **metric_kwargs
        )
        .isnull()
        .any()
    )
    for var in res.data_vars:
        assert not res[var]


def test_compute_hindcast_lead0_lead1(
    hind_ds_initialized_1d, hind_ds_initialized_1d_lead0, reconstruction_ds_1d
):
    """
    Checks that compute hindcast returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_hindcast(
        hind_ds_initialized_1d,
        reconstruction_ds_1d,
        metric="rmse",
        comparison="e2o",
    )
    res2 = compute_hindcast(
        hind_ds_initialized_1d_lead0,
        reconstruction_ds_1d,
        metric="rmse",
        comparison="e2o",
    )
    assert (res1.SST.values == res2.SST.values).all()


@pytest.mark.parametrize("metric", DETERMINISTIC_HINDCAST_METRICS)
def test_persistence(hind_da_initialized_1d, reconstruction_da_1d, metric):
    """
    Checks that compute persistence works without breaking.
    """
    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    else:
        metric_kwargs = {}
    res = compute_persistence(
        hind_da_initialized_1d, reconstruction_da_1d, metric=metric, **metric_kwargs
    )
    assert not res.isnull().any()
    # check persistence metadata
    assert res.attrs["metric"] == metric
    assert res.attrs["skill_calculated_by_function"] == "compute_persistence"
    assert "number of members" not in res.attrs


def test_persistence_lead0_lead1(
    hind_ds_initialized_1d, hind_ds_initialized_1d_lead0, reconstruction_ds_1d
):
    """
    Checks that compute persistence returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_persistence(
        hind_ds_initialized_1d, reconstruction_ds_1d, metric="rmse"
    )
    res2 = compute_persistence(
        hind_ds_initialized_1d_lead0, reconstruction_ds_1d, metric="rmse"
    )
    assert (res1.SST.values == res2.SST.values).all()


def test_bootstrap_hindcast_da1d_not_nan(
    hind_da_initialized_1d, hist_da_uninitialized_1d, reconstruction_da_1d
):
    """
    Checks that there are no NaNs on bootstrap hindcast of 1D da.
    """
    actual = bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        reconstruction_da_1d,
        metric="rmse",
        comparison="e2o",
        sig=50,
        iterations=ITERATIONS,
    )
    actual_init_skill = (
        actual.sel(skill="initialized", results="verify skill").isnull().any()
    )
    assert not actual_init_skill
    actual_uninit_p = actual.sel(skill="uninitialized", results="p").isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize("metric", ("AnomCorr", "test", "None"))
def test_compute_hindcast_metric_keyerrors(
    hind_ds_initialized_1d, reconstruction_ds_1d, metric
):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            hind_ds_initialized_1d,
            reconstruction_ds_1d,
            comparison="e2o",
            metric=metric,
        )
    assert "Specify metric from" in str(excinfo.value)


@pytest.mark.parametrize("comparison", ("ensemblemean", "test", "None"))
def test_compute_hindcast_comparison_keyerrors(
    hind_ds_initialized_1d, reconstruction_ds_1d, comparison
):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            hind_ds_initialized_1d,
            reconstruction_ds_1d,
            comparison=comparison,
            metric="mse",
        )
    assert "Specify comparison from" in str(excinfo.value)


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_compute_hindcast_dask_spatial(
    hind_da_initialized_3d, reconstruction_da_3d, metric
):
    """Chunking along spatial dims."""
    # chunk over dims in both
    for dim in hind_da_initialized_3d.dims:
        if dim in reconstruction_da_3d.dims:
            step = 5
            res_chunked = compute_hindcast(
                hind_da_initialized_3d.chunk({dim: step}),
                reconstruction_da_3d.chunk({dim: step}),
                comparison="e2o",
                metric=metric,
            )
            # check for chunks
            assert dask.is_dask_collection(res_chunked)
            assert res_chunked.chunks is not None


@pytest.mark.skip(reason="not yet implemented")
@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_compute_hindcast_dask_climpred_dims(
    hind_da_initialized_3d, reconstruction_da_3d, metric
):
    """Chunking along climpred dims if available."""
    step = 5
    for dim in CLIMPRED_DIMS:
        if dim in hind_da_initialized_3d.dims:
            hind_da_initialized_3d = hind_da_initialized_3d.chunk({dim: step})
        if dim in reconstruction_da_3d.dims:
            reconstruction_da_3d = reconstruction_da_3d.chunk({dim: step})
        res_chunked = compute_hindcast(
            hind_da_initialized_3d,
            reconstruction_da_3d,
            comparison="e2o",
            metric=metric,
        )
        # check for chunks
        assert dask.is_dask_collection(res_chunked)
        assert res_chunked.chunks is not None


def test_compute_hindcast_CESM_3D_keep_coords(
    hind_da_initialized_3d, reconstruction_da_3d
):
    """Test that no coords are lost in compute_hindcast with the CESM sample data."""
    s = compute_hindcast(hind_da_initialized_3d, reconstruction_da_3d)
    for c in hind_da_initialized_3d.drop("init").coords:
        assert c in s.coords


def test_bootstrap_hindcast_keeps_lead_units(
    hind_da_initialized_1d, hist_da_uninitialized_1d, observations_da_1d
):
    """Test that lead units is kept in compute."""
    sig = 95
    units = "years"
    hind_da_initialized_1d["lead"].attrs["units"] = units
    actual = bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        metric="mse",
        iterations=ITERATIONS,
        comparison="e2o",
        sig=sig,
        dim="init",
    )
    assert actual.lead.attrs["units"] == units
