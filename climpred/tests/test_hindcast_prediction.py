"""Test compute_hindcast."""

import dask
import pytest

from climpred.prediction import compute_hindcast
from climpred.reference import compute_persistence

ITERATIONS = 2


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


def test_compute_hindcast_CESM_3D_keep_coords(
    hind_da_initialized_3d, reconstruction_da_3d
):
    """Test that no coords are lost in compute_hindcast with the CESM sample data."""
    s = compute_hindcast(hind_da_initialized_3d, reconstruction_da_3d)
    for c in hind_da_initialized_3d.drop_vars("init").coords:
        assert c in s.coords


def test_HindcastEnsemble_keeps_lead_units(hindcast_hist_obs_1d):
    """Test that lead units is kept in bootstrap."""
    sig = 95
    actual = hindcast_hist_obs_1d.bootstrap(
        metric="mse",
        iterations=ITERATIONS,
        comparison="e2o",
        sig=sig,
        dim="init",
        alignment="same_verif",
    )
    assert actual.lead.attrs["units"] == "years"
