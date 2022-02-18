"""Test compute_hindcast."""

import dask
import pytest

from climpred.reference import compute_persistence

ITERATIONS = 2

kw = dict(alignment="same_inits", dim="init")


def test_hindcast_verify_lead0_lead1(hindcast_hist_obs_1d):
    """
    Checks that HindcastEnsemble.verify() returns the same results with a lead-0 and
    lead-1 framework.
    """
    res1 = hindcast_hist_obs_1d.verify(
        **kw,
        metric="rmse",
        comparison="e2o",
    )
    res2 = hindcast_hist_obs_1d.verify(
        **kw,
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
def test_hindcast_verify_metric_keyerrors(hindcast_hist_obs_1d, metric):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        hindcast_hist_obs_1d.verify(
            metric=metric,
            **kw,
            comparison="e2o",
        )
    assert "Specify metric from" in str(excinfo.value)


@pytest.mark.parametrize("comparison", ("ensemblemean", "test", "None"))
def test_hindcast_verify_comparison_keyerrors(hindcast_hist_obs_1d, comparison):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        hindcast_hist_obs_1d.verify(
            **kw,
            comparison=comparison,
            metric="mse",
        )
    assert "Specify comparison from" in str(excinfo.value)


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_hindcast_verify_dask_spatial(hindcast_recon_3d, metric):
    """Chunking along spatial dims."""
    # chunk over dims in both
    for dim in hindcast_recon_3d.get_initialized().dims:
        if dim in hindcast_recon_3d.get_observations().dims:
            step = 5
            res_chunked = (
                hindcast_recon_3d.chunk({dim: step})
                .verify(
                    metric=metric,
                    **kw,
                    comparison="e2o",
                )
                .to_array()
            )
            # check for chunks
            assert dask.is_dask_collection(res_chunked)
            assert res_chunked.chunks is not None


def test_hindcast_verify_CESM_3D_keep_coords(hindcast_recon_3d):
    """Test that no coords are lost in compute_hindcast with the CESM sample data."""
    s = hindcast_recon_3d.verify(metric="mse", comparison="e2o", **kw)
    for c in (
        hindcast_recon_3d.get_initialized()
        .drop_vars("init")
        .drop_vars("valid_time")
        .coords
    ):
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
        alignment="same_inits",
    )
    assert actual.lead.attrs["units"] == "years"
