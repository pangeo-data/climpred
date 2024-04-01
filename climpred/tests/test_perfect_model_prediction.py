"""Test compute_perfect_model."""

import pytest
import xarray as xr

from climpred.reference import compute_persistence

xr.set_options(display_style="text")
v = "tos"


@pytest.mark.parametrize("metric", ["mse", "pearson_r"])
def test_compute_persistence_lead0_lead1(
    PM_ds_initialized_1d, PM_ds_initialized_1d_lead0, PM_ds_control_1d, metric
):
    """
    Checks that persistence forecast results are identical for a lead 0 and lead 1 setup
    """
    res1 = compute_persistence(
        PM_ds_initialized_1d, PM_ds_control_1d, metric=metric, alignment="same_inits"
    )
    res2 = compute_persistence(
        PM_ds_initialized_1d_lead0,
        PM_ds_control_1d,
        metric=metric,
        alignment="same_inits",
    )
    assert (res1[v].values == res2[v].values).all()
