"""Test compute_perfect_model."""


import pytest
import xarray as xr

from climpred.prediction import compute_perfect_model
from climpred.reference import compute_persistence

xr.set_options(display_style="text")


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
