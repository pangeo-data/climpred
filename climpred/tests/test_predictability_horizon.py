import numpy as np
import pytest
import xarray as xr

from climpred.predictability_horizon import last_item_cond_true, predictability_horizon


@pytest.mark.parametrize("threshold,expected", [(1, 1), (2, 3), (3, 6), (3.5, 6)])
def test_least_item_cond_true(threshold, expected):
    """"test `last_item_cond_true` on artificial data."""
    ds = xr.DataArray(
        [1, 2, 2, 3, 3, 1, 4], dims="lead", coords={"lead": np.arange(1, 1 + 7)}
    )
    cond = ds <= threshold
    actual = last_item_cond_true(cond, "lead")
    assert actual == expected


def test_predictability_horizon_bootstrap_1d(perfectModelEnsemble_initialized_control):
    """test predictability_horizon for pm.bootstrap for 1d."""
    bskill = perfectModelEnsemble_initialized_control.bootstrap(
        iterations=101,
        metric="rmse",
        comparison="m2e",
        dim=["member", "init"],
        reference="uninitialized",
    )
    ph = predictability_horizon(bskill.sel(results="p", skill="uninitialized") <= 0.05)
    assert ph.tos.attrs["units"] == "years", print(ph.tos)
    assert int(ph.tos) in [5, 6, 7]  # should be 6


def test_predictability_horizon_3d():
    pass
    # test all nan on land

    # test never significant

    # test always significant
