import numpy as np
import pytest
import xarray as xr

from climpred.predictability_horizon import _last_item_cond_true, predictability_horizon


@pytest.mark.parametrize("threshold,expected", [(1, 1), (2, 3), (3, 6), (3.5, 6)])
def test_least_item_cond_true(threshold, expected):
    """"test `last_item_cond_true` on artificial data."""
    ds = xr.DataArray(
        [1, 2, 2, 3, 3, 1, 4], dims="lead", coords={"lead": np.arange(1, 1 + 7)}
    )
    cond = ds <= threshold
    actual = _last_item_cond_true(cond, "lead")
    assert actual == expected


def test_predictability_horizon_bootstrap_1d(perfectModelEnsemble_initialized_control):
    """test predictability_horizon for pm.bootstrap for 1d."""
    bskill = perfectModelEnsemble_initialized_control.bootstrap(
        iterations=201,
        metric="rmse",
        comparison="m2e",
        dim=["member", "init"],
        reference="uninitialized",
    )
    ph = predictability_horizon(bskill.sel(results="p", skill="uninitialized") <= 0.05)
    assert ph.tos.attrs["units"] == "years", print(ph.tos)
    assert int(ph.tos) in [5, 6, 7]  # should be 6, testing on the safe side


def test_predictability_horizon_3d(hindcast_recon_3d):
    he = hindcast_recon_3d
    # set one grid to all nan mimicking land
    land = he._datasets["initialized"]["SST"][:, :, 0, 0]
    he._datasets["initialized"]["SST"][:, :, 0, 0] = land.where(land == 100000)
    # verify
    skill = he.verify(
        metric="acc",
        comparison="e2o",
        dim="init",
        alignment="maximize",
        reference=["persistence"],
    )
    from climpred.predictability_horizon import predictability_horizon

    ph = predictability_horizon(
        skill.sel(skill="initialized") > skill.sel(skill="persistence")
    )
    # test all nan on land
    print(ph)
    assert ph["SST"][0, 0].isnull()
    # test significant everywhere
    assert (ph >= 1).all()
    assert (ph.isel(nlat=-1) == 2).all()
