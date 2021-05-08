import numpy as np
import pytest
import xarray as xr

from climpred.horizon import _last_item_cond_true, horizon


@pytest.mark.parametrize("input", ["DataArray", "Dataset"])
@pytest.mark.parametrize("threshold,expected", [(1, 1), (2, 3), (3, 6), (3.5, 6)])
def test_least_item_cond_true(threshold, expected, input):
    """test `last_item_cond_true` on artificial data."""
    ds = xr.DataArray(
        [1, 2, 2, 3, 3, 1, 4], dims="lead", coords={"lead": np.arange(1, 1 + 7)}
    )
    if input == "Dataset":
        ds = ds.to_dataset(name="test")
    cond = ds <= threshold
    actual = _last_item_cond_true(cond, "lead")
    assert actual == expected
    assert type(ds) == type(actual)


def test_horizon_bootstrap_1d(perfectModelEnsemble_initialized_control):
    """test horizon for pm.bootstrap for 1d."""
    bskill = perfectModelEnsemble_initialized_control.bootstrap(
        iterations=201,
        metric="rmse",
        comparison="m2e",
        dim=["member", "init"],
        reference="uninitialized",
    )
    ph = horizon(bskill.sel(results="p", skill="uninitialized") <= 0.05)
    assert ph.tos.attrs["units"] == "years", print(ph.tos)
    assert int(ph.tos) in [4, 5, 6, 7, 8]  # should be 6, testing on the safe side


def test_horizon_3d(hindcast_recon_3d):
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

    ph = horizon(skill.sel(skill="initialized") > skill.sel(skill="persistence"))
    assert "variable" not in skill.dims
    assert "variable" not in skill.coords
    # test all nan on land
    assert ph["SST"][0, 0].isnull()
    # test significant everywhere
    assert (ph >= 1).all()
    assert (ph.isel(nlat=-1) == 2).all()


@pytest.mark.parametrize("smooth", [True, False])
def test_horizon_smooth(perfectModelEnsemble_initialized_control, smooth):
    """test horizon for pm.smooth(lead).verify."""
    pm = perfectModelEnsemble_initialized_control
    if smooth:
        pm = pm.smooth(
            {"lead": 2}, how="mean"
        )  # converts lead to '1-2', '2-3', ... after verify
    skill = pm.verify(
        metric="rmse",
        comparison="m2e",
        dim=["member", "init"],
        reference="persistence",
    )
    assert skill.lead.attrs["units"] == "years", print(skill.lead.attrs)
    # initialized better than persistence if RMSE smaller
    cond = skill.sel(skill="initialized", drop=True) < skill.sel(
        skill="persistence", drop=True
    )
    ph = horizon(cond)
    assert ph.tos.attrs["units"] == "years", print("ph.tos.attrs = ", ph.tos.attrs)
    assert ph.tos.values == skill.lead.isel(lead=-1).values


def test_horizon_weird_coords():
    """Test horizon for weird coords."""
    cond = xr.DataArray([True] * 10, dims="lead").to_dataset(name="SST")
    # Change leads to something weird
    cond["lead"] = [0.25, 0.75, 1, 3, 4, 5, 6, 7, 8, 9]
    assert _last_item_cond_true(cond, "lead") == 9.0

    cond[0] = False
    assert _last_item_cond_true(cond, "lead").isnull(), print(cond)

    cond[0] = True
    cond[1] = False
    assert _last_item_cond_true(cond, "lead") == cond.lead[0], print(cond)

    cond["lead"] = np.arange(cond.lead.size)
    cond[0] = True
    cond[1] = False
    assert _last_item_cond_true(cond, "lead") == 0.0, print(cond)
