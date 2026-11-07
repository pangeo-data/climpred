"""e2o + reference='uninitialized' must score the uninitialized ENSEMBLE MEAN (GH #969)."""

import numpy as np
import xarray as xr

from climpred import HindcastEnsemble


def _toy_hindcast(nmem=10, seed=1):
    rng = np.random.default_rng(seed)
    years = np.arange(1960, 2021)
    signal = np.cumsum(rng.normal(0, 0.3, years.size))
    obs = xr.DataArray(
        signal + rng.normal(0, 1, years.size),
        dims="time",
        coords={"time": years},
        name="x",
    )
    unin = xr.DataArray(
        signal[None] + rng.normal(0, 1, (nmem, years.size)),
        dims=["member", "time"],
        coords={"member": range(nmem), "time": years},
        name="x",
    )
    inits, leads = years[:-2], [1, 2]
    init = xr.DataArray(
        np.stack(
            [
                signal[np.searchsorted(years, inits + lead)][:, None]
                + rng.normal(0, 1, (inits.size, nmem))
                for lead in leads
            ],
            1,
        ),
        dims=["init", "lead", "member"],
        coords={"init": inits, "lead": leads, "member": range(nmem)},
        name="x",
    )
    init["lead"].attrs["units"] = "years"
    he = HindcastEnsemble(init).add_observations(obs).add_uninitialized(unin)
    return he, obs, unin, years, inits, leads


def test_e2o_uninitialized_reference_uses_ensemble_mean():
    he, obs, unin, years, inits, leads = _toy_hindcast()
    skill = he.verify(
        metric="acc",
        comparison="e2o",
        dim="init",
        alignment="same_verifs",
        reference="uninitialized",
    )
    verif_years = years[
        (years >= inits.min() + max(leads)) & (years <= inits.max() + min(leads))
    ]
    expected = float(
        xr.corr(unin.mean("member").sel(time=verif_years), obs.sel(time=verif_years))
    )
    np.testing.assert_allclose(
        skill.x.sel(skill="uninitialized").values, expected, rtol=1e-6
    )
