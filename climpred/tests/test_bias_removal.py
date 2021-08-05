import numpy as np
import pytest
import xarray as xr

from climpred import set_options
from climpred.constants import BIAS_CORRECTION_METHODS, GROUPBY_SEASONALITIES
from climpred.options import OPTIONS
from climpred.testing import assert_PredictionEnsemble

BIAS_CORRECTION_METHODS = BIAS_CORRECTION_METHODS.copy()
BIAS_CORRECTION_METHODS.remove("normal_mapping")
BIAS_CORRECTION_METHODS.remove("gamma_mapping")
# fails with these conftest files somehow


@pytest.mark.parametrize("how", BIAS_CORRECTION_METHODS)
def test_remove_bias_difference_seasonality(hindcast_recon_1d_mm, how):
    """Test HindcastEnsemble.remove_bias yields different results for different seasonality settings."""
    verify_kwargs = dict(
        metric="rmse", dim="init", comparison="e2o", alignment="same_inits", skipna=True
    )
    hindcast = hindcast_recon_1d_mm.isel(lead=range(3))
    v = "SST"

    bias_reduced_skill = []
    seasonalities = GROUPBY_SEASONALITIES
    for seasonality in seasonalities:
        with set_options(seasonality=seasonality):
            hindcast_rb = hindcast.remove_bias(
                how=how, alignment=verify_kwargs["alignment"], cv=False
            )

            bias_reduced_skill.append(hindcast_rb.verify(**verify_kwargs)[v])
    bias_reduced_skill = xr.concat(bias_reduced_skill, "seasonality").assign_coords(
        seasonality=seasonalities
    )

    # check not identical
    for s in seasonalities:
        assert bias_reduced_skill.sel(seasonality=s).notnull().all()
        for s2 in seasonalities:
            if s != s2:
                assert (
                    bias_reduced_skill.sel(seasonality=[s, s2])
                    .diff("seasonality")
                    .notnull()
                    .any()
                )


@pytest.mark.parametrize("cv", [False, "LOO"])
@pytest.mark.parametrize("seasonality", GROUPBY_SEASONALITIES)
@pytest.mark.parametrize("how", BIAS_CORRECTION_METHODS)
@pytest.mark.parametrize(
    "alignment", ["same_inits", "maximize"]
)  # same_verifs  # no overlap here for same_verifs
def test_remove_bias(hindcast_recon_1d_mm, alignment, how, seasonality, cv):
    """Test remove mean bias, ensure than skill doesnt degrade and keeps attrs."""

    def check_hindcast_coords_maintained_except_init(hindcast, hindcast_bias_removed):
        # init only slighty cut due to alignment
        for c in hindcast.coords:
            if c == "init":
                assert hindcast.coords[c].size >= hindcast_bias_removed.coords[c].size
            else:
                assert hindcast.coords[c].size == hindcast_bias_removed.coords[c].size

    with set_options(seasonality=seasonality):
        metric = "rmse"
        dim = "init"
        comparison = "e2o"
        hindcast = hindcast_recon_1d_mm.isel(lead=range(3))
        hindcast._datasets["initialized"].attrs["test"] = "test"
        hindcast._datasets["initialized"]["SST"].attrs["units"] = "test_unit"
        verify_kwargs = dict(
            metric=metric,
            alignment=alignment,
            dim=dim,
            comparison=comparison,
            keep_attrs=True,
        )

        # add how bias
        if "additive" in how:
            with xr.set_options(keep_attrs=True):
                hindcast._datasets["observations"] = (
                    hindcast._datasets["observations"] + 0.1
                )
        elif "multiplicative" in how:
            with xr.set_options(keep_attrs=True):
                hindcast._datasets["observations"] = (
                    hindcast._datasets["observations"] * 1.1
                )

        biased_skill = hindcast.verify(**verify_kwargs)

        hindcast_bias_removed = hindcast.remove_bias(
            how=how, alignment=alignment, cv=False
        )

        check_hindcast_coords_maintained_except_init(hindcast, hindcast_bias_removed)

        bias_removed_skill = hindcast_bias_removed.verify(**verify_kwargs)

        seasonality = OPTIONS["seasonality"]
        if cv:
            hindcast_bias_removed_properly = hindcast.remove_bias(
                how=how, cv="LOO", alignment=alignment
            )
            check_hindcast_coords_maintained_except_init(
                hindcast, hindcast_bias_removed_properly
            )

            bias_removed_skill_properly = hindcast_bias_removed_properly.verify(
                **verify_kwargs
            )
            # checks
            assert seasonality not in bias_removed_skill_properly.coords
            assert (biased_skill > bias_removed_skill_properly).all()
            assert (bias_removed_skill_properly >= bias_removed_skill).all()

        assert (biased_skill > bias_removed_skill).all()
        assert seasonality not in bias_removed_skill.coords
        # keeps data_vars attrs
        for v in hindcast_bias_removed.get_initialized().data_vars:
            if cv:
                assert (
                    hindcast_bias_removed_properly.get_initialized()[v].attrs
                    == hindcast.get_initialized()[v].attrs
                )
            assert (
                hindcast_bias_removed.get_initialized()[v].attrs
                == hindcast.get_initialized()[v].attrs
            )
        # keeps dataset attrs
        if cv:
            assert (
                hindcast_bias_removed_properly.get_initialized().attrs
                == hindcast.get_initialized().attrs
            )
        assert (
            hindcast_bias_removed.get_initialized().attrs
            == hindcast.get_initialized().attrs
        )
        # keep lead attrs
        assert (
            hindcast_bias_removed.get_initialized().lead.attrs
            == hindcast.get_initialized().lead.attrs
        )


@pytest.mark.parametrize(
    "alignment", ["same_inits", "maximize"]
)  # same_verifs  # no overlap here for same_verifs
@pytest.mark.parametrize("seasonality", GROUPBY_SEASONALITIES)
@pytest.mark.parametrize("how", BIAS_CORRECTION_METHODS)
def test_monthly_leads_remove_bias_LOO(
    hindcast_NMME_Nino34, how, seasonality, alignment
):
    """Get different HindcastEnsemble depending on CV or not."""
    with set_options(seasonality=seasonality):
        he = (
            hindcast_NMME_Nino34.isel(lead=[0, 1])
            .isel(model=2, drop=True)
            .sel(init=slice("2005", "2006"))
        )
        assert not he.remove_bias(how=how, alignment=alignment, cv=False).equals(
            he.remove_bias(how=how, alignment=alignment, cv="LOO")
        )


@pytest.mark.parametrize(
    "alignment", ["same_inits", "maximize", "same_verifs"]
)  # same_verifs  # no overlap here for same_verifs
@pytest.mark.parametrize("seasonality", ["month", "season"])
@pytest.mark.parametrize("how", BIAS_CORRECTION_METHODS)
def test_remove_bias_unfair_artificial_skill_over_fair(
    hindcast_NMME_Nino34, how, seasonality, alignment
):
    """Show how method unfair better skill than fair."""
    with set_options(seasonality=seasonality):
        he = (
            hindcast_NMME_Nino34.sel(lead=[4, 5])
            .sel(model="GEM-NEMO")
            .sel(init=slice("2000", "2009"))
        )
        print("\n unfair \n")
        he_unfair = he.remove_bias(
            how=how, alignment=alignment, train_test_split="unfair"
        )
        unfair_skill = he_unfair.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment=alignment,
            skipna=False,
        )

        print("\n unfair-cv \n")
        he_unfair_cv = he.remove_bias(
            how=how,
            alignment=alignment,
            train_test_split="unfair-cv",
            cv="LOO",
        )
        unfair_cv_skill = he_unfair_cv.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment=alignment,
            skipna=False,
        )

        print("\n fair \n")
        kw = (
            dict(train_time=slice("2000", "2003"))
            if alignment == "same_verifs"
            else dict(train_init=slice("2000", "2003"))
        )
        he_fair = he.remove_bias(
            how=how,
            alignment=alignment,
            train_test_split="fair",
            **kw,
        )

        fair_skill = he_fair.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment=alignment,
            skipna=False,
        )

        assert not unfair_skill.sst.isnull().all()
        assert not fair_skill.sst.isnull().all()

        assert (fair_skill > unfair_skill).sst.all(), print(
            fair_skill.sst, unfair_skill.sst
        )
        print("checking unfair-cv")
        if how not in ["multiplicative_std", "modified_quantile"]:
            assert not unfair_cv_skill.sst.isnull().all()
            assert (fair_skill > unfair_cv_skill).sst.all(), print(
                fair_skill.sst, unfair_cv_skill.sst
            )


def test_remove_bias_errors(hindcast_NMME_Nino34):
    """Test remove_bias error messaging."""
    how = "additive_mean"
    he = (
        hindcast_NMME_Nino34.sel(lead=[4, 5])
        .sel(model="GEM-NEMO")
        .sel(init=slice("2000", "2009"))
    )

    with pytest.raises(ValueError, match="please provide `train_init`"):
        he.remove_bias(how=how, alignment="same_inits", train_test_split="fair")

    with pytest.raises(ValueError, match="please provide `train_init`"):
        he.remove_bias(
            how=how, alignment="same_inits", train_test_split="fair", train_init=2000
        )

    with pytest.raises(ValueError, match="please provide `train_time`"):
        he.remove_bias(how=how, alignment="same_verif", train_test_split="fair")

    with pytest.raises(ValueError, match="please provide `train_time`"):
        he.remove_bias(
            how=how, alignment="same_verif", train_test_split="fair", train_time=2000
        )

    with pytest.raises(ValueError, match="Please provide `cv="):
        he.remove_bias(how=how, alignment="same_verif", train_test_split="unfair-cv")

    with pytest.raises(NotImplementedError, match="please choose from"):
        he.remove_bias(how="new", alignment="same_verif", train_test_split="unfair-cv")

    for tts in ["fair-sliding", "fair-all"]:
        with pytest.raises(
            NotImplementedError, match="Please choose `train_test_split` from"
        ):
            he.remove_bias(how="new", alignment="same_verif", train_test_split="tts")
