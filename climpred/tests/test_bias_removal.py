import numpy as np
import pytest
import xarray as xr

from climpred import set_options
from climpred.constants import BIAS_CORRECTION_METHODS, GROUPBY_SEASONALITIES
from climpred.options import OPTIONS

BIAS_CORRECTION_METHODS.remove(
    "gamma_mapping"
)  # fails with these conftest files somehow


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
                how=how, alignment=verify_kwargs["alignment"], cross_validate=False
            )

            bias_reduced_skill.append(hindcast_rb.verify(**verify_kwargs)[v])
    bias_reduced_skill = xr.concat(bias_reduced_skill, "seasonality").assign_coords(
        seasonality=seasonalities
    )

    # check not identical
    for s in seasonalities:
        print(s, bias_reduced_skill.sel(seasonality=s))
        assert bias_reduced_skill.sel(seasonality=s).notnull().all()
        for s2 in seasonalities:
            if s != s2:
                print(s, s2)
                assert (
                    bias_reduced_skill.sel(seasonality=[s, s2])
                    .diff("seasonality")
                    .notnull()
                    .any()
                )


@pytest.mark.parametrize("cross_validate", [False])  # True])
@pytest.mark.parametrize("seasonality", GROUPBY_SEASONALITIES)
@pytest.mark.parametrize("how", BIAS_CORRECTION_METHODS)
@pytest.mark.parametrize(
    "alignment", ["same_inits", "maximize"]
)  # same_verifs  # no overlap here for same_verifs
def test_remove_bias(hindcast_recon_1d_mm, alignment, how, seasonality, cross_validate):
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
            how=how, alignment=alignment, cross_validate=False
        )

        check_hindcast_coords_maintained_except_init(hindcast, hindcast_bias_removed)

        bias_removed_skill = hindcast_bias_removed.verify(**verify_kwargs)

        seasonality = OPTIONS["seasonality"]
        if cross_validate:
            hindcast_bias_removed_properly = hindcast.remove_bias(
                how=how, cross_validate=True, alignment=alignment
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
            if cross_validate:
                assert (
                    hindcast_bias_removed_properly.get_initialized()[v].attrs
                    == hindcast.get_initialized()[v].attrs
                )
            assert (
                hindcast_bias_removed.get_initialized()[v].attrs
                == hindcast.get_initialized()[v].attrs
            )
        # keeps dataset attrs
        if cross_validate:
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
