import pytest
import xarray as xr

from climpred import set_options


@pytest.mark.parametrize("seasonality", ["month", "season", "dayofyear", "weekofyear"])
@pytest.mark.parametrize("reference", ["persistence", "climatology", "uninitialized"])
def test_HindcastEnsemble_verify_reference(
    hindcast_hist_obs_1d, seasonality, reference
):
    with set_options(seasonality=seasonality):
        hindcast_hist_obs_1d.verify(
            metric="mse",
            comparison="e2o",
            dim="init",
            alignment="same_verifs",
            reference=reference,
        )


@pytest.mark.parametrize("comparison", ["m2m", "m2e", "m2c", "e2c"])
def test_PerfectModelEnsemble_verify_persistence_from_first_lead(
    perfectModelEnsemble_initialized_control, comparison
):
    """Test compute_persistence_from_first_lead vs compute_persistence."""
    kw = dict(
        metric="mse",
        comparison=comparison,
        dim="init" if comparison == "e2c" else ["member", "init"],
        reference=["persistence", "climatology"],
    )
    with set_options(PerfectModel_persistence_from_initialized_lead_0=True):
        new_persistence = perfectModelEnsemble_initialized_control.verify(**kw)
    with set_options(PerfectModel_persistence_from_initialized_lead_0=False):
        old_persistence = perfectModelEnsemble_initialized_control.verify(**kw)
    assert not new_persistence.sel(skill="persistence").equals(
        old_persistence.sel(skill="persistence")
    )
    assert "forecast_member" not in new_persistence.dims
    assert "forecast_member" not in old_persistence.dims


@pytest.mark.parametrize("call", ["verify", "bootstrap"])
def test_PerfectModelEnsemble_persistence_from_first_lead_warning_lead_non_zero(
    perfectModelEnsemble_initialized_control, call
):
    """Test that compute_persistence_from_first_lead warns if first lead not zero."""
    kw = dict(
        metric="mse", comparison="m2e", dim=["member", "init"], reference="persistence"
    )
    if call == "bootstrap":
        kw["iterations"] = 2
    with set_options(PerfectModel_persistence_from_initialized_lead_0=True):
        with pytest.warns(UserWarning, match="Calculate persistence from lead=1"):
            # perfectModelEnsemble_initialized_control starts with lead 1
            getattr(perfectModelEnsemble_initialized_control, call)(**kw)

        # with pytest.does_not_warn(FutureWarning):
        with xr.set_options(keep_attrs=True):
            perfectModelEnsemble_initialized_control._datasets["initialized"][
                "lead"
            ] = (
                perfectModelEnsemble_initialized_control._datasets["initialized"][
                    "lead"
                ]
                - 1
            )
        getattr(perfectModelEnsemble_initialized_control, call)(**kw)
