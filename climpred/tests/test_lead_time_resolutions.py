import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climpred import HindcastEnsemble, PerfectModelEnsemble

NLEADS = 5
NMEMBERS = 3
NINITS = 10
START = "1990"

HindcastEnsemble_verify_kw = dict(
    metric="rmse", comparison="e2o", dim="init", alignment="maximize"
)
PerfectModelEnsemble_verify_kw = dict(metric="rmse", comparison="e2c", dim="init")


@pytest.fixture(
    params=[
        "seconds",
        "minutes",
        "hours",
        "days",
        "pentads",
        "weeks",
        "months",
        "seasons",
        "years",
    ]
)
def HindcastEnsemble_time_resolution(request):
    """Create HindcastEnsemble of given lead time resolution."""
    if request.param == "pentads":
        freq = "5D"
    elif request.param == "weeks":
        freq = "7D"
    elif request.param == "minutes":
        freq = "T"
    elif request.param in "months":
        freq = "MS"
    elif request.param == "seasons":
        freq = "QS"
    elif request.param == "years":
        freq = "YS"
    else:
        freq = request.param[0].upper()
    # create initialized
    init = xr.cftime_range(START, freq=freq, periods=NINITS)
    lead = np.arange(NLEADS)
    member = np.arange(NMEMBERS)
    initialized = xr.DataArray(
        np.random.rand(len(init), len(lead), len(member)),
        dims=["init", "lead", "member"],
        coords=[init, lead, member],
    ).to_dataset(name="var")
    initialized.lead.attrs["units"] = request.param

    # create observations
    time = xr.cftime_range(START, freq=freq, periods=NINITS + NLEADS)
    obs = xr.DataArray(
        np.random.rand(len(time)), dims=["time"], coords=[time]
    ).to_dataset(name="var")
    # climpred.PredictionEnsemble
    hindcast = HindcastEnsemble(initialized).add_observations(obs)
    return hindcast


def test_HindcastEnsemble_time_resolution_verify(HindcastEnsemble_time_resolution):
    """Test that HindcastEnsemble.verify() in any lead time resolution works."""
    assert (
        HindcastEnsemble_time_resolution.verify(**HindcastEnsemble_verify_kw)
        .notnull()
        .any()
    )


def test_PerfectModelEnsemble_time_resolution_verify(HindcastEnsemble_time_resolution):
    """Test that PerfectModelEnsemble.verify() in any lead time resolution works."""
    pm = PerfectModelEnsemble(HindcastEnsemble_time_resolution.get_initialized())
    assert pm.verify(**PerfectModelEnsemble_verify_kw).notnull().any()


@pytest.mark.parametrize(
    "lead_res", ["seconds", "minutes", "hours", "days", "pentads", "weeks"]
)
def test_HindcastEnsemble_lead_pdTimedelta(hind_da_initialized_1d, lead_res):
    """Test to see HindcastEnsemble can be initialized with lead as pd.Timedelta."""
    if lead_res == "pentads":
        n, freq = 5, "d"
    else:
        n, freq = 1, lead_res[0].lower()
    initialized = hind_da_initialized_1d

    initialized["lead"] = [
        pd.Timedelta(f"{i*n} {freq}") for i in initialized.lead.values
    ]
    hindcast = HindcastEnsemble(initialized)

    assert hindcast.get_initialized().lead.attrs["units"] == lead_res


def test_monthly_leads_real_example(hindcast_NMME_Nino34):
    skill = (
        hindcast_NMME_Nino34.isel(lead=[0, 1, 2])
        .sel(init=slice("2005", "2006"))
        .verify(
            metric="crps",
            comparison="m2o",
            dim=["init", "member"],
            alignment="same_inits",
        )
    )
    assert skill.to_array().notnull().all()


def test_daily_leads_real_example(hindcast_S2S_Germany):
    skill = (
        hindcast_S2S_Germany.isel(lead=[0, 1])
        .sel(init=slice("2005", "2006"))
        .verify(
            metric="crps",
            comparison="m2o",
            dim=["init", "member"],
            alignment="same_inits",
        )
    )
    assert skill.to_array().notnull().all()
