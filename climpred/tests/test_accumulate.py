import xarray as xr

from climpred import HindcastEnsemble, PerfectModelEnsemble


def test_HindcastEnsemble_accumulate(hindcast_hist_obs_1d):
    """Test if HindcastEnsemble understands lead attribute aggregate set to cumsum."""
    hindcast_hist_obs_1d = hindcast_hist_obs_1d.rename({"SST": "pr"})
    initialized = xr.ones_like(hindcast_hist_obs_1d.get_initialized())
    obs = xr.ones_like(hindcast_hist_obs_1d.get_observations())

    he = HindcastEnsemble(initialized).add_observations(obs)
    metric_kwargs = {
        "metric": "mae",
        "comparison": "e2o",
        "dim": "init",
        "alignment": "same_verifs",
    }
    skill = he.verify(**metric_kwargs)

    initialized.lead.attrs["aggregate"] = "cumsum"
    obs.time.attrs["aggregate"] = "cumsum"
    he = HindcastEnsemble(initialized).add_observations(obs)
    skill_accum = he.verify(**metric_kwargs)
    print(skill)
    print(skill_accum)
    print(skill - skill_accum)
    assert False
    assert not skill_accum.equals(skill)


def test_PerfectModelEnsemble_accumulate(perfectModelEnsemble_initialized_control):
    """Test if HindcastEnsemble understands lead attribute aggregate set to cumsum."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.rename({"tos": "pr"})
    initialized = pm.get_initialized()

    he = PerfectModelEnsemble(initialized)
    skill = he.verify(metric="rmse", comparison="m2e", dim=["init", "member"])

    initialized.lead.attrs["aggregate"] = "cumsum"
    he = PerfectModelEnsemble(initialized)
    skill_accum = he.verify(metric="rmse", comparison="m2e", dim=["init", "member"])
    assert not skill_accum.equals(skill)
