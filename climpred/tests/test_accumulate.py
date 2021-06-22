from climpred import HindcastEnsemble, PerfectModelEnsemble


def test_HindcastEnsemble_accumulate(hindcast_hist_obs_1d):
    """Test if HindcastEnsemble understands lead attribute aggregate set to cumsum."""
    hindcast_hist_obs_1d = hindcast_hist_obs_1d.rename({"SST": "pr"})
    initialized = hindcast_hist_obs_1d.get_initialized()
    obs = hindcast_hist_obs_1d.get_observations()

    he = HindcastEnsemble(initialized).add_observations(obs)
    skill = he.verify(
        metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
    )

    initialized.lead.attrs["aggregate"] = "cumsum"
    he = HindcastEnsemble(initialized).add_observations(obs)
    skill_accum = he.verify(
        metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
    )
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
