import pytest
import xarray as xr

from climpred import HindcastEnsemble
from climpred.exceptions import CoordinateError, DimensionError


def test_hindcastEnsemble_init(hind_ds_initialized_1d):
    """Test to see hindcast ensemble can be initialized with xr.Dataset."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    assert hindcast


def test_hindcastEnsemble_init_da(hind_da_initialized_1d):
    """Test to see hindcast ensemble can be initialized with xr.DataArray."""
    hindcast = HindcastEnsemble(hind_da_initialized_1d)
    assert hindcast


def test_add_observations(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if observations can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d)
    assert hindcast.get_observations()


def test_add_observations_da_1d(hind_ds_initialized_1d, observations_da_1d):
    """Test to see if observations can be added to the HindcastEnsemble as a da"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(observations_da_1d)
    assert hindcast.get_observations()


def test_add_uninitialized(hind_ds_initialized_1d, hist_ds_uninitialized_1d):
    """Test to see if an uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    assert hindcast.get_uninitialized()


def test_add_hist_da_uninitialized_1d(hind_ds_initialized_1d, hist_da_uninitialized_1d):
    """Test to see if da uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d).add_uninitialized(
        hist_da_uninitialized_1d
    )
    assert hindcast.get_uninitialized()


def test_verify(hindcast_hist_obs_1d):
    """Test to see if HindcastEnsemble.verify() works."""
    hindcast_hist_obs_1d.verify(
        metric="acc", comparison="e2o", dim="init", alignment="same_verif"
    )


def test_isel_xarray_func(hindcast_hist_obs_1d):
    """Test whether applying isel to the objects works."""
    hindcast = hindcast_hist_obs_1d
    hindcast = hindcast.isel(lead=0, init=slice(0, 3)).isel(time=slice(5, 10))
    assert hindcast.get_initialized().init.size == 3
    assert hindcast.get_initialized().lead.size == 1
    assert hindcast.get_observations().time.size == 5


def test_get_initialized(hindcast_hist_obs_1d):
    """Test whether get_initialized method works."""
    assert hindcast_hist_obs_1d.get_initialized().identical(
        hindcast_hist_obs_1d._datasets["initialized"]
    )


def test_get_uninitialized(hindcast_hist_obs_1d):
    """Test whether get_uninitialized method works."""
    assert hindcast_hist_obs_1d.get_uninitialized().identical(
        hindcast_hist_obs_1d._datasets["uninitialized"]
    )


def test_get_observations(hindcast_hist_obs_1d):
    """Tests whether get_observations method works."""
    assert hindcast_hist_obs_1d.get_observations().identical(
        hindcast_hist_obs_1d._datasets["observations"]
    )


def test_inplace(
    hind_ds_initialized_1d, reconstruction_ds_1d, hist_ds_uninitialized_1d
):
    """Tests that inplace operations do not work."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    # Adding observations.
    hindcast.add_observations(reconstruction_ds_1d)
    with_obs = hindcast.add_observations(reconstruction_ds_1d)
    assert hindcast != with_obs
    # Adding an uninitialized ensemble.
    hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    with_uninit = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    assert hindcast != with_uninit
    # Applying arbitrary func.
    hindcast.sum("init")
    summed = hindcast.sum("init")
    assert hindcast != summed


@pytest.mark.parametrize("call", ["verify", "bootstrap"])
@pytest.mark.parametrize(
    "dim",
    [set(["init"]), list(["init"]), tuple(["init"]), "init"],
    ids=["set", "list", "tuple", "str"],
)
def test_dim_input_type(hindcast_hist_obs_1d, dim, call):
    """Test verify and bootstrap for different dim types."""
    kw = dict(iterations=2) if call == "bootstrap" else {}
    assert getattr(hindcast_hist_obs_1d.isel(lead=range(3)), call)(
        metric="rmse", comparison="e2o", dim=dim, alignment="same_verifs", **kw
    )


def test_verify_metric_kwargs(hindcast_hist_obs_1d):
    """Test that HindcastEnsemble works with metrics using metric_kwargs."""
    assert hindcast_hist_obs_1d.verify(
        metric="threshold_brier_score",
        comparison="m2o",
        dim="member",
        threshold=0.5,
        reference="uninitialized",
        alignment="same_verifs",
    )


def test_verify_fails_expected_metric_kwargs(hindcast_hist_obs_1d):
    """Test that HindcastEnsemble fails when metric_kwargs expected but not given."""
    hindcast = hindcast_hist_obs_1d
    with pytest.raises(ValueError, match="Please provide threshold"):
        hindcast.verify(
            metric="threshold_brier_score",
            comparison="m2o",
            dim="member",
            alignment="same_verifs",
        )


def test_calendar_matching_observations(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Tests that error is thrown if calendars mismatch when adding observations."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    reconstruction_ds_1d["time"] = xr.cftime_range(
        start="1950",
        periods=reconstruction_ds_1d.time.size,
        freq="MS",
        calendar="all_leap",
    )
    with pytest.raises(ValueError, match="does not match"):
        hindcast.add_observations(reconstruction_ds_1d)


def test_calendar_matching_uninitialized(
    hind_ds_initialized_1d, hist_ds_uninitialized_1d
):
    """Tests that error is thrown if calendars mismatch when adding uninitialized."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hist_ds_uninitialized_1d["time"] = xr.cftime_range(
        start="1950",
        periods=hist_ds_uninitialized_1d.time.size,
        freq="MS",
        calendar="all_leap",
    )
    with pytest.raises(ValueError, match="does not match"):
        hindcast.add_uninitialized(hist_ds_uninitialized_1d)


@pytest.mark.parametrize("metric", ["mse", "crps"])
def test_verify_reference_same_dims(hindcast_hist_obs_1d, metric):
    """Test that verify returns the same dimensionality regardless of reference."""
    hindcast = hindcast_hist_obs_1d.isel(lead=range(3), init=range(10))
    if metric == "mse":
        comparison = "e2o"
        dim = "init"
    elif metric == "crps":
        comparison = "m2o"
        dim = ["member", "init"]
    alignment = "same_verif"
    actual_no_ref = hindcast.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        alignment=alignment,
        reference=None,
    )
    actual_uninit_ref = hindcast.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        alignment=alignment,
        reference="uninitialized",
    )
    actual_pers_ref = hindcast.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        alignment=alignment,
        reference="persistence",
    )
    actual_clim_ref = hindcast.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        alignment=alignment,
        reference="climatology",
    )
    assert actual_uninit_ref.skill.size == 2
    assert actual_pers_ref.skill.size == 2
    assert actual_clim_ref.skill.size == 2
    # no additional dimension, +1 because initialized squeezed
    assert len(actual_no_ref.dims) + 1 == len(actual_pers_ref.dims)
    assert len(actual_no_ref.dims) + 1 == len(actual_uninit_ref.dims)
    assert len(actual_no_ref.dims) + 1 == len(actual_clim_ref.dims)
    assert len(actual_pers_ref.dims) == len(actual_uninit_ref.dims)
    assert len(actual_pers_ref.dims) == len(actual_clim_ref.dims)


def test_HindcastEnsemble_multidim_initialized_lessdim_verif(hindcast_hist_obs_1d):
    """
    Test HindcastEnsemble allow broadcast over dimensions in initialized only.

    But not in observations, e.g. model dim which is not available in observations.
    """
    initialized = hindcast_hist_obs_1d.get_initialized()
    obs = hindcast_hist_obs_1d.get_observations()
    hind = HindcastEnsemble(
        initialized.expand_dims("model").isel(model=[0] * 2)
    ).add_observations(obs)
    skill = hind.verify(
        metric="acc", dim="init", comparison="e2o", alignment="same_inits"
    )
    assert "model" in skill.dims


def test_HindcastEnsemble_multidim_verif_lessdim_initialized(hindcast_hist_obs_1d):
    """Test HindcastEnsemble.__init__ fails if obs has more dims than initialized."""
    initialized = hindcast_hist_obs_1d.get_initialized()
    obs = hindcast_hist_obs_1d.get_observations()
    with pytest.raises(
        DimensionError, match="Verification contains more dimensions than initialized"
    ):
        HindcastEnsemble(initialized).add_observations(
            obs.expand_dims("model").isel(model=[0] * 2)
        )


@pytest.mark.parametrize(
    "dim,new_dim,cf_standard_name",
    [
        ("init", "forecast_time", "forecast_reference_time"),
        ("lead", "lead_time", "forecast_period"),
        ("member", "number", "realization"),
    ],
)
def test_HindcastEnsemble_instantiating_standard_name(
    da_lead, dim, new_dim, cf_standard_name
):
    """Test PredictionEnsemble without init only works with matching standard name."""
    init = (
        da_lead.to_dataset(name="var").expand_dims("member").assign_coords(member=[1])
    )
    init["init"] = xr.cftime_range(start="2000", periods=init.init.size, freq="YS")
    init["lead"].attrs["units"] = "years"
    # change to non CLIMPRED_DIMS
    init = init.rename({dim: new_dim})

    if dim != "member":  # member not required
        with pytest.raises(
            DimensionError,
            match="PredictionEnsemble object must contain the following dimensions",
        ):
            HindcastEnsemble(init)

    init[new_dim].attrs["standard_name"] = cf_standard_name
    # find renamed after warning
    with pytest.warns(UserWarning, match="but renamed dimension"):
        init = HindcastEnsemble(init).get_initialized()
        assert dim in init.dims, print(init.dims, init.coords)


@pytest.mark.parametrize("lead_freq", ["years", "months", "seasons"])
def test_fractional_leads_360_day(hind_ds_initialized_1d, lead_freq):
    """Test that lead can also contain floats when calendar='360_day'."""
    hind_ds_initialized_1d["init"] = xr.cftime_range(
        start=str(hind_ds_initialized_1d.init[0].values),
        freq="YS",
        periods=hind_ds_initialized_1d.init.size,
        calendar="360_day",
    )
    with xr.set_options(keep_attrs=True):
        hind_ds_initialized_1d["lead"] = hind_ds_initialized_1d["lead"] - 0.5
        hind_ds_initialized_1d["lead"].attrs["units"] = lead_freq
    assert HindcastEnsemble(hind_ds_initialized_1d)


@pytest.mark.parametrize("lead_freq", ["weeks", "pentads", "days", "seconds"])
@pytest.mark.parametrize(
    "calendar", ["standard", "julian", "noleap", "proleptic_gregorian"]
)
def test_fractional_leads_lower_than_month_lead_units(
    hind_ds_initialized_1d, lead_freq, calendar
):
    """Test that lead can contain floats when lead units is lower or equal to weeks'."""
    hind_ds_initialized_1d["init"] = xr.cftime_range(
        start=str(hind_ds_initialized_1d.init[0].values),
        freq="YS",
        periods=hind_ds_initialized_1d.init.size,
        calendar=calendar,
    )
    hind_ds_initialized_1d["lead"] = hind_ds_initialized_1d["lead"] - 0.5
    hind_ds_initialized_1d["lead"].attrs["units"] = lead_freq
    assert HindcastEnsemble(hind_ds_initialized_1d)


def test_fractional_leads_fails(hind_ds_initialized_1d):
    """Test float leads fail for normal calendars and lead units in larger than days."""
    with xr.set_options(keep_attrs=True):
        hind_ds_initialized_1d["lead"] = hind_ds_initialized_1d["lead"] - 0.5
    with pytest.raises(CoordinateError, match="Require integer"):
        HindcastEnsemble(hind_ds_initialized_1d)


def test_HindcastEnsemble_reference_uninitialized_sensitive_to_alignment(
    hindcast_hist_obs_1d,
):
    """Test that reference='uninitialized' is sensitive to alignment."""
    kw = dict(metric="mse", comparison="e2o", dim=[], reference="uninitialized")
    skill_same_inits = hindcast_hist_obs_1d.verify(alignment="same_inits", **kw).sel(
        skill="uninitialized"
    )
    skill_same_verif = hindcast_hist_obs_1d.verify(alignment="same_verif", **kw).sel(
        skill="uninitialized"
    )
    skill_maximize = hindcast_hist_obs_1d.verify(alignment="maximize", **kw).sel(
        skill="uninitialized"
    )

    assert not skill_same_inits.mean("init").equals(skill_same_verif.mean("init"))
    assert not skill_maximize.mean("init").equals(skill_same_verif.mean("init"))
    assert not skill_same_inits.mean("init").equals(skill_maximize.mean("init"))

    # same_inits and same_verif have same skill init size
    assert (
        skill_same_inits.isel(lead=4).dropna("init").init.size
        == skill_same_verif.isel(lead=4).dropna("init").init.size
    )
    # maximize has more inits
    assert (
        skill_same_inits.isel(lead=4).dropna("init").init.size
        < skill_maximize.isel(lead=4).dropna("init").init.size
    )


def test_HindcastEnsemble_remove_seasonality(
    hindcast_hist_obs_1d,
):
    """Test remove_seasonality reduces distance."""
    hindcast_offset = hindcast_hist_obs_1d.copy()
    hindcast_offset._datasets["observations"] += 1.0
    no_seasonality = hindcast_offset.remove_seasonality()
    kw = dict(
        metric="mse",
        comparison="e2o",
        dim="init",
        alignment="same_inits",
        reference="uninitialized",
    )
    skill_no_seasonality = no_seasonality.verify(**kw)
    skill_seasonality = hindcast_offset.verify(**kw)
    assert (skill_seasonality >= skill_no_seasonality).to_array().all()


def test_HindcastEnsemble_verify_groupby(
    hindcast_hist_obs_1d,
):
    """Test groupby keyword."""
    kw = dict(
        metric="mse",
        comparison="e2o",
        dim="init",
        reference="uninitialized",
        alignment="same_inits",
    )
    grouped_skill = hindcast_hist_obs_1d.verify(**kw, groupby="month")
    assert "month" in grouped_skill.dims
    grouped_skill = hindcast_hist_obs_1d.verify(
        **kw, groupby=hindcast_hist_obs_1d.get_initialized().init.dt.month
    )
    assert "month" in grouped_skill.dims
    grouped_skill = hindcast_hist_obs_1d.bootstrap(iterations=2, **kw, groupby="month")
    assert "month" in grouped_skill.dims
