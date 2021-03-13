import pytest
import xarray as xr

from climpred import HindcastEnsemble


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
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_uninitialized(hist_da_uninitialized_1d)
    assert hindcast.get_uninitialized()


def test_verify(hindcast_hist_obs_1d):
    """Test to see if HindcastEnsemble.verify() works."""
    hindcast = hindcast_hist_obs_1d
    hindcast.verify(metric="acc", comparison="e2o", dim="init", alignment="same_verif")


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


@pytest.mark.parametrize("alignment", ["same_inits", "same_verifs", "maximize"])
def test_mean_remove_bias(hindcast_hist_obs_1d, alignment):
    """Test remove mean bias, ensure than skill doesnt degrade and keeps attrs."""
    how = "mean"
    metric = "rmse"
    dim = "init"
    comparison = "e2o"
    hindcast = hindcast_hist_obs_1d.isel(lead=range(3))
    hindcast._datasets["initialized"].attrs["test"] = "test"
    hindcast._datasets["initialized"]["SST"].attrs["units"] = "test_unit"
    verify_kwargs = dict(
        metric=metric,
        alignment=alignment,
        dim=dim,
        comparison=comparison,
        keep_attrs=True,
    )

    biased_skill = hindcast.verify(**verify_kwargs)

    hindcast_bias_removed = hindcast.remove_bias(
        how=how, alignment=alignment, cross_validate=False
    )
    bias_removed_skill = hindcast_bias_removed.verify(**verify_kwargs)

    hindcast_bias_removed_properly = hindcast.remove_bias(
        how=how, cross_validate=True, alignment=alignment
    )
    bias_removed_skill_properly = hindcast_bias_removed_properly.verify(**verify_kwargs)

    assert "dayofyear" not in bias_removed_skill_properly.coords
    assert biased_skill > bias_removed_skill
    assert biased_skill > bias_removed_skill_properly
    assert bias_removed_skill_properly >= bias_removed_skill
    # keeps data_vars attrs
    for v in hindcast_bias_removed.get_initialized().data_vars:
        assert (
            hindcast_bias_removed_properly.get_initialized()[v].attrs
            == hindcast.get_initialized()[v].attrs
        )
        assert (
            hindcast_bias_removed.get_initialized()[v].attrs
            == hindcast.get_initialized()[v].attrs
        )
    # keeps dataset attrs
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
