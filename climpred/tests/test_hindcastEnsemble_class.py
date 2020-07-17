import pytest

from climpred import HindcastEnsemble


def test_hindcastEnsemble_init(hind_ds_initialized_1d):
    """Test to see hindcast ensemble can be initialized with xr.Dataset."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    print(hindcast)
    assert hindcast


def test_hindcastEnsemble_init_da(hind_da_initialized_1d):
    """Test to see hindcast ensemble can be initialized with xr.DataArray."""
    hindcast = HindcastEnsemble(hind_da_initialized_1d)
    assert hindcast


def test_add_observations(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if observations can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    assert hindcast.get_observations()


def test_add_observations_da_1d(hind_ds_initialized_1d, observations_da_1d):
    """Test to see if observations can be added to the HindcastEnsemble as a da"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(observations_da_1d, 'observations')
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


def test_verify(hind_ds_initialized_1d, reconstruction_ds_1d, observations_ds_1d):
    """Test to see if verify can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast = hindcast.add_observations(observations_ds_1d, 'observations')
    # Don't need to check for NaNs, etc. since that's handled in the prediction
    # module testing.
    hindcast.verify()  # compute over all observations
    hindcast.verify('reconstruction')  # compute over single observation
    # test all keywords
    hindcast.verify(metric='rmse', comparison='m2o')


def test_verify_single(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if verify automatically works with a single observational
    product."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast.verify()


def test_isel_xarray_func(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test whether applying isel to the objects works."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'FOSI')
    hindcast = hindcast.isel(lead=0, init=slice(0, 3)).isel(time=slice(5, 10))
    assert hindcast.get_initialized().init.size == 3
    assert hindcast.get_initialized().lead.size == 1
    assert hindcast.get_observations('FOSI').time.size == 5


def test_get_initialized(hind_ds_initialized_1d):
    """Test whether get_initialized method works."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    init = hindcast.get_initialized()
    assert init == hindcast._datasets['initialized']


def test_get_uninitialized(hind_ds_initialized_1d, hist_ds_uninitialized_1d):
    """Test whether get_uninitialized method works."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    uninit = hindcast.get_uninitialized()
    assert uninit == hindcast._datasets['uninitialized']


def test_get_observations(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Tests whether get_observations method works."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'FOSI')
    # Without name keyword.
    obs = hindcast.get_observations()
    assert obs == hindcast._datasets['observations']['FOSI']
    # With name keyword.
    obs = hindcast.get_observations('FOSI')
    assert obs == hindcast._datasets['observations']['FOSI']


def test_inplace(
    hind_ds_initialized_1d, reconstruction_ds_1d, hist_ds_uninitialized_1d
):
    """Tests that inplace operations do not work."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    # Adding observations.
    hindcast.add_observations(reconstruction_ds_1d, 'FOSI')
    with_obs = hindcast.add_observations(reconstruction_ds_1d, 'FOSI')
    assert hindcast != with_obs
    # Adding an uninitialized ensemble.
    hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    with_uninit = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    assert hindcast != with_uninit
    # Applying arbitrary func.
    hindcast.sum('init')
    summed = hindcast.sum('init')
    assert hindcast != summed


@pytest.mark.parametrize('alignment', ['same_inits', 'same_verifs', 'maximize'])
def test_mean_reduce_bias(hindcast_hist_obs_1d, alignment):
    how = 'mean'
    metric = 'rmse'
    hindcast = hindcast_hist_obs_1d
    biased_skill = hindcast.verify(metric=metric, alignment=alignment)
    bias_reduced_skill = hindcast.reduce_bias(
        how=how, alignment=alignment, cross_validate=False
    ).verify(metric=metric, alignment=alignment)
    bias_reduced_skill_properly = hindcast.reduce_bias(
        how=how, cross_validate=True, alignment=alignment
    ).verify(metric=metric, alignment=alignment)
    assert biased_skill > bias_reduced_skill
    assert biased_skill > bias_reduced_skill_properly
    assert bias_reduced_skill_properly >= bias_reduced_skill


def test_verify_metric_kwargs(hindcast_hist_obs_1d):
    """Test that HindcastEnsemble works with metrics using metric_kwargs."""
    assert hindcast_hist_obs_1d.verify(
        metric='threshold_brier_score',
        comparison='m2o',
        threshold=0.5,
        reference='historical',
    )


def test_verify_fails_expected_metric_kwargs(hindcast_hist_obs_1d):
    """Test that HindcastEnsemble fails when metric_kwargs expected but not given."""
    hindcast = hindcast_hist_obs_1d
    with pytest.raises(ValueError) as excinfo:
        hindcast.verify(metric='threshold_brier_score', comparison='m2o')
    assert 'Please provide threshold.' == str(excinfo.value)
