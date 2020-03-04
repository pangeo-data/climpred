import pytest

from climpred import HindcastEnsemble


def test_hindcastEnsemble_init(hind_ds_initialized_1d):
    """Test to see hindcast ensemble can be initialized"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    print(hindcast)
    assert hindcast


def test_hindcastEnsemble_init_da(hind_da_initialized_1d):
    """Test to see hindcast ensemble can be initialized with da"""
    hindcast = HindcastEnsemble(hind_da_initialized_1d)
    assert hindcast


def test_add_reference(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if a reference can be added to the HindcastEnsemble"""
    # TODO: This should be removed once `add_reference` is deprecated.
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_reference(reconstruction_ds_1d, 'reconstruction')
    # Will fail if this comes back empty.
    assert hindcast.get_reference()


def test_add_reference_da(hind_ds_initialized_1d, observations_da_1d):
    """Test to see if a reference can be added to the HindcastEnsemble as a da"""
    # TODO: This should be removed once `add_reference` is deprecated.
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_reference(observations_da_1d, 'observations')
    assert hindcast.get_reference()


def test_add_reference_deprecated(hind_ds_initialized_1d, observations_da_1d):
    """Tests that deprecation warning is thrown for add_reference method."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    with pytest.warns(PendingDeprecationWarning) as record:
        hindcast = hindcast.add_reference(observations_da_1d, 'observations')
    assert 'deprecated' in record[0].message.args[0]


def test_add_observations(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if observations can be added to the HindcastEnsemble"""
    # NOTE: This should be removed once `add_reference` is deprecated.
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    # Will fail if this comes back empty.
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


def test_compute_metric(
    hind_ds_initialized_1d, reconstruction_ds_1d, observations_ds_1d
):
    """Test to see if compute_metric can be run from the HindcastEnsemble"""
    # TODO: Remove this test after deprecating `compute_metric()`
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast = hindcast.add_observations(observations_ds_1d, 'observations')
    # Don't need to check for NaNs, etc. since that's handled in the prediction
    # module testing.
    hindcast.compute_metric()  # compute over all observations
    # compute over single observation
    hindcast.compute_metric('reconstruction')
    # test all keywords
    hindcast.compute_metric(metric='rmse', comparison='m2o')


def test_compute_metric_single(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Test to see if compute_metric automatically works with a single observational
    product."""
    # TODO: Remove this test after deprecating `compute_metric()`
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast.compute_metric()


def test_compute_metric_deprecated(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Tests that deprecation warning is thrown for compute_metric method."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    with pytest.warns(PendingDeprecationWarning) as record:
        hindcast = hindcast.compute_metric()
    assert 'deprecated' in record[0].message.args[0]


@pytest.mark.slow
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


@pytest.mark.skip(reason='will be addressed when refactoring hindcast stuff.')
def test_compute_uninitialized(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    reconstruction_ds_1d,
    observations_ds_1d,
):
    """Test to see if compute_uninitialized can be frun from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast = hindcast.add_uninitialized(hist_ds_uninitialized_1d)
    # single observations, no declaration of name.
    hindcast.compute_uninitialized()
    hindcast = hindcast.add_observations(observations_ds_1d, 'observations')
    # multiple observations, no name declaration.
    hindcast.compute_uninitialized()
    # multiple observations, call one.
    hindcast.compute_uninitialized('reconstruction')
    hindcast.compute_uninitialized(metric='rmse', comparison='m2o')


def test_compute_persistence(
    hind_ds_initialized_1d, reconstruction_ds_1d, observations_ds_1d
):
    """Test to see if compute_persistence can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_observations(reconstruction_ds_1d, 'reconstruction')
    hindcast = hindcast.add_observations(observations_ds_1d, 'observations')
    hindcast.compute_persistence()
    hindcast.compute_persistence('observations')
    hindcast.compute_persistence(metric='rmse')


def test_smooth_goddard(reconstruction_ds_3d, hind_ds_initialized_3d):
    """Test whether goddard smoothing function reduces ntime."""
    hindcast = HindcastEnsemble(hind_ds_initialized_3d.isel(nlat=slice(1, None)))
    hindcast = hindcast.add_observations(
        reconstruction_ds_3d.isel(nlat=slice(1, None)), 'reconstruction'
    )
    hindcast = hindcast.add_uninitialized(
        reconstruction_ds_3d.isel(nlat=slice(1, None))
    )
    initialized_before = hindcast._datasets['initialized']
    hindcast = hindcast.smooth(smooth_kws='goddard2013')
    actual_initialized = hindcast._datasets['initialized']
    dim = 'lead'
    assert actual_initialized[dim].size < initialized_before[dim].size
    for dim in ['nlon', 'nlat']:
        assert actual_initialized[dim[1:]].size < initialized_before[dim].size


def test_smooth_coarsen(reconstruction_ds_3d, hind_ds_initialized_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(hind_ds_initialized_3d)
    hindcast = hindcast.add_observations(reconstruction_ds_3d, 'reconstruction')
    hindcast = hindcast.add_uninitialized(reconstruction_ds_3d)
    initialized_before = hindcast._datasets['initialized']
    dim = 'nlon'
    hindcast = hindcast.smooth(smooth_kws={dim: 2})
    actual_initialized = hindcast._datasets['initialized']
    assert initialized_before[dim].size // 2 == actual_initialized[dim].size


def test_smooth_temporal(reconstruction_ds_3d, hind_ds_initialized_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(hind_ds_initialized_3d)
    hindcast = hindcast.add_observations(reconstruction_ds_3d, 'reconstruction')
    hindcast = hindcast.add_uninitialized(reconstruction_ds_3d)
    initialized_before = hindcast._datasets['initialized']
    dim = 'lead'
    hindcast = hindcast.smooth(smooth_kws={dim: 4})
    actual_initialized = hindcast._datasets['initialized']
    assert initialized_before[dim].size > actual_initialized[dim].size


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


def test_get_reference(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Tests whether get_reference method works."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_reference(reconstruction_ds_1d, 'FOSI')
    # Without name keyword.
    ref = hindcast.get_reference()
    assert ref == hindcast._datasets['observations']['FOSI']
    # With name keyword.
    ref = hindcast.get_reference('FOSI')
    assert ref == hindcast._datasets['observations']['FOSI']


def test_get_reference_deprecated(hind_ds_initialized_1d, reconstruction_ds_1d):
    """Tests that deprecation warning is thrown for get_reference method."""
    hindcast = HindcastEnsemble(hind_ds_initialized_1d)
    hindcast = hindcast.add_reference(reconstruction_ds_1d, 'FOSI')
    with pytest.warns(PendingDeprecationWarning) as record:
        hindcast = hindcast.get_reference()
    assert 'deprecated' in record[0].message.args[0]


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
