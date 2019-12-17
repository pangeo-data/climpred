import numpy as np
import pytest

from climpred import HindcastEnsemble
from climpred.tutorial import load_dataset


@pytest.fixture
def initialized_ds():
    da = load_dataset('CESM-DP-SST')
    return da


@pytest.fixture
def initialized_da():
    da = load_dataset('CESM-DP-SST')['SST']
    da = da.sel(init=slice(1955, 2015))
    da = da - da.mean('init')
    return da


@pytest.fixture
def fosi_3d():
    ds = load_dataset('FOSI-SST-3D')
    return ds


@pytest.fixture
def dple_3d():
    ds = load_dataset('CESM-DP-SST-3D')
    return ds


@pytest.fixture
def observations_ds():
    da = load_dataset('ERSST')
    return da


@pytest.fixture
def reconstruction_ds():
    da = load_dataset('FOSI-SST')
    # same timeframe as DPLE
    return da


@pytest.fixture
def uninitialized_ds():
    da = load_dataset('CESM-LE')
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    return da


@pytest.fixture
def uninitialized_da():
    da = load_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


def test_hindcastEnsemble_init(initialized_ds):
    """Test to see hindcast ensemble can be initialized"""
    hindcast = HindcastEnsemble(initialized_ds)
    print(hindcast)
    assert hindcast


def test_hindcastEnsemble_init_da(initialized_da):
    """Test to see hindcast ensemble can be initialized with da"""
    hindcast = HindcastEnsemble(initialized_da)
    assert hindcast


def test_add_reference(initialized_ds, reconstruction_ds):
    """Test to see if a reference can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'reconstruction')
    # Will fail if this comes back empty.
    assert hindcast.get_reference()


def test_add_reference_da(initialized_ds, observations_da):
    """Test to see if a reference can be added to the HindcastEnsemble as a da"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(observations_da, 'observations')
    assert hindcast.get_reference()


def test_add_uninitialized(initialized_ds, uninitialized_ds):
    """Test to see if an uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_uninitialized(uninitialized_ds)
    assert hindcast.get_uninitialized()


def test_add_uninitialized_da(initialized_ds, uninitialized_da):
    """Test to see if da uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_uninitialized(uninitialized_da)
    assert hindcast.get_uninitialized()


def test_compute_metric(initialized_ds, reconstruction_ds, observations_ds):
    """Test to see if compute_metric can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast = hindcast.add_reference(observations_ds, 'observations')
    # Don't need to check for NaNs, etc. since that's handled in the prediction
    # module testing.
    hindcast.compute_metric()  # compute over all references
    hindcast.compute_metric('reconstruction')  # compute over single reference
    # test all keywords
    hindcast.compute_metric(max_dof=True, metric='rmse', comparison='m2r')


def test_compute_metric_single(initialized_ds, reconstruction_ds):
    """Test to see if compute_metric automatically works with a single reference"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.compute_metric()


def test_compute_uninitialized(
    initialized_ds, uninitialized_ds, reconstruction_ds, observations_ds
):
    """Test to see if compute_uninitialized can be frun from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast = hindcast.add_uninitialized(uninitialized_ds)
    # single reference, no declaration of name.
    hindcast.compute_uninitialized()
    hindcast = hindcast.add_reference(observations_ds, 'observations')
    # multiple references, no name declaration.
    hindcast.compute_uninitialized()
    # multiple references, call one.
    hindcast.compute_uninitialized('reconstruction')
    hindcast.compute_uninitialized(metric='rmse', comparison='m2r')


def test_compute_persistence(initialized_ds, reconstruction_ds, observations_ds):
    """Test to see if compute_persistence can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast = hindcast.add_reference(observations_ds, 'observations')
    hindcast.compute_persistence()
    hindcast.compute_persistence('observations')
    hindcast.compute_persistence(metric='rmse')


def test_smooth_goddard(fosi_3d, dple_3d):
    """Test whether goddard smoothing function reduces ntime."""
    hindcast = HindcastEnsemble(dple_3d.isel(nlat=slice(1, None)))
    hindcast = hindcast.add_reference(
        fosi_3d.isel(nlat=slice(1, None)), 'reconstruction'
    )
    hindcast = hindcast.add_uninitialized(fosi_3d.isel(nlat=slice(1, None)))
    initialized_before = hindcast._datasets['initialized']
    hindcast = hindcast.smooth(smooth_kws='goddard2013')
    actual_initialized = hindcast._datasets['initialized']
    dim = 'lead'
    assert actual_initialized[dim].size < initialized_before[dim].size
    for dim in ['nlon', 'nlat']:
        assert actual_initialized[dim[1:]].size < initialized_before[dim].size


def test_smooth_coarsen(fosi_3d, dple_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(dple_3d)
    hindcast = hindcast.add_reference(fosi_3d, 'reconstruction')
    hindcast = hindcast.add_uninitialized(fosi_3d)
    initialized_before = hindcast._datasets['initialized']
    dim = 'nlon'
    hindcast = hindcast.smooth(smooth_kws={dim: 2})
    actual_initialized = hindcast._datasets['initialized']
    assert initialized_before[dim].size // 2 == actual_initialized[dim].size


def test_smooth_temporal(fosi_3d, dple_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(dple_3d)
    hindcast = hindcast.add_reference(fosi_3d, 'reconstruction')
    hindcast = hindcast.add_uninitialized(fosi_3d)
    initialized_before = hindcast._datasets['initialized']
    dim = 'lead'
    hindcast = hindcast.smooth(smooth_kws={dim: 4})
    actual_initialized = hindcast._datasets['initialized']
    assert initialized_before[dim].size > actual_initialized[dim].size


def test_isel_xarray_func(initialized_ds, reconstruction_ds):
    """Test whether applying isel to the objects works."""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'FOSI')
    hindcast = hindcast.isel(lead=0, init=slice(0, 3)).isel(time=slice(5, 10))
    assert hindcast.get_initialized().init.size == 3
    assert hindcast.get_initialized().lead.size == 1
    assert hindcast.get_reference('FOSI').time.size == 5


def test_get_initialized(initialized_ds):
    """Test whether get_initialized function works."""
    hindcast = HindcastEnsemble(initialized_ds)
    init = hindcast.get_initialized()
    assert init == hindcast._datasets['initialized']


def test_get_uninitialized(initialized_ds, uninitialized_ds):
    """Test whether get_uninitialized function works."""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_uninitialized(uninitialized_ds)
    uninit = hindcast.get_uninitialized()
    assert uninit == hindcast._datasets['uninitialized']


def test_get_reference(initialized_ds, reconstruction_ds):
    """Tests whether get_reference function works."""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast = hindcast.add_reference(reconstruction_ds, 'FOSI')
    # Without name keyword.
    ref = hindcast.get_reference()
    assert ref == hindcast._datasets['reference']['FOSI']
    # With name keyword.
    ref = hindcast.get_reference('FOSI')
    assert ref == hindcast._datasets['reference']['FOSI']


def test_inplace(initialized_ds, reconstruction_ds, uninitialized_ds):
    """Tests that inplace operations do not work."""
    hindcast = HindcastEnsemble(initialized_ds)
    # Adding a reference.
    hindcast.add_reference(reconstruction_ds, 'FOSI')
    with_ref = hindcast.add_reference(reconstruction_ds, 'FOSI')
    assert hindcast != with_ref
    # Adding an uninitialized ensemble.
    hindcast.add_uninitialized(uninitialized_ds)
    with_uninit = hindcast.add_uninitialized(uninitialized_ds)
    assert hindcast != with_uninit
    # Applying arbitrary func.
    hindcast.sum('init')
    summed = hindcast.sum('init')
    assert hindcast != summed
