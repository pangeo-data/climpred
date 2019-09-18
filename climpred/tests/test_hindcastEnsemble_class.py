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
    hindcast.add_reference(reconstruction_ds, 'reconstruction')


def test_add_reference_da(initialized_ds, observations_da):
    """Test to see if a reference can be added to the HindcastEnsemble as a da"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(observations_da, 'observations')


def test_add_uninitialized(initialized_ds, uninitialized_ds):
    """Test to see if an uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_uninitialized(uninitialized_ds)


def test_add_uninitialized_da(initialized_ds, uninitialized_da):
    """Test to see if da uninitialized ensemble can be added to the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_uninitialized(uninitialized_da)


def test_compute_metric(initialized_ds, reconstruction_ds, observations_ds):
    """Test to see if compute_metric can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.add_reference(observations_ds, 'observations')
    # Don't need to check for NaNs, etc. since that's handled in the prediction
    # module testing.
    hindcast.compute_metric()  # compute over all references
    hindcast.compute_metric('reconstruction')  # compute over single reference
    # test all keywords
    hindcast.compute_metric(max_dof=True, metric='rmse', comparison='m2r')


def test_compute_metric_single(initialized_ds, reconstruction_ds):
    """Test to see if compute_metric automatically works with a single reference"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.compute_metric()


def test_compute_uninitialized(
    initialized_ds, uninitialized_ds, reconstruction_ds, observations_ds
):
    """Test to see if compute_uninitialized can be frun from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.add_uninitialized(uninitialized_ds)
    # single reference, no declaration of name.
    hindcast.compute_uninitialized()
    hindcast.add_reference(observations_ds, 'observations')
    # multiple references, no name declaration.
    hindcast.compute_uninitialized()
    # multiple references, call one.
    hindcast.compute_uninitialized('reconstruction')
    hindcast.compute_uninitialized(metric='rmse', comparison='m2r')


def test_compute_persistence(initialized_ds, reconstruction_ds, observations_ds):
    """Test to see if compute_persistence can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.add_reference(observations_ds, 'observations')
    hindcast.compute_persistence()
    hindcast.compute_persistence('observations')
    hindcast.compute_persistence(metric='rmse')


def test_smooth_goddard(fosi_3d, dple_3d):
    """Test whether goddard smoothing function reduces ntime."""
    hindcast = HindcastEnsemble(dple_3d.isel(nlat=slice(1, None)))
    hindcast.add_reference(fosi_3d.isel(nlat=slice(1, None)), 'reconstruction')
    hindcast.add_uninitialized(fosi_3d.isel(nlat=slice(1, None)))
    initialized_before = hindcast.initialized
    hindcast.smooth(smooth_kws='goddard2013')
    actual_initialized = hindcast.initialized
    dim = 'lead'
    assert actual_initialized[dim].size < initialized_before[dim].size
    for dim in ['nlon', 'nlat']:
        assert actual_initialized[dim[1:]].size < initialized_before[dim].size


def test_smooth_coarsen(fosi_3d, dple_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(dple_3d)
    hindcast.add_reference(fosi_3d, 'reconstruction')
    hindcast.add_uninitialized(fosi_3d)
    initialized_before = hindcast.initialized
    dim = 'nlon'
    hindcast.smooth(smooth_kws={dim: 2})
    actual_initialized = hindcast.initialized
    assert initialized_before[dim].size // 2 == actual_initialized[dim].size


def test_smooth_temporal(fosi_3d, dple_3d):
    """Test whether coarsening reduces dim.size."""
    hindcast = HindcastEnsemble(dple_3d)
    hindcast.add_reference(fosi_3d, 'reconstruction')
    hindcast.add_uninitialized(fosi_3d)
    initialized_before = hindcast.initialized
    dim = 'lead'
    hindcast.smooth(smooth_kws={dim: 4})
    actual_initialized = hindcast.initialized
    assert initialized_before[dim].size > actual_initialized[dim].size
