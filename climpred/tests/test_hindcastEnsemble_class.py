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
    hindcast.compute_uninitialized()  # single reference, no declaration of name.
    hindcast.add_reference(observations_ds, 'observations')
    hindcast.compute_uninitialized()  # multiple references, no name declaration.
    hindcast.compute_uninitialized('reconstruction')  # multiple references, call one.
    hindcast.compute_uninitialized(metric='rmse', comparison='m2r')


def test_compute_persistence(initialized_ds, reconstruction_ds, observations_ds):
    """Test to see if compute_persistence can be run from the HindcastEnsemble"""
    hindcast = HindcastEnsemble(initialized_ds)
    hindcast.add_reference(reconstruction_ds, 'reconstruction')
    hindcast.add_reference(observations_ds, 'observations')
    hindcast.compute_persistence()
    hindcast.compute_persistence('observations')
    hindcast.compute_persistence(metric='rmse')
