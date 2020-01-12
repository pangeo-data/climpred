import pytest

from climpred import PerfectModelEnsemble
from climpred.tutorial import load_dataset


@pytest.fixture
def pm_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_ds_ds1d():
    ds = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    return ds


@pytest.fixture
def pm_ds_control1d():
    ds = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    return ds


def test_perfectModelEnsemble_init(pm_ds_ds1d):
    """Test to see if perfect model ensemble can be initialized"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    print(PerfectModelEnsemble)
    assert pm


def test_perfectModelEnsemble_init_da(pm_da_ds1d):
    """Test to see if perfect model ensemble can be initialized with da"""
    pm = PerfectModelEnsemble(pm_da_ds1d)
    assert pm


def test_add_control(pm_ds_ds1d, pm_ds_control1d):
    """Test to see if control can be added to PerfectModelEnsemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    assert pm.get_control()


def test_generate_uninit(pm_ds_ds1d, pm_ds_control1d):
    """Test to see if uninitialized ensemble can be bootstrapped"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm = pm.generate_uninitialized()
    assert pm.get_uninitialized()


def test_compute_metric(pm_ds_ds1d, pm_ds_control1d):
    """Test that metric can be computed for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm.compute_metric()


def test_compute_uninitialized(pm_ds_ds1d, pm_ds_control1d):
    """Test that compute uninitialized can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm = pm.generate_uninitialized()
    pm.compute_uninitialized()


def test_compute_persistence(pm_ds_ds1d, pm_ds_control1d):
    """Test that compute persistence can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm.compute_persistence()


def test_bootstrap(pm_ds_ds1d, pm_ds_control1d):
    """Test that perfect model ensemble object can be bootstrapped"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm.bootstrap(bootstrap=2)


def test_get_initialized(pm_ds_ds1d):
    """Test whether get_initialized function works."""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    init = pm.get_initialized()
    assert init == pm._datasets['initialized']


def test_get_uninitialized(pm_ds_ds1d, pm_ds_control1d):
    """Test whether get_uninitialized function works."""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    pm = pm.generate_uninitialized()
    uninit = pm.get_uninitialized()
    assert uninit == pm._datasets['uninitialized']


def test_get_control(pm_ds_ds1d, pm_ds_control1d):
    """Test whether get_control function works."""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm = pm.add_control(pm_ds_control1d)
    ctrl = pm.get_control()
    assert ctrl == pm._datasets['control']


def test_inplace(pm_ds_ds1d, pm_ds_control1d):
    """Tests that inplace operations do not work."""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    # Adding a control.
    pm.add_control(pm_ds_control1d)
    with_ctrl = pm.add_control(pm_ds_control1d)
    assert pm != with_ctrl
    # Adding an uninitialized ensemble.
    pm = pm.add_control(pm_ds_control1d)
    pm.generate_uninitialized()
    with_uninit = pm.generate_uninitialized()
    assert pm != with_uninit
    # Applying arbitrary func.
    pm.sum('init')
    summed = pm.sum('init')
    assert pm != summed
