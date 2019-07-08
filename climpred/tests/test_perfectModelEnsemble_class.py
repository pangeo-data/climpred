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
    """Test tos ee if perfect model ensemble can be initialized with da"""
    pm = PerfectModelEnsemble(pm_da_ds1d)
    assert pm


def test_add_control(pm_ds_ds1d, pm_ds_control1d):
    """Test to see if control can be added to PerfectModelEnsemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)


def test_generate_uninit(pm_ds_ds1d, pm_ds_control1d):
    """Test to see if uninitialized ensemble can be bootstrapped"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)
    pm.generate_uninitialized()


def test_compute_metric(pm_ds_ds1d, pm_ds_control1d):
    """Test that metric can be computed for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)
    pm.compute_metric()


def test_compute_uninitialized(pm_ds_ds1d, pm_ds_control1d):
    """Test that compute uninitialized can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)
    pm.generate_uninitialized()
    pm.compute_uninitialized()


def test_compute_persistence(pm_ds_ds1d, pm_ds_control1d):
    """Test that compute persistence can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)
    pm.compute_persistence()


def test_bootstrap(pm_ds_ds1d, pm_ds_control1d):
    """Test that perfect model ensemble object can be bootstrapped"""
    pm = PerfectModelEnsemble(pm_ds_ds1d)
    pm.add_control(pm_ds_control1d)
    pm.bootstrap(bootstrap=2)
