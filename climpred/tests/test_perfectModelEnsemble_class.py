import pytest

from climpred import PerfectModelEnsemble


def test_perfectModelEnsemble_init(PM_ds_initialized_1d):
    """Test to see if perfect model ensemble can be initialized."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    assert pm


def test_perfectModelEnsemble_init_da(PM_da_initialized_1d):
    """Test to see if perfect model ensemble can be initialized with da."""
    pm = PerfectModelEnsemble(PM_da_initialized_1d)
    assert pm


@pytest.mark.parametrize(
    'handle', ['get_uninitialized', 'get_control', 'get_initialized'],
)
def test_perfectModelEnsemble_initialized_control_handles_returns(
    perfectModelEnsemble_initialized_control, handle
):
    """Test that perfect model ensemble object gets a return from `handle`."""
    assert getattr(perfectModelEnsemble_initialized_control, handle)


@pytest.mark.parametrize(
    'handle',
    [
        'compute_persistence',
        'compute_uninitialized',
        'bootstrap(bootstrap=2)',
        'compute_metric',
    ],
)
def test_perfectModelEnsemble_initialized_control_handles(
    perfectModelEnsemble_initialized_control, handle
):
    """Test that perfect model ensemble object can use `handle`."""
    getattr(perfectModelEnsemble_initialized_control, handle)


def test_get_initialized(PM_ds_initialized_1d):
    """Test whether get_initialized function works.."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    init = pm.get_initialized()
    assert init == pm._datasets['initialized']


def test_get_uninitialized(perfectModelEnsemble_initialized_control):
    """Test whether get_uninitialized function works.."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    uninit = pm.get_uninitialized()
    assert uninit == pm._datasets['uninitialized']


def test_get_control(perfectModelEnsemble_initialized_control):
    """Test whether get_control function works.."""
    ctrl = perfectModelEnsemble_initialized_control.get_control()
    assert ctrl == perfectModelEnsemble_initialized_control._datasets['control']


def test_inplace(PM_ds_initialized_1d, PM_ds_control_1d):
    """Tests that inplace operations do not work.."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    # Adding a control.
    pm.add_control(PM_ds_control_1d)
    with_ctrl = pm.add_control(PM_ds_control_1d)
    assert pm != with_ctrl
    # Adding an uninitialized ensemble.
    pm = pm.add_control(PM_ds_control_1d)
    pm.generate_uninitialized()
    with_uninit = pm.generate_uninitialized()
    assert pm != with_uninit
    # Applying arbitrary func.
    pm.sum('init')
    summed = pm.sum('init')
    assert pm != summed
