from climpred import PerfectModelEnsemble


def test_perfectModelEnsemble_init(PM_ds_initialized_1d):
    """Test to see if perfect model ensemble can be initialized"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    print(PerfectModelEnsemble)
    assert pm


def test_perfectModelEnsemble_init_da(PM_da_initialized_1d):
    """Test to see if perfect model ensemble can be initialized with da"""
    pm = PerfectModelEnsemble(PM_da_initialized_1d)
    assert pm


def test_add_control(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test to see if control can be added to PerfectModelEnsemble"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    assert pm.get_control()


def test_generate_uninit(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test to see if uninitialized ensemble can be bootstrapped"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm = pm.generate_uninitialized()
    assert pm.get_uninitialized()


def test_compute_metric(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test that metric can be computed for perfect model ensemble"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm.compute_metric()


def test_compute_uninitialized(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test that compute uninitialized can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm = pm.generate_uninitialized()
    pm.compute_uninitialized()


def test_compute_persistence(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test that compute persistence can be run for perfect model ensemble"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm.compute_persistence()


def test_bootstrap(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test that perfect model ensemble object can be bootstrapped"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm.bootstrap(bootstrap=2)


def test_get_initialized(PM_ds_initialized_1d):
    """Test whether get_initialized function works."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    init = pm.get_initialized()
    assert init == pm._datasets['initialized']


def test_get_uninitialized(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test whether get_uninitialized function works."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    pm = pm.generate_uninitialized()
    uninit = pm.get_uninitialized()
    assert uninit == pm._datasets['uninitialized']


def test_get_control(PM_ds_initialized_1d, PM_ds_control_1d):
    """Test whether get_control function works."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    pm = pm.add_control(PM_ds_control_1d)
    ctrl = pm.get_control()
    assert ctrl == pm._datasets['control']


def test_inplace(PM_ds_initialized_1d, PM_ds_control_1d):
    """Tests that inplace operations do not work."""
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
