import pytest
import xarray as xr

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


def test_add_control(perfectModelEnsemble_initialized_control):
    """Test to see if control can be added to PerfectModelEnsemble"""
    assert perfectModelEnsemble_initialized_control.get_control()


def test_generate_uninit(perfectModelEnsemble_initialized_control):
    """Test to see if uninitialized ensemble can be bootstrapped"""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    assert pm.get_uninitialized()


@pytest.mark.skip(reason="skip now until uninit is refactored")
def test_compute_uninitialized(perfectModelEnsemble_initialized_control):
    """Test that compute uninitialized can be run for perfect model ensemble"""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    pm._compute_uninitialized()


def test_compute_persistence(perfectModelEnsemble_initialized_control):
    """Test that compute persistence can be run for perfect model ensemble"""
    perfectModelEnsemble_initialized_control._compute_persistence(metric="acc")


@pytest.mark.slow
def test_bootstrap(perfectModelEnsemble_initialized_control):
    """Test that perfect model ensemble object can be bootstrapped"""
    perfectModelEnsemble_initialized_control.bootstrap(
        iterations=2, metric="acc", comparison="m2e", dim=["init", "member"]
    )


def test_get_initialized(PM_ds_initialized_1d):
    """Test whether get_initialized function works."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    init = pm.get_initialized()
    assert init == pm._datasets["initialized"]


def test_get_uninitialized(perfectModelEnsemble_initialized_control):
    """Test whether get_uninitialized function works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    uninit = pm.get_uninitialized()
    assert uninit == pm._datasets["uninitialized"]


def test_get_control(perfectModelEnsemble_initialized_control):
    """Test whether get_control function works."""
    ctrl = perfectModelEnsemble_initialized_control.get_control()
    assert ctrl == perfectModelEnsemble_initialized_control._datasets["control"]


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
    pm.sum("init")
    summed = pm.sum("init")
    assert pm != summed


def test_verify(perfectModelEnsemble_initialized_control):
    """Test that verify works."""
    assert perfectModelEnsemble_initialized_control.verify(
        metric="mse", comparison="m2e", dim=["init", "member"]
    )


def test_verify_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that verify with metric_kwargs works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    assert pm.verify(
        metric="threshold_brier_score",
        comparison="m2c",
        dim=["init", "member"],
        threshold=0.5,
    )


@pytest.mark.parametrize(
    "reference",
    ["historical", ["historical"], "persistence", None, ["historical", "persistence"]],
)
def test_verify_reference(perfectModelEnsemble_initialized_control, reference):
    """Test that verify works with references given."""
    pm = perfectModelEnsemble_initialized_control.generate_uninitialized()
    skill = pm.verify(
        metric="rmse", comparison="m2e", dim=["init", "member"], reference=reference
    )
    if isinstance(reference, str):
        reference = [reference]
    elif reference is None:
        reference = []
    if len(reference) == 0:
        assert "skill" not in skill.dims
    else:
        assert skill.skill.size == len(reference) + 1


def test_verify_fails_expected_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that verify without metric_kwargs fails."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    with pytest.raises(ValueError) as excinfo:
        pm.verify(
            metric="threshold_brier_score", comparison="m2c", dim=["init", "member"]
        )
    assert "Please provide threshold." == str(excinfo.value)


def test_compute_uninitialized_metric_kwargs(perfectModelEnsemble_initialized_control):
    "Test that _compute_uninitialized with metric_kwargs works"
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    pm = pm.generate_uninitialized()
    assert pm._compute_uninitialized(
        metric="threshold_brier_score",
        comparison="m2c",
        threshold=0.5,
        dim=["init", "member"],
    )


def test_bootstrap_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that bootstrap with metric_kwargs works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    pm = pm.generate_uninitialized()
    assert pm.bootstrap(
        metric="threshold_brier_score",
        comparison="m2c",
        threshold=0.5,
        iterations=3,
        dim=["init", "member"],
    )


def test_calendar_matching_control(PM_da_initialized_1d, PM_ds_control_1d):
    """Tests that error is thrown if calendars mismatch when adding observations."""
    pm = PerfectModelEnsemble(PM_da_initialized_1d)
    PM_ds_control_1d["time"] = xr.cftime_range(
        start="1950", periods=PM_ds_control_1d.time.size, freq="MS", calendar="all_leap"
    )
    with pytest.raises(ValueError) as excinfo:
        pm = pm.add_control(PM_ds_control_1d)
    assert "does not match" in str(excinfo.value)
