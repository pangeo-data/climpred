"""One retained EOF still needs a two-dimensional covariance matrix."""

import numpy as np
import pytest

from climpred import relative_entropy as module


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("neofs", [1, 2])
def test_public_relative_entropy_covariance_shape(
    PM_ds_initialized_3d, PM_ds_control_3d, monkeypatch, dataset, anomaly, neofs
):
    pytest.importorskip("eofs")
    initialized = PM_ds_initialized_3d.tos.isel(init=slice(0, 2))
    control = PM_ds_control_3d.tos
    if dataset:
        initialized, control = initialized.to_dataset(), control.to_dataset()
    original = module._relative_entropy_formula
    expected = []

    def observe(b, x, mx, mb, count):
        # Promote only the independent oracle inputs, not the production inputs.
        original_b, original_x = b, x
        b, x = np.atleast_2d(b), np.atleast_2d(x)
        delta = np.asarray(mx - mb)
        dispersion = 0.5 * (
            np.linalg.slogdet(b)[1]
            - np.linalg.slogdet(x)[1]
            + np.trace(np.linalg.solve(b, x))
            - count
        )
        signal = 0.5 * delta @ np.linalg.solve(b, delta)
        expected.append((dispersion + signal, dispersion, signal))
        # Multi-mode covariance arithmetic is separate from scalar shape.
        return original(original_b, original_x, mx, mb, count)

    monkeypatch.setattr(module, "_relative_entropy_formula", observe)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        actual = module.compute_relative_entropy(
            initialized,
            control,
            nmember_control=8,
            neofs=neofs,
            nlead=2,
            anomaly_data=anomaly,
        )
    finally:
        np.random.set_state(state)
    assert actual.sizes == {"init": 2, "lead": 2}
    assert np.isfinite(actual.to_array()).all()
    if neofs == 1:
        expected = np.asarray(expected).reshape(2, 2, 3)
        np.testing.assert_allclose(actual.R, expected[:, :, 0])
        np.testing.assert_allclose(actual.D, expected[:, :, 1])
        np.testing.assert_allclose(actual.S, expected[:, :, 2])


@pytest.mark.parametrize("dataset", [False, True])
def test_bootstrap_relative_entropy_with_one_mode(
    PM_ds_initialized_3d, PM_ds_control_3d, dataset
):
    pytest.importorskip("eofs")
    initialized = PM_ds_initialized_3d.tos.isel(init=slice(0, 2), lead=slice(0, 2))
    control = PM_ds_control_3d.tos
    if dataset:
        initialized, control = initialized.to_dataset(), control.to_dataset()
    state = np.random.get_state()
    try:
        np.random.seed(42)
        actual = module.bootstrap_relative_entropy(
            initialized, control, nmember_control=8, neofs=1, nlead=2, bootstrap=2
        )
    finally:
        np.random.set_state(state)
    assert set(actual.data_vars) == {"R", "D", "S"}
    assert np.isfinite(actual.to_array()).all()
