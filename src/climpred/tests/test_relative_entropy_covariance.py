"""Gaussian dispersion uses a matrix solve, not elementwise covariance division."""

import numpy as np
import pytest
import xarray as xr

from climpred import relative_entropy as module


def gaussian_oracle(b, x, mb, mx):
    difference = np.asarray(mx) - np.asarray(mb)
    dispersion = 0.5 * (
        np.linalg.slogdet(b)[1]
        - np.linalg.slogdet(x)[1]
        + np.trace(np.linalg.solve(b, x))
        - len(difference)
    )
    signal = 0.5 * difference @ np.linalg.solve(b, difference)
    return dispersion + signal, dispersion, signal


@pytest.mark.parametrize(
    "b",
    [
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.array([[2.0, -0.8], [-0.8, 1.0]]),
        np.diag([2.0, 2.0]),
    ],
)
@pytest.mark.parametrize("shift", [False, True])
def test_correlated_covariance_matches_gaussian_kl(b, shift):
    x = np.array([[3.0, 0.2], [0.2, 1.0]])
    mb = xr.DataArray([0.0, 0.0], dims="mode")
    mx = xr.DataArray([1.0, -0.3] if shift else [0.0, 0.0], dims="mode")
    actual = module._relative_entropy_formula(b, x, mx, mb, 2)
    np.testing.assert_allclose(actual, gaussian_oracle(b, x, mb, mx), atol=1e-12)


@pytest.mark.parametrize("anomaly", [False, True])
def test_public_entropy_uses_full_projected_covariance(
    PM_ds_initialized_3d, PM_ds_control_3d, monkeypatch, anomaly
):
    pytest.importorskip("eofs")
    original = module._relative_entropy_formula
    expected = []

    def observe(b, x, mx, mb, neofs):
        expected.append(gaussian_oracle(np.asarray(b), np.asarray(x), mb, mx))
        return original(b, x, mx, mb, neofs)

    monkeypatch.setattr(module, "_relative_entropy_formula", observe)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        actual = module.compute_relative_entropy(
            PM_ds_initialized_3d.tos.isel(init=slice(0, 2)),
            PM_ds_control_3d.tos,
            nmember_control=8,
            neofs=2,
            nlead=2,
            anomaly_data=anomaly,
        )
    finally:
        np.random.set_state(state)
    expected = np.asarray(expected).reshape(2, 2, 3)
    np.testing.assert_allclose(actual.R, expected[:, :, 0], atol=1e-10)
    np.testing.assert_allclose(actual.D, expected[:, :, 1], atol=1e-10)
    np.testing.assert_allclose(actual.S, expected[:, :, 2], atol=1e-10)
