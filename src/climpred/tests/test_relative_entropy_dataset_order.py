"""EOF inputs should put samples first, not the Dataset variable axis."""

import numpy as np
import pytest
import xarray as xr

from climpred.relative_entropy import compute_relative_entropy


@pytest.mark.parametrize("variables", [1, 2])
@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("layout", ["native", "spatial_first"])
def test_multivariable_dataset_matches_explicit_variable_axis(
    PM_ds_initialized_3d, PM_ds_control_3d, anomaly, layout, variables
):
    pytest.importorskip("eofs")
    initialized = PM_ds_initialized_3d.isel(init=slice(0, 2))
    control = PM_ds_control_3d
    if variables == 1:
        initialized, control = initialized[["tos"]], control[["tos"]]
    if layout == "spatial_first":
        initialized = initialized.transpose("y", "x", "member", "init", "lead")
        control = control.transpose("y", "x", "time")
    # Dataset variables are independent spatial channels for EOF projection.
    array_init = initialized.to_array().transpose("member", "init", "lead", ...)
    array_control = control.to_array().transpose("time", ...)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        expected = compute_relative_entropy(
            array_init,
            array_control,
            nmember_control=8,
            neofs=2,
            nlead=2,
            anomaly_data=anomaly,
        )
        np.random.seed(42)
        actual = compute_relative_entropy(
            initialized,
            control,
            nmember_control=8,
            neofs=2,
            nlead=2,
            anomaly_data=anomaly,
        )
    finally:
        np.random.set_state(state)
    xr.testing.assert_allclose(actual, expected)
