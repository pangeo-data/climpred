"""Temporal window labels must retain endpoint signs when computing centers."""

import numpy as np
import pytest
import xarray as xr

from climpred.smoothing import _reset_temporal_axis, _set_center_coord


@pytest.mark.parametrize("kind", ["array", "dataset"])
@pytest.mark.parametrize("start,window", [(-4, 3), (-2, 3), (-1, 2), (0, 3), (1, 4)])
def test_reset_axis_keeps_signed_window_centers(kind, start, window):
    leads = np.arange(start, start + 4)
    original = xr.DataArray(
        np.arange(4.0), dims="lead", coords={"lead": leads}, name="v"
    )
    original.lead.attrs["units"] = "years"
    if kind == "dataset":
        original = original.to_dataset()
    actual = _reset_temporal_axis(original.copy(deep=True), {"lead": window})
    np.testing.assert_array_equal(actual.lead_center, leads + (window - 1) / 2)
    np.testing.assert_array_equal(actual.lead, [f"{i}-{i + window - 1}" for i in leads])
    xr.testing.assert_identical(
        actual.drop_vars("lead_center").assign_coords(lead=leads),
        original.assign_coords(lead=leads),
    )
    assert actual.lead_center.attrs["long_name"] == "lead center of rolling window"


@pytest.mark.parametrize(
    "label,expected",
    [
        ("-4--2", -3),
        ("-1-1", 0),
        ("-0.5-1.5", 0.5),
        ("1e-05-3e-05", 2e-05),
        ("1-3", 2),
        ("0-2", 1),
    ],
)
def test_center_coord_reads_numeric_endpoints(label, expected):
    original = xr.DataArray([7.0], dims="lead", coords={"lead": [label]})
    original.lead.attrs["units"] = "years"
    actual = _set_center_coord(original)
    assert actual.lead_center.attrs["units"] == "years"
    np.testing.assert_allclose(actual.lead_center, [expected])


@pytest.mark.parametrize("start", [-2, 0, 1])
def test_public_smoothed_verify_reports_correct_centers(
    perfectModelEnsemble_initialized_control_1d_ym_cftime, start
):
    from climpred import PerfectModelEnsemble

    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    initialized = pm.get_initialized().copy(deep=True)
    initialized["lead"] = np.arange(start, start + initialized.sizes["lead"])
    initialized.lead.attrs["units"] = "years"
    shifted = PerfectModelEnsemble(initialized).add_control(pm.get_control())
    result = shifted.smooth({"lead": 3}).verify(
        metric="rmse", comparison="m2e", dim="init"
    )
    np.testing.assert_array_equal(
        result.lead_center, np.arange(start, start + result.sizes["lead"]) + 1
)
