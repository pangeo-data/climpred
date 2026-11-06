"""Control block support for annual perfect-model reference ensembles."""

import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble
from climpred.bootstrap import _bootstrap_by_stacking


def _inputs(length=5, dataset=False):
    initialized = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("init", "member", "lead"),
        coords={
            "init": xr.date_range("2000", periods=2, freq="YS", use_cftime=True),
            "member": [1, 2],
            "lead": [0, 1, 2],
        },
        name="temperature",
    )
    initialized.lead.attrs["units"] = "years"
    control = xr.DataArray(
        np.arange(length, dtype=float),
        dims="time",
        coords={
            "time": xr.date_range("1900", periods=length, freq="YS", use_cftime=True)
        },
        name="temperature",
    )
    if dataset:
        return initialized.to_dataset(), control.to_dataset()
    return initialized, control


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("length", [3, 5])
def test_annual_sampling_includes_last_fitting_block(monkeypatch, dataset, length):
    initialized, control = _inputs(length, dataset)
    calls = []

    def choose_last(low, high, size):
        calls.append((low, high, size))
        return np.full(size, high - 1)

    monkeypatch.setattr(np.random, "randint", choose_last)
    actual = _bootstrap_by_stacking(initialized, control)
    assert calls == [(0, length - 3 + 1, 4)]
    expected = (
        initialized.copy(data=np.broadcast_to(np.arange(length - 3, length), (2, 2, 3)))
        if not dataset
        else initialized.copy()
    )
    if dataset:
        expected["temperature"] = initialized.temperature.copy(
            data=np.broadcast_to(np.arange(length - 3, length), (2, 2, 3))
        )
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("dataset", [False, True])
def test_annual_control_exactly_one_block(dataset):
    initialized, control = _inputs(3, dataset)
    actual = _bootstrap_by_stacking(initialized, control)
    values = actual.temperature.values if dataset else actual.values
    np.testing.assert_array_equal(values, np.broadcast_to([0, 1, 2], (2, 2, 3)))


def test_public_perfect_model_can_use_exact_control_block():
    initialized, control = _inputs(3, True)
    ensemble = PerfectModelEnsemble(initialized).add_control(control)
    actual = ensemble.generate_uninitialized().get_uninitialized()
    np.testing.assert_array_equal(
        actual.temperature, np.broadcast_to([0, 1, 2], (2, 2, 3))
    )
    assert actual.lead.attrs["units"] == "years"


def test_annual_first_block_still_available(monkeypatch):
    initialized, control = _inputs()
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.zeros(size, dtype=int)
    )
    actual = _bootstrap_by_stacking(initialized, control)
    np.testing.assert_array_equal(actual, np.broadcast_to([0, 1, 2], (2, 2, 3)))


@pytest.mark.parametrize("chunked", [False, True])
def test_annual_last_block_with_multiple_variables_and_spatial_chunks(
    monkeypatch, chunked
):
    initialized, control = _inputs(5, True)
    initialized["salinity"] = initialized.temperature + 20
    control["salinity"] = control.temperature + 20
    initialized = initialized.expand_dims(site=["west", "east"]).transpose(
        "site", "init", "member", "lead"
    )
    control = control.expand_dims(site=["west", "east"]).transpose("time", "site")
    if chunked:
        initialized = initialized.chunk({"lead": 2, "site": 1})
        control = control.chunk({"time": 2, "site": 1})
    monkeypatch.setattr(
        np.random, "randint", lambda low, high, size: np.full(size, high - 1)
    )
    actual = _bootstrap_by_stacking(initialized, control)
    expected = initialized.copy()
    for variable, offset in [("temperature", 0), ("salinity", 20)]:
        expected[variable] = xr.ones_like(initialized[variable]) * xr.DataArray(
            np.arange(2, 5) + offset, dims="lead", coords={"lead": initialized.lead}
        )
    expected.lead.attrs = initialized.lead.attrs
    # Dataset stacking already places non-stacked axes first. This test checks
    # sampled values and coordinates, not an unrelated dimension-order guarantee.
    xr.testing.assert_identical(
        actual.compute(), expected.transpose(*actual.temperature.dims).compute()
    )
    if chunked:
        assert actual.temperature.chunks is not None
