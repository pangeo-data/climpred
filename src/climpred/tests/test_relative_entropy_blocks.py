"""Every contiguous control block that fits must be eligible for resampling."""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_identical

from climpred.relative_entropy import _bootstrap_dim


@pytest.mark.parametrize("length,nlead", [(3, 3), (4, 3), (5, 3), (4, 1)])
def test_all_valid_block_starts_are_eligible(monkeypatch, length, nlead):
    control = xr.DataArray(
        np.arange(length, dtype=float),
        dims="time",
        coords={"time": np.arange(length)},
        name="temperature",
    )
    labels = np.arange(length - nlead + 1)
    before = control.copy(deep=True)

    def choose_every_start(low, high, size):
        assert low == 0
        assert high == length - nlead + 1
        assert size == len(labels)
        return np.arange(low, high)

    monkeypatch.setattr(np.random, "randint", choose_every_start)
    actual = _bootstrap_dim(control, nlead, "member", labels)
    expected = xr.DataArray(
        np.stack([np.arange(start, start + nlead, dtype=float) for start in labels]),
        dims=("member", "lead"),
        coords={"member": labels, "lead": np.arange(1, nlead + 1)},
        name="temperature",
    )
    assert_identical(actual, expected)
    assert_identical(control, before)


def test_single_fitting_block_works_with_real_random_generator():
    control = xr.DataArray([10.0, 20.0, 30.0], dims="time")
    actual = _bootstrap_dim(control, 3, "member", ["a", "b"])
    np.testing.assert_array_equal(actual.values, [[10, 20, 30], [10, 20, 30]])
    assert actual.member.values.tolist() == ["a", "b"]


def test_control_shorter_than_block_remains_invalid():
    with pytest.raises(ValueError):
        _bootstrap_dim(xr.DataArray([1.0, 2.0], dims="time"), 3, "member", [0])


def test_public_relative_entropy_uses_complete_start_range(monkeypatch):
    pytest.importorskip("eofs")
    from climpred.relative_entropy import compute_relative_entropy

    rng = np.random.default_rng(42)
    initialized = xr.DataArray(
        rng.normal(size=(2, 3, 6, 2, 3)),
        dims=("init", "lead", "member", "lat", "lon"),
        coords={
            "init": [2000, 2001],
            "lead": [1, 2, 3],
            "member": np.arange(6),
            "lat": [0, 10],
            "lon": [0, 10, 20],
        },
    )
    control = xr.DataArray(
        rng.normal(size=(30, 2, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(30), "lat": [0, 10], "lon": [0, 10, 20]},
    )
    calls = []

    def choose_starts(low, high, size):
        assert (low, high, size) == (0, 28, 12)
        calls.append((low, high, size))
        return rng.integers(low, high, size)

    monkeypatch.setattr(np.random, "randint", choose_starts)
    actual = compute_relative_entropy(initialized, control, neofs=2, nmember_control=12)
    assert len(calls) == 2
    assert actual.sizes == {"init": 2, "lead": 3}
    assert bool(np.isfinite(actual.to_array()).all())
