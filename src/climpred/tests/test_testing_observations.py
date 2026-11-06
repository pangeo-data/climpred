"""Prediction ensemble assertions must compare both observation datasets."""

import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble
from climpred.testing import assert_PredictionEnsemble


@pytest.fixture
def hindcast_with_observations():
    initialized = xr.Dataset(
        {"a": (("init", "lead", "member"), np.arange(24).reshape(4, 2, 3))},
        coords={
            "init": xr.date_range("2000", periods=4, freq="YS", use_cftime=True),
            "lead": [1, 2],
            "member": [1, 2, 3],
        },
    )
    initialized.lead.attrs["units"] = "years"
    observations = xr.Dataset(
        {"a": ("time", np.arange(6.0))},
        coords={"time": xr.date_range("2000", periods=6, freq="YS", use_cftime=True)},
    )
    return HindcastEnsemble(initialized).add_observations(observations)


@pytest.mark.parametrize("how", ["equal", "allclose", "identical"])
def test_changed_observation_values_rejected(hindcast_with_observations, how):
    before = hindcast_with_observations
    after = before.copy()
    after._datasets["observations"]["a"] = before._datasets["observations"]["a"] + 10
    for left, right in ((before, after), (after, before)):
        with pytest.raises(AssertionError):
            assert_PredictionEnsemble(left, right, how=how)
    assert_PredictionEnsemble(before, before.copy(), how=how)


@pytest.mark.parametrize("change", ["coordinate", "variable", "empty"])
def test_changed_observation_structure_rejected(hindcast_with_observations, change):
    before = hindcast_with_observations
    after = before.copy()
    observations = after._datasets["observations"]
    if change == "coordinate":
        after._datasets["observations"] = observations.assign_coords(
            time=observations.time.shift(time=1)
        )
    elif change == "variable":
        after._datasets["observations"] = observations.rename({"a": "b"})
    else:
        after._datasets["observations"] = {}
    for left, right in ((before, after), (after, before)):
        with pytest.raises(AssertionError):
            assert_PredictionEnsemble(left, right)


def test_observation_tolerances_are_forwarded(hindcast_with_observations):
    before = hindcast_with_observations
    after = before.copy()
    after._datasets["observations"]["a"] = before._datasets["observations"]["a"] + 0.01
    assert_PredictionEnsemble(before, after, how="allclose", atol=0.02, rtol=0)
    with pytest.raises(AssertionError):
        assert_PredictionEnsemble(before, after, how="allclose", atol=0.001, rtol=0)


def test_observation_attrs_require_identical(hindcast_with_observations):
    before = hindcast_with_observations
    after = before.copy()
    after._datasets["observations"]["a"].attrs["units"] = "changed"
    assert_PredictionEnsemble(before, after, how="equal")
    assert_PredictionEnsemble(before, after, how="allclose")
    with pytest.raises(AssertionError):
        assert_PredictionEnsemble(before, after, how="identical")


def test_existing_initialized_difference_rejected(hindcast_with_observations):
    before = hindcast_with_observations
    after = before.copy()
    after._datasets["initialized"]["a"] = before._datasets["initialized"]["a"] + 10
    with pytest.raises(AssertionError):
        assert_PredictionEnsemble(before, after)


def test_missing_observations_remain_supported(hindcast_with_observations):
    before = hindcast_with_observations
    before._datasets["observations"] = {}
    assert_PredictionEnsemble(before, before.copy())
