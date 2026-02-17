"""Testing checks.py."""

import pytest

from climpred.checks import (
    has_dataset,
    has_dims,
    has_min_len,
    has_valid_lead_units,
    is_in_list,
    match_initialized_dims,
    match_initialized_vars,
)
from climpred.constants import VALID_LEAD_UNITS
from climpred.exceptions import DatasetError, DimensionError, VariableError


def test_has_dims_str(da1):
    """Test if check works for a string."""
    assert has_dims(da1, "x", "arbitrary")


def test_has_dims_list(da1):
    """Test if check works for a list."""
    # returns None if no errors
    assert has_dims(da1, ["x", "y"], "arbitrary")


def test_has_dims_str_fail(da1):
    """Test if check fails properly for a string."""
    with pytest.raises(DimensionError) as e:
        has_dims(da1, "z", "arbitrary")
    assert "Your arbitrary object must contain" in str(e.value)


def test_has_dims_list_fail(da1):
    """Test if check fails properly for a list."""
    with pytest.raises(DimensionError) as e:
        has_dims(da1, ["z"], "arbitrary")
    assert "Your arbitrary object must contain" in str(e.value)


def test_has_min_len_arr(da1):
    """Test if check works for min len."""
    assert has_min_len(da1.values, 2, "arbitrary")


def test_has_min_len_fail(da1):
    """Test if check fails properly."""
    with pytest.raises(DimensionError) as e:
        has_min_len(da1.values, 5, "arbitrary")
    assert "Your arbitrary array must be at least" in str(e.value)


def test_has_dataset():
    """Test if check works for a non-empty list."""
    obj = [5]
    assert has_dataset(obj, "list", "something")


def test_has_dataset_fail():
    """Test if check works to fail for an empty list."""
    obj = []
    with pytest.raises(DatasetError) as e:
        has_dataset(obj, "test", "something")
    assert "You need to add at least one test dataset" in str(e.value)


def test_match_initialized_dims(da1, da2):
    """Test if check works if both da has the proper dims."""
    assert match_initialized_dims(da1.rename({"y": "init"}), da2.rename({"y": "time"}))


def test_match_initialized_dims_fail(da1, da2):
    """Test if check works if the da does not have the proper dims."""
    with pytest.raises(DimensionError) as e:
        match_initialized_dims(da1.rename({"y": "init"}), da2.rename({"y": "not_time"}))
    assert "Verification contains more dimensions than initialized" in str(e.value)


def test_match_initialized_vars(ds1, ds2):
    """Test if check works if both have the same variables."""
    assert match_initialized_vars(ds1, ds2)


def test_match_initialized_vars_fail(ds1, ds2):
    """Test if check works if both do not have the same variables."""
    with pytest.raises(VariableError) as e:
        match_initialized_vars(ds1, ds2.rename({"air": "tmp"}))
    assert "Please provide a Dataset/DataArray with at least" in str(e.value)


def test_is_in_list():
    """Test if check works if key is in dict."""
    some_list = ["key"]
    assert is_in_list("key", some_list, "metric")


def test_is_in_list_fail():
    """Test if check works if key is not in dict."""
    some_list = ["key"]
    with pytest.raises(KeyError) as e:
        is_in_list("not_key", some_list, "metric")
    assert "Specify metric from" in str(e.value)


@pytest.mark.parametrize("lead_units", VALID_LEAD_UNITS)
def test_valid_lead_units(da_lead, lead_units):
    """Test that lead units check passes with appropriate lead units."""
    da_lead["lead"].attrs["units"] = lead_units
    assert has_valid_lead_units(da_lead)


def test_valid_lead_units_no_units(da_lead):
    """Test that valid lead units check breaks if there are no units."""
    with pytest.raises(AttributeError):
        has_valid_lead_units(da_lead)


def test_valid_lead_units_invalid_units(da_lead):
    """Test that valid lead units check breaks if invalid units provided."""
    da_lead["lead"].attrs["units"] = "dummy"
    with pytest.raises(AttributeError):
        has_valid_lead_units(da_lead)


@pytest.mark.parametrize("lead_units", VALID_LEAD_UNITS)
def test_nonplural_lead_units_works(da_lead, lead_units):
    """Test that non-plural lead units work on lead units check."""
    da_lead["lead"].attrs["units"] = lead_units[:-1]
    with pytest.warns(UserWarning) as record:
        has_valid_lead_units(da_lead)
    expected = f'The letter "s" was appended to the lead units; now {lead_units}.'
    assert record[0].message.args[0] == expected
