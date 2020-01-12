import numpy as np
import pytest
import xarray as xr

from climpred.checks import (
    has_dataset,
    has_dims,
    has_min_len,
    has_valid_lead_units,
    is_in_list,
    is_xarray,
    match_initialized_dims,
    match_initialized_vars,
)
from climpred.constants import VALID_LEAD_UNITS
from climpred.exceptions import DatasetError, DimensionError, VariableError


@pytest.fixture
def ds1():
    return xr.Dataset(
        {'air': (('lon', 'lat'), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={'lon': [1, 3, 4], 'lat': [5, 6, 7]},
    )


@pytest.fixture
def ds2():
    return xr.Dataset(
        {'air': (('lon', 'lat'), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={'lon': [1, 3, 6], 'lat': [5, 6, 9]},
    )


@pytest.fixture
def da1():
    return xr.DataArray([[0, 1], [3, 4], [6, 7]], dims=('x', 'y'))


@pytest.fixture
def da2():
    return xr.DataArray([[0, 1], [5, 6], [6, 7]], dims=('x', 'y'))


@pytest.fixture
def da_lead_time():
    lead = np.arange(5)
    init = np.arange(5)
    return xr.DataArray(
        np.random.rand(len(lead), len(init)), dims=['init', 'lead'], coords=[init, lead]
    )


@is_xarray(0)
def _arbitrary_ds_da_func(ds_da, *args, **kwargs):
    """Function for testing if checking the first item in arg list is ds/da."""
    return ds_da, args, kwargs


@is_xarray([0, 1])
def _arbitrary_two_xr_func(ds_da1, ds_da2, *args, **kwargs):
    """Function for testing if checking the first two items in arg list is ds/da."""
    return ds_da1, ds_da2, args, kwargs


@is_xarray([0, 2])
def _arbitrary_two_xr_func_random_loc(ds_da1, some_arg, ds_da2, **kwargs):
    """Function for testing if checking the 1st and 3rd item in arg list is ds/da."""
    return ds_da1, some_arg, ds_da2, kwargs


@is_xarray([0, 'da', 'other_da'])
def _arbitrary_three_xr_func_args_keys(ds, da=None, other_da=None, **kwargs):
    """Function for testing if checking the first in arg list and the
    keywords da/other_da is ds/da."""
    return ds, da, other_da, kwargs


def test_is_xarray_ds(ds1):
    """Test if checking the first item in arg list is ds."""
    ds, args, kwargs = _arbitrary_ds_da_func(ds1, 'arg1', 'arg2', kwarg1='kwarg1')
    assert (ds1 == ds).all()
    assert args == ('arg1', 'arg2')
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_not_ds():
    """Test if checking the first item in arg list is not a ds/da, raise an error."""
    not_a_ds = 'not_a_ds'
    with pytest.raises(IOError) as e:
        _arbitrary_ds_da_func(not_a_ds, 'arg1', 'arg2', kwarg1='kwarg1')
    assert 'The input data is not an xarray' in str(e.value)


def test_is_xarray_da(da1):
    """Test if checking the first item in arg list is da."""
    da, args, kwargs = _arbitrary_ds_da_func(da1, 'arg1', 'arg2', kwarg1='kwarg1')
    assert (da1 == da).all()
    assert args == ('arg1', 'arg2')
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_ds_da(ds1, da1):
    """Test if checking the first two items in arg list is ds/da."""
    ds, da, args, kwargs = _arbitrary_two_xr_func(
        ds1, da1, 'arg1', kwarg1='kwarg1', kwarg2='kwarg2'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert args == ('arg1',)
    assert kwargs == {'kwarg1': 'kwarg1', 'kwarg2': 'kwarg2'}


def test_is_xarray_ds_da_random_loc(ds1, da1):
    """Test if checking the first and third items in arg list is ds/da."""
    ds, arg, da, kwargs = _arbitrary_two_xr_func_random_loc(
        ds1, 'arg1', da1, kwarg1='kwarg1', kwarg2='kwarg2'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert arg == 'arg1'
    assert kwargs == {'kwarg1': 'kwarg1', 'kwarg2': 'kwarg2'}


def test_is_xarray_ds_da_args_keys(ds1, da1, da2):
    """Test if checking the args and kwargs are ds/da."""
    ds, da, other_da, kwargs = _arbitrary_three_xr_func_args_keys(
        ds1, da=da1, other_da=da2, kwarg1='kwarg1'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert (da2 == other_da).all()
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_ds_da_args_keys_not(ds1, da2):
    """Test if checking the args and kwargs are not ds/da, it raises an error."""
    not_a_da = np.array([0, 1, 2])
    with pytest.raises(IOError) as e:
        _arbitrary_three_xr_func_args_keys(
            ds1, da=not_a_da, other_da=da2, kwarg1='kwarg1'
        )
    assert 'The input data is not an xarray' in str(e.value)


class _ArbitraryClass:
    @is_xarray(1)
    def __init__(self, xobj):
        pass


def test_is_xarray_class_not():
    """Function for testing if checking class init is ds/da, it raises an error."""
    with pytest.raises(IOError) as e:
        _ArbitraryClass('totally not a ds')
    assert 'The input data is not an xarray' in str(e.value)


def test_has_dims_str(da1):
    """Test if check works for a string."""
    assert has_dims(da1, 'x', 'arbitrary')


def test_has_dims_list(da1):
    """Test if check works for a list."""
    # returns None if no errors
    assert has_dims(da1, ['x', 'y'], 'arbitrary')


def test_has_dims_str_fail(da1):
    """Test if check fails properly for a string."""
    with pytest.raises(DimensionError) as e:
        has_dims(da1, 'z', 'arbitrary')
    assert 'Your arbitrary object must contain' in str(e.value)


def test_has_dims_list_fail(da1):
    """Test if check fails properly for a list."""
    with pytest.raises(DimensionError) as e:
        has_dims(da1, ['z'], 'arbitrary')
    assert 'Your arbitrary object must contain' in str(e.value)


def test_has_min_len_arr(da1):
    """Test if check works for min len."""
    assert has_min_len(da1.values, 2, 'arbitrary')


def test_has_min_len_fail(da1):
    """Test if check fails properly."""
    with pytest.raises(DimensionError) as e:
        has_min_len(da1.values, 5, 'arbitrary')
    assert 'Your arbitrary array must be at least' in str(e.value)


def test_has_dataset():
    """Test if check works for a non-empty list."""
    obj = [5]
    assert has_dataset(obj, 'list', 'something')


def test_has_dataset_fail():
    """Test if check works to fail for an empty list."""
    obj = []
    with pytest.raises(DatasetError) as e:
        has_dataset(obj, 'test', 'something')
    assert 'You need to add at least one test dataset' in str(e.value)


def test_match_initialized_dims(da1, da2):
    """Test if check works if both da has the proper dims."""
    assert match_initialized_dims(da1.rename({'y': 'init'}), da2.rename({'y': 'time'}))


def test_match_initialized_dims_fail(da1, da2):
    """Test if check works if the da does not have the proper dims."""
    with pytest.raises(DimensionError) as e:
        match_initialized_dims(da1.rename({'y': 'init'}), da2.rename({'y': 'not_time'}))
    assert 'Dimensions must match initialized prediction' in str(e.value)


def test_match_initialized_vars(ds1, ds2):
    """Test if check works if both have the same variables."""
    assert match_initialized_vars(ds1, ds2)


def test_match_initialized_vars_fail(ds1, ds2):
    """Test if check works if both do not have the same variables."""
    with pytest.raises(VariableError) as e:
        match_initialized_vars(ds1, ds2.rename({'air': 'tmp'}))
    assert 'Please provide a Dataset/DataArray with at least' in str(e.value)


def test_is_in_list():
    """Test if check works if key is in dict."""
    some_list = ['key']
    assert is_in_list('key', some_list, 'metric')


def test_is_in_list_fail():
    """Test if check works if key is not in dict."""
    some_list = ['key']
    with pytest.raises(KeyError) as e:
        is_in_list('not_key', some_list, 'metric')
    assert 'Specify metric from' in str(e.value)


@pytest.mark.parametrize('lead_units', VALID_LEAD_UNITS)
def test_valid_lead_units(da_lead_time, lead_units):
    """Test that lead units check passes with appropriate lead units."""
    da_lead_time['lead'].attrs['units'] = lead_units
    assert has_valid_lead_units(da_lead_time)


def test_valid_lead_units_no_units(da_lead_time):
    """Test that valid lead units check breaks if there are no units."""
    with pytest.raises(AttributeError):
        has_valid_lead_units(da_lead_time)


def test_valid_lead_units_invalid_units(da_lead_time):
    """Test that valid lead units check breaks if invalid units provided."""
    da_lead_time['lead'].attrs['units'] = 'dummy'
    with pytest.raises(AttributeError):
        has_valid_lead_units(da_lead_time)


@pytest.mark.parametrize('lead_units', VALID_LEAD_UNITS)
def test_nonplural_lead_units_works(da_lead_time, lead_units):
    """Test that non-plural lead units work on lead units check."""
    da_lead_time['lead'].attrs['units'] = lead_units[:-1]
    with pytest.warns(UserWarning) as record:
        has_valid_lead_units(da_lead_time)
    expected = f'The letter "s" was appended to the lead units; now {lead_units}.'
    assert record[0].message.args[0] == expected
