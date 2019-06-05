import numpy as np
import pytest
import xarray as xr

from climpred.checks import (
    is_xarray, has_dims, has_min_len, is_initialized,
    match_initialized_dims, match_initialized_vars, is_in_dict
)
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


@is_xarray(0)
def _arbitrary_ds_da_func(ds_da, *args, **kwargs):
    return ds_da, args, kwargs


@is_xarray([0, 1])
def _arbitrary_two_xr_func(ds_da1, ds_da2, *args, **kwargs):
    return ds_da1, ds_da2, args, kwargs


@is_xarray([0, 2])
def _arbitrary_two_xr_func_random_loc(ds_da1, some_arg, ds_da2, **kwargs):
    return ds_da1, some_arg, ds_da2, kwargs


@is_xarray([0, 'da', 'other_da'])
def _arbitrary_three_xr_func_args_keys(ds, da=None, other_da=None, **kwargs):
    return ds, da, other_da, kwargs


def test_is_xarray_ds(ds1):
    ds, args, kwargs = _arbitrary_ds_da_func(ds1, 'arg1', 'arg2', kwarg1='kwarg1')
    assert (ds1 == ds).all()
    assert args == ('arg1', 'arg2')
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_not_ds():
    not_a_ds = 'not_a_ds'
    with pytest.raises(IOError) as e:
        _arbitrary_ds_da_func(not_a_ds, 'arg1', 'arg2', kwarg1='kwarg1')
    assert 'The input data is not an xarray' in str(e.value)


def test_is_xarray_da(da1):
    da, args, kwargs = _arbitrary_ds_da_func(da1, 'arg1', 'arg2', kwarg1='kwarg1')
    assert (da1 == da).all()
    assert args == ('arg1', 'arg2')
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_ds_da(ds1, da1):
    ds, da, args, kwargs = _arbitrary_two_xr_func(
        ds1, da1, 'arg1', kwarg1='kwarg1', kwarg2='kwarg2'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert args == ('arg1',)
    assert kwargs == {'kwarg1': 'kwarg1', 'kwarg2': 'kwarg2'}


def test_is_xarray_ds_da_random_loc(ds1, da1):
    ds, arg, da, kwargs = _arbitrary_two_xr_func_random_loc(
        ds1, 'arg1', da1, kwarg1='kwarg1', kwarg2='kwarg2'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert arg == 'arg1'
    assert kwargs == {'kwarg1': 'kwarg1', 'kwarg2': 'kwarg2'}


def test_is_xarray_ds_da_args_keys(ds1, da1, da2):
    ds, da, other_da, kwargs = _arbitrary_three_xr_func_args_keys(
        ds1, da=da1, other_da=da2, kwarg1='kwarg1'
    )
    assert (ds1 == ds).all()
    assert (da1 == da).all()
    assert (da2 == other_da).all()
    assert kwargs == {'kwarg1': 'kwarg1'}


def test_is_xarray_ds_da_args_keys_not(ds1, da2):
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
    with pytest.raises(IOError) as e:
        _ArbitraryClass('totally not a ds')
    assert 'The input data is not an xarray' in str(e.value)


def test_has_dims_str(da1):
    # returns None if no errors
    assert has_dims(da1, 'x', 'arbitrary')


def test_has_dims_list(da1):
    # returns None if no errors
    assert has_dims(da1, ['x', 'y'], 'arbitrary')


def test_has_dims_str_fail(da1):
    with pytest.raises(DimensionError) as e:
        has_dims(da1, 'z', 'arbitrary')
    assert 'Your arbitrary object must contain' in str(e.value)


def test_has_dims_list_fail(da1):
    with pytest.raises(DimensionError) as e:
        has_dims(da1, ['z'], 'arbitrary')
    assert 'Your arbitrary object must contain' in str(e.value)


def test_has_min_len_arr(da1):
    assert has_min_len(da1.values, 2, 'arbitrary')


def test_has_min_len_fail(da1):
    with pytest.raises(DimensionError) as e:
        has_min_len(da1.values, 5, 'arbitrary')
    assert 'Your arbitrary array must be at least' in str(e.value)


def test_is_initialized():
    obj = [5]
    assert is_initialized(obj, 'list', 'something')


def test_is_initialized_fail():
    obj = []
    with pytest.raises(DatasetError) as e:
        is_initialized(obj, 'test', 'something')
    assert 'You need to add at least one test dataset' in str(e.value)


def test_match_initialized_dims(da1, da2):
    assert match_initialized_dims(
        da1.rename({'y': 'init'}),
        da2.rename({'y': 'time'})
    )


def test_match_initialized_dims_fail(da1, da2):
    with pytest.raises(DimensionError) as e:
        match_initialized_dims(
            da1.rename({'y': 'init'}),
            da2.rename({'y': 'not_time'}),
        )
    assert 'Dimensions must match initialized prediction' in str(e.value)


def test_match_initialized_vars(ds1, ds2):
    assert match_initialized_vars(ds1, ds2)


def test_match_initialized_vars_fail(ds1, ds2):
    with pytest.raises(VariableError) as e:
        match_initialized_vars(ds1, ds2.rename({'air': 'tmp'}))
    assert 'Please provide a Dataset/DataArray with at least' in str(e.value)


def test_is_in_dict():
    some_dict = {'key': 'value'}
    assert is_in_dict('key', some_dict, 'metric')


def test_is_in_dict_fail():
    some_dict = {'key': 'value'}
    with pytest.raises(KeyError) as e:
        is_in_dict('not_key', some_dict, 'metric')
    assert 'Specify metric from' in str(e.value)
