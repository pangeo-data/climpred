import numpy as np
import pytest
import xarray as xr

from climpred.checks import is_xarray


@pytest.fixture
def ds1():
    return xr.Dataset(
        {'air': (('lon', 'lat'), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])},
        coords={'lon': [1, 3, 4], 'lat': [5, 6, 7]},
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
