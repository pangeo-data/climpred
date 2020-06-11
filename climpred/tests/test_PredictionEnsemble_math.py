import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.classes import PredictionEnsemble
from climpred.exceptions import VariableError

ALLOWED_TYPES_FOR_MATH_OPERATORS = [
    int,
    float,
    np.ndarray,
    xr.DataArray,
    xr.Dataset,
    PredictionEnsemble,
]

OTHER_ALLOWED_HINDCAST = [
    2,
    2.0,
    np.array(2),
    xr.DataArray(2),
    xr.DataArray(2).chunk(),
    xr.Dataset({'SST': 2}),
    xr.Dataset({'SST': 2}).chunk(),
]
OTHER_ALLOWED_PM = [
    2,
    2.0,
    np.array(2),
    xr.DataArray(2),
    xr.DataArray(2).chunk(),
    xr.Dataset({'tos': 2}),
    xr.Dataset({'tos': 2}).chunk(),
]
OTHER_ALLOWED_IDS = [
    'int',
    'float',
    'np.array',
    'xr.DataArray',
    'chunked xr.DataArray',
    'xr.Dataset',
    'chunked xr.Dataset',
]

OTHER_NOT_ALLOWED = ['2', list(), dict(), set()]
OTHER_NOT_ALLOWED_IDS = ['str', 'list', 'dict', 'set']

MATH_OPERATORS = ['add', 'sub', 'mul', 'div']


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def assert_equal_PredictionEnsemble(he, he2):
    # check for same non-empty datasets
    def non_empty_datasets(he):
        return [k for k in he._datasets.keys() if he._datasets[k]]

    assert non_empty_datasets(he) == non_empty_datasets(he2)
    # check all datasets
    for dataset in he._datasets:
        if he._datasets[dataset]:
            if dataset == 'observations':
                for obs_dataset in he._datasets['observations']:
                    print('check observations', obs_dataset)
                    assert_equal(
                        he2._datasets['observations'][obs_dataset],
                        he2._datasets['observations'][obs_dataset],
                    )
            else:
                print('check', dataset)
                assert_equal(he2._datasets[dataset], he._datasets[dataset])


def check_dataset_dims_and_data_vars(before, after, dataset):
    if dataset not in ['initialized', 'uninitialized', 'control']:
        before = before._datasets['observations'][dataset]
        after = after._datasets['observations'][dataset]
    else:
        before = before._datasets[dataset]
        after = after._datasets[dataset]
    assert before.dims == after.dims
    assert list(before.data_vars) == list(after.data_vars)


def gen_error_str(he, operator, other):
    return f'Cannot use {type(he)} {operator.__name__} {type(other)}'


# HindcastEnsemble
@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize('other', OTHER_ALLOWED_HINDCAST, ids=OTHER_ALLOWED_IDS)
def test_hindcastEnsemble_plus_defined(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    observations_ds_1d,
    other,
    operator,
):
    """Test that HindcastEnsemble math operator (+-*/) other works correctly for
    allowed other types."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he = he.add_observations(observations_ds_1d, 'obs')
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            if dataset == 'observations':
                for obs_dataset in he._datasets['observations']:
                    print('check observations', obs_dataset)
                    assert_equal(
                        he2._datasets['observations'][obs_dataset],
                        operator(he._datasets['observations'][obs_dataset], other),
                    )
                    # check same dims and data_vars as before
                    check_dataset_dims_and_data_vars(he, he2, obs_dataset)
            else:
                print('check', dataset)
                assert_equal(
                    he2._datasets[dataset], operator(he._datasets[dataset], other)
                )
                # check same dims and data_vars as before
                check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
def test_hindcastEnsemble_plus_hindcastEnsemble(
    hind_ds_initialized_1d, hist_ds_uninitialized_1d, observations_ds_1d, operator
):
    """Test that HindcastEnsemble math operator (+-*/) HindcastEnsemble works
    correctly."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he = he.add_observations(observations_ds_1d, 'obs')
    other = he.mean('init')
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            if dataset == 'observations':
                for obs_dataset in he._datasets['observations']:
                    print('check observations', obs_dataset)
                    assert_equal(
                        he2._datasets['observations'][obs_dataset],
                        operator(
                            he._datasets['observations'][obs_dataset],
                            other._datasets['observations'][obs_dataset],
                        ),
                    )
                    # check same dims and data_vars as before
                    check_dataset_dims_and_data_vars(he, he2, obs_dataset)
            else:
                print('check', dataset)
                assert_equal(
                    he2._datasets[dataset],
                    operator(he._datasets[dataset], other._datasets[dataset]),
                )
                # check same dims and data_vars as before
                check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize('other', OTHER_NOT_ALLOWED, ids=OTHER_NOT_ALLOWED_IDS)
def test_hindcastEnsemble_plus_not_defined(hind_ds_initialized_1d, other, operator):
    """Test that HindcastEnsemble math operator (+-*/) other raises error for
    non-defined others."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    operator = eval(operator)
    error_str = gen_error_str(he, operator, other)
    with pytest.raises(TypeError) as excinfo:
        operator(he, other)
    assert f'{error_str} because type {type(other)} not supported' in str(excinfo.value)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize('other', OTHER_ALLOWED_PM[-2:], ids=OTHER_ALLOWED_IDS[-2:])
def test_hindcastEnsemble_plus_Dataset_different_name(
    hind_ds_initialized_1d, other, operator
):
    """Test that HindcastEnsemble math operator (+-*/) other raises error for
    non-defined others."""

    he = HindcastEnsemble(hind_ds_initialized_1d)
    operator = eval(operator)
    error_str = gen_error_str(he, operator, other)
    with pytest.raises(VariableError) as excinfo:
        operator(he, other)
    assert f'{error_str} with new `data_vars`' in str(excinfo.value)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
def test_hindcastEnsemble_plus_broadcast(hind_ds_initialized_3d, operator):
    """Test that HindcastEnsemble math operator (+-*/) other also broadcasts
    correctly."""
    he = HindcastEnsemble(hind_ds_initialized_3d)
    operator = eval(operator)
    # minimal adding an offset or like multiplying area
    he2 = operator(
        he, xr.ones_like(hind_ds_initialized_3d.isel(init=1, lead=1, drop=True))
    )
    he3 = operator(he, 1)
    assert_equal_PredictionEnsemble(he2, he3)


# basically all copied
# PerfectModelEnsemble
@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize('other', OTHER_ALLOWED_PM, ids=OTHER_ALLOWED_IDS)
def test_PerfectModelEnsemble_plus_defined(
    PM_ds_initialized_1d, PM_ds_control_1d, other, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) other works correctly for
    allowed other types."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    he = he.add_control(PM_ds_control_1d)
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print('check', dataset)
            assert_equal(he2._datasets[dataset], operator(he._datasets[dataset], other))
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
def test_PerfectModelEnsemble_plus_PerfectModelEnsemble(
    PM_ds_initialized_1d, PM_ds_control_1d, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) PerfectModelEnsemble works
    correctly."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    he = he.add_control(PM_ds_control_1d)
    other = he.mean('init')
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print('check', dataset)
            assert_equal(
                he2._datasets[dataset],
                operator(he._datasets[dataset], other._datasets[dataset]),
            )
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize('other', OTHER_NOT_ALLOWED, ids=OTHER_NOT_ALLOWED_IDS)
def test_PerfectModelEnsemble_plus_not_defined(PM_ds_initialized_1d, other, operator):
    """Test that PerfectModelEnsemble math operator (+-*/) other raises error for
    non-defined others."""

    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    operator = eval(operator)
    error_str = gen_error_str(he, operator, other)
    with pytest.raises(TypeError) as excinfo:
        operator(he, other)
    assert f'{error_str} because type {type(other)} not supported' in str(excinfo.value)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize(
    'other', OTHER_ALLOWED_HINDCAST[-2:], ids=OTHER_ALLOWED_IDS[-2:]
)
def test_PerfectModelEnsemble_plus_Dataset_different_name(
    PM_ds_initialized_1d, other, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) other raises error for
    Dataset with other dims and/or variables."""

    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    operator = eval(operator)
    error_str = gen_error_str(he, operator, other)
    with pytest.raises(VariableError) as excinfo:
        operator(he, other)
    assert f'{error_str} with new `data_vars`' in str(excinfo.value)


@pytest.mark.parametrize('operator', MATH_OPERATORS, ids=MATH_OPERATORS)
def test_PerfectModelEnsemble_plus_broadcast(PM_ds_initialized_3d, operator):
    """Test that PerfectModelEnsemble math operator (+-*/) other also broadcasts
    correctly."""
    he = PerfectModelEnsemble(PM_ds_initialized_3d)
    operator = eval(operator)
    # minimal adding an offset or like multiplying area
    he2 = operator(
        he, xr.ones_like(PM_ds_initialized_3d.isel(init=1, lead=1, drop=True))
    )
    he3 = operator(he, 1)
    assert_equal_PredictionEnsemble(he2, he3)
