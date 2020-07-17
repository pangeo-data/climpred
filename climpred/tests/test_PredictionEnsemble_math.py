import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.classes import PredictionEnsemble
from climpred.constants import CLIMPRED_DIMS
from climpred.exceptions import VariableError
from climpred.testing import assert_PredictionEnsemble, check_dataset_dims_and_data_vars

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


def gen_error_str(he, operator, other):
    print(type(operator))
    OPERATOR_STR = {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': '/',
    }
    return f'Cannot use {type(he)} {OPERATOR_STR[operator]} {type(other)}'


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
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
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
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
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
    assert_PredictionEnsemble(he2, he3)


def test_HindcastEnsemble_area_weighted_mean(hind_ds_initialized_3d):
    """Test area weighted mean HindcastEnsemble."""
    he = HindcastEnsemble(hind_ds_initialized_3d)
    # fake area
    area = hind_ds_initialized_3d['TAREA']
    spatial_dims = [d for d in hind_ds_initialized_3d.dims if d not in CLIMPRED_DIMS]
    # PredictionEnsemble doesnt like other data_vars
    he_self_spatial_mean = (he * area).sum(spatial_dims) / area.sum()
    # weighted requires Dataset
    area = area.to_dataset(name='area')
    he_xr_spatial_mean = he.weighted(area).mean(spatial_dims)
    assert_PredictionEnsemble(
        he_self_spatial_mean, he_xr_spatial_mean, how='allclose', rtol=0.03, atol=0.05
    )


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
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
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
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
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
    assert_PredictionEnsemble(he2, he3)


def test_PerfectModelEnsemble_area_weighted_mean(PM_ds_initialized_3d):
    """Test area weighted mean PerfectModelEnsemble."""
    he = PerfectModelEnsemble(PM_ds_initialized_3d)
    # fake area
    area = np.cos(PM_ds_initialized_3d.lat) + 1
    spatial_dims = [d for d in PM_ds_initialized_3d.dims if d not in CLIMPRED_DIMS]
    # PredictionEnsemble doesnt like other data_vars
    he_self_spatial_mean = (he * area).sum(spatial_dims) / area.sum()
    # weighted requires Dataset
    area = area.to_dataset(name='area')
    he_xr_spatial_mean = he.weighted(area).mean(spatial_dims)
    assert_PredictionEnsemble(
        he_self_spatial_mean, he_xr_spatial_mean, how='allclose', rtol=0.03, atol=0.05
    )
