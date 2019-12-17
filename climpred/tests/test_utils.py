import numpy as np
import pytest

from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import __m2c
from climpred.constants import DETERMINISTIC_PM_METRICS, PM_COMPARISONS
from climpred.metrics import __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from climpred.utils import (
    copy_coords_from_to,
    get_comparison_class,
    get_metric_class,
    intersect,
)


@pytest.fixture
def control_ds_3d():
    """North Atlantic xr.Dataset."""
    ds = load_dataset('MPI-control-3D').sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.fixture
def control_da_3d():
    """North Atlantic xr.DataArray."""
    da = load_dataset('MPI-control-3D').sel(x=slice(120, 130), y=slice(50, 60))['tos']
    return da


def test_get_metric_class():
    """Test if passing in a string gets the right metric function."""
    actual = get_metric_class('pearson_r', DETERMINISTIC_PM_METRICS).name
    expected = __pearson_r.name
    assert actual == expected


def test_get_metric_class_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_metric_class('not_metric', DETERMINISTIC_PM_METRICS)
    assert 'Specify metric from' in str(excinfo.value)


def test_get_comparison_class():
    """Test if passing in a string gets the right comparison function."""
    actual = get_comparison_class('m2c', PM_COMPARISONS).name
    expected = __m2c.name
    assert actual == expected


def test_get_comparison_class_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_comparison_class('not_comparison', PM_COMPARISONS)
    assert 'Specify comparison from' in str(excinfo.value)


def test_intersect():
    """Test if the intersect (overlap) of two lists work."""
    x = [1, 5, 6]
    y = [1, 6, 7]
    actual = intersect(x, y)
    expected = np.array([1, 6])
    assert all(a == e for a, e in zip(actual, expected))


def test_da_assign_attrs():
    """Test assigning attrs for compute_perfect_model and dataarrays."""
    v = 'tos'
    metric = 'pearson_r'
    comparison = 'm2e'
    da = load_dataset('MPI-PM-DP-1D')[v].isel(area=1, period=-1)
    control = load_dataset('MPI-control-1D')[v].isel(area=1, period=-1)
    actual = compute_perfect_model(
        da, control, metric=metric, comparison=comparison
    ).attrs
    assert actual['metric'] == metric
    assert actual['comparison'] == comparison
    if metric == 'pearson_r':
        assert actual['units'] == 'None'
    assert actual['skill_calculated_by_function'] == 'compute_perfect_model'
    assert (
        actual['prediction_skill']
        == 'calculated by climpred https://climpred.readthedocs.io/'
    )


def test_ds_assign_attrs():
    """Test assigning attrs for datasets."""
    metric = 'mse'
    comparison = 'm2e'
    v = 'tos'
    da = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)[v]
    control = load_dataset('MPI-control-1D').isel(area=1, period=-1)[v]
    da.attrs['units'] = 'C'
    actual = compute_perfect_model(
        da, control, metric=metric, comparison=comparison
    ).attrs
    assert actual['metric'] == metric
    assert actual['comparison'] == comparison
    if metric == 'pearson_r':
        assert actual['units'] == 'None'
    assert actual['skill_calculated_by_function'] == 'compute_perfect_model'
    assert actual['units'] == '(C)^2'


def test_bootstrap_pm_assign_attrs():
    """Test assigning attrs for bootstrap_perfect_model."""
    v = 'tos'
    metric = 'pearson_r'
    comparison = 'm2e'
    bootstrap = 3
    sig = 95
    da = load_dataset('MPI-PM-DP-1D')[v].isel(area=1, period=-1)
    control = load_dataset('MPI-control-1D')[v].isel(area=1, period=-1)
    actual = bootstrap_perfect_model(
        da, control, metric=metric, comparison=comparison, bootstrap=bootstrap, sig=sig
    ).attrs
    assert actual['metric'] == metric
    assert actual['comparison'] == comparison
    assert actual['bootstrap_iterations'] == bootstrap
    assert str(round((1 - sig / 100) / 2, 3)) in actual['confidence_interval_levels']
    if metric == 'pearson_r':
        assert actual['units'] == 'None'
    assert 'bootstrap' in actual['skill_calculated_by_function']


def test_hindcast_assign_attrs():
    """Test assigning attrs for compute_hindcast."""
    metric = 'pearson_r'
    comparison = 'e2r'
    da = load_dataset('CESM-DP-SST')
    control = load_dataset('ERSST')
    actual = compute_hindcast(da, control, metric=metric, comparison=comparison).attrs
    assert actual['metric'] == metric
    assert actual['comparison'] == comparison
    if metric == 'pearson_r':
        assert actual['units'] == 'None'
    assert actual['skill_calculated_by_function'] == 'compute_hindcast'


def test_copy_coords_from_to_ds(control_ds_3d):
    """Test whether coords are copied from one xr object to another."""
    #
    xro = control_ds_3d
    c_1time = xro.isel(time=4).drop_vars('time')
    assert 'time' not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_da(control_da_3d):
    """Test whether coords are copied from one xr object to another."""
    #
    xro = control_da_3d
    c_1time = xro.isel(time=4).drop_vars('time')
    assert 'time' not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_ds_chunk(control_ds_3d):
    """Test whether coords are copied from one xr object to another."""
    #
    xro = control_ds_3d.chunk({'time': 5})
    c_1time = xro.isel(time=4).drop_vars('time')
    assert 'time' not in c_1time.coords
    c_1time = copy_coords_from_to(xro.isel(time=2), c_1time)
    assert (c_1time.time == xro.isel(time=2).time).all()


def test_copy_coords_from_to_da_different_xro(control_ds_3d):
    xro = control_ds_3d.chunk({'time': 5})
    c_1time = xro.isel(time=4).drop_vars('time')
    with pytest.raises(ValueError) as excinfo:
        copy_coords_from_to(xro.isel(time=2).tos, c_1time)
    assert 'xro_from and xro_to must be both either' in str(excinfo.value)
