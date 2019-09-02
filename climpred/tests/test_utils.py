import numpy as np
import pytest
from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import _m2c
from climpred.constants import DETERMINISTIC_PM_METRICS, PM_COMPARISONS
from climpred.metrics import _pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from climpred.utils import get_comparison_function, get_metric_function, intersect


def test_get_metric_function():
    """Test if passing in a string gets the right metric function."""
    actual = get_metric_function('pearson_r', DETERMINISTIC_PM_METRICS)
    expected = _pearson_r
    assert actual == expected


def test_get_metric_function_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_metric_function('not_metric', DETERMINISTIC_PM_METRICS)
    assert 'Specify metric from' in str(excinfo.value)


def test_get_comparison_function():
    """Test if passing in a string gets the right comparison function."""
    actual = get_comparison_function('m2c', PM_COMPARISONS)
    expected = _m2c
    assert actual == expected


def test_get_comparison_function_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_comparison_function('not_comparison', PM_COMPARISONS)
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
    print(actual)
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
    metric = 'pearson_r'
    comparison = 'm2e'
    da = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    control = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    actual = compute_perfect_model(
        da, control, metric=metric, comparison=comparison
    ).attrs
    assert actual['metric'] == metric
    assert actual['comparison'] == comparison
    if metric == 'pearson_r':
        assert actual['units'] == 'None'
    assert actual['skill_calculated_by_function'] == 'compute_perfect_model'


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
