import pytest

import numpy as np

from climpred.utils import get_metric_function, get_comparison_function, intersect
from climpred.constants import PM_METRICS, PM_COMPARISONS
from climpred.metrics import _pearson_r
from climpred.comparisons import _m2c


def test_get_metric_function():
    """Test if passing in a string gets the right metric function."""
    actual = get_metric_function('pearson_r', PM_METRICS)
    expected = _pearson_r
    assert actual == expected


def test_get_metric_function_fail():
    """Test if passing something not in the dict raises the right error."""
    with pytest.raises(KeyError) as excinfo:
        get_metric_function('not_metric', PM_METRICS)
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
