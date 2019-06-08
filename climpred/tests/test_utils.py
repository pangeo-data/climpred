import pytest

import numpy as np

from climpred.utils import get_metric_function, get_comparison_function, intersect
from climpred.constants import ALL_PM_METRICS_DICT, ALL_PM_COMPARISONS_DICT
from climpred.metrics import _pearson_r
from climpred.comparisons import _m2c


def test_get_metric_function():
    actual = get_metric_function('pearson_r', ALL_PM_METRICS_DICT)
    expected = _pearson_r
    assert actual == expected


def test_get_metric_function_fail():
    with pytest.raises(KeyError) as excinfo:
        get_metric_function('not_metric', ALL_PM_METRICS_DICT)
    assert 'Specify metric from' in str(excinfo.value)


def test_get_comparison_function():
    actual = get_comparison_function('m2c', ALL_PM_COMPARISONS_DICT)
    expected = _m2c
    assert actual == expected


def test_get_comparison_function_fail():
    with pytest.raises(KeyError) as excinfo:
        get_comparison_function('not_comparison', ALL_PM_COMPARISONS_DICT)
    assert 'Specify comparison from' in str(excinfo.value)


def test_intersect():
    x = [1, 5, 6]
    y = [1, 6, 7]
    actual = intersect(x, y)
    expected = np.array([1, 6])
    assert all(a == e for a, e in zip(actual, expected))
