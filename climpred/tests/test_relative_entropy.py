"""Test relative_entropy.py"""

from climpred.graphics import plot_relative_entropy
from climpred.relative_entropy import (
    bootstrap_relative_entropy,
    compute_relative_entropy,
)

from . import requires_eofs


@requires_eofs
def test_compute_relative_entropy(PM_da_initialized_3d, PM_da_control_3d):
    """
    Checks that there are no NaNs.
    """
    actual = compute_relative_entropy(
        PM_da_initialized_3d, PM_da_control_3d, nmember_control=5, neofs=2
    )
    actual_any_nan = actual.isnull().any()
    for var in actual_any_nan.data_vars:
        assert not actual_any_nan[var]


@requires_eofs
def test_bootstrap_relative_entropy(PM_da_initialized_3d, PM_da_control_3d):
    """
    Checks that there are no NaNs.
    """
    actual = bootstrap_relative_entropy(
        PM_da_initialized_3d,
        PM_da_control_3d,
        nmember_control=5,
        neofs=2,
        bootstrap=2,
    )
    actual_any_nan = actual.isnull()
    for var in actual_any_nan.data_vars:
        assert not actual_any_nan[var]


@requires_eofs
def test_plot_relative_entropy(PM_da_initialized_3d, PM_da_control_3d):
    res = compute_relative_entropy(
        PM_da_initialized_3d, PM_da_control_3d, nmember_control=5, neofs=2
    )
    threshold = bootstrap_relative_entropy(
        PM_da_initialized_3d,
        PM_da_control_3d,
        nmember_control=5,
        neofs=2,
        bootstrap=2,
    )
    res_ax = plot_relative_entropy(res, threshold)
    assert res_ax is not None
