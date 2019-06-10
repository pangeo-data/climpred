import types

import numpy as np

from .checks import is_in_dict


def get_metric_function(metric, dict_):
    """
    This allows the user to submit a string representing the desired function
    to anything that takes a metric.

    Currently compatable with functions:
    * compute_persistence()
    * compute_perfect_model()
    * compute_hindcast()

    Args:
        metric (str): name of metric.

    Returns:
        metric (function): function object of the metric.

    Raises:
        KeyError: if metric not implemented.
    """
    # catches issues with wrappers, etc. that actually submit the
    # proper underscore function
    if isinstance(metric, types.FunctionType):
        return metric
    else:
        is_in_dict(metric, dict_, 'metric')
        return dict_[metric]


def get_comparison_function(comparison, dict_):
    """
    Converts a string comparison entry from the user into an actual
     function for the package to interpret.

    PERFECT MODEL:
    m2m: Compare all members to all other members.
    m2c: Compare all members to the control.
    m2e: Compare all members to the ensemble mean.
    e2c: Compare the ensemble mean to the control.

    HINDCAST:
    e2r: Compare the ensemble mean to the reference.
    m2r: Compare each ensemble member to the reference.

    Args:
        comparison (str): name of comparison.

    Returns:
        comparison (function): comparison function.

    """
    if isinstance(comparison, types.FunctionType):
        return comparison
    else:
        is_in_dict(comparison, dict_, 'comparison')
        return dict_[comparison]


def intersect(lst1, lst2):
    """
    Custom intersection, since `set.intersection()` changes type of list.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)
