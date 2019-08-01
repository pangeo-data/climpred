import datetime
import types

import numpy as np
import xarray as xr

from . import comparisons, metrics
from .checks import is_in_list
from .constants import (
    HINDCAST_COMPARISONS,
    HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_COMPARISONS,
    PM_METRICS,
)


def get_metric_function(metric, list_):
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
        # equivalent of: `if metric in METRIC_ALIASES;
        # METRIC_ALIASES[metric]; else metric`
        metric = METRIC_ALIASES.get(metric, metric)
        is_in_list(metric, list_, 'metric')
        return getattr(metrics, '_' + metric)


def get_comparison_function(comparison, list_):
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
        is_in_list(comparison, list_, 'comparison')
        return getattr(comparisons, '_' + comparison)


def intersect(lst1, lst2):
    """
    Custom intersection, since `set.intersection()` changes type of list.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def reduce_time_series(forecast, reference, nlags):
    """Reduces forecast and reference to common time frame for prediction and lag.

    Args:
        forecast (`xarray` object): prediction ensemble with inits.
        reference (`xarray` object): reference being compared to (for skill,
                                     persistence, etc.)
        nlags (int): number of lags being computed

    Returns:
       forecast (`xarray` object): prediction ensemble reduced to
       reference (`xarray` object):
    """
    imin = max(forecast.time.min(), reference.time.min())
    imax = min(forecast.time.max(), reference.time.max() - nlags)
    imax = xr.DataArray(imax).rename('time')
    forecast = forecast.where(forecast.time <= imax, drop=True)
    forecast = forecast.where(forecast.time >= imin, drop=True)
    reference = reference.where(reference.time >= imin, drop=True)
    return forecast, reference


def assign_climpred_compute_to_attrs(
    skill,
    ds,
    function_name='compute_perfect_model',
    metadata_dict=None,
    metric=None,
    comparison=None,
):
    """Write information about prediction skill into attrs."""
    # assign old attrs
    skill.attrs = ds.attrs

    # climpred info
    skill.attrs[
        'Prediction skill'
    ]: f'calculated by climpred https://climpred.readthedocs.io/'
    skill.attrs['skill calculated by function'] = function_name
    if 'init' in ds.coords:
        skill.attrs['number of initializations'] = ds.init.size
    if 'member' in ds.coords:
        skill.attrs['number of members'] = ds.member.size

    if 'perfect_model' in function_name:
        comparison = get_comparison_function(comparison, PM_COMPARISONS).__name__[1:]
        metric = get_metric_function(metric, PM_METRICS).__name__[1:]
    elif 'hindcast' in function_name:
        comparison = get_comparison_function(comparison, HINDCAST_COMPARISONS).__name__[
            1:
        ]
        metric = get_metric_function(metric, HINDCAST_METRICS).__name__[1:]

    skill.attrs['metric'] = metric
    skill.attrs['comparison'] = comparison

    if metric in ['pearson_r']:
        skill.attrs['units'] = ' '

    # write optional information
    if metadata_dict is None:
        metadata_dict = dict()
    for key, val in metadata_dict.items():
        skill.attrs[key] = val

    skill.attrs[
        'date'
    ] = f'created on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]}'
    return skill
