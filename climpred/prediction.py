import inspect

import xarray as xr

from .checks import is_xarray
from .comparisons import _e2c
from .constants import (
    HINDCAST_COMPARISONS,
    HINDCAST_METRICS,
    METRICS_PROBABISTIC,
    PM_COMPARISONS,
    PM_METRICS,
)
from .utils import (
    assign_attrs,
    get_comparison_function,
    get_metric_function,
    intersect,
    reduce_time_series,
)


# --------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
# --------------------------------------------#
@is_xarray([0, 1])
def compute_perfect_model(
    ds, control, metric='pearson_r', comparison='m2e', add_attrs=True, **kwargs
):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Args:
        ds (xarray object): ensemble with dims ``lead``, ``init``, ``member``.
        control (xarray object): control with dimension ``time``.
        metric (str): `metric` name, see :py:func:`climpred.utils.get_metric_function`.
        comparison (str): `comparison` name, see
                          :py:func:`climpred.utils.get_comparison_function`.
        add_attrs (bool): write climpred compute args to attrs. default: True

    Returns:
        skill (xarray object): skill score.

    """
    supervector_dim = 'svd'
    if metric in METRICS_PROBABISTIC:
        if comparison is ['e2c', 'm2e']:
            raise ValueError(
                'Probabilistic metrics cannot work with comparison', comparison
            )
        stack = False
    else:
        stack = True
    metric = get_metric_function(metric, PM_METRICS)
    comparison = get_comparison_function(comparison, PM_COMPARISONS)

    forecast, reference = comparison(ds, supervector_dim, stack=stack)

    skill = metric(
        forecast, reference, dim=supervector_dim, comparison=comparison, **kwargs
    )
    # Attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            ds,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_hindcast(
    hind,
    reference,
    metric='pearson_r',
    comparison='e2r',
    max_dof=False,
    add_attrs=True,
    **kwargs
):
    """Compute a predictability skill score against a reference

    Args:
        hind (xarray object):
            Expected to follow package conventions:

            * ``init`` : dim of initialization dates
            * ``lead`` : dim of lead time from those initializations

            Additional dims can be member, lat, lon, depth, ...
        reference (xarray object):
            reference output/data over same time period.
        metric (str):
            Metric used in comparing the decadal prediction ensemble with the
            reference (see :py:func:`climpred.utils.get_metric_function`).
        comparison (str):
            How to compare the decadal prediction ensemble to the reference:

                * e2r : ensemble mean to reference (Default)
                * m2r : each member to the reference
        nlags (int): How many lags to compute skill/potential predictability out
                     to. Default: length of `lead` dim
        max_dof (bool):
            If True, maximize the degrees of freedom by slicing `hind` and `reference`
            to a common time frame at each lead.

            If False (default), then slice to a common time frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
        add_attrs (bool): write climpred compute args to attrs. default: True


    Returns:
        skill (xarray object):
            Predictability with main dimension ``lag``

    """
    nlags = max(hind.lead.values)
    comparison = get_comparison_function(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_function(metric, HINDCAST_METRICS)

    forecast, reference = comparison(hind, reference)
    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})
    # take only inits for which we have references at all leahind
    if not max_dof:
        forecast, reference = reduce_time_series(forecast, reference, nlags)

    plag = []
    # iterate over all leads (accounts for lead.min() in [0,1])
    for i in forecast.lead.values:
        if max_dof:
            forecast, reference = reduce_time_series(forecast, reference, i)
        # take lead year i timeseries and convert to real time
        a = forecast.sel(lead=i).drop('lead')
        a['time'] = [t + i for t in a.time.values]
        # take real time reference of real time forecast years
        b = reference.sel(time=a.time.values)
        plag.append(metric(a, b, dim='time', comparison=comparison, **kwargs))
    skill = xr.concat(plag, 'lead')
    skill['lead'] = forecast.lead.values
    # Attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            hind,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_persistence(hind, reference, metric='pearson_r', max_dof=False, **kwargs):
    """Computes the skill of a persistence forecast from a simulation.

    Args:
        hind (xarray object): The initialized ensemble.
        reference (xarray object): The reference time series.
        metric (str): Metric name to apply at each lag for the persistence
                      computation. Default: 'pearson_r'
        max_dof (bool):
            If True, maximize the degrees of freedom by slicing `hind` and `reference`
            to a common time frame at each lead.

            If False (default), then slice to a common time frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.

    Returns:
        pers (xarray object): Results of persistence forecast with the input metric
        applied.

    Reference:
        * Chapter 8 (Short-Term Climate Prediction) in Van den Dool, Huug.
          Empirical methods in short-term climate prediction.
          Oxford University Press, 2007.
    """
    if metric in METRICS_PROBABISTIC:
        raise ValueError(
            'probabilistic metric ', metric, 'cannot compute persistence forecast.'
        )
    metric = get_metric_function(metric, HINDCAST_METRICS)
    # If lead 0, need to make modifications to get proper persistence, since persistence
    # at lead 0 is == 1.
    if [0] in hind.lead.values:
        hind = hind.copy()
        hind['lead'] += 1
        hind['init'] -= 1
    nlags = max(hind.lead.values)
    # temporarily change `init` to `time` for comparison to reference time.
    hind = hind.rename({'init': 'time'})
    if not max_dof:
        # slices down to inits in common with hindcast, plus gives enough room
        # for maximum lead time forecast.
        a, _ = reduce_time_series(hind, reference, nlags)
        inits = a['time']

    plag = []
    for lag in hind.lead.values:
        if max_dof:
            # slices down to inits in common with hindcast, but only gives enough
            # room for lead from current forecast
            a, _ = reduce_time_series(hind, reference, lag)
            inits = a['time']
        ref = reference.sel(time=inits + lag)
        fct = reference.sel(time=inits)
        ref['time'] = fct['time']
        plag.append(metric(ref, fct, dim='time', comparison=_e2c, **kwargs))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    return pers


@is_xarray([0, 1])
def compute_uninitialized(
    uninit, reference, metric='pearson_r', comparison='e2r', add_attrs=True, **kwargs
):
    """Compute a predictability score between an uninitialized ensemble and a reference.

    .. note::
        Based on Decadal Prediction protocol, this should only be computed for the
        first lag and then projected out to any further lags being analyzed.

    Args:
        uninit (xarray object):
            uninitialized ensemble.
        reference (xarray object):
            reference output/data over same time period.
        metric (str):
            Metric used in comparing the uninitialized ensemble with the reference.
        comparison (str):
            How to compare the uninitialized ensemble to the reference:
                * e2r : ensemble mean to reference (Default)
                * m2r : each member to the reference
        add_attrs (bool): write climpred compute args to attrs. default: True


    Returns:
        u (xarray object): Results from comparison at the first lag.

    """
    comparison = get_comparison_function(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_function(metric, HINDCAST_METRICS)
    forecast, reference = comparison(uninit, reference)
    # Find common times between two for proper comparison.
    common_time = intersect(forecast['time'].values, reference['time'].values)
    forecast = forecast.sel(time=common_time)
    reference = reference.sel(time=common_time)
    uninit_skill = metric(
        forecast, reference, dim='time', comparison=comparison, **kwargs
    )
    # Attach climpred compute information to skill
    if add_attrs:
        uninit_skill = assign_attrs(
            uninit_skill,
            uninit,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=kwargs,
        )
    return uninit_skill
