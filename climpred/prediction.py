import inspect
import warnings

import xarray as xr

from .checks import is_xarray
from .comparisons import _e2c
from .constants import (
    DETERMINISTIC_HINDCAST_METRICS,
    HINDCAST_COMPARISONS,
    HINDCAST_METRICS,
    PM_METRICS,
    PM_COMPARISONS,
    PROBABILISTIC_METRICS,
    PROBABILISTIC_PM_COMPARISONS,
    METRIC_ALIASES,
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
    ds,
    control,
    metric='pearson_r',
    comparison='m2e',
    dim=None,
    add_attrs=True,
    **metric_kwargs,
):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Args:
        ds (xarray object): ensemble with dims ``lead``, ``init``, ``member``.
        control (xarray object): control with dimension ``time``.
        metric (str): `metric` name, see
         :py:func:`climpred.utils.get_metric_function` and (see :ref:`Metrics`).
        comparison (str): `comparison` name defines what to take as forecast
            and verification (see
            :py:func:`climpred.utils.get_comparison_function` and :ref:`Comparisons`).
        dim (str or list): dimension to apply metric over. default: ['member', 'init']
        add_attrs (bool): write climpred compute args to attrs. default: True
        ** metric_kwargs (dict): additional keywords to be passed to metric.
            (see the arguments required for a given metric in metrics.py)

    Returns:
        skill (xarray object): skill score with dimensions as input `ds`
                               without `dim`.

    """
    if dim is None:
        dim = ['init', 'member']
    # get metric function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    # if stack_dims, comparisons return forecast with member dim and reference
    # without member dim which is needed for probabilistic
    # if not stack_dims, comparisons return forecast and reference with member dim
    # which is neeeded for deterministic
    if metric in PROBABILISTIC_METRICS:
        if comparison not in PROBABILISTIC_PM_COMPARISONS:
            raise ValueError(
                f'Probabilistic metric {metric} cannot work with '
                f'comparison {comparison}.'
            )
        stack_dims = False
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    elif set(dim) == set(['init', 'member']):
        dim_to_apply_metric_to = 'svd'
        stack_dims = True
    else:
        if metric == 'pearson_r':
            # it doesnt fail though; we could also raise ValueError here
            warnings.warn(
                'ACC doesnt work on dim other than'
                '["init", "member"] in perfect-model framework.'
            )
        stack_dims = False

    if not stack_dims:
        dim_to_apply_metric_to = dim

    comparison = get_comparison_function(comparison, PM_COMPARISONS)

    forecast, reference = comparison(ds, dim_to_apply_metric_to, stack_dims=stack_dims)

    # in case you want to compute skill over member dim
    if (forecast.dims != reference.dims) and (metric not in PROBABILISTIC_METRICS):
        # broadcast when deterministic dim=member
        forecast, reference = xr.broadcast(forecast, reference)
        # m2m creates additional forecast_member when over dim member
        if comparison.__name__ == '_m2m':
            dim_to_apply_metric_to = 'forecast_member'
        else:
            dim_to_apply_metric_to = dim

    metric = get_metric_function(metric, PM_METRICS)

    skill = metric(
        forecast,
        reference,
        dim=dim_to_apply_metric_to,
        comparison=comparison,
        **metric_kwargs,
    )

    # correction for distance based metrics in m2m comparison
    comparison_name = comparison.__name__[1:]
    # fix for m2m TODO
    if comparison_name == 'm2m':
        if 'forecast_member' in skill.dims:
            skill = skill.mean('forecast_member')
        if dim == 'member' and 'member' in skill.dims:
            skill = skill.mean('member')
        if dim == 'init' and 'init' in skill.dims:
            skill = skill.mean('init')
        # m2m stack_dims=False has one identical comparison
        skill = skill * (forecast.member.size / (forecast.member.size - 1))
    # Attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            ds,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=metric_kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_hindcast(
    hind,
    reference,
    metric='pearson_r',
    comparison='e2r',
    dim='init',
    max_dof=False,
    add_attrs=True,
    **metric_kwargs,
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
            reference
            (see :py:func:`climpred.utils.get_metric_function` and :ref:`Metrics`).
        comparison (str):
            How to compare the decadal prediction ensemble to the reference:

                * e2r : ensemble mean to reference (Default)
                * m2r : each member to the reference
                (see :ref:`Comparisons`)
        dim (str or list): dimension to apply metric over. default: 'init'
        max_dof (bool):
            If True, maximize the degrees of freedom by slicing `hind` and `reference`
            to a common time frame at each lead.

            If False (default), then slice to a common time frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
        add_attrs (bool): write climpred compute args to attrs. default: True
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        skill (xarray object):
            Predictability with main dimension ``lag`` without dimension ``dim``

    """
    # get metric function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)

    # if stack_dims, comparisons return forecast with member dim and reference
    # without member dim which is needed for probabilistic
    # if not stack_dims, comparisons return forecast and reference with member dim
    # which is neeeded for deterministic
    if metric in PROBABILISTIC_METRICS:
        if comparison != 'm2r':
            raise ValueError(
                f'Probabilistic metric `{metric}` requires comparison `m2r`.'
            )
        stack_dims = False
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    elif dim == 'init':
        stack_dims = True
    elif dim == 'member':
        stack_dims = False
    else:
        raise ValueError(
            f'Please use a probabilistic metric [now {metric}] ',
            f'and the comparison `m2r` [now {comparison}] or ',
            f'specify dim from ["init", "member"], now: {dim}.',
        )
    nlags = max(hind.lead.values)
    comparison = get_comparison_function(comparison, HINDCAST_COMPARISONS)

    forecast, reference = comparison(hind, reference, stack_dims=stack_dims)

    # in case you want to compute skill over member dim
    if (
        (forecast.dims != reference.dims)
        and not stack_dims
        and metric in DETERMINISTIC_HINDCAST_METRICS
    ):
        dim_to_apply_metric_to = 'member'
    else:
        dim_to_apply_metric_to = 'time'

    metric = get_metric_function(metric, HINDCAST_METRICS)

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
        a['time'] = [int(t + i) for t in a.time.values]
        # take real time reference of real time forecast years
        b = reference.sel(time=a.time.values)
        # broadcast dims when apply over member
        if (a.dims != b.dims) and dim_to_apply_metric_to == 'member':
            a, b = xr.broadcast(a, b)
        plag.append(
            metric(
                a, b, dim=dim_to_apply_metric_to, comparison=comparison, **metric_kwargs
            )
        )
    skill = xr.concat(plag, 'lead')
    skill['lead'] = forecast.lead.values
    # rename back to init
    if 'time' in skill.dims:  # when dim was 'member'
        skill = skill.rename({'time': 'init'})
    # attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            hind,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=metric_kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_persistence(
    hind, reference, metric='pearson_r', max_dof=False, **metric_kwargs
):
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
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        pers (xarray object): Results of persistence forecast with the input metric
        applied.

    Reference:
        * Chapter 8 (Short-Term Climate Prediction) in Van den Dool, Huug.
          Empirical methods in short-term climate prediction.
          Oxford University Press, 2007.
    """
    if metric in PROBABILISTIC_METRICS:
        raise ValueError(
            'probabilistic metric ', metric, 'cannot compute persistence forecast.'
        )
    metric = get_metric_function(metric, DETERMINISTIC_HINDCAST_METRICS)
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
        plag.append(metric(ref, fct, dim='time', comparison=_e2c, **metric_kwargs))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    return pers


@is_xarray([0, 1])
def compute_uninitialized(
    uninit,
    reference,
    metric='pearson_r',
    comparison='e2r',
    dim='time',
    add_attrs=True,
    **metric_kwargs,
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
        ** metric_kwargs (dict): additional keywords to be passed to metric


    Returns:
        u (xarray object): Results from comparison at the first lag.

    """
    comparison = get_comparison_function(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_function(metric, DETERMINISTIC_HINDCAST_METRICS)
    forecast, reference = comparison(uninit, reference)
    # Find common times between two for proper comparison.
    common_time = intersect(forecast['time'].values, reference['time'].values)
    forecast = forecast.sel(time=common_time)
    reference = reference.sel(time=common_time)
    uninit_skill = metric(
        forecast, reference, dim=dim, comparison=comparison, **metric_kwargs
    )
    # Attach climpred compute information to skill
    if add_attrs:
        uninit_skill = assign_attrs(
            uninit_skill,
            uninit,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            metadata_dict=metric_kwargs,
        )
    return uninit_skill
