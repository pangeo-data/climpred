import inspect
import warnings

import xarray as xr

from .checks import is_in_list, is_xarray
from .comparisons import __e2c
from .constants import (
    CLIMPRED_DIMS,
    DETERMINISTIC_HINDCAST_METRICS,
    HINDCAST_COMPARISONS,
    HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_COMPARISONS,
    PM_METRICS,
)
from .utils import (
    assign_attrs,
    copy_coords_from_to,
    get_comparison_class,
    get_metric_class,
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
         :py:func:`climpred.utils.get_metric_class` and (see :ref:`Metrics`).
        comparison (str): `comparison` name defines what to take as forecast
            and verification (see
            :py:func:`climpred.utils.get_comparison_class` and :ref:`Comparisons`).
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
    is_in_list(dim, ['member', 'init', ['init', 'member']], '')
    # get metric function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    # get class metric(Metric)
    metric = get_metric_class(metric, PM_METRICS)
    # get class comparison(Comparison)
    comparison = get_comparison_class(comparison, PM_COMPARISONS)
    # if stack_dims, comparisons return forecast with member dim and reference
    # without member dim which is needed for probabilistic
    # if not stack_dims, comparisons return forecast and reference with member dim
    # which is neeeded for deterministic
    if metric.probabilistic:
        if not comparison.probabilistic:
            raise ValueError(
                f'Probabilistic metric {metric.name} cannot work with '
                f'comparison {comparison.name}.'
            )
        stack_dims = False
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric.name} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    else:
        # prevent comparison e2c and member in dim
        if (comparison.name == 'e2c') and (
            set(dim) == set(['init', 'member']) or dim == 'member'
        ):
            warnings.warn(
                f'comparison `e2c` does not work on `member` in dims, found '
                f'{dim}, automatically changed to dim=`init`.'
            )
            dim = 'init'
        stack_dims = False
    dim_to_apply_metric_to = dim

    # stack_dims = True when metric probabilistic
    forecast, reference = comparison.function(ds, stack_dims=stack_dims)

    # in case you want to compute skill over member dim
    if (forecast.dims != reference.dims) and (not metric.probabilistic):
        # broadcast when deterministic dim=member
        forecast, reference = xr.broadcast(forecast, reference)

    skill = metric.function(
        forecast,
        reference,
        dim=dim_to_apply_metric_to,
        comparison=comparison,
        **metric_kwargs,
    )

    # correction for distance based metrics in m2m comparison
    # fix for m2m TODO
    if comparison.name == 'm2m':
        if 'forecast_member' in skill.dims:
            skill = skill.mean('forecast_member')
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
            (see :py:func:`climpred.utils.get_metric_class` and :ref:`Metrics`).
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
    is_in_list(dim, ['member', 'init'], str)
    # get metric function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    # get class metric(Metric)
    metric = get_metric_class(metric, HINDCAST_METRICS)
    # get class comparison(Comparison)
    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)

    # if stack_dims, comparisons return forecast with member dim and reference
    # without member dim which is needed for probabilistic
    # if not stack_dims, comparisons return forecast and reference with member dim
    # which is neeeded for deterministic
    if metric.probabilistic:
        if comparison.name != 'm2r':
            raise ValueError(
                f'Probabilistic metric `{metric.name}` requires comparison'
                f' `m2r`, found {comparison.name}.'
            )
        stack_dims = False
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric.name} requires to be '
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
            f'Please use a probabilistic metric [now {metric.name}] ',
            f'and the comparison `m2r` [now {comparison.name}] or ',
            f'specify dim from ["init", "member"], now: {dim}.',
        )
    nlags = max(hind.lead.values)

    forecast, reference = comparison.function(hind, reference, stack_dims=stack_dims)

    # in case you want to compute skill over member dim
    if (
        (forecast.dims != reference.dims)
        and not stack_dims
        and not metric.probabilistic
    ):
        dim_to_apply_metric_to = 'member'
    else:
        dim_to_apply_metric_to = 'time'

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
        a = forecast.sel(lead=i).drop_vars('lead')
        a['time'] = [int(t + i) for t in a.time.values]
        # take real time reference of real time forecast years
        b = reference.sel(time=a.time.values)
        # adapt weights to shorter time
        if 'weights' in metric_kwargs:
            metric_kwargs.update(
                {
                    'weights': metric_kwargs['weights'].isel(
                        time=slice(None, a.time.size)
                    )
                }
            )
        # broadcast dims when apply over member
        if (a.dims != b.dims) and dim_to_apply_metric_to == 'member':
            a, b = xr.broadcast(a, b)
        plag.append(
            metric.function(
                a, b, dim=dim_to_apply_metric_to, comparison=comparison, **metric_kwargs
            )
        )
    skill = xr.concat(plag, 'lead')
    skill['lead'] = forecast.lead.values
    # rename back to init
    if 'time' in skill.dims:  # when dim was 'member'
        skill = skill.rename({'time': 'init'})
    # keep coords from hind
    drop_dims = [d for d in hind.coords if d in CLIMPRED_DIMS]
    skill = copy_coords_from_to(hind.drop_vars(drop_dims), skill)
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
    # get metric function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    # get class metric(Metric)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    if metric.probabilistic:
        raise ValueError(
            'probabilistic metric ', metric.name, 'cannot compute persistence forecast.'
        )
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
        plag.append(
            metric.function(ref, fct, dim='time', comparison=__e2c, **metric_kwargs)
        )
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    # keep coords from hind
    drop_dims = [d for d in hind.coords if d in CLIMPRED_DIMS]
    pers = copy_coords_from_to(hind.drop_vars(drop_dims), pers)
    # TODO: add climpred metadata
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
    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    forecast, reference = comparison.function(uninit, reference)
    # Find common times between two for proper comparison.
    common_time = intersect(forecast['time'].values, reference['time'].values)
    forecast = forecast.sel(time=common_time)
    reference = reference.sel(time=common_time)
    uninit_skill = metric.function(
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
