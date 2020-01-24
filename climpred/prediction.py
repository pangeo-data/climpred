import inspect
import warnings

import dask
import xarray as xr

from .checks import has_valid_lead_units, is_in_list, is_xarray
from .comparisons import __e2c
from .constants import (
    CLIMPRED_DIMS,
    COMPARISON_ALIASES,
    DETERMINISTIC_HINDCAST_METRICS,
    HINDCAST_COMPARISONS,
    HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_COMPARISONS,
    PM_METRICS,
)
from .utils import (
    assign_attrs,
    convert_time_index,
    copy_coords_from_to,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    intersect,
    reduce_time_series,
    shift_cftime_index,
)

# from .constants import M2M_MEMBER_DIM
M2M_MEMBER_DIM = 'forecast_member'

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
    # get metric and comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    # get class metric(Metric)
    metric = get_metric_class(metric, PM_METRICS)
    # get class comparison(Comparison)
    comparison = get_comparison_class(comparison, PM_COMPARISONS)

    if metric.probabilistic:
        if not comparison.probabilistic:
            raise ValueError(
                f'Probabilistic metric {metric.name} cannot work with '
                f'comparison {comparison.name}.'
            )
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric.name} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    else:  # deterministic metric
        # prevent comparison e2c and member in dim
        if (comparison.name == 'e2c') and (
            set(dim) == set(['init', 'member']) or dim == 'member'
        ):
            warnings.warn(
                f'comparison `e2c` does not work on `member` in dims, found '
                f'{dim}, automatically changed to dim=`init`.'
            )
            dim = 'init'

    forecast, reference = comparison.function(ds, metric=metric)

    # in case you want to compute deterministic skill over member dim
    if (forecast.dims != reference.dims) and not metric.probabilistic:
        forecast, reference = xr.broadcast(forecast, reference)

    skill = metric.function(
        forecast, reference, dim=dim, comparison=comparison, **metric_kwargs
    )
    if comparison.name == 'm2m':
        skill = skill.mean(M2M_MEMBER_DIM)
    # Attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            ds,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            dim=dim,
            metadata_dict=metric_kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_hindcast(
    hind,
    verif,
    metric='pearson_r',
    comparison='e2o',
    dim='init',
    max_dof=False,
    add_attrs=True,
    **metric_kwargs,
):
    """Verify hindcast predictions against verification data.

    Args:
        hind (xarray object): Hindcast ensemble.
            Expected to follow package conventions:
            * ``init`` : dim of initialization dates
            * ``lead`` : dim of lead time from those initializations
            Additional dims can be member, lat, lon, depth, ...
        verif (xarray object): Verification data with some temporal overlap with the
            hindcast.
        metric (str): Metric used in comparing the decadal prediction ensemble with the
            verification data. (see :py:func:`~climpred.utils.get_metric_class` and
            :ref:`Metrics`).
        comparison (str):
            How to compare the decadal prediction ensemble to the verification data:

                * e2o : ensemble mean to verification data (Default)
                * m2o : each member to the verification data
                (see :ref:`Comparisons`)
        dim (str or list): dimension to apply metric over. default: 'init'
        max_dof (bool):
            If True, maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.

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
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, 'init', 'hind[init]')
    verif = convert_time_index(verif, 'time', 'verif[time]')
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    # get class metric(Metric)
    metric = get_metric_class(metric, HINDCAST_METRICS)
    # get class comparison(Comparison)
    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)

    if metric.probabilistic:
        if not comparison.probabilistic:
            raise ValueError(
                f'Probabilistic metric `{metric.name}` requires comparison'
                f' e.g. `m2o`, found `{comparison.name}`.'
            )
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric.name} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    else:
        if dim == 'init':
            # for later thinking in real time
            dim = 'time'
    nlags = max(hind.lead.values)

    forecast, verif = comparison.function(hind, verif, metric=metric)

    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})

    # If dask, then chunk in time.
    if dask.is_dask_collection(forecast):
        forecast = forecast.chunk({'time': -1})
    if dask.is_dask_collection(verif):
        verif = verif.chunk({'time': -1})

    # take only inits for which we have verification data at all leads
    if not max_dof:
        forecast, verif = reduce_time_series(forecast, verif, nlags)

    plag = []
    # iterate over all leads (accounts for lead.min() in [0,1])
    for i in forecast.lead.values:
        if max_dof:
            forecast, verif = reduce_time_series(forecast, verif, i)
        # take lead year i timeseries and convert to real time based on temporal
        # resolution of lead.
        n, freq = get_lead_cftime_shift_args(forecast.lead.attrs['units'], i)
        a = forecast.sel(lead=i).drop_vars('lead')
        a['time'] = shift_cftime_index(a, 'time', n, freq)
        # Take real time verification data using real time forecast dates.
        b = verif.sel(time=a.time.values)

        # adapt weights to shorter time
        if 'weights' in metric_kwargs:
            metric_kwargs.update(
                {
                    'weights': metric_kwargs['weights'].isel(
                        time=slice(None, a.time.size)
                    )
                }
            )

        # broadcast dims when deterministic metric and apply over member
        if (a.dims != b.dims) and (dim == 'member') and not metric.probabilistic:
            a, b = xr.broadcast(a, b)
        plag.append(
            metric.function(a, b, dim=dim, comparison=comparison, **metric_kwargs,)
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
            dim=dim,
            metadata_dict=metric_kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_persistence(
    hind, verif, metric='pearson_r', max_dof=False, **metric_kwargs
):
    """Computes the skill of a persistence forecast from a simulation.

    Args:
        hind (xarray object): The initialized ensemble.
        verif (xarray object): Verification data.
        metric (str): Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        max_dof (bool):
            If True, maximize the degrees of freedom by slicing ``hind`` and ``verif``
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
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, 'init', 'hind[init]')
    verif = convert_time_index(verif, 'time', 'verif[time]')
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)

    # get class metric(Metric)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    if metric.probabilistic:
        raise ValueError(
            'probabilistic metric ',
            metric.name,
            'cannot compute persistence forecast.',
        )
    # If lead 0, need to make modifications to get proper persistence, since persistence
    # at lead 0 is == 1.
    if [0] in hind.lead.values:
        hind = hind.copy()
        hind['lead'] += 1
        n, freq = get_lead_cftime_shift_args(hind.lead.attrs['units'], 1)
        # Shift backwards shift for lead zero.
        hind['init'] = shift_cftime_index(hind, 'init', -1 * n, freq)
    nlags = max(hind.lead.values)
    # temporarily change `init` to `time` for comparison to verification data time.
    hind = hind.rename({'init': 'time'})
    if not max_dof:
        # slices down to inits in common with hindcast, plus gives enough room
        # for maximum lead time forecast.
        a, _ = reduce_time_series(hind, verif, nlags)
        inits = a['time']

    plag = []
    for lag in hind.lead.values:
        if max_dof:
            # slices down to inits in common with hindcast, but only gives enough
            # room for lead from current forecast
            a, _ = reduce_time_series(hind, verif, lag)
            inits = a['time']
        n, freq = get_lead_cftime_shift_args(hind.lead.attrs['units'], lag)
        target_dates = shift_cftime_index(a, 'time', n, freq)

        o = verif.sel(time=target_dates)
        f = verif.sel(time=inits)
        o['time'] = f['time']
        plag.append(
            metric.function(o, f, dim='time', comparison=__e2c, **metric_kwargs)
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
    verif,
    metric='pearson_r',
    comparison='e2o',
    dim='time',
    add_attrs=True,
    **metric_kwargs,
):
    """Verify an uninitialized ensemble against verification data.

    .. note::
        Based on Decadal Prediction protocol, this should only be computed for the
        first lag and then projected out to any further lags being analyzed.

    Args:
        uninit (xarray object): Uninitialized ensemble.
        verif (xarray object): Verification data with some temporal overlap with the
            uninitialized ensemble.
        metric (str):
            Metric used in comparing the uninitialized ensemble with the verification
            data.
        comparison (str):
            How to compare the uninitialized ensemble to the verification data:
                * e2o : ensemble mean to verification data (Default)
                * m2o : each member to the verification data
        add_attrs (bool): write climpred compute args to attrs. default: True
        ** metric_kwargs (dict): additional keywords to be passed to metric


    Returns:
        u (xarray object): Results from comparison at the first lag.

    """
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    uninit = convert_time_index(uninit, 'time', 'uninit[time]')
    verif = convert_time_index(verif, 'time', 'verif[time]')

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    forecast, verif = comparison.function(uninit, verif, metric=metric)
    # Find common times between two for proper comparison.
    common_time = intersect(forecast['time'].values, verif['time'].values)
    forecast = forecast.sel(time=common_time)
    verif = verif.sel(time=common_time)
    uninit_skill = metric.function(
        forecast, verif, dim=dim, comparison=comparison, **metric_kwargs
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
