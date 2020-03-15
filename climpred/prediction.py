import inspect
import warnings

import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_in_list, is_xarray
from .comparisons import (
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from .constants import CLIMPRED_DIMS, CONCAT_KWARGS, M2M_MEMBER_DIM, PM_CALENDAR_STR
from .logging import log_compute_hindcast_header, log_compute_hindcast_inits_and_verifs
from .metrics import HINDCAST_METRICS, METRIC_ALIASES, PM_METRICS
from .reference import historical, persistence
from .utils import (
    assign_attrs,
    convert_to_cftime_index,
    copy_coords_from_to,
    get_comparison_class,
    get_metric_class,
)


def _get_metric_comparison_dim(metric, comparison, dim, kind):
    """Returns `metric`, `comparison` and `dim` for compute functions.

    Args:
        metric (str): metric or alias string
        comparison (str): Description of parameter `comparison`.
        dim (list of str or str): dimension to apply metric to.
        kind (str): experiment type from ['hindcast', 'PM'].

    Returns:
        metric (Metric): metric class
        comparison (Comparison): comparison class.
        dim (list of str or str): corrected dimension to apply metric to.
    """
    # check kind allowed
    is_in_list(kind, ['hindcast', 'PM'], 'kind')
    # set default dim
    if dim is None:
        dim = 'init' if kind == 'hindcast' else ['init', 'member']
    # check allowed dims
    if kind == 'hindcast':
        is_in_list(dim, ['member', 'init'], 'dim')
    elif kind == 'PM':
        is_in_list(dim, ['member', 'init', ['init', 'member']], 'dim')

    # get metric and comparison strings incorporating alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    METRICS = HINDCAST_METRICS if kind == 'hindcast' else PM_METRICS
    COMPARISONS = HINDCAST_COMPARISONS if kind == 'hindcast' else PM_COMPARISONS
    metric = get_metric_class(metric, METRICS)
    comparison = get_comparison_class(comparison, COMPARISONS)

    # check whether combination of metric and comparison works
    PROBABILISTIC_COMPARISONS = (
        PROBABILISTIC_HINDCAST_COMPARISONS
        if kind == 'hindcast'
        else PROBABILISTIC_PM_COMPARISONS
    )
    if metric.probabilistic:
        if not comparison.probabilistic:
            raise ValueError(
                f'Probabilistic metric `{metric.name}` requires comparison '
                f'accepting multiple members e.g. `{PROBABILISTIC_COMPARISONS}`, '
                f'found `{comparison.name}`.'
            )
        if dim != 'member':
            warnings.warn(
                f'Probabilistic metric {metric.name} requires to be '
                f'computed over dimension `dim="member"`. '
                f'Set automatically.'
            )
            dim = 'member'
    else:  # determinstic metric
        if kind == 'hindcast':
            if dim == 'init':
                # for thinking in real time # verify_hindcast renames init to time
                dim = 'time'
        elif kind == 'PM':
            # prevent comparison e2c and member in dim
            if (comparison.name == 'e2c') and (
                set(dim) == set(['init', 'member']) or dim == 'member'
            ):
                warnings.warn(
                    f'comparison `{comparison.name}` does not work on `member` in dims,'
                    f' found {dim}, automatically changed to dim=`init`.'
                )
                dim = 'init'
    return metric, comparison, dim


@is_xarray([0, 1])
def compute_perfect_model(
    init_pm,
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
        init_pm (xarray object): ensemble with dims ``lead``, ``init``, ``member``.
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
    if 'weights' in metric_kwargs:
        warnings.warn(
            'Weights is not currently supported by climpred and will be ignored.'
        )
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    init_pm = convert_to_cftime_index(
        init_pm, 'init', 'init_pm[init]', calendar=PM_CALENDAR_STR
    )

    # check args compatible with each other
    metric, comparison, dim = _get_metric_comparison_dim(
        metric, comparison, dim, kind='PM'
    )

    forecast, verif = comparison.function(init_pm, metric=metric)

    # in case you want to compute deterministic skill over member dim
    if (forecast.dims != verif.dims) and not metric.probabilistic:
        forecast, verif = xr.broadcast(forecast, verif)

    skill = metric.function(
        forecast, verif, dim=dim, comparison=comparison, **metric_kwargs
    )
    if comparison.name == 'm2m':
        skill = skill.mean(M2M_MEMBER_DIM)
    # Attach climpred compute information to skill
    if add_attrs:
        skill = assign_attrs(
            skill,
            init_pm,
            function_name=inspect.stack()[0][3],
            metric=metric,
            comparison=comparison,
            dim=dim,
            metadata_dict=metric_kwargs,
        )
    return skill


def _apply_hindcast_metric(
    forecast,
    verif,
    inits,
    verif_dates,
    lead,
    hist=None,
    reference=None,
    metric=None,
    comparison=None,
    dim=None,
    **metric_kwargs,
):
    """Temporary docstring. Will clean up args and document this properly."""
    if reference is None:
        # Use `.where()` instead of `.sel()` to account for resampled inits when
        # bootstrapping.
        a = (
            forecast.sel(lead=lead)
            .where(forecast['time'].isin(inits[lead]), drop=True)
            .drop_vars('lead')
        )
        b = verif.sel(time=verif_dates[lead])
        a['time'] = b['time']
    elif reference == 'persistence':
        a, b = persistence(verif, inits, verif_dates, lead)
    elif reference == 'historical':
        a, b = historical(hist, verif, inits, verif_dates, lead)

    if a.time.size > 0:
        log_compute_hindcast_inits_and_verifs(dim, lead, inits, verif_dates)

    # broadcast dims when deterministic metric and apply over member
    if (a.dims != b.dims) and (dim == 'member') and not metric.probabilistic:
        a, b = xr.broadcast(a, b)
    result = metric.function(a, b, dim=dim, comparison=comparison, **metric_kwargs,)
    return result


@is_xarray([0, 1])
def verify_hindcast(
    hind,
    verif,
    metric='pearson_r',
    comparison='e2o',
    dim='init',
    alignment='same_verifs',
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
        alignment (str): which inits or verification times should be aligned?
            - maximize/None: maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.
            - same_inits: slice to a common init frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
            - same_verif: slice to a common/consistent verification time frame prior to
            computing metric. This philosophy follows the thought that each lead
            should be based on the same set of verification dates.
        add_attrs (bool): write climpred compute args to attrs. default: True
        **metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        result (xarray object):
            Verification metric over ``lead`` reduced by dimension(s) ``dim``.
    """
    if 'weights' in metric_kwargs:
        warnings.warn(
            'Weights is not currently supported by climpred and will be ignored.'
        )

    metric, comparison, dim = _get_metric_comparison_dim(
        metric, comparison, dim, kind='hindcast'
    )

    hind = convert_to_cftime_index(hind, 'init', 'hind[init]')
    verif = convert_to_cftime_index(verif, 'time', 'verif[time]')
    has_valid_lead_units(hind)

    forecast, verif = comparison.function(hind, verif, metric=metric)

    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})

    inits, verif_dates = return_inits_and_verif_dates(
        forecast, verif, alignment=alignment
    )

    log_compute_hindcast_header(metric, comparison, dim, alignment)

    # NOTE: Here we can just do list comprehension looping. The apply_metric function
    # should just handle alignment. This will open up the pathway to inserting some
    # reference function more easily.
    metric_over_leads = [
        _apply_hindcast_metric(
            forecast,
            verif,
            inits,
            verif_dates,
            i,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        for i in forecast['lead'].data
    ]
    result = xr.concat(metric_over_leads, dim='lead', **CONCAT_KWARGS)
    result['lead'] = forecast['lead']
    # rename back to 'init'
    if 'time' in result.dims:  # If dim is 'member'
        result = result.rename({'time': 'init'})
    # These computations sometimes drop coordinates along the way. This appends them
    # back onto the results of the metric.
    drop_dims = [d for d in hind.coords if d in CLIMPRED_DIMS]
    result = copy_coords_from_to(hind.drop_vars(drop_dims), result)

    # Attach climpred compute information to result
    if add_attrs:
        result = assign_attrs(
            result,
            hind,
            function_name=inspect.stack()[0][3],
            alignment=alignment,
            metric=metric,
            comparison=comparison,
            dim=dim,
            metadata_dict=metric_kwargs,
        )
    return result


def hindcast_reference(
    hind,
    verif,
    hist=None,
    reference='persistence',
    metric='pearson_r',
    comparison='e2o',
    dim='init',
    alignment='same_verifs',
    **metric_kwargs,
):
    """Work in progress on a generic reference forecast function."""
    metric, comparison, dim = _get_metric_comparison_dim(
        metric, comparison, dim, kind='hindcast'
    )

    if (metric.probabilistic) and (reference == 'persistence'):
        raise ValueError(
            'probabilistic metric ',
            metric.name,
            'cannot compute persistence forecast.',
        )

    hind = convert_to_cftime_index(hind, 'init', 'hind[init]')
    verif = convert_to_cftime_index(verif, 'time', 'verif[time]')
    hist = convert_to_cftime_index(hist, 'time', 'hist[time]')
    has_valid_lead_units(hind)

    forecast, verif = comparison.function(hind, verif, metric=metric)

    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})

    inits, verif_dates = return_inits_and_verif_dates(
        forecast, verif, alignment=alignment
    )

    metric_over_leads = [
        _apply_hindcast_metric(
            forecast,
            verif,
            inits,
            verif_dates,
            i,
            hist=hist,
            reference=reference,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        for i in forecast['lead'].data
    ]
    result = xr.concat(metric_over_leads, dim='lead', **CONCAT_KWARGS)
    result['lead'] = forecast['lead']
    return result
