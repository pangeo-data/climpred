import inspect
import warnings

import dask
import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_in_list, is_xarray
from .comparisons import (
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
    __e2c,
)
from .constants import CLIMPRED_DIMS, M2M_MEMBER_DIM
from .logging import log_compute_hindcast_header, log_compute_hindcast_inits_and_verifs
from .metrics import (
    DETERMINISTIC_HINDCAST_METRICS,
    HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_METRICS,
)
from .utils import (
    assign_attrs,
    convert_to_cftime_index,
    copy_coords_from_to,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)


def get_metric_comparison_dim(metric, comparison, dim, kind):
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
                # for thinking in real time # compute_hindcast renames init to time
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
    # check args compatible with each other
    metric, comparison, dim = get_metric_comparison_dim(
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


@is_xarray([0, 1])
def compute_hindcast(
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
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        skill (xarray object):
            Predictability with main dimension ``lag`` without dimension ``dim``

    """
    metric, comparison, dim = get_metric_comparison_dim(
        metric, comparison, dim, kind='hindcast'
    )
    hind = convert_to_cftime_index(hind, 'init', 'hind[init]')
    verif = convert_to_cftime_index(verif, 'time', 'verif[time]')
    has_valid_lead_units(hind)

    forecast, verif = comparison.function(hind, verif, metric=metric)

    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})

    # If dask, then only one chunk in time.
    if dask.is_dask_collection(forecast):
        forecast = forecast.chunk({'time': -1})
    if dask.is_dask_collection(verif):
        verif = verif.chunk({'time': -1})

    inits, verif_dates = return_inits_and_verif_dates(
        forecast, verif, alignment=alignment
    )

    log_compute_hindcast_header(metric, comparison, dim, alignment)
    plag = []
    # iterate over all leads (accounts for lead.min() in [0,1])
    for i in forecast['lead'].values:
        # Use `.where()` instead of `.sel()` to account for resampled inits.
        a = forecast.sel(lead=i).where(forecast['time'].isin(inits[i]), drop=True)
        b = verif.sel(time=verif_dates[i])
        # Align time coordinate for metric computations.
        a['time'] = b['time']
        if a.time.size > 0:
            log_compute_hindcast_inits_and_verifs(dim, i, inits, verif_dates)

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
            alignment=alignment,
            metric=metric,
            comparison=comparison,
            dim=dim,
            metadata_dict=metric_kwargs,
        )
    return skill


@is_xarray([0, 1])
def compute_persistence(
    hind,
    verif,
    metric='pearson_r',
    alignment='same_verifs',
    add_attrs=True,
    **metric_kwargs,
):
    """Computes the skill of a persistence forecast from a simulation.

    Args:
        hind (xarray object): The initialized ensemble.
        verif (xarray object): Verification data.
        metric (str): Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        alignment (str): which inits or verification times should be aligned?
            - maximize/None: maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.
            - same_inits: slice to a common init frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
            - same_verif: slice to a common/consistent verification time frame prior to
            computing metric. This philosophy follows the thought that each lead
            should be based on the same set of verification dates.
        add_attrs (bool): write climpred compute_persistence args to attrs.
            default: True
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
    hind = convert_to_cftime_index(hind, 'init', 'hind[init]')
    verif = convert_to_cftime_index(verif, 'time', 'verif[time]')
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
    # temporarily change `init` to `time` for comparison to verification data time.
    hind = hind.rename({'init': 'time'})

    inits, verif_dates = return_inits_and_verif_dates(hind, verif, alignment=alignment)

    plag = []
    for i in hind.lead.values:
        a = verif.sel(time=inits[i])
        b = verif.sel(time=verif_dates[i])
        a['time'] = b['time']
        plag.append(
            metric.function(a, b, dim='time', comparison=__e2c, **metric_kwargs)
        )
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    # keep coords from hind
    drop_dims = [d for d in hind.coords if d in CLIMPRED_DIMS]
    pers = copy_coords_from_to(hind.drop_vars(drop_dims), pers)
    if add_attrs:
        pers = assign_attrs(
            pers,
            hind,
            function_name=inspect.stack()[0][3],
            alignment=alignment,
            metric=metric,
            metadata_dict=metric_kwargs,
        )
    return pers


@is_xarray([0, 1])
def compute_uninitialized(
    hind,
    uninit,
    verif,
    metric='pearson_r',
    comparison='e2o',
    dim='time',
    alignment='same_verifs',
    add_attrs=True,
    **metric_kwargs,
):
    """Verify an uninitialized ensemble against verification data.

    .. note::
        Based on Decadal Prediction protocol, this should only be computed for the
        first lag and then projected out to any further lags being analyzed.

    Args:
        hind (xarray object): Initialized ensemble.
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
        ** metric_kwargs (dict): additional keywords to be passed to metric


    Returns:
        u (xarray object): Results from comparison at the first lag.

    """
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_to_cftime_index(hind, 'init', 'hind[init]')
    uninit = convert_to_cftime_index(uninit, 'time', 'uninit[time]')
    verif = convert_to_cftime_index(verif, 'time', 'verif[time]')
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    forecast, verif = comparison.function(uninit, verif, metric=metric)

    hind = hind.rename({'init': 'time'})

    _, verif_dates = return_inits_and_verif_dates(hind, verif, alignment=alignment)

    plag = []
    # TODO: Refactor this, getting rid of `compute_uninitialized` completely.
    # `same_verifs` does not need to go through the loop, since it's a fixed
    # skill over all leads.
    for i in hind['lead'].values:
        # Ensure that the uninitialized reference has all of the
        # dates for alignment.
        dates = list(set(forecast['time'].values) & set(verif_dates[i]))
        a = forecast.sel(time=dates)
        b = verif.sel(time=dates)
        a['time'] = b['time']
        plag.append(
            metric.function(a, b, dim='time', comparison=comparison, **metric_kwargs)
        )
    uninit_skill = xr.concat(plag, 'lead')
    uninit_skill['lead'] = hind.lead.values

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
