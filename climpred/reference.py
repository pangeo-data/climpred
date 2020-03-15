import inspect

import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_xarray
from .comparisons import __e2c
from .constants import CLIMPRED_DIMS
from .metrics import DETERMINISTIC_HINDCAST_METRICS, METRIC_ALIASES
from .utils import (
    assign_attrs,
    convert_to_cftime_index,
    copy_coords_from_to,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)


def persistence(verif, inits, verif_dates, lead):
    a = verif.where(verif.time.isin(inits[lead]), drop=True)
    b = verif.sel(time=verif_dates[lead])
    a['time'] = b['time']
    return a, b


def historical(hist, verif, inits, verif_dates, lead):
    a = hist.sel(time=verif_dates[lead])
    b = verif.sel(time=verif_dates[lead])
    a['time'] = b['time']
    return a, b


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
