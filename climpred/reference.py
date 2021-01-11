import inspect

import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_xarray
from .comparisons import COMPARISON_ALIASES, HINDCAST_COMPARISONS, __e2c
from .constants import CLIMPRED_DIMS
from .metrics import DETERMINISTIC_HINDCAST_METRICS, METRIC_ALIASES, _rename_dim
from .utils import (
    assign_attrs,
    convert_time_index,
    copy_coords_from_to,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)


def persistence(verif, inits, verif_dates, lead):
    a = verif.where(verif.time.isin(inits[lead]), drop=True)
    b = verif.sel(time=verif_dates[lead])
    return a, b


def uninitialized(hist, verif, verif_dates, lead):
    """also called historical in some communities."""
    a = hist.sel(time=verif_dates[lead])
    b = verif.sel(time=verif_dates[lead])
    return a, b


# LEGACY CODE BELOW -- WILL BE DELETED DURING INHERITANCE REFACTORING #
@is_xarray([0, 1])
def compute_persistence(
    hind,
    verif,
    metric="pearson_r",
    alignment="same_verifs",
    add_attrs=True,
    dim="init",
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
        dim (str or list of str): dimension to apply metric over.
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
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, "init", "hind[init]")
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)

    # get class metric(Metric)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    if metric.probabilistic:
        raise ValueError(
            "probabilistic metric ",
            metric.name,
            "cannot compute persistence forecast.",
        )
    # If lead 0, need to make modifications to get proper persistence, since persistence
    # at lead 0 is == 1.
    if [0] in hind.lead.values:
        hind = hind.copy()
        with xr.set_options(keep_attrs=True):  # keeps lead.attrs['units']
            hind["lead"] = hind["lead"] + 1
        n, freq = get_lead_cftime_shift_args(hind.lead.attrs["units"], 1)
        # Shift backwards shift for lead zero.
        hind["init"] = shift_cftime_index(hind, "init", -1 * n, freq)
    # temporarily change `init` to `time` for comparison to verification data time.
    hind = hind.rename({"init": "time"})

    inits, verif_dates = return_inits_and_verif_dates(hind, verif, alignment=alignment)

    if metric.normalize:
        metric_kwargs["comparison"] = __e2c
    dim = _rename_dim(dim, hind, verif)
    if "member" in dim:
        dim.remove("member")

    plag = []
    for i in hind.lead.values:
        a = verif.sel(time=inits[i])
        b = verif.sel(time=verif_dates[i])
        a["time"] = b["time"]
        # comparison expected for normalized metrics
        plag.append(metric.function(a, b, dim=dim, **metric_kwargs))
    pers = xr.concat(plag, "lead")
    if "time" in pers:
        pers = pers.dropna(dim="time").rename({"time": "init"})
    pers["lead"] = hind.lead.values
    # keep coords from hind
    drop_dims = [d for d in hind.coords if d in CLIMPRED_DIMS]
    if "iteration" in hind.dims:
        drop_dims += ["iteration"]
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
    metric="pearson_r",
    comparison="e2o",
    dim="time",
    alignment="same_verifs",
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
        dim (str or list of str): dimension to apply metric over.
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
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, "init", "hind[init]")
    uninit = convert_time_index(uninit, "time", "uninit[time]")
    verif = convert_time_index(verif, "time", "verif[time]")
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)
    metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    forecast, verif = comparison.function(uninit, verif, metric=metric)

    hind = hind.rename({"init": "time"})

    _, verif_dates = return_inits_and_verif_dates(hind, verif, alignment=alignment)

    if metric.normalize:
        metric_kwargs["comparison"] = comparison

    plag = []
    # TODO: Refactor this, getting rid of `compute_uninitialized` completely.
    # `same_verifs` does not need to go through the loop, since it's a fixed
    # skill over all leads
    for i in hind["lead"].values:
        # Ensure that the uninitialized reference has all of the
        # dates for alignment.
        dates = list(set(forecast["time"].values) & set(verif_dates[i]))
        a = forecast.sel(time=dates)
        b = verif.sel(time=dates)
        a["time"] = b["time"]
        # comparison expected for normalized metrics
        plag.append(metric.function(a, b, dim=dim, **metric_kwargs))
    uninit_skill = xr.concat(plag, "lead")
    uninit_skill["lead"] = hind.lead.values

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
