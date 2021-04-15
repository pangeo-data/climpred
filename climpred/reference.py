import inspect

import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_xarray
from .comparisons import (
    ALL_COMPARISONS,
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    __e2c,
)
from .constants import CLIMPRED_DIMS, M2M_MEMBER_DIM
from .metrics import (
    ALL_METRICS,
    DETERMINISTIC_HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_METRICS,
    _rename_dim,
)
from .options import OPTIONS
from .utils import (
    assign_attrs,
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)


def persistence(verif, inits, verif_dates, lead):
    lforecast = verif.where(verif.time.isin(inits[lead]), drop=True)
    lverif = verif.sel(time=verif_dates[lead])
    return lforecast, lverif


def climatology(verif, inits, verif_dates, lead):
    seasonality_str = OPTIONS["seasonality"]
    if seasonality_str == "weekofyear":
        raise NotImplementedError
    climatology_day = verif.groupby(f"time.{seasonality_str}").mean()
    # enlarge times to get climatology_forecast times
    # this prevents errors if verification.time and hindcast.init are too much apart
    verif_hind_union = xr.DataArray(
        verif.time.to_index().union(inits[lead].time.to_index()), dims="time"
    )

    climatology_forecast = climatology_day.sel(
        {seasonality_str: getattr(verif_hind_union.time.dt, seasonality_str)},
        method="nearest",
    ).drop(seasonality_str)

    lforecast = climatology_forecast.where(
        climatology_forecast.time.isin(inits[lead]), drop=True
    )
    lverif = verif.sel(time=verif_dates[lead])
    return lforecast, lverif


def uninitialized(hist, verif, verif_dates, lead):
    """also called historical in some communities."""
    lforecast = hist.sel(time=verif_dates[lead])
    lverif = verif.sel(time=verif_dates[lead])
    return lforecast, lverif


# needed for PerfectModelEnsemble.verify(reference=...) and PredictionEnsemble.bootstrap
# TODO: should be refactored for better non-functional use within verify and bootstrap


def _adapt_member_for_reference_forecast(lforecast, lverif, metric, comparison, dim):
    """Maybe drop member from dim or add single-member dimension. Used in reference forecasts: climatology, uninitialized, persistence."""
    # persistence or climatology forecasts wont have member dimension, create if required
    # some metrics dont allow member dimension, remove and try mean
    # delete member from dim if needed
    if "member" in dim:
        if (
            "member" in lforecast.dims
            and "member" not in lverif.dims
            and not metric.requires_member_dim
        ):
            dim = dim.copy()
            dim.remove("member")
        elif "member" not in lforecast.dims and "member" not in lverif.dims:
            dim = dim.copy()
            dim.remove("member")
    # for probabilistic metrics requiring member dim, add single-member dimension
    if metric.requires_member_dim:
        if "member" not in lforecast.dims:
            lforecast = lforecast.expand_dims("member")  # add fake member dim
            if "member" not in dim:
                dim = dim.copy()
                dim.append("member")
        assert "member" in lforecast.dims and "member" not in lverif.dims
    # member not required by metric and not in dim but present in forecast
    if not metric.requires_member_dim:
        if "member" in lforecast.dims and "member" not in dim:
            lforecast = lforecast.mean("member")
    # multi member comparisons expect member dim
    if (
        comparison.name in ["m2o", "m2m", "m2c", "m2e"]
        and "member" not in lforecast.dims
        and metric.requires_member_dim
    ):
        lforecast = lforecast.expand_dims("member")  # add fake member dim
    return lforecast, dim


def compute_climatology(
    hind,
    verif=None,
    metric="pearson_r",
    comparison="m2e",
    alignment="same_inits",
    add_attrs=True,
    dim="init",
    **metric_kwargs,
):
    """Computes the skill of a climatology forecast.

    Args:
        hind (xarray object): The initialized ensemble.
        verif (xarray object): control data, not needed
        metric (str): Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        dim (str or list of str): dimension to apply metric over.
        add_attrs (bool): write climpred compute_persistence args to attrs.
            default: True
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        clim (xarray object): Results of climatology forecast with the input metric
            applied.
    """
    seasonality_str = OPTIONS["seasonality"]

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
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, ALL_COMPARISONS)
    metric = get_metric_class(metric, ALL_METRICS)

    if "iteration" in hind.dims:
        hind = hind.isel(iteration=0, drop=True)

    if comparison.hindcast:
        kind = "hindcast"
    else:
        kind = "perfect"

    if kind == "perfect":
        forecast, verif = comparison.function(hind, metric=metric)
        climatology_day = verif.groupby(f"init.{seasonality_str}").mean()
    else:
        forecast, verif = comparison.function(hind, verif, metric=metric)
        climatology_day = verif.groupby(f"time.{seasonality_str}").mean()

    climatology_day_forecast = climatology_day.sel(
        {seasonality_str: getattr(forecast.init.dt, seasonality_str)}, method="nearest"
    ).drop(seasonality_str)

    if kind == "hindcast":
        climatology_day_forecast = climatology_day_forecast.rename({"init": "time"})
    dim = _rename_dim(dim, climatology_day_forecast, verif)
    if metric.normalize:
        metric_kwargs["comparison"] = __e2c

    climatology_day_forecast, dim = _adapt_member_for_reference_forecast(
        climatology_day_forecast, verif, metric, comparison, dim
    )

    clim_skill = metric.function(
        climatology_day_forecast, verif, dim=dim, **metric_kwargs
    )
    if M2M_MEMBER_DIM in clim_skill.dims:
        clim_skill = clim_skill.mean(M2M_MEMBER_DIM)
    return clim_skill


@is_xarray([0, 1])
def compute_persistence(
    hind,
    verif,
    metric="pearson_r",
    alignment="same_verifs",
    add_attrs=True,
    dim="init",
    comparison="m2o",
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
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    # get class metric(Metric)
    metric = get_metric_class(metric, ALL_METRICS)
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

    plag = []
    for i in hind.lead.values:
        lforecast = verif.sel(time=inits[i])
        lverif = verif.sel(time=verif_dates[i])
        lforecast, dim = _adapt_member_for_reference_forecast(
            lforecast, lverif, metric, comparison, dim
        )
        lverif["time"] = lforecast["time"]
        # comparison expected for normalized metrics
        plag.append(metric.function(lforecast, lverif, dim=dim, **metric_kwargs))
    pers = xr.concat(plag, "lead")
    if "time" in pers:
        pers = pers.dropna(dim="time").rename({"time": "init"})
    pers["lead"] = hind.lead.values
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
    # TODO: `same_verifs` does not need to go through the loop, since it's a fixed
    # skill over all leads
    for i in hind["lead"].values:
        # Ensure that the uninitialized reference has all of the
        # dates for alignment.
        dates = list(set(forecast["time"].values) & set(verif_dates[i]))
        lforecast = forecast.sel(time=dates)
        lverif = verif.sel(time=dates)
        lforecast, dim = _adapt_member_for_reference_forecast(
            lforecast, lverif, metric, comparison, dim
        )
        lforecast["time"] = lverif["time"]
        # comparison expected for normalized metrics
        plag.append(metric.function(lforecast, lverif, dim=dim, **metric_kwargs))
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
