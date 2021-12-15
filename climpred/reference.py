"""Reference forecasts: climatology, persistence, uninitialized."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units
from .comparisons import (
    ALL_COMPARISONS,
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    Comparison,
    __e2c,
)
from .constants import M2M_MEMBER_DIM
from .metrics import (
    ALL_METRICS,
    DETERMINISTIC_HINDCAST_METRICS,
    METRIC_ALIASES,
    Metric,
    _rename_dim,
)
from .options import OPTIONS
from .utils import (
    convert_cftime_to_datetime_coords,
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)

metricType = Union[str, Metric]
comparisonType = Union[str, Comparison]
dimType = Optional[Union[str, List[str]]]
alignmentType = str
metric_kwargsType = Optional[Any]


def _maybe_seasons_to_int(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """Set season str values or coords to int."""
    seasonal = False
    for season in ["DJF", "MAM", "JJA", "SON"]:
        if season in ds:
            seasonal = True
    if seasonal:
        ds = (
            ds.str.replace("DJF", "1")
            .str.replace("MAM", "2")
            .str.replace("JJA", "3")
            .str.replace("SON", "4")
            .astype("int")
        )
    elif "season" in ds.coords:  # set season coords to int
        seasonal = False
        for season in ["DJF", "MAM", "JJA", "SON"]:
            if season in ds.coords["season"]:
                seasonal = True
        if seasonal:
            ds.coords["season"] = (
                ds.coords.get("season")
                .str.replace("DJF", "1")
                .str.replace("MAM", "2")
                .str.replace("JJA", "3")
                .str.replace("SON", "4")
                .astype("int")
            )
    return ds


def persistence(
    verif: xr.Dataset,
    inits: Dict[float, xr.DataArray],
    verif_dates: Dict[float, xr.DataArray],
    lead: float,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Create forecast, verification tuple at lead for persistence forecast."""
    lforecast = verif.where(verif.time.isin(inits[lead]), drop=True)
    lverif = verif.sel(time=verif_dates[lead])
    return lforecast, lverif


def climatology(
    verif: xr.Dataset,
    inits: Dict[float, xr.DataArray],
    verif_dates: Dict[float, xr.DataArray],
    lead: float,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Create forecast, verification tuple at lead for climatology forecast."""
    init_lead = inits[lead].copy()
    seasonality_str = OPTIONS["seasonality"]
    if seasonality_str == "weekofyear":
        # convert to datetime for weekofyear operations
        verif = convert_cftime_to_datetime_coords(verif, "time")
        init_lead["time"] = init_lead["time"].to_index().to_datetimeindex()
        init_lead = init_lead["time"]
    climatology_day = verif.groupby(f"time.{seasonality_str}").mean()
    with warnings.catch_warnings():  # ignore numpy warning https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur#46721064 # noqa: E501
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # enlarge times to get climatology_forecast times
        # this prevents errors if verification.time and hindcast.init are too much apart
        verif_hind_union = xr.DataArray(
            verif.time.to_index().union(init_lead.time.to_index()), dims="time"
        )

        climatology_forecast = (
            _maybe_seasons_to_int(climatology_day)
            .sel(
                {
                    seasonality_str: _maybe_seasons_to_int(
                        getattr(verif_hind_union.time.dt, seasonality_str)  # type: ignore
                    )
                },
                method="nearest",  # nearest may be a bit incorrect but doesnt error
            )
            .drop_vars(seasonality_str)
        )
        lforecast = climatology_forecast.where(
            climatology_forecast.time.isin(init_lead), drop=True
        )
    lverif = verif.sel(time=verif_dates[lead])
    # convert back to CFTimeIndex if needed
    if isinstance(lforecast["time"].to_index(), pd.DatetimeIndex):
        lforecast = convert_time_index(lforecast, "time")
    if isinstance(lverif["time"].to_index(), pd.DatetimeIndex):
        lverif = convert_time_index(lverif, "time")
    return lforecast, lverif


def uninitialized(
    hist: xr.Dataset,
    verif: xr.Dataset,
    verif_dates: Dict[float, xr.DataArray],
    lead: float,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Create forecast, verification tuple at lead for uninitialized forecast.

    Uninitialized forecast uses a simulation without any initialization
    (assimilation/nudging). Also called historical in some communities.
    """
    lforecast = hist.sel(time=verif_dates[lead])
    lverif = verif.sel(time=verif_dates[lead])
    return lforecast, lverif


# needed for PerfectModelEnsemble.verify(reference=...) and PredictionEnsemble.bootstrap
# TODO: should be refactored for better non-functional use within verify and bootstrap


def _adapt_member_for_reference_forecast(lforecast, lverif, metric, comparison, dim):
    """
    Maybe drop member from dim or add single-member dimension.

    Used in reference forecasts: climatology, uninitialized, persistence.
    """
    # persistence or climatology forecasts wont have member dimension, create if
    # required
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
    initialized,
    verif=None,
    metric="pearson_r",
    comparison="m2e",
    alignment="same_inits",
    dim="init",
    **metric_kwargs,
):
    """Compute the skill of a climatology forecast.

    Args:
        initialized: The initialized ensemble.
        verif: control data, not needed
        metric: Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        dim: dimension to apply metric over.
        ** metric_kwargs: additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        clim: Results of climatology forecast with the input metric
            applied.
    """
    seasonality_str = OPTIONS["seasonality"]

    if isinstance(dim, str):
        dim = [dim]
    has_valid_lead_units(initialized)

    # get metric/comparison function name, not the alias
    if isinstance(metric, str):
        metric = METRIC_ALIASES.get(metric, metric)
        metric = get_metric_class(metric, ALL_METRICS)

    if isinstance(comparison, str):
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    if "iteration" in initialized.dims:
        initialized = initialized.isel(iteration=0, drop=True)

    if comparison.hindcast:
        kind = "hindcast"
    else:
        kind = "perfect"

    if kind == "perfect":
        forecast, verif = comparison.function(initialized, metric=metric)
        climatology_day = verif.groupby(f"init.{seasonality_str}").mean()
    else:
        forecast, verif = comparison.function(initialized, verif, metric=metric)
        climatology_day = verif.groupby(f"time.{seasonality_str}").mean()

    climatology_day_forecast = climatology_day.sel(
        {seasonality_str: getattr(forecast.init.dt, seasonality_str)}, method="nearest"
    ).drop_vars(seasonality_str)

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


def compute_persistence(
    initialized: xr.Dataset,
    verif: xr.Dataset,
    metric: metricType = "acc",
    comparison: comparisonType = "m2o",
    dim: dimType = "init",
    alignment: alignmentType = "same_verifs",
    **metric_kwargs: Any,
) -> xr.Dataset:
    """Compute the skill of a persistence forecast from a simulation.

    This function unlike
    :py:func:`~climpred.reference.compute_persistence_from_first_lead` is
    not sensitive to ``comparison``. Requires
    ``climpred.set_options(PerfectModel_persistence_from_initialized_lead_0=False)``.

    Args:
        initialized: The initialized ensemble.
        verif: Verification data.
        metric: Metric name to apply at each lag for the persistence computation.
            Default: ``"pearson_r"``.
        alignment: which inits or verification times should be aligned?

            - ``"maximize"``: maximize the degrees of freedom by slicing
              ``initialized`` and ``verif`` to a common time frame at each lead.
            - ``"same_inits"``: slice to a common init frame prior to computing
              metric. This philosophy follows the thought that each lead should be
              based on the same set of initializations.
            - ``"same_verif"``: slice to a common/consistent verification time frame
              prior to computing metric. This philosophy follows the thought that
              each lead should be based on the same set of verification dates.

        dim: dimension to apply metric over.
        ** metric_kwargs: additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        pers: Results of persistence forecast with the input metric
            applied.

    Reference:
        * Chapter 8 (Short-Term Climate Prediction) in Van den Dool, Huug.
          Empirical methods in short-term climate prediction.
          Oxford University Press, 2007.

    See also:
        * :py:func:`~climpred.reference.compute_persistence_from_first_lead`
    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    initialized = convert_time_index(initialized, "init", "initialized[init]")
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(initialized)

    # get metric/comparison function name, not the alias
    if isinstance(metric, str):
        metric = METRIC_ALIASES.get(metric, metric)
        metric = get_metric_class(metric, ALL_METRICS)

    if isinstance(comparison, str):
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    # If lead 0, need to make modifications to get proper persistence, since persistence
    # at lead 0 is == 1.
    if [0] in initialized.lead.values:
        initialized = initialized.copy()
        with xr.set_options(keep_attrs=True):  # keeps lead.attrs['units']
            initialized["lead"] = initialized["lead"] + 1
        n, freq = get_lead_cftime_shift_args(initialized.lead.attrs["units"], 1)
        # Shift backwards shift for lead zero.
        initialized["init"] = shift_cftime_index(initialized, "init", -1 * n, freq)
    # temporarily change `init` to `time` for comparison to verification data time.
    initialized = initialized.rename({"init": "time"})

    inits, verif_dates = return_inits_and_verif_dates(
        initialized, verif, alignment=alignment
    )

    if metric.normalize:
        metric_kwargs["comparison"] = __e2c
    dim = _rename_dim(dim, initialized, verif)

    plag = []
    for i in initialized.lead.values:
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
    pers["lead"] = initialized.lead.values
    return pers


def compute_persistence_from_first_lead(
    initialized: xr.Dataset,
    verif: Optional[xr.Dataset] = None,
    metric: metricType = "pearson_r",
    alignment: alignmentType = "same_inits",
    dim: dimType = "init",
    comparison: comparisonType = "m2e",
    **metric_kwargs: metric_kwargsType,
):
    """Compute persistence skill based on first ``lead`` in ``initialized``.

    This function unlike :py:func:`~climpred.reference.compute_persistence` is
    sensitive to ``comparison``. Requires
    ``climpred.set_options(PerfectModel_persistence_from_initialized_lead_0=True)``.

    Args:
        initialized: The initialized ensemble.
        verif: Verification data. Not used.
        metric: Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        alignment: which inits or verification times should be aligned?

            - ``maximize``: maximize the degrees of freedom by slicing ``initialized``
              and ``verif`` to a common time frame at each lead.
            - ``same_inits``: slice to a common ``init`` frame prior to computing
              metric. This philosophy follows the thought that each lead should be
              based on the same set of initializations.
            - ``same_verif``: slice to a common/consistent verification time frame
              prior to computing metric. This philosophy follows the thought that
              each lead should be based on the same set of verification dates.

        dim: dimension to apply metric over.
        ** metric_kwargs: additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        pers: Results of persistence forecast with the input metric
            applied.

    Example:
        >>> with climpred.set_options(
        ...     PerfectModel_persistence_from_initialized_lead_0=True
        ... ):
        ...     PerfectModelEnsemble.verify(
        ...         metric="mse",
        ...         comparison="m2e",
        ...         dim=["init", "member"],
        ...         reference="persistence",
        ...     ).sel(
        ...         skill="persistence"
        ...     )  # persistence sensitive to comparison
        <xarray.Dataset>
        Dimensions:  (lead: 20)
        Coordinates:
          * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
            skill    <U11 'persistence'
        Data variables:
            tos      (lead) float32 0.01056 0.01962 0.02925 ... 0.08033 0.08731 0.07578
        Attributes:
            prediction_skill_software:                         climpred https://clim...
            skill_calculated_by_function:                      PerfectModelEnsemble....
            number_of_initializations:                         12
            number_of_members:                                 10
            metric:                                            mse
            comparison:                                        m2e
            dim:                                               ['init', 'member']
            reference:                                         ['persistence']
            PerfectModel_persistence_from_initialized_lead_0:  True


        >>> with climpred.set_options(
        ...     PerfectModel_persistence_from_initialized_lead_0=False
        ... ):
        ...     PerfectModelEnsemble.verify(
        ...         metric="mse",
        ...         comparison="m2e",
        ...         dim=["init", "member"],
        ...         reference="persistence",
        ...     ).sel(
        ...         skill="persistence"
        ...     )  # persistence not sensitive to comparison
        <xarray.Dataset>
        Dimensions:  (lead: 20)
        Coordinates:
          * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
            skill    <U11 'persistence'
        Data variables:
            tos      (lead) float32 0.02794 0.04554 0.08024 ... 0.06327 0.09077 0.05898
        Attributes:
            prediction_skill_software:                         climpred https://clim...
            skill_calculated_by_function:                      PerfectModelEnsemble....
            number_of_initializations:                         12
            number_of_members:                                 10
            metric:                                            mse
            comparison:                                        m2e
            dim:                                               ['init', 'member']
            reference:                                         ['persistence']
            PerfectModel_persistence_from_initialized_lead_0:  False

    Reference:
        * Chapter 8 (Short-Term Climate Prediction) in Van den Dool, Huug.
          Empirical methods in short-term climate prediction.
          Oxford University Press, 2007.

    See also:
        * :py:func:`~climpred.reference.compute_persistence`

    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    initialized = convert_time_index(initialized, "init", "initialized[init]")
    has_valid_lead_units(initialized)

    # get metric/comparison function name, not the alias
    if isinstance(metric, str):
        metric = METRIC_ALIASES.get(metric, metric)
        metric = get_metric_class(metric, ALL_METRICS)
    if isinstance(comparison, str):
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    forecast, observations = comparison.function(initialized, metric=metric)  # type: ignore
    forecast, dim = _adapt_member_for_reference_forecast(
        forecast, observations, metric, comparison, dim
    )
    # persistence is forecast lead 0
    persistence_skill = metric.function(
        forecast.isel(lead=0, drop=True), observations, dim=dim, **metric_kwargs
    )
    persistence_skill["lead"] = persistence_skill.lead.values
    return persistence_skill


def compute_uninitialized(
    initialized: xr.Dataset,
    uninit: xr.Dataset,
    verif: xr.Dataset,
    metric: metricType = "pearson_r",
    comparison: comparisonType = "e2o",
    dim: dimType = "time",
    alignment: alignmentType = "same_verifs",
    **metric_kwargs: metric_kwargsType,
):
    """Verify an uninitialized ensemble against verification data.

    .. note::
        Based on Decadal Prediction protocol, this should only be computed for the
        first lag and then projected out to any further lags being analyzed.

    Args:
        initialized: Initialized ensemble.
        uninit: Uninitialized ensemble.
        verif: Verification data with some temporal overlap with the
            uninitialized ensemble.
        metric:
            Metric used in comparing the uninitialized ensemble with the verification
            data.
        comparison:
            How to compare the uninitialized ensemble to the verification data:
                * `"e2o"` : ensemble mean to verification data (Default)
                * `"m2o"` : each member to the verification data
        dim: dimension to apply metric over.
        alignment: which inits or verification times should be aligned?

            - ``"maximize"``: maximize the degrees of freedom by slicing
              ``initialized`` and ``verif`` to a common time frame at each lead.
            - ``"same_inits"``: slice to a common init frame prior to computing
              metric. This philosophy follows the thought that each lead
              should be based on the same set of initializations.
            - ``same_verif``: slice to a common/consistent verification time
              frame prior to computing metric. This philosophy follows the
              thought that each lead should be based on the same set of
              verification dates.

        ** metric_kwargs: additional keywords to be passed to metric

    Returns:
        uninit_skill: Results from comparison at the first lag.

    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    initialized = convert_time_index(initialized, "init", "initialized[init]")
    uninit = convert_time_index(uninit, "time", "uninit[time]")
    verif = convert_time_index(verif, "time", "verif[time]")
    has_valid_lead_units(initialized)

    # get metric/comparison function name, not the alias
    if isinstance(metric, str):
        metric = METRIC_ALIASES.get(metric, metric)
        metric = get_metric_class(metric, DETERMINISTIC_HINDCAST_METRICS)
    if isinstance(comparison, str):
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        comparison = get_comparison_class(comparison, HINDCAST_COMPARISONS)

    forecast, verif = comparison.function(uninit, verif, metric)

    initialized = initialized.rename({"init": "time"})

    _, verif_dates = return_inits_and_verif_dates(
        initialized, verif, alignment=alignment
    )

    if metric.normalize:
        metric_kwargs["comparison"] = comparison

    plag = []
    # TODO: `same_verifs` does not need to go through the loop, since it's a fixed
    # skill over all leads
    for i in initialized["lead"].values:
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
    uninit_skill["lead"] = initialized.lead.values
    return uninit_skill
