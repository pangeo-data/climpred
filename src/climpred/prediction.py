"""Prediction module: _apply_metric_at_given_lead and compute functions."""

from typing import List, Optional, Set, Tuple, Union

from dask import is_dask_collection

from .checks import is_in_list
from .comparisons import (
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from .constants import M2M_MEMBER_DIM, PM_CALENDAR_STR
from .exceptions import DimensionError
from .logging import log_hindcast_verify_inits_and_verifs
from .metrics import HINDCAST_METRICS, METRIC_ALIASES, PM_METRICS
from .reference import (
    _adapt_member_for_reference_forecast,
    climatology,
    persistence,
    uninitialized,
)
from .utils import (
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_singular,
)


def _apply_metric_at_given_lead(
    verif,
    verif_dates,
    lead,
    initialized=None,
    hist=None,
    inits=None,
    reference=None,
    metric=None,
    comparison=None,
    dim=None,
    **metric_kwargs,
):
    """Apply a metric between two time series at a given lead.

    Args:
        verif (xr.Dataset): Verification data.
        verif_dates (dict): Lead-dependent verification dates for alignment.
        lead (int): Given lead to score.
        initialized (xr.Dataset): Initialized hindcast. Not required in a persistence
            forecast.
        hist (xr.Dataset): Uninitialized/historical simulation. Required when
            ``reference='uninitialized'``.
        inits (dict): Lead-dependent initialization dates for alignment.
        reference (str): If not ``None``, return score for this reference forecast.
            * 'persistence'
            * 'uninitialized'
        metric (Metric): Metric class for scoring.
        comparison (Comparison): Comparison class.
        dim (str): Dimension to apply metric over.

    Returns:
        result (xr.Dataset): Metric results for the given lead for the initialized
            forecast or reference forecast.
    """
    # naming:: lforecast: forecast at lead; lverif: verification at lead
    if reference is None:
        # Use `.where()` instead of `.sel()` to account for resampled inits when
        # bootstrapping.
        lforecast = initialized.sel(lead=lead).where(
            initialized["time"].isin(inits[lead]), drop=True
        )
        lverif = verif.sel(time=verif_dates[lead])
    elif reference == "persistence":
        lforecast, lverif = persistence(verif, inits, verif_dates, lead)
    elif reference == "uninitialized":
        lforecast, lverif = uninitialized(hist, verif, verif_dates, lead)
    elif reference == "climatology":
        lforecast, lverif = climatology(verif, inits, verif_dates, lead)
    if reference is not None:
        lforecast, dim = _adapt_member_for_reference_forecast(
            lforecast, lverif, metric, comparison, dim
        )
    assert lforecast.time.size == lverif.time.size, print(
        lforecast.time.to_index(), lverif.time.to_index(), reference
    )
    lforecast["time"] = lverif[
        "time"
    ]  # a bit dangerous: what if different? more clear once implemented
    # https://github.com/pangeo-data/climpred/issues/523#issuecomment-728951645
    dim = _rename_dim(
        dim, initialized, verif
    )  # dim should be much clearer once time_valid in initialized.coords
    if metric.normalize or metric.allows_logical:
        metric_kwargs["comparison"] = comparison

    result = metric.function(lforecast, lverif, dim=dim, **metric_kwargs)

    log_hindcast_verify_inits_and_verifs(dim, lead, inits, verif_dates, reference)
    # push time (later renamed to init) back by lead
    if "time" in result.dims:
        n, freq = get_lead_cftime_shift_args(initialized.lead.attrs["units"], lead)
        result = result.assign_coords(time=shift_cftime_singular(result.time, -n, freq))
    if "valid_time" in result.coords:
        if is_dask_collection(result.coords["valid_time"]):
            result.coords["valid_time"] = result.coords["valid_time"].compute()
    return result


def _rename_dim(dim, forecast, verif):
    """Rename `dim` to `time` or `init` if forecast and verif dims requires."""
    if "init" in dim and "time" in forecast.dims and "time" in verif.dims:
        dim = dim.copy()
        dim.remove("init")
        dim = dim + ["time"]
    elif "time" in dim and "init" in forecast.dims and "init" in verif.dims:
        dim = dim.copy()
        dim.remove("time")
        dim = dim + ["init"]
    elif "init" in dim and "time" in forecast.dims and "time" in verif.dims:
        dim = dim.copy()
        dim.remove("init")
        dim = dim + ["time"]
    return dim


def _sanitize_to_list(dim: Union[str, Set, Tuple, List, None]) -> Optional[List[str]]:
    """Ensure dim is List, raises ValueError if not str, set, tuple or None."""
    if isinstance(dim, str):
        dim = [dim]
    elif isinstance(dim, set):
        dim = list(dim)
    elif isinstance(dim, tuple):
        dim = list(dim)
    elif isinstance(dim, list) or dim is None:
        pass
    else:
        raise ValueError(
            f"Expected `dim` as `str`, `list` or None, found {dim} as type {type(dim)}."
        )
    return dim


def _get_metric_comparison_dim(
    initialized,
    metric,
    comparison,
    dim: Optional[Union[List[str], str]],
    kind,
):
    """Return `metric`, `comparison` and `dim` for compute functions.

    Args:
        initialized (xr.Dataset): initialized dataset
        metric (str): metric or alias string
        comparison (str): Description of parameter `comparison`.
        dim (list of str or str): dimension to apply metric to.
        kind (str): experiment type from ['hindcast', 'PM'].

    Returns:
        metric (Metric): metric class
        comparison (Comparison): comparison class.
        dim (list of str or str): corrected dimension to apply metric to.
    """
    dim = _sanitize_to_list(dim)

    # check kind allowed
    is_in_list(kind, ["hindcast", "perfect"], "kind")

    if dim is None:  # take all dimension from initialized except lead
        dim = list(initialized.dims)
        if "lead" in dim:
            dim.remove("lead")
        # adjust dim for e2c comparison when member not in forecast or verif
        if comparison in ["e2c"] and "member" in dim:
            dim.remove("member")

    # check that initialized contains all dims from dim
    if not set(dim).issubset(initialized.dims):
        raise DimensionError(
            f"`dim`={dim} is expected to be a subset of "
            f"`initialized.dims`={initialized.dims}."
        )

    # get metric and comparison strings incorporating alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    METRICS = HINDCAST_METRICS if kind == "hindcast" else PM_METRICS
    COMPARISONS = HINDCAST_COMPARISONS if kind == "hindcast" else PM_COMPARISONS
    metric = get_metric_class(metric, METRICS)
    comparison = get_comparison_class(comparison, COMPARISONS)

    # check whether combination of metric and comparison works
    PROBABILISTIC_COMPARISONS = (
        PROBABILISTIC_HINDCAST_COMPARISONS
        if kind == "hindcast"
        else PROBABILISTIC_PM_COMPARISONS
    )
    if metric.probabilistic:
        if not comparison.probabilistic and "member" in initialized.dims:
            raise ValueError(
                f"Probabilistic metric `{metric.name}` requires comparison "
                f"accepting multiple members e.g. `{PROBABILISTIC_COMPARISONS}`, "
                f"found `{comparison.name}`."
            )
        if "member" not in dim and "member" in initialized.dims:
            raise ValueError(
                f"Probabilistic metric {metric.name} requires to be "
                f"computed over dimension `member`, which is not found in {dim}."
            )
    else:  # determinstic metric
        if kind == "hindcast":
            # for thinking in real time as compute_hindcast renames init to time
            dim = ["time" if d == "init" else d for d in dim]
    return metric, comparison, dim


def compute_perfect_model(
    initialized,
    control=None,
    metric="pearson_r",
    comparison="m2e",
    dim=["member", "init"],
    **metric_kwargs,
):
    """
    Compute a predictability skill score in a perfect-model framework.

    Args:
        initialized (xr.Dataset): ensemble with dims ``lead``, ``init``, ``member``.
        control (xr.Dataset): NOTE that this is a legacy argument from a former
            release. ``control`` is not used in ``compute_perfect_model`` anymore.
        metric (str): `metric` name, see
         :py:func:`climpred.utils.get_metric_class` and (see :ref:`Metrics`).
        comparison (str): `comparison` name defines what to take as forecast
            and verification (see
            :py:func:`climpred.utils.get_comparison_class` and :ref:`Comparisons`).
        dim (str or list of str): dimension to apply metric over.
            default: ['member', 'init']
        ** metric_kwargs (dict): additional keywords to be passed to metric.
            (see the arguments required for a given metric in metrics.py)

    Returns:
        skill (xr.Dataset): skill score with dimensions as input `ds`
                               without `dim`.

    """
    # Check that init is int, cftime, or datetime; convert ints or datetime to cftime
    initialized = convert_time_index(
        initialized, "init", "initialized[init]", calendar=PM_CALENDAR_STR
    )

    # check args compatible with each other
    metric, comparison, dim = _get_metric_comparison_dim(
        initialized, metric, comparison, dim, kind="perfect"
    )

    forecast, verif = comparison.function(initialized, metric=metric)

    if metric.normalize or metric.allows_logical:
        metric_kwargs["comparison"] = comparison
    skill = metric.function(forecast, verif, dim=dim, **metric_kwargs)
    if comparison.name == "m2m" and M2M_MEMBER_DIM in skill.dims:
        skill = skill.mean(M2M_MEMBER_DIM)
    return skill
