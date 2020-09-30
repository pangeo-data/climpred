import inspect

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
from .exceptions import DimensionError
from .logging import log_compute_hindcast_header, log_compute_hindcast_inits_and_verifs
from .metrics import HINDCAST_METRICS, METRIC_ALIASES, PM_METRICS
from .reference import persistence, uninitialized
from .utils import (
    assign_attrs,
    convert_time_index,
    copy_coords_from_to,
    get_comparison_class,
    get_metric_class,
)


def _apply_metric_at_given_lead(
    verif,
    verif_dates,
    lead,
    hind=None,
    hist=None,
    inits=None,
    reference=None,
    metric=None,
    comparison=None,
    dim=None,
    **metric_kwargs,
):
    """Applies a metric between two time series at a given lead.

    .. note::

        This will be moved to a method of the `Scoring()` class in the next PR.

    Args:
        verif (xr object): Verification data.
        verif_dates (dict): Lead-dependent verification dates for alignment.
        lead (int): Given lead to score.
        hind (xr object): Initialized hindcast. Not required in a persistence forecast.
        hist (xr object): Uninitialized/historical simulation. Required when
            ``reference='uninitialized'``.
        inits (dict): Lead-dependent initialization dates for alignment.
        reference (str): If not ``None``, return score for this reference forecast.
            * 'persistence'
            * 'uninitialized'
        metric (Metric): Metric class for scoring.
        comparison (Comparison): Comparison class.
        dim (str): Dimension to apply metric over.

    Returns:
        result (xr object): Metric results for the given lead for the initialized
            forecast or reference forecast.
    """
    if reference is None:
        # Use `.where()` instead of `.sel()` to account for resampled inits when
        # bootstrapping.
        a = (
            hind.sel(lead=lead)
            .where(hind["time"].isin(inits[lead]), drop=True)
            .drop_vars("lead")
        )
        b = verif.sel(time=verif_dates[lead])
    elif reference == "persistence":
        a, b = persistence(verif, inits, verif_dates, lead)
    elif reference == "uninitialized":
        a, b = uninitialized(hist, verif, verif_dates, lead)
    a["time"] = b["time"]

    dim = _rename_dim(dim, hind, verif)
    if metric.normalize or metric.allows_logical:
        metric_kwargs["comparison"] = comparison
    result = metric.function(a, b, dim=dim, **metric_kwargs)
    log_compute_hindcast_inits_and_verifs(dim, lead, inits, verif_dates)
    return result


def _rename_dim(dim, forecast, verif):
    """rename `dim` to `time` or `init` if forecast and verif dims require."""
    if "init" in dim and "time" in forecast.dims and "time" in verif.dims:
        dim = dim.copy()
        dim.remove("init")
        dim = dim + ["time"]
    elif "time" in dim and "init" in forecast.dims and "init" in verif.dims:
        dim = dim.copy()
        dim.remove("time")
        dim = dim + ["init"]
    return dim


def _sanitize_str_to_list(dim):
    """Make dim to list if string, pass if None else raise ValueError."""
    if isinstance(dim, str):
        dim = [dim]
    elif isinstance(dim, list) or dim is None:
        dim = dim
    else:
        raise ValueError(
            f"Expected `dim` as `str`, `list` or None, found {dim} as type {type(dim)}."
        )
    return dim


def _get_metric_comparison_dim(initialized, metric, comparison, dim, kind):
    """Returns `metric`, `comparison` and `dim` for compute functions.

    Args:
        initialized (xr.object): initialized dataset: init_pm or hind
        metric (str): metric or alias string
        comparison (str): Description of parameter `comparison`.
        dim (list of str or str): dimension to apply metric to.
        kind (str): experiment type from ['hindcast', 'PM'].

    Returns:
        metric (Metric): metric class
        comparison (Comparison): comparison class.
        dim (list of str or str): corrected dimension to apply metric to.
    """
    dim = _sanitize_str_to_list(dim)

    # check kind allowed
    is_in_list(kind, ["hindcast", "PM"], "kind")

    if dim is None:  # take all dimension from initialized except lead
        dim = list(initialized.dims)
        if "lead" in dim:
            dim.remove("lead")

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


@is_xarray([0])
def compute_perfect_model(
    init_pm,
    control=None,
    metric="pearson_r",
    comparison="m2e",
    dim=["member", "init"],
    add_attrs=True,
    **metric_kwargs,
):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Args:
        init_pm (xarray object): ensemble with dims ``lead``, ``init``, ``member``.
        control (xarray object): NOTE that this is a legacy argument from a former
            release. ``control`` is not used in ``compute_perfect_model`` anymore.
        metric (str): `metric` name, see
         :py:func:`climpred.utils.get_metric_class` and (see :ref:`Metrics`).
        comparison (str): `comparison` name defines what to take as forecast
            and verification (see
            :py:func:`climpred.utils.get_comparison_class` and :ref:`Comparisons`).
        dim (str or list of str): dimension to apply metric over.
            default: ['member', 'init']
        add_attrs (bool): write climpred compute args to attrs. default: True
        ** metric_kwargs (dict): additional keywords to be passed to metric.
            (see the arguments required for a given metric in metrics.py)

    Returns:
        skill (xarray object): skill score with dimensions as input `ds`
                               without `dim`.

    """
    # Check that init is int, cftime, or datetime; convert ints or datetime to cftime
    init_pm = convert_time_index(
        init_pm, "init", "init_pm[init]", calendar=PM_CALENDAR_STR
    )

    # check args compatible with each other
    metric, comparison, dim = _get_metric_comparison_dim(
        init_pm, metric, comparison, dim, kind="PM"
    )

    forecast, verif = comparison.function(init_pm, metric=metric)

    if metric.normalize or metric.allows_logical:
        metric_kwargs["comparison"] = comparison
    skill = metric.function(forecast, verif, dim=dim, **metric_kwargs)
    if comparison.name == "m2m" and M2M_MEMBER_DIM in skill.dims:
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
    metric="pearson_r",
    comparison="e2o",
    dim="init",
    alignment="same_verifs",
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
    metric, comparison, dim = _get_metric_comparison_dim(
        hind, metric, comparison, dim, kind="hindcast"
    )
    hind = convert_time_index(hind, "init", "hind[init]")
    verif = convert_time_index(verif, "time", "verif[time]")
    has_valid_lead_units(hind)

    forecast, verif = comparison.function(hind, verif, metric=metric)

    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({"init": "time"})

    inits, verif_dates = return_inits_and_verif_dates(
        forecast, verif, alignment=alignment
    )

    log_compute_hindcast_header(metric, comparison, dim, alignment)

    metric_over_leads = [
        _apply_metric_at_given_lead(
            verif,
            verif_dates,
            lead,
            hind=forecast,
            inits=inits,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        for lead in forecast["lead"].data
    ]
    result = xr.concat(metric_over_leads, dim="lead", **CONCAT_KWARGS)
    result["lead"] = forecast["lead"]
    # rename back to 'init'
    if "time" in result.dims:
        result = result.rename({"time": "init"})
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
