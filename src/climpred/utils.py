"""Utility functions used by other modules."""

import logging
import warnings
from typing import List, Union

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import to_offset

from . import comparisons, metrics
from .checks import is_in_list
from .comparisons import COMPARISON_ALIASES, Comparison
from .constants import FREQ_LIST_TO_INFER_STRIDE, HINDCAST_CALENDAR_STR
from .exceptions import CoordinateError
from .metrics import ALL_METRICS, METRIC_ALIASES, Metric
from .options import OPTIONS


def add_attrs_to_climpred_coords(results):
    """Write attrs for coords added by climpred."""
    try:
        from . import __version__ as version

        version = f"v{version}"
    except ImportError:
        version = "stable"

    if "results" in results.coords:
        results["results"] = results["results"].assign_attrs(
            {
                "description": "new coordinate created by .bootstrap()",
                "verify skill": "skill from verify",
                "p": "probability that reference performs better than initialized",
                "low_ci": "lower confidence interval threshold based on resampling with replacement",  # noqa: E501
                "high_ci": "higher confidence interval threshold based on resampling with replacement",  # noqa: E501
            }
        )
    if "skill" in results.coords:
        results["skill"] = results["skill"].assign_attrs(
            {
                "description": "new dimension prediction skill of initialized and reference forecasts created by .verify() or .bootstrap()",  # noqa: E501
                "documentation": f"https://climpred.readthedocs.io/en/{version}/reference_forecast.html",  # noqa: E501
            }
        )
    if "skill" in results.dims:
        results["skill"] = results["skill"].assign_attrs(
            {f: f"{f} forecast skill" for f in results.skill.values}
        )
    return results


def assign_attrs(
    skill,
    ds,
    function_name=None,
    alignment=None,
    reference=None,
    metric=None,
    comparison=None,
    dim=None,
    **kwargs,
):
    r"""Write information about prediction skill into attrs.

    Args:
        skill (`xarray` object): prediction skill.
        ds (`xarray` object): prediction ensemble with inits.
        function_name (str): name of compute function
        alignment (str): method used to align inits and verification data.
        reference (str): reference forecasts
        metric (class) : metric used in comparing the forecast and verification data.
        comparison (class): how to compare the forecast and verification data.
        dim (str): Dimension over which metric was applied.
        \*\*kwargs : other information

    Returns:
       skill (`xarray` object): prediction skill with additional attrs.
    """
    # assign old attrs
    skill.attrs = ds.attrs
    for v in skill.data_vars:
        skill[v].attrs.update(ds[v].attrs)

    # climpred info
    skill.attrs["prediction_skill_software"] = (
        "climpred https://climpred.readthedocs.io/"
    )
    if function_name:
        skill.attrs["skill_calculated_by_function"] = function_name
    if "init" in ds.coords and "init" not in skill.dims:
        skill.attrs["number_of_initializations"] = ds.init.size
    if "member" in ds.coords and "member" not in skill.coords:
        skill.attrs["number_of_members"] = ds.member.size
    if alignment is not None:
        skill.attrs["alignment"] = alignment

    metric = METRIC_ALIASES.get(metric, metric)
    metric = get_metric_class(metric, ALL_METRICS)
    skill.attrs["metric"] = metric.name
    if comparison is not None:
        skill.attrs["comparison"] = comparison
    if dim is not None:
        if isinstance(dim, list):
            if len(dim) == 1:
                dim = dim[0]
        skill.attrs["dim"] = dim
    if reference is not None:
        skill.attrs["reference"] = reference
    if "persistence" in reference and "PerfectModelEnsemble" in function_name:
        skill.attrs["PerfectModel_persistence_from_initialized_lead_0"] = OPTIONS[
            "PerfectModel_persistence_from_initialized_lead_0"
        ]

    # change unit power in all variables
    if metric.unit_power == 0:
        for v in skill.data_vars:
            skill[v].attrs["units"] = "None"
    elif metric.unit_power >= 2:
        for v in skill.data_vars:
            if "units" in skill[v].attrs:
                p = metric.unit_power
                p = int(p) if int(p) == p else p
                skill[v].attrs["units"] = f"({skill[v].attrs['units']})^{p}"

    if "logical" in kwargs:
        kwargs["logical"] = "Callable"

    from .bootstrap import _p_ci_from_sig

    if "sig" in kwargs:
        if kwargs["sig"] is not None:
            _, ci_low, ci_high = _p_ci_from_sig(kwargs["sig"])
            kwargs["confidence_interval_levels"] = f"{ci_high}-{ci_low}"
    if "pers_sig" in kwargs:
        if kwargs["pers_sig"] is not None:
            _, ci_low_pers, ci_high_pers = _p_ci_from_sig(kwargs["pers_sig"])
            kwargs["confidence_interval_levels_persistence"] = (
                f"{ci_high_pers}-{ci_low_pers}"
            )
    # check for none attrs and remove
    del_list = []
    for key, value in kwargs.items():
        if value is None:
            del_list.append(key)
    for entry in del_list:
        del kwargs[entry]
    # write optional information
    skill.attrs.update(kwargs)

    skill = add_attrs_to_climpred_coords(skill)
    return skill


def convert_time_index(
    xobj, time_string, kind="object", calendar=HINDCAST_CALENDAR_STR
):
    """Convert incoming time index to a :py:class:`~xarray.CFTimeIndex`.

    Args:
        xobj (xarray.Dataset): with a time dimension to convert.
        time_string (str): Name of time dimension.
        kind (str): Kind of object for error message.
        calendar (str): calendar to set time dimension to.

    Returns:
        Dataset or DataArray with converted time dimension. If incoming time index is
        ``xr.CFTimeIndex``, returns the same index. If ``pd.DatetimeIndex``, converts to
        ``cftime.ProlepticGregorian``. If ``pd.Int64Index`` or ``pd.Float64Index``,
        assumes annual resolution and returns year-start ``cftime.ProlepticGregorian``.

    Raises:
        ValueError: If ``time_index`` is not an ``xr.CFTimeIndex``, ``pd.Int64Index``,
            ``pd.Float64Index``, or ``pd.DatetimeIndex``.
    """
    xobj = xobj.copy()  # Ensures that main object index is not overwritten.
    time_index = xobj[time_string].to_index()

    if not isinstance(time_index, xr.CFTimeIndex):
        if isinstance(time_index, pd.DatetimeIndex):
            # Extract year, month, day strings from datetime.
            time_strings = [str(t) for t in time_index]
            split_dates = [d.split(" ")[0].split("-") for d in time_strings]
            split_dates = [
                d.split("-") for d in time_index.strftime("%Y-%m-%d-%H-%M-%S")
            ]

        # If pd.Index, assume annual and convert accordingly.
        elif isinstance(time_index, pd.Index):
            if OPTIONS["warn_for_init_coords_int_to_annual"]:
                warnings.warn(
                    "Assuming annual resolution starting Jan 1st due to numeric inits. "
                    "Please change ``init`` to a datetime if it is another resolution. "
                    "We recommend using xr.CFTimeIndex as ``init``, see "
                    "https://climpred.readthedocs.io/en/stable/setting-up-data.html."
                )
            # TODO: What about decimal time? E.g. someone has 1955.5 or something?
            # hard to maintain a clear rule below seasonality
            dates = [str(int(t)) + "-01-01-00-00-00" for t in time_index]
            split_dates = [d.split("-") for d in dates]
            if "lead" in xobj.dims:
                # Probably the only case we can assume lead units, since `lead` does not
                # give us any information on this.
                xobj["lead"].attrs["units"] = "years"

        else:
            raise ValueError(
                f"Your {kind} object must be pd.Float64Index, "
                "pd.Int64Index, xr.CFTimeIndex or "
                "pd.DatetimeIndex."
            )
        cftime_dates = [
            getattr(cftime, calendar)(int(y), int(m), int(d), int(H), int(M), int(S))
            for (y, m, d, H, M, S) in split_dates
        ]
        time_index = xr.CFTimeIndex(cftime_dates)
        xobj[time_string] = time_index
        if time_string == "time":
            xobj["time"].attrs.setdefault("long_name", "time")
            xobj["time"].attrs.setdefault("standard_name", "time")

    return xobj


def convert_cftime_to_datetime_coords(ds, dim):
    """Convert dimension coordinate dim from CFTimeIndex to pd.DatetimeIndex."""
    return ds.assign_coords(
        {dim: xr.DataArray(ds[dim].to_index().to_datetimeindex(), dims=dim)}
    )


def convert_init_lead_to_valid_time_lead(
    skill: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """Convert ``data(init,lead)`` to ``data(valid_time,lead)`` visualizing predict barrier.

    Args:
        skill with dimensions init and lead and coordinate valid_time(init, lead).

    Returns:
        skill with dimensions valid_time and lead

    Examples:
        Calculate skill at each ``init``, i.e. do not reduce ``init`` and set ``dim=[]``.

        >>> skill_init_lead = HindcastEnsemble.sel(
        ...     lead=[1, 2, 3], init=slice("1990", "2000")
        ... ).verify(metric="rmse", comparison="e2o", dim=[], alignment="same_verifs")
        >>> skill_init_lead.SST
        <xarray.DataArray 'SST' (lead: 3, init: 11)>
        array([[       nan,        nan, 0.07668081, 0.06826989, 0.08174487,
                0.06208846, 0.1537402 , 0.15632479, 0.01302786, 0.06343324,
                0.13758603],
               [       nan, 0.07732193, 0.06369554, 0.08282175, 0.0761979 ,
                0.20424354, 0.18043845, 0.06553673, 0.00906034, 0.13045045,
                       nan],
               [0.06212777, 0.11822992, 0.15282457, 0.05752934, 0.20133476,
                0.19931679, 0.00987793, 0.06375334, 0.07705835,        nan,
                       nan]])
        Coordinates:
          * init        (init) object 1990-01-01 00:00:00 ... 2000-01-01 00:00:00
          * lead        (lead) int32 1 2 3
            valid_time  (lead, init) object 1991-01-01 00:00:00 ... 2003-01-01 00:00:00
            skill       <U11 'initialized'
        Attributes:
            units:    C
        >>> climpred.utils.convert_init_lead_to_valid_time_lead(skill_init_lead).SST
        <xarray.DataArray 'SST' (lead: 3, valid_time: 9)>
        array([[0.07668081, 0.06826989, 0.08174487, 0.06208846, 0.1537402 ,
                0.15632479, 0.01302786, 0.06343324, 0.13758603],
               [0.07732193, 0.06369554, 0.08282175, 0.0761979 , 0.20424354,
                0.18043845, 0.06553673, 0.00906034, 0.13045045],
               [0.06212777, 0.11822992, 0.15282457, 0.05752934, 0.20133476,
                0.19931679, 0.00987793, 0.06375334, 0.07705835]])
        Coordinates:
          * valid_time  (valid_time) object 1993-01-01 00:00:00 ... 2001-01-01 00:00:00
          * lead        (lead) int32 1 2 3
            skill       <U11 'initialized'
            init        (lead, valid_time) object 1992-01-01 00:00:00 ... 1998-01-01 ...
        Attributes:
            units:    C

    See also:
        https://github.com/pydata/xarray/discussions/6943
        :py:func:`climpred.utils.convert_valid_time_lead_to_init_lead`
    """
    # ensure valid_time 2d
    assert "valid_time" in skill.coords
    assert len(skill.coords["valid_time"].dims) == 2
    swapped = xr.concat(
        [skill.sel(lead=lead).swap_dims({"init": "valid_time"}) for lead in skill.lead],
        "lead",
    )
    return add_init_from_time_lead(swapped.drop_vars("init")).dropna(
        "valid_time", how="all"
    )


def convert_valid_time_lead_to_init_lead(
    skill: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """Convert ``data(valid_time,lead)`` to ``data(init,lead)``.
    Args:
        skill with dimensions valid_time and lead and coordinate init(valid_time, lead).
    Returns:
        skill with dimensions init and lead
    Examples:
        Calculate skill at each ``init``, i.e. do not reduce ``init`` and set ``dim=[]``.
        >>> skill_init_lead = HindcastEnsemble.sel(
        ...     lead=[1, 2, 3], init=slice("1990", "2000")
        ... ).verify(metric="rmse", comparison="e2o", dim=[], alignment="same_verifs")
        >>> skill_init_lead.SST
        <xarray.DataArray 'SST' (lead: 3, init: 11)>
        array([[       nan,        nan, 0.07668081, 0.06826989, 0.08174487,
                0.06208846, 0.1537402 , 0.15632479, 0.01302786, 0.06343324,
                0.13758603],
               [       nan, 0.07732193, 0.06369554, 0.08282175, 0.0761979 ,
                0.20424354, 0.18043845, 0.06553673, 0.00906034, 0.13045045,
                       nan],
               [0.06212777, 0.11822992, 0.15282457, 0.05752934, 0.20133476,
                0.19931679, 0.00987793, 0.06375334, 0.07705835,        nan,
                       nan]])
        Coordinates:
          * init        (init) object 1990-01-01 00:00:00 ... 2000-01-01 00:00:00
          * lead        (lead) int32 1 2 3
            valid_time  (lead, init) object 1991-01-01 00:00:00 ... 2003-01-01 00:00:00
            skill       <U11 'initialized'
        Attributes:
            units:    C
        >>> assert climpred.utils.convert_valid_time_lead_to_init_lead(
        ...     climpred.utils.convert_init_lead_to_valid_time_lead(skill_init_lead)
        ... ).equals(skill_init_lead)

    See also:
        https://github.com/pydata/xarray/discussions/6943
        :py:func:`climpred.utils.convert_init_lead_to_valid_time_lead`
    """
    # ensure init 2d
    assert "init" in skill.coords
    assert len(skill.coords["init"].dims) == 2
    swapped = xr.concat(
        [skill.sel(lead=lead).swap_dims({"valid_time": "init"}) for lead in skill.lead],
        "lead",
    )
    return add_time_from_init_lead(swapped.drop_vars("valid_time")).dropna(
        "init", how="all"
    )


def find_start_dates_for_given_init(control, single_init):
    """
    Find same start dates for cftime single_init in different years in control.

    Return control.time. Requires calendar=Datetime(No)Leap for consistent
    `dayofyear`.
    """
    # check that Leap or NoLeap calendar
    for dim in [single_init.init, control.time]:
        # dirty workaround .values requires a dimension but single_init is only a
        # single initialization and therefore without init dim
        dim = dim.expand_dims("init") if "time" not in dim.coords else dim
        calendar = type(dim.values[0]).__name__
        if "Leap" not in calendar:
            warnings.warn(
                f"inputs to `find_start_dates_for_given_init` should be `Leap` "
                f" or `NoLeap` calendar, found {calendar} in {dim}."
                f" Otherwise dayofyear is not static and can lead to slight shifts."
            )
    # could also just take first of month or even a random number day in month
    take_same_time = "dayofyear"
    return control.sel(
        time=getattr(control.time.dt, take_same_time).values
        == getattr(single_init.init.dt, take_same_time).values
    ).time


def return_time_series_freq(ds, dim):
    """Return the temporal frequency of the input time series.

    Finds the frequency starting from high frequencies at which all ds.dim are
    not equal.

    Args:
        ds (xr.Dataset): input with dimension `dim`.
        dim (str): name of dimension.

    Returns:
        str: frequency string from FREQ_LIST_TO_INFER_STRIDE

    """
    for freq in FREQ_LIST_TO_INFER_STRIDE:
        # first dim values not equal all others
        if not (
            getattr(ds.isel({dim: 0})[dim].dt, freq) == getattr(ds[dim].dt, freq)
        ).all():
            return freq


def get_metric_class(metric: Union[str, Metric], list_: List) -> Metric:
    """
    Convert string representing the desired metric to corresponding metric class.

    Args:
        metric: name of metric.
        list_: check whether metric in list

    Returns:
        class object of the metric.
    """
    if isinstance(metric, metrics.Metric):
        return metric
    elif isinstance(metric, str):
        # check if metric allowed
        is_in_list(metric, list_, "metric")
        metric = METRIC_ALIASES.get(metric, metric)
        return getattr(metrics, f"__{metric}")
    else:
        raise ValueError(
            f"Please provide metric as str or Metric class, found {type(metric)}"
        )


def get_comparison_class(comparison: Union[str, Comparison], list_: List) -> Comparison:
    """
    Convert string comparison entry into a Comparison class.

    Perfect Model:

        * m2m: Compare all members to all other members.
        * m2c: Compare all members to the control.
        * m2e: Compare all members to the ensemble mean.
        * e2c: Compare the ensemble mean to the control.

    Hindcast:

        * e2o: Compare the ensemble mean to the verification data.
        * m2o: Compare each ensemble member to the verification data.

    Args:
        comparison: name of comparison.

    Returns:
        comparison: comparison class.

    """
    if isinstance(comparison, comparisons.Comparison):
        return comparison
    elif isinstance(comparison, str):
        # check if comparison allowed
        is_in_list(comparison, list_, "comparison")
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        return getattr(comparisons, f"__{comparison}")
    else:
        is_in_list(comparison, list_, "comparison")
        return getattr(comparisons, f"__{comparison}")


def get_lead_cftime_shift_args(units, lead):
    """
    Determine date increment when adding lead to init based on units attribute.

    Args:
        units (str): Units associated with the lead dimension. Must be
            years, seasons, months, weeks, pentads, days, hours, minutes.
        lead (int): Increment of lead being computed.

    Returns:
       n (int): Number of units to shift. ``value`` for
           ``CFTimeIndex.shift(value, str)``.
       freq (str): Pandas frequency alias. ``str`` for
           ``CFTimeIndex.shift(value, str)``.
    """
    lead = int(lead)

    d = {
        # Currently assumes yearly aligns with year start.
        "years": (lead, "YS"),
        "seasons": (lead * 3, "MS"),
        # Currently assumes monthly aligns with month start.
        "months": (lead, "MS"),
        "weeks": (lead * 7, "D"),
        "pentads": (lead * 5, "D"),
        "days": (lead, "D"),
        "hours": (lead, "H"),
        "minutes": (lead, "T"),
        "seconds": (lead, "S"),
    }

    try:
        n, freq = d[units]
    except KeyError:
        print(f"{units} is not a valid choice.")
        print(f"Accepted `units` values include: {d.keys()}")
    return n, freq


def get_multiple_lead_cftime_shift_args(units, leads):
    """Return ``CFTimeIndex.shift()`` offset for arbitrary number of leads.

    Args:
        units (str): Units associated with the lead dimension. Must be one of
            years, seasons, months, weeks, pentads, days, hours, minutes.
        leads (list, array, xr.DataArray of ints): Leads to return offset for.

    Returns:
        n (tuple of ints): Numbers of units to shift for ``leads``. ``value`` for
            ``CFTimeIndex.shift(value, str)``.
        freq (str): Pandas frequency alias. ``str`` for
            ``CFTimeIndex.shift(value, str)``.
    """
    n_freq_tuples = [get_lead_cftime_shift_args(units, lead) for lead in leads]
    n, freq = list(zip(*n_freq_tuples))
    return n, freq[0]


def intersect(lst1, lst2):
    """Return custom intersection as `set.intersection()` changes type."""
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def lead_units_equal_control_time_stride(init, verif):
    """
    Check that lead units of initialized has same frequency as control stride.

    Args:
        init (xr.Dataset): initialized ensemble with lead units.
        verif (xr.Dataset): control, uninitialized historical simulation / observations.

    Returns:
        bool: Possible to continue or raise warning.

    """
    verif_time_stride = return_time_series_freq(verif, "time")
    lead_units = init.lead.attrs["units"].strip("s")
    if verif_time_stride != lead_units:
        raise ValueError(
            "Please provide the same temporal resolution for verif.time",
            f"(found {verif_time_stride}) and init.init (found",
            f"{lead_units}).",
        )
    else:
        return True


def _load_into_memory(res):
    """Compute if res is lazy data."""
    if dask.is_dask_collection(res):
        res = res.compute()
    return res


def rechunk_to_single_chunk_if_more_than_one_chunk_along_dim(ds, dim):
    """Rechunk an xarray object more than one chunk along dim."""
    if dask.is_dask_collection(ds) and dim in ds.chunks:
        if isinstance(ds, xr.Dataset):
            nchunks = len(ds.chunks[dim])
        elif isinstance(ds, xr.DataArray):
            nchunks = len(ds.chunks[ds.get_axis_num(dim)])
        if nchunks > 1:
            ds = ds.chunk({dim: -1})
    return ds


def shift_cftime_index(xobj, time_string, n, freq):
    """
    Shift a ``CFTimeIndex`` over n time steps at a given temporal frequency.

    This leverages the handy ``.shift()`` method from ``xarray.CFTimeIndex``. It's a
    simple call, but is used throughout ``climpred`` so it is documented here clearly
    for convenience.

    Args:
        xobj (xarray object): Dataset or DataArray with the ``CFTimeIndex`` to shift.
        time_string (str): Name of time dimension to be shifted.
        n (int): Number of units to shift.
            Returned from :py:func:`get_lead_cftime_shift_args`.
        freq (str): Pandas frequency alias.
            Returned from :py:func:`get_lead_cftime_shift_args`.

    Returns:
        ``CFTimeIndex`` shifted by ``n`` steps at time frequency ``freq``.
    """
    time_index = xobj[time_string].to_index()
    return time_index.shift(n, freq)


def shift_cftime_singular(cftime, n, freq):
    """Shifts a singular ``cftime`` by the desired frequency.

    This directly pulls the ``shift`` method from ``CFTimeIndex`` in ``xarray``. This
    is useful if you need to shift a singular ``cftime`` by some offset, but are not
    working with a full ``CFTimeIndex``.

    Args:
        cftime (``cftime``): ``cftime`` object to shift.
        n (int): Number of steps to shift by.
        freq (str): Frequency string, per ``pandas`` convention.

    See:
    https://github.com/pydata/xarray/blob/master/xarray/coding/cftimeindex.py#L376.
    """
    if not isinstance(n, int):
        raise TypeError(f"'n' must be an int, got {n}.")
    if not isinstance(freq, str):
        raise TypeError(f"'freq' must be a str, got {freq}.")
    return cftime + n * to_offset(freq)


def _transpose_and_rechunk_to(new_chunk_ds, ori_chunk_ds):
    """
    Ensure same chunks and dimension order.

    This is needed after some operations which reduce chunks to size 1.
    First transpose a to ds.dims then apply ds chunking to a.
    """
    return new_chunk_ds.transpose(*ori_chunk_ds.dims).chunk(ori_chunk_ds.chunks)


def convert_Timedelta_to_lead_units(ds):
    """
    Convert lead as pd.Timedelta to int and corresponding lead.attrs['units'].

    Converts to longest integer lead unit possible.
    """
    if ds["lead"].dtype == "<m8[ns]":
        ds["lead"] = (ds.lead * 1e-9).astype(int)
        ds["lead"].attrs["units"] = "seconds"

    if (ds["lead"] % 60 == 0).all() and ds["lead"].attrs["units"] == "seconds":
        ds["lead"] = ds["lead"] / 60
        ds["lead"].attrs["units"] = "minutes"
    if (ds["lead"] % 60 == 0).all() and ds["lead"].attrs["units"] == "minutes":
        ds["lead"] = ds["lead"] / 60
        ds["lead"].attrs["units"] = "hours"
    if (ds["lead"] % 24 == 0).all() and ds["lead"].attrs["units"] == "hours":
        ds["lead"] = ds["lead"] / 24
        ds["lead"].attrs["units"] = "days"
    if (ds["lead"] % 5 == 0).all() and ds["lead"].attrs["units"] == "days":
        ds["lead"] = ds["lead"] / 5
        ds["lead"].attrs["units"] = "pentads"
    if (ds["lead"] % 7 == 0).all() and ds["lead"].attrs["units"] == "days":
        ds["lead"] = ds["lead"] / 7
        ds["lead"].attrs["units"] = "weeks"
    return ds


def broadcast_time_grouped_to_time(forecast, category_edges, dim):
    """
    Help function for rps metric.

    Broadcast time.groupby('time.month/dayofyear/weekofyear').mean() from
    category_edges back to dim matching forecast.
    """
    category_edges_time_dim = [
        d
        for d in category_edges.dims
        if d in ["season", "month", "weekofyear", "dayofyear"]
    ]
    if isinstance(category_edges_time_dim, list):
        if len(category_edges_time_dim) > 0:
            category_edges_time_dim = category_edges_time_dim[0]
            logging.debug(f"found category_edges_time_dim = {category_edges_time_dim}")
            category_edges = category_edges.sel(
                {
                    category_edges_time_dim: getattr(
                        forecast[dim].dt, category_edges_time_dim
                    )
                }
            )
    return category_edges


def broadcast_metric_kwargs_for_rps(forecast, verif, metric_kwargs):
    """
    Help function for rps metric.

    Apply broadcast_time_grouped_to_time to category_edges in metric_kwargs.
    """
    category_edges = metric_kwargs.get("category_edges", None)
    logging.debug("enter climpred.utils.broadcast_metric_kwargs_for_rps")
    if category_edges is not None:
        if isinstance(category_edges, tuple):
            logging.debug("category_edges is tuple")
            verif_edges = category_edges[0]
            verif_edges = broadcast_time_grouped_to_time(verif, verif_edges, dim="time")
            forecast_edges = category_edges[1]
            forecast_edges = broadcast_time_grouped_to_time(
                forecast, forecast_edges, dim="init"
            )
            metric_kwargs["category_edges"] = (verif_edges, forecast_edges)
        elif isinstance(category_edges, xr.Dataset):
            logging.debug("category_edges is xr.Dataset")
            metric_kwargs["category_edges"] = broadcast_time_grouped_to_time(
                verif, category_edges, dim="time"
            )
        elif isinstance(category_edges, np.ndarray):
            logging.debug("category_edges is np.array")
        else:
            raise ValueError(
                "excepted category edges as tuple, xr.Dataset or np.array, "
                f"found {type(category_edges)}"
            )
        return metric_kwargs


def my_shift(dim, other_dim="lead", operator="add"):
    """operator(dim,other_dim) adds/subtracts lead to/from time (init/valid_time)."""
    assert operator in ["add", "subtract"]

    if isinstance(dim, xr.DataArray):
        dim = dim.to_index()
    dim_calendar = dim.calendar
    if isinstance(other_dim, xr.DataArray):
        other_dim_unit = other_dim.attrs["units"]
        other_dim = other_dim.values

    if other_dim_unit in ["years", "seasons", "months"] and "360" not in dim_calendar:
        if int(other_dim) != float(other_dim):
            raise CoordinateError(
                f'Require integer leads if lead.attrs["units"]="{other_dim_unit}" in '
                f'["years", "seasons", "months"] and calendar="{dim_calendar}" '
                'not "360_day".'
            )
        other_dim = int(other_dim)

    if "360" in dim_calendar:  # use pd.Timedelta
        if other_dim_unit == "years":
            other_dim = other_dim * 360
            other_dim_unit = "D"
        elif other_dim_unit == "seasons":
            other_dim = other_dim * 90
            other_dim_unit = "D"
        elif other_dim_unit == "months":
            other_dim_unit = "D"
            other_dim = other_dim * 30

    if other_dim_unit in ["years", "seasons", "months"]:
        # use init_freq reconstructed from anchor and lead unit
        from xarray.coding.frequencies import month_anchor_check

        anchor_check = month_anchor_check(dim)  # returns None, ce or cs
        if anchor_check is not None:
            other_dim_freq_string = other_dim_unit[0].upper()  # A for years, D for days
            # go down to monthly freq
            if other_dim_freq_string == "Y":
                other_dim_freq_string = "12M"
            elif other_dim_freq_string == "S":
                other_dim_freq_string = "3M"
            anchor = anchor_check[-1].upper()  # S/E for start/end of month
            if anchor == "E":
                anchor = ""
            other_dim_freq = f"{other_dim_freq_string}{anchor}"
            if other_dim_freq_string in ["A", "Q"]:  # add month info again
                dim_freq = xr.infer_freq(dim)
                if dim_freq:
                    if "-" in dim_freq:
                        other_dim_freq = other_dim_freq + "-" + dim_freq.split("-")[-1]
        else:
            raise ValueError(
                f"could not shift dim={dim} in calendar={dim_calendar} by "
                f" other_dim={other_dim} {other_dim_unit}"
            )
        if operator == "subtract":
            other_dim = other_dim * -1
        return dim.shift(other_dim, other_dim_freq)
    else:  # lower freq
        # reducing pentads, weeks (W) to days
        if other_dim_unit == "weeks":
            other_dim_unit = "W"
        elif other_dim_unit == "pentads":
            other_dim = other_dim * 5
            other_dim_unit = "D"
        if operator == "subtract":
            other_dim = other_dim * -1
        return dim + pd.Timedelta(float(other_dim), other_dim_unit)


def _add_coord2d_from_coords1d(
    ds, lead_dim="lead", init_dim="init", time_dim="valid_time", operator="add"
):
    """Add valid_time = init + lead to ds coords."""
    if "time" not in ds.dims:
        new_time = xr.concat(
            [
                xr.DataArray(
                    my_shift(ds[init_dim], lead, operator=operator),
                    dims=init_dim,
                    coords={init_dim: ds[init_dim]},
                )
                for lead in ds[lead_dim]
            ],
            dim=lead_dim,
            join="inner",
            compat="broadcast_equals",
        )
        new_time[lead_dim] = ds[lead_dim]
        ds = ds.copy()  # otherwise inplace coords setting
        if dask.is_dask_collection(new_time):
            new_time = new_time.compute()
        ds = ds.assign_coords({time_dim: new_time})
    return ds


def add_time_from_init_lead(
    ds, lead_dim="lead", init_dim="init", time_dim="valid_time", operator="add"
):
    """Add init(valid_time, lead) from valid_time and lead."""
    ds = _add_coord2d_from_coords1d(
        ds, operator="add", init_dim="init", lead_dim="lead", time_dim="valid_time"
    )
    ds.coords["valid_time"].attrs.update(
        {
            "long_name": "validity time",
            "standard_name": "time",
            "description": "time for which the forecast is valid",
            "calculate": "init + lead",
        }
    )
    return ds


def add_init_from_time_lead(ds):
    """Add init(valid_time, lead) from valid_time and lead."""
    ds = _add_coord2d_from_coords1d(
        ds, operator="subtract", init_dim="valid_time", lead_dim="lead", time_dim="init"
    )
    ds.coords["init"].attrs.update(
        {
            "long_name": "forecast_reference_time",
            "standard_name": "initialization",
            "description": "The forecast reference time in NWP is the 'data time', "
            "the time of the analysis from which the forecast was made. It is not the "
            "time for which the forecast is valid; the standard name of time should be "
            "used for that time.",
            "calculate": "valid_time - lead",
        }
    )
    return ds
