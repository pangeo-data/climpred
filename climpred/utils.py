import datetime
import logging
import warnings

import cftime
import dask
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import to_offset

from . import comparisons, metrics
from .checks import is_in_list
from .comparisons import COMPARISON_ALIASES
from .constants import FREQ_LIST_TO_INFER_STRIDE, HINDCAST_CALENDAR_STR
from .metrics import METRIC_ALIASES
from .options import OPTIONS


def assign_attrs(
    skill,
    ds,
    function_name=None,
    metadata_dict=None,
    alignment=None,
    metric=None,
    comparison=None,
    dim=None,
):
    """Write information about prediction skill into attrs.

    Args:
        skill (`xarray` object): prediction skill.
        ds (`xarray` object): prediction ensemble with inits.
        function_name (str): name of compute function
        metadata_dict (dict): optional attrs
        alignment (str): method used to align inits and verification data.
        metric (class) : metric used in comparing the forecast and verification data.
        comparison (class): how to compare the forecast and verification data.
        dim (str): Dimension over which metric was applied.

    Returns:
       skill (`xarray` object): prediction skill with additional attrs.
    """
    # assign old attrs
    skill.attrs = ds.attrs

    # climpred info
    skill.attrs[
        "prediction_skill"
    ] = "calculated by climpred https://climpred.readthedocs.io/"
    if function_name:
        skill.attrs["skill_calculated_by_function"] = function_name
    if "init" in ds.coords:
        skill.attrs["number_of_initializations"] = ds.init.size
    if "member" in ds.coords and function_name != "compute_persistence":
        skill.attrs["number_of_members"] = ds.member.size

    if alignment is not None:
        skill.attrs["alignment"] = alignment
    skill.attrs["metric"] = metric.name
    if comparison is not None:
        skill.attrs["comparison"] = comparison.name
    if dim is not None:
        skill.attrs["dim"] = dim

    # change unit power
    if metric.unit_power == 0:
        skill.attrs["units"] = "None"
    if metric.unit_power >= 2 and "units" in skill.attrs:
        p = metric.unit_power
        p = int(p) if int(p) == p else p
        skill.attrs["units"] = f"({skill.attrs['units']})^{p}"

    # check for none attrs and remove
    del_list = []
    for key, value in metadata_dict.items():
        if value is None and key != "units":
            del_list.append(key)
    for entry in del_list:
        del metadata_dict[entry]

    # write optional information
    if metadata_dict is None:
        metadata_dict = dict()
    skill.attrs.update(metadata_dict)

    return skill


def convert_time_index(xobj, time_string, kind, calendar=HINDCAST_CALENDAR_STR):
    """Converts incoming time index to a standard xr.CFTimeIndex.

    Args:
        xobj (xarray object): Dataset or DataArray with a time dimension to convert.
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

        # If Float64Index or Int64Index, assume annual and convert accordingly.
        elif isinstance(time_index, pd.Float64Index) | isinstance(
            time_index, pd.Int64Index
        ):
            if OPTIONS["warn_for_init_coords_int_to_annual"]:
                warnings.warn(
                    "Assuming annual resolution due to numeric inits. "
                    "Change init to a datetime if it is another resolution."
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

    return xobj


def convert_cftime_to_datetime_coords(ds, dim):
    """Convert dimension coordinate dim from CFTimeIndex to pd.DatetimeIndex."""
    return ds.assign_coords(
        {dim: xr.DataArray(ds[dim].to_index().to_datetimeindex(), dims=dim)}
    )


def find_start_dates_for_given_init(control, single_init):
    """Find the same start dates for cftime single_init across different years in
    control. Return control.time. Requires calendar=Datetime(No)Leap for consistent
    `dayofyear`."""
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
    """Return the temporal frequency of the input time series. Finds the frequency
    starting from high frequencies at which all ds.dim are not equal.

    Args:
        ds (xr.object): input with dimension `dim`.
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


def get_metric_class(metric, list_):
    """
    This allows the user to submit a string representing the desired metric
    to the corresponding metric class.

    Currently compatable with functions:
    * compute_persistence()
    * compute_perfect_model()
    * compute_hindcast()

    Args:
        metric (str): name of metric.
        list_ (list): check whether metric in list

    Returns:
        metric (Metric): class object of the metric.
    """
    if isinstance(metric, metrics.Metric):
        return metric
    elif isinstance(metric, str):
        # check if metric allowed
        is_in_list(metric, list_, "metric")
        metric = METRIC_ALIASES.get(metric, metric)
        return getattr(metrics, "__" + metric)
    else:
        raise ValueError(
            f"Please provide metric as str or Metric class, found {type(metric)}"
        )


def get_comparison_class(comparison, list_):
    """
    Converts a string comparison entry from the user into a Comparison class
     for the package to interpret.

    Perfect Model:

        * m2m: Compare all members to all other members.
        * m2c: Compare all members to the control.
        * m2e: Compare all members to the ensemble mean.
        * e2c: Compare the ensemble mean to the control.

    Hindcast:

        * e2o: Compare the ensemble mean to the verification data.
        * m2o: Compare each ensemble member to the verification data.

    Args:
        comparison (str): name of comparison.

    Returns:
        comparison (Comparison): comparison class.

    """
    if isinstance(comparison, comparisons.Comparison):
        return comparison
    elif isinstance(comparison, str):
        # check if comparison allowed
        is_in_list(comparison, list_, "comparison")
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        return getattr(comparisons, "__" + comparison)
    else:
        is_in_list(comparison, list_, "comparison")
        return getattr(comparisons, "__" + comparison)


def get_lead_cftime_shift_args(units, lead):
    """Determines the date increment to use when adding the lead time to init time based
    on the units attribute.

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
    """Returns ``CFTimeIndex.shift()`` offset increment for an arbitrary number of
    leads.

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
    """
    Custom intersection, since `set.intersection()` changes type of list.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def lead_units_equal_control_time_stride(init, verif):
    """Check that the lead units of the initialized ensemble have the same frequency as
    the control stride.

    Args:
        init (xr.object): initialized ensemble with lead units.
        verif (xr.object): control, uninitialized historical simulation / observations.

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
    """Shifts a ``CFTimeIndex`` over a specified number of time steps at a given
    temporal frequency.

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
    """Chunk xr.object `new_chunk_ds` as another xr.object `ori_chunk_ds`.
    This is needed after some operations which reduce chunks to size 1.
    First transpose a to ds.dims then apply ds chunking to a."""
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(new_chunk_ds, xr.DataArray) else {}
    )
    return new_chunk_ds.transpose(*ori_chunk_ds.dims, **transpose_kwargs).chunk(
        ori_chunk_ds.chunks
    )


def convert_Timedelta_to_lead_units(ds):
    """Convert lead as pd.Timedelta to lead as int and corresponding lead.attrs['units'] and convert to longest integer lead unit possible."""
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
    """Broadcast time.groupby('time.month/dayofyear/weekofyear').mean() from
    category_edges back to dim matching forecast."""
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
    """Apply broadcast_time_grouped_to_time to category_edges in metric_kwargs."""
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
                f"excepted category edges as tuple, xr.Dataset or np.array, found {type(category_edges)}"
            )
        return metric_kwargs
