import datetime
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
from .metrics import METRIC_ALIASES


def assign_attrs(
    skill,
    ds,
    function_name,
    metadata_dict=None,
    alignment='same_inits',
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
        'prediction_skill'
    ] = f'calculated by climpred https://climpred.readthedocs.io/'
    skill.attrs['skill_calculated_by_function'] = function_name
    if 'init' in ds.coords:
        skill.attrs['number_of_initializations'] = ds.init.size
    if 'member' in ds.coords and function_name != 'compute_persistence':
        skill.attrs['number_of_members'] = ds.member.size

    skill.attrs['alignment'] = alignment
    skill.attrs['metric'] = metric.name
    if comparison is not None:
        skill.attrs['comparison'] = comparison.name
    if dim is not None:
        skill.attrs['dim'] = dim

    # change unit power
    if metric.unit_power == 0:
        skill.attrs['units'] = 'None'
    if metric.unit_power >= 2 and 'units' in skill.attrs:
        p = metric.unit_power
        p = int(p) if int(p) == p else p
        skill.attrs['units'] = f"({skill.attrs['units']})^{p}"

    # check for none attrs and remove
    del_list = []
    for key, value in metadata_dict.items():
        if value is None and key != 'units':
            del_list.append(key)
    for entry in del_list:
        del metadata_dict[entry]

    # write optional information
    if metadata_dict is None:
        metadata_dict = dict()
    skill.attrs.update(metadata_dict)

    skill.attrs[
        'created'
    ] = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%f")[:-6]}'
    return skill


def copy_coords_from_to(xro_from, xro_to):
    """Copy coords from one xr object to another."""
    if isinstance(xro_from, xr.DataArray) and isinstance(xro_to, xr.DataArray):
        for c in xro_from.coords:
            xro_to[c] = xro_from[c]
        return xro_to
    elif isinstance(xro_from, xr.Dataset) and isinstance(xro_to, xr.Dataset):
        xro_to = xro_to.assign_coords(**xro_from.coords)
    else:
        raise ValueError(
            f'xro_from and xro_to must be both either xr.DataArray or',
            f'xr.Dataset, found {type(xro_from)} {type(xro_to)}.',
        )
    return xro_to


def convert_time_index(xobj, time_string, kind):
    """Converts incoming time index to a standard xr.CFTimeIndex.

    Args:
        xobj (xarray object): Dataset or DataArray with a time dimension to convert.
        time_string (str): Name of time dimension.
        kind (str): Kind of object for error message.

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
            split_dates = [d.split(' ')[0].split('-') for d in time_strings]

        # If Float64Index or Int64Index, assume annual and convert accordingly.
        elif isinstance(time_index, pd.Float64Index) | isinstance(
            time_index, pd.Int64Index
        ):
            warnings.warn(
                'Assuming annual resolution due to numeric inits. '
                'Change init to a datetime if it is another resolution.'
            )
            # TODO: What about decimal time? E.g. someone has 1955.5 or something?
            dates = [str(int(t)) + '-01-01' for t in time_index]
            split_dates = [d.split('-') for d in dates]
            if 'lead' in xobj.dims:
                # Probably the only case we can assume lead units, since `lead` does not
                # give us any information on this.
                xobj['lead'].attrs['units'] = 'years'

        else:
            raise ValueError(
                f'Your {kind} object must be pd.Float64Index, '
                'pd.Int64Index, xr.CFTimeIndex or '
                'pd.DatetimeIndex.'
            )
        # TODO: Account for differing calendars. Currently assuming `Gregorian`.
        cftime_dates = [
            cftime.DatetimeProlepticGregorian(int(y), int(m), int(d))
            for (y, m, d) in split_dates
        ]
        time_index = xr.CFTimeIndex(cftime_dates)
        xobj[time_string] = time_index

    return xobj


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
        is_in_list(metric, list_, 'metric')
        metric = METRIC_ALIASES.get(metric, metric)
        return getattr(metrics, '__' + metric)
    else:
        raise ValueError(
            f'Please provide metric as str or Metric class, found {type(metric)}'
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
        is_in_list(comparison, list_, 'comparison')
        comparison = COMPARISON_ALIASES.get(comparison, comparison)
        return getattr(comparisons, '__' + comparison)
    else:
        is_in_list(comparison, list_, 'comparison')
        return getattr(comparisons, '__' + comparison)


def get_lead_cftime_shift_args(units, lead):
    """Determines the date increment to use when adding the lead time to init time based
    on the units attribute.

    Args:
        units (str): Units associated with the lead dimension. Must be
            years, seasons, months, weeks, pentads, days.
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
        'years': (lead, 'YS'),
        'seasons': (lead * 3, 'MS'),
        # Currently assumes monthly aligns with month start.
        'months': (lead, 'MS'),
        'weeks': (lead * 7, 'D'),
        'pentads': (lead * 5, 'D'),
        'days': (lead, 'D'),
    }

    try:
        n, freq = d[units]
    except KeyError:
        print(f'{units} is not a valid choice.')
        print(f'Accepted `units` values include: {d.keys()}')
    return n, freq


def get_multiple_lead_cftime_shift_args(units, leads):
    """Returns ``CFTimeIndex.shift()`` offset increment for an arbitrary number of
    leads.

    Args:
        units (str): Units associated with the lead dimension. Must be one of
            years, seasons, months, weeks, pentads, days.
        leads (list, array, xr.DataArray of ints): Leads to return offset for.

    Returns:
        n (tuple of ints): Number of units to shift for ``leads``. ``value`` for
            ``CFTimeIndex.shift(value, str)``.
        freq (str): Pandas frequency alias. ``str`` for
            ``CFTimeIndex.shift(value, str)``.
    """
    n_freq_tuples = [get_lead_cftime_shift_args(units, l) for l in leads]
    n, freq = list(zip(*n_freq_tuples))
    return n, freq[0]


def intersect(lst1, lst2):
    """
    Custom intersection, since `set.intersection()` changes type of list.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def _load_into_memory(res):
    """Compute if res is lazy data."""
    if dask.is_dask_collection(res):
        res = res.compute()
    return res


def reduce_forecast_to_same_inits(forecast, verif):
    """Reduces the forecast to common set of initializations that verify over all lags.

    Args:
        forecast (``xarray`` object): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        verif (``xarray`` object): Verification data with ``time`` dim.

    Returns:
        forecast (``xarray`` object): Prediction ensemble with ``init`` sub-selected for
            those that verify over all leads.
        verif (``xarray`` object): Original verification data.
    """
    # Construct list of `n` offset over all leads.
    n, freq = get_multiple_lead_cftime_shift_args(
        forecast['lead'].attrs['units'], forecast['lead'].values
    )
    n = list(n)
    # Add lead 0 to check that init exists in the observations so that reference
    # forecasts have the same set of inits.
    if 0 not in n:
        n.insert(0, 0)
    # Note that `init` is renamed to `time` in the compute function to compute metrics.
    init_lead_matrix = xr.concat(
        [
            xr.DataArray(
                shift_cftime_index(forecast, 'time', n, freq),
                dims=['time'],
                coords=[forecast['time']],
            )
            for n in n
        ],
        'lead',
    )
    # Checks at each `init` if all leads can verify.
    verifies_at_all_leads = init_lead_matrix.isin(verif['time']).all('lead')
    forecast = forecast.where(verifies_at_all_leads, drop=True)
    return forecast, verif


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
    return new_chunk_ds.transpose(*ori_chunk_ds.dims).chunk(ori_chunk_ds.chunks)
