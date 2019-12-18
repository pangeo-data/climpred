from functools import wraps

import xarray as xr
import pandas as pd

import warnings
from .exceptions import DatasetError, DimensionError, VariableError

# from .constants import VALID_LEAD_UNITS

# the import of CLIMPRED_DIMS from constants fails. currently fixed manually.
VALID_LEAD_UNITS = ['years', 'seasons', 'months', 'weeks', 'pentads', 'days']


# https://stackoverflow.com/questions/10610824/
# python-shortcut-for-writing-decorators-which-accept-arguments
def dec_args_kwargs(wrapper):
    return lambda *dec_args, **dec_kwargs: lambda func: wrapper(
        func, *dec_args, **dec_kwargs
    )


# --------------------------------------#
# CHECKS
# --------------------------------------#
@dec_args_kwargs
def is_xarray(func, *dec_args):
    """
    Decorate a function to ensure the first arg being submitted is
    either a Dataset or DataArray.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            ds_da_locs = dec_args[0]
            if not isinstance(ds_da_locs, list):
                ds_da_locs = [ds_da_locs]

            for loc in ds_da_locs:
                if isinstance(loc, int):
                    ds_da = args[loc]
                elif isinstance(loc, str):
                    ds_da = kwargs[loc]

                is_ds_da = isinstance(ds_da, (xr.Dataset, xr.DataArray))
                if not is_ds_da:
                    typecheck = type(ds_da)
                    raise IOError(
                        f"""The input data is not an xarray DataArray or
                        Dataset. climpred is built to wrap xarray to make
                        use of its awesome features. Please input an xarray
                        object and retry the function.

                        Your input was of type: {typecheck}"""
                    )
        except IndexError:
            pass
        # this is outside of the try/except so that the traceback is relevant
        # to the actual function call rather than showing a simple Exception
        # (probably IndexError from trying to subselect an empty dec_args list)
        return func(*args, **kwargs)

    return wrapper


def has_dims(xobj, dims, kind):
    """
    Checks that at the minimum, the object has provided dimensions.
    """
    if isinstance(dims, str):
        dims = [dims]

    if not all(dim in xobj.dims for dim in dims):
        raise DimensionError(
            f'Your {kind} object must contain the '
            f'following dimensions at the minimum: {dims}'
        )
    return True


def has_min_len(arr, len_, kind):
    """
    Checks that the array is at least the specified length.
    """
    arr_len = len(arr)
    if arr_len < len_:
        raise DimensionError(
            f'Your {kind} array must be at least {len_}, '
            f'but has only length {arr_len}!'
        )
    return True


def has_dataset(obj, kind, what):
    """Checks that the PredictionEnsemble has a specific dataset in it."""
    if len(obj) == 0:
        raise DatasetError(
            f'You need to add at least one {kind} dataset before '
            f'attempting to {what}.'
        )
    return True


def is_in_list(item, list_, kind):
    """Check whether an item is in a list; kind is just a string."""
    if item not in list_:
        raise KeyError(f'Specify {kind} from {list_}: got {item}')
    return True


def match_initialized_dims(init, ref, uninitialized=False):
    """Checks that the reference dimensions match appropriate initialized dimensions.

    If uninitialized, ignore 'member'. Otherwise, ignore 'lead' and 'member'.
    """
    # since reference products won't have the initialization dimension,
    # temporarily rename to time.
    init = init.rename({'init': 'time'})
    # Check that the reference time dimension is units of DateTimeIndex,
    # convertm or raise exception
    ref = convert_time_index(ref, 'time', 'ref[time]')

    init_dims = list(init.dims)
    if 'lead' in init_dims:
        init_dims.remove('lead')
    if ('member' in init_dims) and not uninitialized:
        init_dims.remove('member')
    if not (set(ref.dims) == set(init_dims)):
        unmatch_dims = set(ref.dims) ^ set(init_dims)
        raise DimensionError(
            'Dimensions must match initialized prediction ensemble '
            f'dimensions; these dimensions do not match: {unmatch_dims}.'
        )
    return True


def match_initialized_vars(init, ref):
    """
    Checks that a new reference (or control) dataset has at least one variable
    in common with the initialized dataset. This ensures that they can be
    compared pairwise.
    ref: new addition
    init: dp.initialized
    """
    init_vars = init.data_vars
    ref_vars = ref.data_vars
    # https://stackoverflow.com/questions/10668282/
    # one-liner-to-check-if-at-least-one-item-in-list-exists-in-another-list
    if set(init_vars).isdisjoint(ref_vars):
        raise VariableError(
            'Please provide a Dataset/DataArray with at least '
            'one matching variable to the initialized prediction ensemble; '
            f'got {init_vars} for init and {ref_vars} for ref.'
        )
    return True


def has_valid_lead_units(xobj):
    """
    Checks that the object has valid units for the lead dimension

    """
    if hasattr(xobj['lead'], 'units'):

        units = getattr(xobj['lead'], 'units')

        # Check if letter s is appended to lead units string and add it if needed
        if not units.endswith('s'):
            units += 's'
            xobj['lead'].attrs['units'] = units
            warnings.warn(
                f'The letter "s" was appended to the lead units; now {units}.'
            )

        # Raise Error if lead units is not valid
        if not getattr(xobj['lead'], 'units') in VALID_LEAD_UNITS:
            raise DimensionError(
                'The lead dimension must must have a valid '
                f'units attribute. Valid options are: {VALID_LEAD_UNITS}'
            )
    else:
        raise DimensionError(
            'The lead dimension must must have a '
            f'units attribute. Valid options are: {VALID_LEAD_UNITS}'
        )
    return True


def convert_time_index(xobj, time_string, kind):
    """
    Checks that the time dimension coming through is a DatetimeIndex,
    CFTimeIndex, Float64Index, or Int64Index
    Raises exception and exits if none of these.
    Converts CFTimeIndex, Float64Index, or Int64Index to DatetimeIndex.

    """

    time_index = xobj[time_string].to_index()

    # If a DatetimeIndex, nothing to do, otherwise check for other
    # options and convert or raise error
    if not isinstance(time_index, pd.DatetimeIndex):

        # If time_index is Float64Index or Int64Index, treat as
        # annual data and convert to DateTimeIndex
        if isinstance(time_index, pd.Float64Index) | isinstance(
            time_index, pd.Int64Index
        ):

            warnings.warn(
                'Assuming annual resolution due to numeric inits. '
                'Change init to a datetime if it is another resolution.'
            )

            startdate = str(int(time_index[0])) + '-01-01'
            enddate = str(int(time_index[-1])) + '-01-01'
            time_index = pd.date_range(start=startdate, end=enddate, freq='AS')
            xobj[time_string] = time_index

        # If time_index type is CFTimeIndex, convert to pd.DatetimeIndex
        elif isinstance(time_index, xr.CFTimeIndex):
            xobj = xr.decode_cf(xobj, decode_times=True)

        # Raise error if time_index is not integer, CFTimeIndex, or pd.DattimeIndex
        else:
            raise ValueError(
                f'Your {kind} object must be pd.Float64Index, '
                'pd.Int64Index, xr.CFTimeIndex or '
                'pd.DatetimeIndex.'
            )

    return xobj
