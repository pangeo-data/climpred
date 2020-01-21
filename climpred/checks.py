import warnings
from functools import wraps

import xarray as xr

from .exceptions import DatasetError, DimensionError, VariableError

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


def match_initialized_dims(init, verif, uninitialized=False):
    """Checks that the verification data dimensions match appropriate initialized
    dimensions.

    If uninitialized, ignore ``member``. Otherwise, ignore ``lead`` and ``member``.
    """
    # Since verification data won't have the ``init``` dimension, temporarily rename to
    # time.
    init = init.rename({'init': 'time'})
    init_dims = list(init.dims)
    if 'lead' in init_dims:
        init_dims.remove('lead')
    if ('member' in init_dims) and not uninitialized:
        init_dims.remove('member')
    if not (set(verif.dims) == set(init_dims)):
        unmatch_dims = set(verif.dims) ^ set(init_dims)
        raise DimensionError(
            'Dimensions must match initialized prediction ensemble '
            f'dimensions; these dimensions do not match: {unmatch_dims}.'
        )
    return True


def match_initialized_vars(init, verif):
    """Checks that a new verification dataset has at least one variable
    in common with the initialized dataset.

    This ensures that they can be compared pairwise.

    Args:
        init (xarray object): Initialized forecast ensemble.
        verif (xarray object): Product to be added.
    """
    init_vars = init.data_vars
    verif_vars = verif.data_vars
    # https://stackoverflow.com/questions/10668282/
    # one-liner-to-check-if-at-least-one-item-in-list-exists-in-another-list
    if set(init_vars).isdisjoint(verif_vars):
        raise VariableError(
            'Please provide a Dataset/DataArray with at least '
            'one matching variable to the initialized prediction ensemble; '
            f'got {init_vars} for init and {verif_vars} for verif.'
        )
    return True


def has_valid_lead_units(xobj):
    """
    Checks that the object has valid units for the lead dimension.
    """
    LEAD_UNIT_ERROR = (
        'The lead dimension must must have a valid '
        f'units attribute. Valid options are: {VALID_LEAD_UNITS}'
    )
    # Use `hasattr` here, as it doesn't throw an error if `xobj` doesn't have a
    # coordinate for lead.
    if hasattr(xobj['lead'], 'units'):

        units = xobj['lead'].attrs['units']

        # Check if letter s is appended to lead units string and add it if needed
        if not units.endswith('s'):
            units += 's'
            xobj['lead'].attrs['units'] = units
            warnings.warn(
                f'The letter "s" was appended to the lead units; now {units}.'
            )

        # Raise Error if lead units is not valid
        if not xobj['lead'].attrs['units'] in VALID_LEAD_UNITS:
            raise AttributeError(LEAD_UNIT_ERROR)
    else:
        raise AttributeError(LEAD_UNIT_ERROR)
    return True
