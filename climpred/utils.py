import sys
import traceback
from functools import wraps

import xarray as xr


# https://stackoverflow.com/questions/10610824/
# python-shortcut-for-writing-decorators-which-accept-arguments
def dec_args_kwargs(wrapper):
    return (
        lambda *dec_args, **dec_kwargs:
            lambda func:
                wrapper(func, *dec_args, **dec_kwargs)
    )

# --------------------------------------#
# CHECKS
# --------------------------------------#
@dec_args_kwargs
def check_xarray(func, *dec_args):
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

                        Your input was of type: {typecheck}""")
        except IndexError:
            pass
        # this is outside of the try/except so that the traceback is relevant
        # to the actual function call rather than showing a simple Exception
        # (probably IndexError from trying to subselect an empty dec_args list)
        return func(*args, **kwargs)
    return wrapper


# --------------------------------------#
# Simple get commands for xarray objects
# --------------------------------------#
def get_coords(da):
    return list(da.coords)


def get_dims(da):
    return list(da.dims)


def get_vars(ds):
    return list(ds.data_vars)
