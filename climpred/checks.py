from functools import wraps

import xarray as xr

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


def has_prediction_ensemble_dims(xobj):
    """
    Checks that at the minimum, the climate prediction  object has dimensions
    `init` and `lead` (i.e., it's a time series with lead times.
    """
    cond = all(dims in xobj.dims for dims in ['init', 'lead'])
    if not cond:
        # create custom error here.
        raise DimensionError(
            'Your prediction object must contain the '
            'dimensions `lead` and `init` at the minimum.'
        )


def match_initialized_dims(init, ref):
    """Checks that the reference matches all initialized dimensions except
    for 'lead' and 'member'"""
    # since reference products won't have the initialization dimension,
    # temporarily rename to time.
    init = init.rename({'init': 'time'})
    init_dims = list(init.dims)
    if 'lead' in init_dims:
        init_dims.remove('lead')
    if 'member' in init_dims:
        init_dims.remove('member')
    if not (set(ref.dims) == set(init_dims)):
        unmatch_dims = set(ref.dims) ^ set(init_dims)
        raise DimensionError(
            'Dimensions must match initialized prediction ensemble '
            f'dimensions; these dimensions do not match: {unmatch_dims}.'
        )


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


def is_in_dict(key, dict_, kind):
    """Check whether a key is in provided dictionary; kind is just a string."""
    if key not in dict_.values():
        raise KeyError(f'Specify {kind} from {dict_.keys()}')
