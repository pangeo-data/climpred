import xarray as xr


# --------------------------------------#
# CHECKS
# --------------------------------------#
def check_xarray(x):
    """Check if the object being submitted is either a Dataset or DataArray."""
    if not (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset)):
        typecheck = type(x)
        raise IOError(f"""The input data is not an xarray object (an xarray
            DataArray or Dataset). esmtools is built to wrap xarray to make
            use of its awesome features. Please input an xarray object and
            retry the function.
            Your input was of type: {typecheck}""")


# --------------------------------------#
# Simple get commands for xarray objects
# --------------------------------------#
def get_coords(da):
    return list(da.coords)


def get_dims(da):
    return list(da.dims)


def get_vars(ds):
    return list(ds.data_vars)
