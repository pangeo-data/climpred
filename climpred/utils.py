# --------------------------------------#
# Simple get commands for xarray objects
# --------------------------------------#
def get_coords(da):
    return list(da.coords)


def get_dims(da):
    return list(da.dims)


def get_vars(ds):
    return list(ds.data_vars)
