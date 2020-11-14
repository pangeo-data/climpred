import numpy as np
import xarray as xr


def last_item_cond_true(cond, dim, shift=-1):
    reached = cond.idxmin(dim) + shift  # to get the last True lead
    # fix below one
    reached = reached.where(reached >= 1, np.nan)
    # reset that never reach to nan
    # reached = reached.where(reached!=cond[dim].min(),other=np.nan)
    # reset where always true to max dim
    reached = reached.where(~cond.all("lead"), other=cond[dim].max())
    # fix locations where always nan to nan
    mask = ~(cond == False).all("lead")
    reached = reached.where(mask, other=np.nan)
    return reached


def predictability_horizon(cond):
    ph = last_item_cond_true(cond, "lead")
    if isinstance(ph, xr.DataArray):
        ph.attrs["units"] = cond["lead"].attrs["units"]
    elif isinstance(ph, xr.Dataset):
        for v in ph.data_vars:
            ph[v].attrs["units"] = cond["lead"].attrs["units"]
    return ph
