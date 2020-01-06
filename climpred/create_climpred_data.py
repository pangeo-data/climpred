import glob as _glob
import os as _os

import numpy as np
import xarray as xr

from .constants import CLIMPRED_ENSEMBLE_DIMS


def get_path(
    dir_base_experiment="/work/bm1124/m300086/CMIP6/experiments",
    member=1,
    init=1960,
    model="hamocc",
    output_stream="monitoring_ym",
    timestr="*1231",
    ending="nc",
    verbose=False,
):
    """Get the path of a file of for MPI-ESM standard output file names and directory."""
    # get experiment_id
    dirs = _os.listdir(dir_base_experiment)
    experiment_id = [
        x for x in dirs if (f"{init}" in x and "r" + str(member) + "i" in x)
    ]
    assert len(experiment_id) == 1
    experiment_id = experiment_id[0]
    dir_outdata = f"{dir_base_experiment}/{experiment_id}/outdata/{model}"
    path = f"{dir_outdata}/{experiment_id}_{model}_{output_stream}_{timestr}.{ending}"
    if verbose:
        print(f"Path: {path}")
    if _os.path.exists(_glob.glob(path)[0]):
        return path
    else:
        raise ValueError(f"Path not found or no access: {path}")


def preprocess(ds, v="global_primary_production"):
    """Subselect one variable from multi-variable input files. Needs to return a dataset."""
    return ds[v].to_dataset(name=v).squeeze()


def load_hindcast(
    inits=range(1961, 1965),
    members=range(1, 3),
    preprocess=None,
    lead_offset=1,
    parallel=True,
    load_grb_func=None,
    **get_path_kwargs,
):
    """Load multi-member, multi-initialization hindcast experiment into one xr.Dataset compatible with `climpred`."""
    init_list = []
    for init in inits:
        print(f"Processing init {init} ...")
        member_list = []
        for member in members:
            # get path p
            p = get_path(member=member, init=init, **get_path_kwargs)
            ending = get_path_kwargs.get('ending', 'nc')
            # open all leads for specified member and init
            if ending is 'nc':
                member_ds = xr.open_mfdataset(
                    p,
                    combine="nested",
                    concat_dim="time",
                    preprocess=preprocess,
                    parallel=parallel,
                    coords="minimal",  # expecting identical coords
                    data_vars="minimal",  # expecting identical vars
                    compat="override",  # speed up
                ).squeeze()
            elif ending is 'grb':
                # fix for MPI-ESM grb files
                if callable(load_grb_func):
                    member_ds = load_grb_func(p)
                else:
                    raise ValueError(
                        f'Please provide a function as `load_grb_func(path)`, found {type(load_grb_func)}')
            # set new integer time
            member_ds["time"] = np.arange(
                lead_offset, lead_offset + member_ds.time.size
            )
            member_list.append(member_ds)
        member_ds = xr.concat(member_list, "member")
        init_list.append(member_ds)
    ds = xr.concat(init_list, "init").rename({"time": "lead"})
    ds["member"] = members
    ds["init"] = inits
    return ds


def climpred_preprocess_internal(ds):
    """CMIP6 DCPP preprocessing before the aggreatations of intake-esm happen."""
    # set time to integers starting at 1
    ds['time'] = np.arange(1, 1 + ds.time.size)
    return ds


def climpred_preprocess_post(ds):
    """Rename existing dimension to CLIMPRED_ENSEMBLE_DIMS."""
    for cdim in CLIMPRED_ENSEMBLE_DIMS:
        renamed = False
        if cdim not in ds.dims:
            for c in ds.dims:
                if cdim in c:
                    ds = ds.rename({c: cdim})
                    renamed = True
        if 'time' in ds.dims and 'lead' not in ds.dims:
            ds = ds.rename({'time': 'lead'})
            renamed = True
        if not renamed:
            raise ValueError(
                f"Couldn't find a variable to rename to `{cdim}`, found {ds.dims}.")
    return ds
