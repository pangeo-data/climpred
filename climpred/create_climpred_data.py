import glob as _glob
import os as _os

import numpy as np
import xarray as xr

from .constants import CLIMPRED_ENSEMBLE_DIMS


def get_path(
    dir_base_experiment='/work/bm1124/m300086/CMIP6/experiments',
    member=1,
    init=1960,
    model='hamocc',
    output_stream='monitoring_ym',
    timestr='*1231',
    ending='nc',
):
    """Get the path of a file of for MPI-ESM standard output file names and directory.

    Args:
        dir_base_experiment (str): Path of experiments folder. Defaults to
        "/work/bm1124/m300086/CMIP6/experiments".
        member (int): `member`. Defaults to 1.
        init (init): `init`. Defaults to 1960.
        model (str): submodel name. Defaults to "hamocc".
        Allowed: ['echam6', 'jsbach', 'mpiom', 'hamocc'].
        output_stream (str): output_stream name. Defaults to "monitoring_ym".
        Allowed: ['data_2d_mm', 'data_3d_ym', 'BOT_mm', ...]
        timestr (str): timestr likely including *. Defaults to "*1231".
        ending (str): ending indicating file format. Defaults to "nc".
        Allowed: ['nc', 'grb'].

    Returns:
        str: path of requested file(s)

    """
    # get experiment_id
    dirs = _os.listdir(dir_base_experiment)
    experiment_id = [
        x for x in dirs if (f'{init}' in x and 'r' + str(member) + 'i' in x)
    ]
    assert len(experiment_id) == 1
    experiment_id = experiment_id[0]
    dir_outdata = f'{dir_base_experiment}/{experiment_id}/outdata/{model}'
    path = f'{dir_outdata}/{experiment_id}_{model}_{output_stream}_{timestr}.{ending}'
    if _os.path.exists(_glob.glob(path)[0]):
        return path
    else:
        raise ValueError(f'Path not found or no access: {path}')


def climpred_preprocess_internal(ds, lead_offset=1, time_dim='time'):
    """CMIP6 DCPP preprocessing before the aggreatations of intake-esm happen."""
    # set time_dim to integers starting at lead_offset
    ds[time_dim] = np.arange(lead_offset, lead_offset + ds[time_dim].size)
    return ds


def load_hindcast(
    inits=range(1961, 1965),
    members=range(1, 3),
    preprocess=None,
    lead_offset=1,
    parallel=True,
    engine=None,
    **get_path_kwargs,
):
    """Load multi-member, multi-initialization hindcast experiment into one
    `xr.Dataset` compatible with `climpred`.

    Args:
        inits (list, array): List of initializations to be loaded.
        Defaults to range(1961, 1965).
        members (list, array): List of initializations to be loaded.
        Defaults to range(1, 3).
        preprocess (function): `preprocess` function accepting and returning
        `xr.Dataset` only. To be passed to `xr.open_dataset`. Defaults to None.
        lead_offset (int): Number of first lead. Defaults to 1.
        parallel (bool): passed to `xr.open_dataset`. Defaults to True.
        engine (str): passed to `xr.open_dataset`. Defaults to None.

        .. note::
            To load MPI-ESM grb files `conda install pynio`, pass `engine='pynio'` and
            rename dimension to `time` in `preprocess`.

        **get_path_kwargs (dict): parameters passed to `**get_path`.

    Returns:
        xr.Dataset: climpred compatible dataset with dims: `member`, `init`, `lead`

    """
    init_list = []
    for init in inits:
        print(f'Processing init {init} ...')
        member_list = []
        for member in members:
            # get path p
            p = get_path(member=member, init=init, **get_path_kwargs)
            # open all leads for specified member and init
            member_ds = xr.open_mfdataset(
                p,
                combine='nested',
                concat_dim='time',
                preprocess=preprocess,
                parallel=parallel,
                engine=engine,
                coords='minimal',  # expecting identical coords
                data_vars='minimal',  # expecting identical vars
                compat='override',  # speed up
            ).squeeze()
            # set new integer time
            member_ds = climpred_preprocess_internal(member_ds)
            member_list.append(member_ds)
        member_ds = xr.concat(member_list, 'member')
        init_list.append(member_ds)
    ds = xr.concat(init_list, 'init').rename({'time': 'lead'})
    ds['member'] = members
    ds['init'] = inits
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
        elif 'lead' in ds.dims:
            renamed = True
        if not renamed:
            raise ValueError(
                f"Couldn't find a variable to rename to `{cdim}`, found {ds.dims}."
            )
    return ds
