import numpy as np
import xarray as xr

from ..constants import CLIMPRED_ENSEMBLE_DIMS
from .mpi import get_path


def set_integer_time_axis(ds, offset=1, time_dim='time'):
    """Set time axis to integers starting with `offset`. Used in hindcast preprocessing
    before the concatination of `intake-esm` happens."""
    ds[time_dim] = np.arange(offset, offset + ds[time_dim].size)
    return ds


def load_hindcast(
    inits=range(1961, 1965),
    members=range(1, 3),
    preprocess=None,
    lead_offset=1,
    parallel=True,
    engine=None,
    get_path=get_path,
    **get_path_kwargs,
):  # “pragma: no cover”
    """Load multi-member, multi-initialization hindcast experiment into one
    `xr.Dataset` compatible with `climpred`.

    Args:
        inits (list, array): List of initializations to be loaded.
            Defaults to range(1961, 1965).
        members (list, array): List of initializations to be loaded.
            Defaults to range(1, 3).
        preprocess (function): `preprocess` function accepting and returning
            `xr.Dataset` only. To be passed to :py:func:`xarray.open_dataset`.
            Defaults to None.
        parallel (bool): passed to `xr.open_dataset`. Defaults to True.
        engine (str): passed to `xr.open_dataset`. Defaults to None.

        .. note::
            To load MPI-ESM grb files, pass `engine='pynio'`.

        get_path (callable): `get_path` function specific to modelling center output
            format. Defaults to :py:func:`climpred.preprocessing.mpi.get_path`.
        **get_path_kwargs (dict): parameters passed to `**get_path`.

    Returns:
        xr.Dataset: `climpred` compatible dataset with dims: `member`, `init`, `lead`

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
            member_ds = set_integer_time_axis(member_ds)
            member_list.append(member_ds)
        member_ds = xr.concat(member_list, 'member')
        init_list.append(member_ds)
    ds = xr.concat(init_list, 'init').rename({'time': 'lead'})
    ds['member'] = members
    ds['init'] = inits
    return ds


def rename_SLM_to_climpred_dims(ds):
    """Rename ensemble dimensions common to SubX or CESM output."""
    dim_dict = {'S': 'init', 'L': 'lead', 'M': 'member'}
    for dim in dim_dict.keys():
        if dim in ds.dims:
            ds = ds.rename({dim: dim_dict[dim]})
    return ds


def rename_to_climpred_dims(ds):
    """Rename existing dimension in xr.object `ds` to CLIMPRED_ENSEMBLE_DIMS from
    existing dimension names."""
    for cdim in CLIMPRED_ENSEMBLE_DIMS:
        renamed = False  # set renamed flag to false initiallly
        if cdim not in ds.dims:  # if a CLIMPRED_ENSEMBLE_DIMS is not found
            for c in ds.dims:  # check in ds.dims for dims
                if cdim in c:  # containing the string of this CLIMPRED_ENSEMBLE_DIMS
                    ds = ds.rename({c: cdim})
                    renamed = True
        # special case for hindcast when containing time
        if 'time' in ds.dims and 'lead' not in ds.dims:
            ds = ds.rename({'time': 'lead'})
            renamed = True
        elif 'lead' in ds.dims:
            renamed = True
        if not renamed:
            raise ValueError(
                f"Couldn't find a dimension to rename to `{cdim}`, found {ds.dims}."
            )
    return ds
