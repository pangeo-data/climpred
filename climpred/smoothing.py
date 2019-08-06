import xarray as xr

try:
    import xesmf as xe

    def spatial_smoothing_xesmf(ds, boxsize=(5, 5)):
        """Apply spatial smoothing by regridding to `boxsize` grid.

        Reference:
          * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
                Gonzalez, V. Kharin, et al. “A Verification Framework for
                Interannual-to-Decadal Predictions Experiments.” Climate
                Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
                https://doi.org/10/f4jjvf.

        Args:
            ds (xr.object): input. xr.DataArray prefered.
            boxsize (tuple): target grid.
                             Defaults to `(5, 5)` (see Goddard et al. 2013).

        Returns:
            ds_smoothed (xr.object): boxsize-regridded input

        """
        if isinstance(ds, xr.Dataset):
            was_dataset = True
            ds = ds.to_array().squeeze()
        else:
            was_dataset = False
        ds_out = xe.util.grid_global(*boxsize)
        regridder = xe.Regridder(ds, ds_out, 'bilinear')
        res = regridder(ds)

        if was_dataset:

            def _variable_dataarray_to_dataset(da):
                """Return dataarray from dataset.to_darray() to dataset."""
                data_vars = [i for i in da['variable'].values]
                print(data_vars)
                da_list = []
                for i in data_vars:
                    a = da.sel(variable=i).squeeze()
                    a.name = i
                    del a['variable']
                    da_list.append(a)
                res = xr.merge(da_list)
                return res

            res = _variable_dataarray_to_dataset(res)
        return res


except:
    print('xesmf couldnt be loaded.')


def spatial_smoothing_xrcoarsen(ds, coarsen_dict=None, how='mean'):
    """Apply spatial smoothing by regridding to `boxsize` grid.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.object): input. xr.DataArray prefered.
        coarsen_dict (dict): coarsen spatial latitudes.
        how (str): aggregation type for coarsening. default: 'mean'

    Returns:
        ds_smoothed (xr.object): boxsize-regridded input

    """
    if coarsen_dict is None:
        # guess spatial dims
        spatial_dims_to_smooth = ds.dims.drop(
            ['time', 'lead', 'member', 'init'])
        # write coarsen to dict to coarsen similar to 5x5 degree
        pass  # not implemented
    # check whether coarsen dims are possible
    for dim in coarsen_dict:
        if dim not in ds.dims:
            raise ValueError(dim, 'not in ds')
        else:
            if ds[dim].size % coarsen_dict[dim] != 0:
                raise ValueError(
                    coarsen_dict[dim], 'does not divide', ds[dim].size, 'in', dim
                )
    ds_out = ds.coarsen(coarsen_dict)
    if how == 'mean':
        ds_out = ds_out.mean()
    elif how == 'sum':
        ds_out = ds_out.sum()
    return ds_out


def temporal_smoothing(ds, smooth=4, dim='time', how='mean', rename_dim=True):
    """Apply temporal smoothing by creating rolling smooth-timestep means.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.object): input.
        smooth (int): length of smoothing of timesteps.
                      Defaults to 4 (see Goddard et al. 2013).
        dim (str): temporal dimension to be smoothed. default: 'time'
        how (str): aggregation type for coarsening. default: 'mean'

    Returns:
        ds_smoothed (xr.object): input with `smooth` timesteps less and
                                 labeling '1-(smooth-1)','...', ... .

    """
    ds_smoothed = ds.rolling({dim: smooth}, center=False)
    if how == 'mean':
        ds_smoothed = ds_smoothed.mean()
    elif how == 'sum':
        ds_smoothed = ds_smoothed.sum()
    # remove first all-nans
    ds_smoothed = ds_smoothed.isel({dim: slice(smooth - 1, None)})
    if rename_dim:
        _reset_temporal_axis(ds_smoothed)
    return ds_smoothed


def _reset_temporal_axis(ds_smoothed, smooth=4, dim='time'):
    """Reduce and reset temporal axis. See temporal_smoothing(). Might be
     used after calculation of skill to maintain readable labels for skill
      computation."""
    new_time = [str(t) + '-' + str(t + smooth - 1)
                for t in ds_smoothed[dim].values]
    ds_smoothed[dim] = new_time
    return ds_smoothed


def smooth_Goddard_2013(
    ds,
    smooth_time=4,
    time_dim='time',
    boxsize=(5, 5),
    coarsen_dict={'x': 4, 'y': 4},
    how='mean',
):
    """Wrapper to smooth as suggested by Goddard et al. 2013."""
    # first temporal smoothing
    ds = temporal_smoothing(ds, smooth=smooth_time, dim=time_dim)
    try:  # xesmf has priority
        ds = spatial_smoothing_xesmf(ds, boxsize=boxsize)
    except:  # otherwise use coarsen
        ds = spatial_smoothing_xrcoarsen(ds, coarsen_dict, how=how)
        print('spatial xesmf smoothing didnt work, tried ')
    return ds
