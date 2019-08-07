import numpy as np

try:
    import xesmf as xe
except ImportError:
    xe = None


def spatial_smoothing_xesmf(
    ds,
    d_lon_lat_dict={'lon': 5, 'lat': 5},
    method='bilinear',
    periodic=False,
    filename=None,
    reuse_weights=True,
):
    """
    Quick regridding

    Parameters
    ----------
    ds : xarray DataSet
        Contain input and output grid coordinates. Look for variables
        ``lon``, ``lat``, and optionally ``lon_b``, ``lat_b`` for
        conservative method.
         Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
        or 2D (Ny, Nx) for general curvilinear grids.
        Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
     d_lon_lat_dict : dict, optional
        Longitude/Latitude step size, i.e. grid resolution; if not provided,
        Longitude will equal 1 and Latitude will equal Longitude
     method : str
        Regridding method. Options are
        - 'bilinear'
        - 'conservative', **need grid corner information**
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'
     periodic : bool, optional
        Periodic in longitude? Default to False.
        Only useful for global grids with non-conservative regridding.
        Will be forced to False for conservative regridding.
     filename : str, optional
        Name for the weight file. The default naming scheme is::
             {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc
         e.g. bilinear_400x600_300x400.nc
     reuse_weights : bool, optional
        Whether to read existing weight file to save computing time.
        False by default (i.e. re-compute, not reuse).

    Returns
    -------
    ds : xarray DataSet with coordinate values or DataArray
    """

    if xe is None:
        raise ImportError(
            'xesmf is not installed; see https://xesmf.readthedocs.io/en/latest/installation.html'
        )

    def _regrid_it(da, d_lon, d_lat, **kwargs):
        """
        Global 2D rectilinear grid centers and bounds
         Parameters
        ----------
        da : xarray DataArray
            Contain input and output grid coordinates. Look for variables
            ``lon``, ``lat``, and optionally ``lon_b``, ``lat_b`` for
            conservative method.
             Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
            or 2D (Ny, Nx) for general curvilinear grids.
            Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
         d_lon : float
            Longitude step size, i.e. grid resolution
         d_lat : float
            Latitude step size, i.e. grid resolution
         Returns
        -------
        da : xarray DataArray with coordinate values
        """

        def warn_lon_lat_dne(da):
            if 'lat' in ds.coords and 'lon' in ds.coords:
                return da
            # elif 'lat_b' in ds.coords and 'lon_b' in ds.coords:
            #    da = da.rename({'lat_b': 'lat', 'lon_b': 'lon'})
            #    return da
            elif 'TLAT' in ds.coords and 'TLONG' in ds.coords:
                da = da.rename({'TLAT': 'lat', 'TLONG': 'lon'})
                return da
            else:
                raise ValueError(
                    'lon/lat or lon_b/lat_b or TLAT/TLON not found, please rename'
                )

        da = warn_lon_lat_dne(da)
        grid_out = {
            'lon': np.arange(da.lon.min(), da.lon.max() + d_lon, d_lon),
            'lat': np.arange(da.lat.min(), da.lat.max() + d_lat, d_lat),
        }
        regridder = xe.Regridder(da, grid_out, **kwargs)
        return regridder(da)

    if 'lon' not in d_lon_lat_dict:
        d_lon_lat_dict['lon'] = d_lon_lat_dict['lat']
    elif 'lat' not in d_lon_lat_dict:
        d_lon_lat_dict['lat'] = d_lon_lat_dict['lon']
    else:
        raise ValueError('please provide either lon or lat in d_lon_lat_dict.')

    kwargs = {
        'd_lon': d_lon_lat_dict['lon'],
        'd_lat': d_lon_lat_dict['lat'],
        'method': method,
        'periodic': periodic,
        'filename': filename,
        'reuse_weights': reuse_weights,
    }

    ds = _regrid_it(ds, **kwargs)

    return ds


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
        spatial_dims_to_smooth = list(ds.dims)
        for dim in ['time', 'lead', 'member', 'init']:
            if dim in ds.dims:
                spatial_dims_to_smooth.remove(dim)
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
    ds_out = getattr(ds.coarsen(coarsen_dict), how)()
    return ds_out


def temporal_smoothing(ds, smooth_dict={'time': 4}, how='mean', rename_dim=True):
    """Apply temporal smoothing by creating rolling smooth-timestep means.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.object): input.
        smooth_dict (dict): length of smoothing of timesteps.
                      Defaults to {'time':4} (see Goddard et al. 2013).
        how (str): aggregation type for smoothing. default: 'mean'

    Returns:
        ds_smoothed (xr.object): input with `smooth` timesteps less and
                                 labeling '1-(smooth-1)','...', ... .

    """
    # unpack dict
    if len(smooth_dict) != 1:
        raise ValueError('smooth_dict doesnt contain only entry.', smooth_dict)
    smooth = [i for i in smooth_dict.values()][0]
    dim = [i for i in smooth_dict.keys()][0]
    # aggreate based on how
    ds_smoothed = getattr(ds.rolling(smooth_dict, center=False), how)()
    # remove first all-nans
    ds_smoothed = ds_smoothed.isel({dim: slice(smooth - 1, None)})
    if rename_dim:
        ds_smoothed = _reset_temporal_axis(ds_smoothed, smooth_dict=smooth_dict)
    return ds_smoothed


def _reset_temporal_axis(ds_smoothed, smooth_dict={'time': 4}):
    """Reduce and reset temporal axis. See temporal_smoothing(). Might be
    used after calculation of skill to maintain readable labels for skill
    computation."""
    if len(smooth_dict) != 1:
        raise ValueError('smooth_dict doesnt contain only entry.', smooth_dict)
    dim = [i for i in smooth_dict.keys()][0]
    smooth = [i for i in smooth_dict.values()][0]
    print('dim', dim)
    print('smooth', smooth)
    new_time = [str(t) + '-' + str(t + smooth - 1) for t in ds_smoothed[dim].values]
    ds_smoothed[dim] = new_time
    return ds_smoothed


def smooth_goddard_2013(
    ds,
    smooth_dict={'time': 4},
    d_lon_lat_dict={'lon': 5},
    coarsen_dict={'x': 2, 'y': 2},
    how='mean',
):
    """Wrapper to smooth as suggested by Goddard et al. 2013."""
    # first temporal smoothing
    ds = temporal_smoothing(ds, smooth_dict=smooth_dict)
    try:  # xesmf has priority
        ds = spatial_smoothing_xesmf(ds, d_lon_lat_dict=d_lon_lat_dict)
    except:  # otherwise use coarsen
        ds = spatial_smoothing_xrcoarsen(ds, coarsen_dict=coarsen_dict, how=how)
        print('spatial xesmf smoothing didnt work, tried ')
    return ds
