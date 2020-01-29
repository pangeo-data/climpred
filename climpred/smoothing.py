import warnings

import numpy as np

from .checks import is_xarray

try:
    import xesmf as xe
except ImportError:
    xe = None


@is_xarray(0)
def spatial_smoothing_xesmf(
    ds,
    d_lon_lat_kws={'lon': 5, 'lat': 5},
    method='bilinear',
    periodic=False,
    filename=None,
    reuse_weights=True,
):
    """
    Quick regridding function. Adapted from
    https://github.com/JiaweiZhuang/xESMF/pull/27/files#diff-b537ef68c98c2ec11e64e4803fe4a113R105.

    Args:
        ds (xarray-object): Contain input and output grid coordinates.
            Look for variables ``lon``, ``lat``, and optionally ``lon_b``,
            ``lat_b`` for conservative method.
            Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
            or 2D (Ny, Nx) for general curvilinear grids.
            Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
         d_lon_lat_kws (dict): optional
            Longitude/Latitude step size (grid resolution); if not provided,
            lon will equal 5 and lat will equal lon
            (optional)
         method (str): Regridding method. Options are:
            - 'bilinear'
            - 'conservative', **need grid corner information**
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'
         periodic (bool): Periodic in longitude? Default to False. optional
            Only useful for global grids with non-conservative regridding.
            Will be forced to False for conservative regridding.
         filename (str): Name for the weight file. (optional)
            The default naming scheme is:
                 {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc
                 e.g. bilinear_400x600_300x400.nc
         reuse_weights (bool) Whether to read existing weight file to save
            computing time. False by default. (optional)

        Returns:
            ds (xarray.object) regridded
    """

    if xe is None:
        raise ImportError(
            'xesmf is not installed; see'
            'https://xesmf.readthedocs.io/en/latest/installation.html'
        )

    def _regrid_it(da, d_lon, d_lat, **kwargs):
        """
        Global 2D rectilinear grid centers and bounds

        Args:
            da (xarray.DataArray): Contain input and output grid coords.
                Look for variables ``lon``, ``lat``, ``lon_b``, ``lat_b`` for
                conservative method, and ``TLAT``, ``TLON`` for CESM POP grid
                Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
                or 2D (Ny, Nx) for general curvilinear grids.
                Shape of bounds should be (N+1,) or (Ny+1, Nx+1).
            d_lon (float): Longitude step size, i.e. grid resolution
            d_lat (float): Latitude step size, i.e. grid resolution
        Returns:
            da : xarray DataArray with coordinate values
        """

        def check_lon_lat_present(da):
            if method == 'conservative':
                if 'lat_b' in ds.coords and 'lon_b' in ds.coords:
                    return da
                else:
                    raise ValueError(
                        'if method == "conservative", lat_b and lon_b are required.'
                    )
            else:
                if 'lat' in ds.coords and 'lon' in ds.coords:
                    return da
                elif 'lat_b' in ds.coords and 'lon_b' in ds.coords:
                    return da
                # for CESM POP grid
                elif 'TLAT' in ds.coords and 'TLONG' in ds.coords:
                    da = da.rename({'TLAT': 'lat', 'TLONG': 'lon'})
                    return da
                else:
                    raise ValueError(
                        'lon/lat or lon_b/lat_b or TLAT/TLON not found, please rename.'
                    )

        da = check_lon_lat_present(da)
        grid_out = {
            'lon': np.arange(da.lon.min(), da.lon.max() + d_lon, d_lon),
            'lat': np.arange(da.lat.min(), da.lat.max() + d_lat, d_lat),
        }
        regridder = xe.Regridder(da, grid_out, **kwargs)
        return regridder(da)

    # check if lon or/and lat missing
    if ('lon' in d_lon_lat_kws) and ('lat' in d_lon_lat_kws):
        pass
    elif ('lon' not in d_lon_lat_kws) and ('lat' in d_lon_lat_kws):
        d_lon_lat_kws['lon'] = d_lon_lat_kws['lat']
    elif ('lat' not in d_lon_lat_kws) and ('lon' in d_lon_lat_kws):
        d_lon_lat_kws['lat'] = d_lon_lat_kws['lon']
    else:
        raise ValueError('please provide either `lon` or/and `lat` in d_lon_lat_kws.')

    kwargs = {
        'd_lon': d_lon_lat_kws['lon'],
        'd_lat': d_lon_lat_kws['lat'],
        'method': method,
        'periodic': periodic,
        'filename': filename,
        'reuse_weights': reuse_weights,
    }

    ds = _regrid_it(ds, **kwargs)

    return ds


@is_xarray(0)
def spatial_smoothing_xrcoarsen(ds, coarsen_kws=None, how='mean'):
    """Apply spatial smoothing by regridding to `boxsize` grid.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.object): input. xr.DataArray prefered.
        coarsen_kws (dict): coarsen spatial latitudes.
        how (str): aggregation type for coarsening. default: 'mean'

    Returns:
        ds_smoothed (xr.object): boxsize-regridded input

    """
    if coarsen_kws is None:
        # guess spatial dims
        spatial_dims_to_smooth = list(ds.dims)
        for dim in ['time', 'lead', 'member', 'init']:
            if dim in ds.dims:
                spatial_dims_to_smooth.remove(dim)
        # write coarsen to dict to coarsen similar to 5x5 degree
        coarsen_kws = dict()
        step = 2
        for dim in spatial_dims_to_smooth:
            coarsen_kws[dim] = step
        warnings.warn(
            f'no coarsen_kws given. created for dims \
            {spatial_dims_to_smooth} with step {step}'
        )
    # check whether coarsen dims are possible
    for dim in coarsen_kws:
        if dim not in ds.dims:
            raise ValueError(f'{dim} not in ds')
        else:
            if ds[dim].size % coarsen_kws[dim] != 0:
                raise ValueError(
                    f'{coarsen_kws[dim]} does not divide',
                    f'evenly {ds[dim].size} in {dim}',
                )
    # equivalent of doing ds.mean() if how == 'mean'
    ds_out = getattr(ds.coarsen(coarsen_kws), how)()
    return ds_out


@is_xarray(0)
def temporal_smoothing(ds, smooth_kws=None, how='mean', rename_dim=True):
    """Apply temporal smoothing by creating rolling smooth-timestep means.

    Reference:
    * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
     Gonzalez, V. Kharin, et al. “A Verification Framework for
     Interannual - to - Decadal Predictions Experiments.” Climate
     Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
     https: // doi.org / 10 / f4jjvf.

    Args:
        ds(xr.object): input.
        smooth_kws(dict): length of smoothing of timesteps.
                          Defaults to {'time': 4} (see Goddard et al. 2013).
        how(str): aggregation type for smoothing. default: 'mean'
        rename_dim(bool): Whether labels should be changed to
                          `'1-(smooth-1)', '...', ...`. default: True.

    Returns:
        ds_smoothed(xr.object): input with `smooth` timesteps less
        and labeling '1-(smooth-1)', '...', ... .

    """
    # unpack dict
    if not isinstance(smooth_kws, dict):
        raise ValueError('Please provide smooth_kws as dict, found ', type(smooth_kws))
    if not ('time' in smooth_kws or 'lead' in smooth_kws):
        raise ValueError(
            'smooth_kws doesnt contain a time dimension \
            (either "lead" or "time").',
            smooth_kws,
        )
    smooth = list(smooth_kws.values())[0]
    dim = list(smooth_kws.keys())[0]
    # fix to smooth either lead or time depending
    time_dims = ['time', 'lead']
    if dim not in ds.dims:
        time_dims.remove(dim)
        dim = time_dims[0]
        smooth_kws = {dim: smooth}
    # aggreate based on how
    ds_smoothed = getattr(ds.rolling(smooth_kws, center=False), how)()
    # remove first all-nans
    ds_smoothed = ds_smoothed.isel({dim: slice(smooth - 1, None)})
    ds_smoothed[dim] = ds.isel({dim: slice(None, -smooth + 1)})[dim]
    if rename_dim:
        ds_smoothed = _reset_temporal_axis(ds_smoothed, smooth_kws=smooth_kws, dim=dim)
    return ds_smoothed


@is_xarray(0)
def _reset_temporal_axis(ds_smoothed, smooth_kws=None, dim=None):
    """Reduce and reset temporal axis. See temporal_smoothing(). Might be
    used after calculation of skill to maintain readable labels for skill
    computation.

    Args:
        ds_smoothed (xarray object): Smoothed dataset.
        smooth_kws (dict): Keywords smoothing is performed over.
            Default is {'time': 4.
        dim (str): Dimension smoothing is performed over ('time' or 'lead').

    Returns:
        Smoothed Dataset with updated labels for smoothed temporal dimension.
    """
    if smooth_kws is None:
        smooth_kws = {'time': 4}
    if not ('time' in smooth_kws or 'lead' in smooth_kws):
        raise ValueError('smooth_kws doesnt contain a time dimension.', smooth_kws)
    smooth = list(smooth_kws.values())[0]
    dim = list(smooth_kws.keys())[0]
    try:
        # TODO: This assumes that smoothing is only done in years. Is this fair?
        composite_values = ds_smoothed[dim].to_index().year
    except AttributeError:
        composite_values = ds_smoothed[dim].values
    new_time = [f'{t}-{t + smooth - 1}' for t in composite_values]
    ds_smoothed[dim] = new_time
    return ds_smoothed


@is_xarray(0)
def smooth_goddard_2013(
    ds,
    smooth_kws={'lead': 4},
    d_lon_lat_kws={'lon': 5, 'lat': 5},
    coarsen_kws=None,
    how='mean',
    rename_dim=True,
):
    """Wrapper to smooth as suggested by Goddard et al. 2013:
        - 4-year composites
        - 5x5 degree regridding

    Reference:
    * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
        Gonzalez, V. Kharin, et al. “A Verification Framework for
        Interannual - to - Decadal Predictions Experiments.” Climate
        Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
        https: // doi.org / 10 / f4jjvf.

    Args:
        ds(xr.object): input.
        smooth_kws(dict): length of smoothing of timesteps (applies to ``lead``
                          in forecast and ``time`` in verification data).
                          Default: {'time': 4} (see Goddard et al. 2013).
        d_lon_lat_kws (dict): target grid for regridding.
                              Default: {'lon':5 , 'lat': 5}
        coarsen_kws (dict): grid coarsening steps in case xesmf regridding
                            fails. default: None.
        how(str): aggregation type for smoothing. default: 'mean'
        rename_dim(bool): Whether labels should be changed to
                          `'1-(smooth-1)', '...', ...`. default: True.

    Returns:
        ds_smoothed_regridded (xr.object): input with `smooth` timesteps less
                                           and labeling '1-(smooth-1)', '...' .

    """
    # first temporal smoothing
    ds_smoothed = temporal_smoothing(ds, smooth_kws=smooth_kws)
    try:  # xesmf has priority
        ds_smoothed_regridded = spatial_smoothing_xesmf(
            ds_smoothed, d_lon_lat_kws=d_lon_lat_kws
        )
    except Exception as e:  # otherwise use coarsen
        ds_smoothed_regridded = spatial_smoothing_xrcoarsen(
            ds_smoothed, coarsen_kws=coarsen_kws, how=how
        )
        print(
            f'spatial xesmf smoothing didnt work. \
            tried spatial_smoothing_xesmf and got {e}.\
            then spatial_smoothing_xrcoarsen'
        )
    return ds_smoothed_regridded
