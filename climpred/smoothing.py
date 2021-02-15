import numpy as np
import xarray as xr

from .checks import is_xarray

try:
    import xesmf as xe
except ImportError:
    xe = None


@is_xarray(0)
def spatial_smoothing_xesmf(
    ds,
    d_lon_lat_kws={"lon": 5, "lat": 5},
    method="bilinear",
    periodic=False,
    filename=None,
    reuse_weights=False,
    tsmooth_kws=None,
    how=None,
):
    """
    Quick regridding function. Adapted from
    https://github.com/JiaweiZhuang/xESMF/pull/27/files#diff-b537ef68c98c2ec11e64e4803fe4a113R105.

    Args:
        ds (xarray-object): Contain input and output grid coordinates.
            Look for coordinates ``lon``, ``lat``, and optionally ``lon_b``,
            ``lat_b`` for conservative method. Also any coordinate which is C/F
            compliant, .i.e. standard_name in ['longitude', 'latitude'] is allowed.
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
        tsmooth_kws (None): leads nowhere but consistent with `temporal_smoothing`.
        how (None): leads nowhere but consistent with `temporal_smoothing`.

        Returns:
            ds (xarray.object) regridded
    """

    if xe is None:
        raise ImportError(
            "xesmf is not installed; see"
            "https://xesmf.readthedocs.io/en/latest/installation.html"
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

        if "lon" in da.coords:
            lon = da.lon
        else:
            try:
                lon = da.cf["longitude"]
            except KeyError:
                raise KeyError(
                    "Could not find `lon` as coordinate or any C/F compliant `latitude` coordinate, see https://pangeo-xesmf.readthedocs.io and https://cf-xarray.readthedocs.io"
                )

        if "lat" in da.coords:
            lat = da.lat
        else:
            try:
                lat = da.cf["latitude"]
            except KeyError:
                raise KeyError(
                    "C/F compliant or `lat` as coordinate, see https://pangeo-xesmf.readthedocs.io"
                )

        grid_out = xr.Dataset(
            {
                "lat": (["lat"], np.arange(lat.min(), lat.max() + d_lat, d_lat)),
                "lon": (["lon"], np.arange(lon.min(), lon.max() + d_lon, d_lon)),
            }
        )
        regridder = xe.Regridder(da, grid_out, **kwargs)
        return regridder(da)

    # check if lon or/and lat missing
    if ("lon" in d_lon_lat_kws) and ("lat" in d_lon_lat_kws):
        pass
    elif ("lon" not in d_lon_lat_kws) and ("lat" in d_lon_lat_kws):
        d_lon_lat_kws["lon"] = d_lon_lat_kws["lat"]
    elif ("lat" not in d_lon_lat_kws) and ("lon" in d_lon_lat_kws):
        d_lon_lat_kws["lat"] = d_lon_lat_kws["lon"]
    else:
        raise ValueError("please provide either `lon` or/and `lat` in d_lon_lat_kws.")

    kwargs = {
        "d_lon": d_lon_lat_kws["lon"],
        "d_lat": d_lon_lat_kws["lat"],
        "method": method,
        "periodic": periodic,
        "filename": filename,
        "reuse_weights": reuse_weights,
    }

    ds = _regrid_it(ds, **kwargs)

    return ds


@is_xarray(0)
def temporal_smoothing(ds, tsmooth_kws=None, how="mean", d_lon_lat_kws=None):
    """Apply temporal smoothing by creating rolling smooth-timestep means.

    Reference:
    * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
     Gonzalez, V. Kharin, et al. “A Verification Framework for
     Interannual - to - Decadal Predictions Experiments.” Climate
     Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
     https://doi.org/10/f4jjvf.

    Args:
        ds(xr.object): input.
        tsmooth_kws(dict): length of smoothing of timesteps.
            Defaults to {'time': 4} (see Goddard et al. 2013).
        how(str): aggregation type for smoothing. default: 'mean'
        d_lon_lat_kws (None): leads nowhere but consistent with
            `spatial_smoothing_xesmf`.

    Returns:
        ds_smoothed(xr.object): input with `smooth` timesteps less
            and labeling '1-(smooth-1)', '...', ... .

    """
    # unpack dict
    if not isinstance(tsmooth_kws, dict):
        raise ValueError(
            "Please provide tsmooth_kws as dict, found ", type(tsmooth_kws)
        )
    if not ("time" in tsmooth_kws or "lead" in tsmooth_kws):
        raise ValueError(
            'tsmooth_kws doesnt contain a time dimension \
            (either "lead" or "time").',
            tsmooth_kws,
        )
    smooth = list(tsmooth_kws.values())[0]
    dim = list(tsmooth_kws.keys())[0]
    # fix to smooth either lead or time depending
    time_dims = ["time", "lead"]
    if dim not in ds.dims:
        time_dims.remove(dim)
        dim = time_dims[0]
        tsmooth_kws = {dim: smooth}
    # aggreate based on how
    ds_smoothed = getattr(ds.rolling(tsmooth_kws, center=False), how)()
    # remove first all-nans
    ds_smoothed = ds_smoothed.isel({dim: slice(smooth - 1, None)})
    ds_smoothed[dim] = ds.isel({dim: slice(None, -smooth + 1)})[dim]
    return ds_smoothed


def _reset_temporal_axis(ds_smoothed, tsmooth_kws, dim="lead", set_lead_center=True):
    """Reduce and reset temporal axis. See temporal_smoothing(). Should be
    used after calculation of skill to maintain readable labels for skill
    computation.

    Args:
        ds_smoothed (xarray object): Smoothed dataset.
        tsmooth_kws (dict): Keywords smoothing is performed over.
        dim (str): Dimension smoothing is performed over. Defaults to 'lead'.
        set_center (bool): Whether to set new coord `{dim}_center`.
            Defaults to True.

    Returns:
        Smoothed Dataset with updated labels for smoothed temporal dimension.
    """
    # bugfix: actually tsmooth_kws should only dict
    if tsmooth_kws is None or callable(tsmooth_kws):
        return ds_smoothed
    if not ("time" in tsmooth_kws.keys() or "lead" in tsmooth_kws.keys()):
        raise ValueError("tsmooth_kws does not contain a time dimension.", tsmooth_kws)
    for c in ["time", "lead"]:
        if c in tsmooth_kws.keys():
            smooth = tsmooth_kws[c]
    ds_smoothed[dim] = [f"{t}-{t + smooth - 1}" for t in ds_smoothed[dim].values]
    if set_lead_center:
        _set_center_coord(ds_smoothed, dim)
    return ds_smoothed


def _set_center_coord(ds, dim="lead"):
    """Set lead_center as a new coordinate."""
    new_dim = []
    old_dim = ds[dim].values
    for i in old_dim:
        new_dim.append(eval(i.replace("-", "+")) / 2)
    new_dim = np.array(new_dim)
    ds.coords[f"{dim}_center"] = (dim, new_dim)
    return ds


@is_xarray(0)
def smooth_goddard_2013(
    ds,
    tsmooth_kws={"lead": 4},
    d_lon_lat_kws={"lon": 5, "lat": 5},
    how="mean",
    **xesmf_kwargs,
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
        tsmooth_kws(dict): length of smoothing of timesteps (applies to ``lead``
                          in forecast and ``time`` in verification data).
                          Default: {'time': 4} (see Goddard et al. 2013).
        d_lon_lat_kws (dict): target grid for regridding.
                              Default: {'lon':5 , 'lat': 5}
        how(str): aggregation type for smoothing. default: 'mean'
        **xesmf_kwargs (kwargs): kwargs passed to `spatial_smoothing_xesmf`.

    Returns:
        ds_smoothed_regridded (xr.object): input with `smooth` timesteps less
                                           and labeling '1-(smooth-1)', '...' .

    """
    # first temporal smoothing
    ds_smoothed = temporal_smoothing(ds, tsmooth_kws=tsmooth_kws)
    ds_smoothed_regridded = spatial_smoothing_xesmf(
        ds_smoothed, d_lon_lat_kws=d_lon_lat_kws, **xesmf_kwargs
    )
    return ds_smoothed_regridded
