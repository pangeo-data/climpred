"""
Objects dealing with plotting and analysis of output from the Model for
Prediction Across Scales (MPAS). This submodule will generally be built for 
MPAS-Ocean, but theoretically should work with other MPAS modules.

References
----------
MPAS-Ocean: Ringler, T., Petersen, M., Higdon, R. L., Jacobsen, D., 
Jones, P. W., & Maltrud, M. (2013). Ocean Modelling. Ocean Modelling, 69(C), 
211–232. doi:10.1016/j.ocemod.2013.04.010. 

Conversion
----------
`xyz_to_lat_lon` : Converts xyz coordinate output to lat/lon coordinates.

Visualization
-------------
`scatter` : Plots output onto a global (or regional) cartopy map.

"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from esmtools.vis import make_cartopy


def scatter(lon, lat, data, cmap, vmin, vmax, stride=5, projection=ccrs.Robinson(),
        colorbar=True):
    """
    Create a cartopy map of MPAS output on the native unstructured grid, using
    matplotlib's scatter function.

    Input
    -----
    lon : xarray DataArray
        1D da of longitudes (should be 'lonCell' in MPAS output)
    lat : xarray DataArray
        1D da of latitudes (should be 'latCell' in MPAS output)
    data : xarray DataArray
        1D da of output data
    cmap : str
        Native matplotlib colormap or cmocean colormap
    vmin : double
        Minimum color bound
    vmax : double
        Maximum color bound
    stride : int
        Stride in plotting data to avoid plotting too much
    projection : ccrs map projection
        Map projection to use 
    colorbar : logical
        Whether or not to add a colorbar to the figure. Generally want to set this 
        to off and do it manually if you need more advanced changes to it.

    Examples
    --------
    from esmtools.mpas import scatter
    import xarray as xr
    ds = xr.open_dataset('some_BGC_output.nc')
    scatter(ds['lonCell'], ds['latCell'], ds['FG_CO2'], "RdBu_r")
    """
    f, ax = make_cartopy(projection=projection, grid_lines=False, frameon=False)
    lon = lon[0::stride]
    lat = lat[0::stride]
    data = data[0::stride]

    p = ax.scatter(lon, lat, s=1, c=data, cmap=cmap, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.08)


def xyz_to_lat_lon(x, y, z, radians=False):
    """
    Convert xyz output (e.g. from Lagrangian Particle Tracking) to conventional
    lat/lon coordinates.

    Input
    -----
    x : array_like
        Array of x values
    y : array_like
        Array of y values
    z : array_like
        Array of z values
    radians : boolean (optional)
        If true, return lat/lon as radians

    Returns
    -------
    lon : array_like
        Array of longitude values
    lat : array_like
        Array of latitude values

    Examples
    --------
    from esmtools.mpas import xyz_to_lat_lon
    import xarray as xr
    ds = xr.open_dataset('particle_output.nc')
    lon, lat = xyz_to_lat_lon(ds.xParticle, ds.yParticle, ds.zParticle)
    """
    plat = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    plon = np.arctan2(y, x)
    # longitude adjustment
    plon[plon < 0.0] = 2*np.pi + plon[plon < 0.0]
    if radians:
        return plon, plat
    else:
        return plon * 180./np.pi, \
               plat * 180./np.pi
