"""
Objects dealing with plotting and analysis of output from the Model for
Prediction Across Scales (MPAS). This submodule will generally be built for 
MPAS-Ocean, but theoretically should work with other MPAS modules.

References
----------
MPAS-Ocean: Ringler, T., Petersen, M., Higdon, R. L., Jacobsen, D., 
Jones, P. W., & Maltrud, M. (2013). Ocean Modelling. Ocean Modelling, 69(C), 
211–232. doi:10.1016/j.ocemod.2013.04.010. 

Visualization
-------------
`scatter` : Plots output onto a global (or regional) cartopy map.

"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from esmtools.vis import make_cartopy

def scatter(lon, lat, data, cmap, stride=5, projection=ccrs.Robinson()):
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
    stride : int
        Stride in plotting data to avoid plotting too much
    projection : ccrs map projection
        Map projection to use 

    Examples
    --------

    """
    f, ax = make_cartopy(projection=projection, grid_lines=False)
    lon = lon[0::stride]
    lat = lat[0::stride]
    data = data[0::stride]

    p = ax.scatter(lon, lat, s=1, c=data, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.08)

