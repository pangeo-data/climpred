"""
Objects dealing with anything visualization.

Color
-----
- `discrete_cmap` : Create a discrete colorbar for the visualization.

Figures
-------
- `outer_legend` : Add a legend in the upper right outside of the figure.
- `savefig` : Matplotlib savefig command with all the right features.

Mapping
-------
- `add_box` : Add a box to highlight an area in a Cartopy plot.
- `deseam` : Get rid of the seam that occurs around the Prime Meridian.
- `make_cartopy` : Create a global Cartopy projection.
- `meshgrid` : Take a 1D lon/lat grid and save as a meshgrid in the dataset.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry.polygon import LinearRing

def deseam(lon, lat, data):
    """
    Returns a "bookended" longitude, latitude, and data array that 
    gets rid of the seam around the Prime Meridian.

    Parameters
    ----------
    lon : numpy array
        2D array of longitude values.
    lat : numpy array
        2D array of latitude values.
    data : numpy array
        MASKED 2D array of data.

    Returns
    -------
    new_lon : numpy array
        2D array of appended longitude values.
    new_lat : numpy array
        2D array of appended latitude values.
    new_data : numpy array
        2D array of appended data values.
    
    Examples
    --------

    """
    i, j = lat.shape
    new_lon = np.zeros((i, j+1))
    new_lon[:, :-1] = lon
    new_lon[:, -1] = lon[:, 0]

    new_lat = np.zeros((i, j+1))
    new_lat[:, :-1] = lat
    new_lat[:, -1] = lat[:, 0]

    new_data = np.zeros((i, j+1))
    new_data[:, :-1] = data
    new_data[:, -1] = data[:, 0]
    new_data = np.ma.array(new_data, mask=np.isnan(new_data))
    return new_lon, new_lat, new_data

def discrete_cmap(levels, base_cmap):
    """
    Returns a discretized colormap based on the specified input colormap.

    Parameters
    ----------
    levels : int
           Number of divisions for the color bar.
    base_cmap : string
           Colormap to discretize (can pull from cmocean, matplotlib, etc.)

    Returns
    ------
    discrete_cmap : LinearSegmentedColormap
           Discretized colormap for plotting

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import esmtools as et
    data = np.random.randn(50,50)
    plt.pcolor(data, vmin=-3, vmax=3, cmap=et.vis.discrete_cmap(10,
               "RdBu"))
    plt.colorbar()
    plt.show()
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, levels))
    cmap_name = base.name + str(levels)
    discrete_cmap = base.from_list(cmap_name, color_list, levels)
    return discrete_cmap

def make_cartopy(projection=ccrs.Robinson(), land_color='k', grid_color='#D3D3D3',
                 grid_lines=True, figsize=(12,8), frameon=True):
    """
    Returns a global cartopy projection with the defined projection style.

    Parameters
    ----------
    projection : ccrs projection instance (optional)
            Named map projection from Cartopy
    land_color : HEX or char string (optional)
            Color string to fill in continents with
    figsize : tuple (optional)
            Size of figure
    grid_color : HEX or char string (optional)
            Color string for the color of the grid lines
    grid_lines : boolean (optional)
            Whether or not to plot gridlines.
    frameon : boolean (optional)
            Whether or not to have a frame around the projection.

    Returns
    -------
    fig : Figure instance
    ax : Axes instance
    
    Examples
    --------
    import esmtools as et
    import cartopy.crs as ccrs
    f, ax, gl = et.vis.make_cartopy(land_color='#D3D3D3', projection=ccrs.Mercator()))
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=projection))
    if grid_lines == True:
        ax.gridlines(draw_labels=False, color=grid_color)
    ax.add_feature(cfeature.LAND, facecolor=land_color)
    if frameon == False:
        ax.outline_patch.set_edgecolor('white')
    return fig, ax

def add_box(ax, x0, x1, y0, y1, **kwargs):
    """
    Add a polygon/box to any cartopy projection. 
 
    Parameters
    ----------
    ax : axes instance (should be from make_cartopy command)
    x0: float; western longitude bound of box.
    x1: float; eastern longitude bound of box.
    y0: float; southern latitude bound of box.
    y1: float; northern latitude bound of box.
    **kwargs: optional keywords
        Will modify the color, etc. of the bounding box.
 
    Returns
    -------
    None
 
    Examples
    --------
    import esmtools as et
    fig, ax = et.vis.make_cartopy()
    et.visualization.add_box(ax, [-150, -110, 30, 50], edgecolor='k', facecolor='#D3D3D3',
                             linewidth=2, alpha=0.5)
    """
    lons = [x0, x0, x1, x1]
    lats = [y0, y1, y1, y0]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], ccrs.PlateCarree(), **kwargs)

def savefig(filename, directory=None, extension='.png', transparent=True,
           dpi=300):
    """
    Save a publication-ready figure.

    Parameters
    ----------
    filename : str
        The name of the file (without the extension)
    directory : str (optional)
        Identify a directory to place the file. Otherwise it will be placed locally.
    extension : str (optional)
        Identify a filetype. Defaults to .png
    transparent : boolean (optional)
        Whether or not to save with a transparent background. Default is True.
    dpi : int (optional)
        Dots per inch, or resolution. Defaults to 300.

    Returns
    -------
    A saved file.
    """
    # Need to identify a directory to place the file.
    if  directory != None:
        plt.savefig(directory + filename + extension, bbox_inches='tight',
                    pad_inches=1, transparent=transparent, dpi=dpi)
    else:
        plt.savefig(filename + extension, bbox_inches='tight', pad_inches=1,
                    transparent=transparent, dpi=dpi)

def meshgrid(x, y, d):
    """
    Returns a Dataset or DataArray with a 2D lat/lon field.

    Parameters
    ----------
    x : array_like
        1D longitude array
    y : array_like
        1D latitude array
    d : xr.Dataset or xr.DataArray
        Structure to tack the gridded lat/lon onto.

    Returns
    -------
    d : xr.Dataset or xr.DataArray
        Original structure with appended gridded lat/lon coordinates.
    """
    (xx, yy) = np.meshgrid(x, y)
    d.coords['gridlon'] = (('lat', 'lon'), xx)
    d.coords['gridlat'] = (('lat', 'lon'), yy)
    return d

def outer_legend():
    """
    Creates a legend outside of the figure in the upper right.
    """
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
