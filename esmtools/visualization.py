"""
Objects dealing with anything visualization.

Color
-----
- `discrete_cmap` : Create a discrete colorbar for the visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    plt.pcolor(data, vmin=-3, vmax=3, cmap=et.visualization.discrete_cmap(10,
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
                 figsize=(12, 8)):
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

    
    Returns
    -------
    fig : Figure instance
    ax : Axes instance
    
    Examples
    --------
    import esmtools as et
    import cartopy.crs as ccrs
    f, ax, gl = et.visualization.make_cartopy(land_color='#D3D3D3', projection=ccrs.Mercator()))
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=projection))
    ax.gridlines(draw_labels=False, color=grid_color)
    ax.add_feature(cfeature.LAND, facecolor=land_color)
    return fig, ax

def pcolormesh(ax, lon, lat, data, global_field=True, extent=None,
               **kwargs):
    """
    Plots a pcolormesh map, given a lon, lat, and data structure. 

    Parameters
    ----------
    ax : axes instance (should be from make_cartopy command)
    lon : 2D longitude array
    lat : 2D latitude array
    data : 2D data array
    global_field : boolean (optional)
        Set to False if only regional data is being plotted.
    extent : array (optional)
        [x0, x1, y0, y1] to set bounds of regional projection
    **kwargs : optional keyword arguments
        Any arguments that could be passed to plt.pcolormesh

    Returns
    -------
    pc : pcolor plot instance
    cb : colorbar instance

    Examples
    --------
    import esmtools as et
    fig, ax, gl = et.visualization.make_cartopy()
    pc, cb = et.visualization.pcolormesh(ax, ds.lon, ds.lat, ds.data)

    """
    if global_field == True:
        lon, lat, data = deseam(lon, lat, data)
    else:
        # Need to mask it manually if we aren't deseaming.
        data = np.ma.array(data, mask=np.isnan(data))
    pc = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                        **kwargs)
    cb = plt.colorbar(pc, ax=ax, orientation='horizontal', pad=0.05)
    if extent != None:
        ax.set_extent(extent)
    return pc, cb
   
