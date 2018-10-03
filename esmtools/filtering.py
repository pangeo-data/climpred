"""
Objects dealing with any sort of spatial and temporal filtering of output.

Spatial
-------
- `_find_indices` -- Find indices for a given lat/lon point on a meshgrid.
- `extract_region` -- Directly extracts a subset of a dataset.
"""

import numpy as np

def _find_indices(xgrid, ygrid, xpoint, ypoint):
    """
    Returns the i,j index for a latitude/longitude point on a grid.
    
    Parameters
    ----------
    xgrid, ygrid : array_like, shape (`M`, `N`).
                   Longitude and latitude meshgrid. 
    xpoint, ypoint : int or 1-D array_like
                   Longitude and latitude of point searching for on grid. 
                   Should be in the same range as the grid itself (e.g.,
                   if the longitude grid is 0-360, should be 200 instead
                   of -160)

    Returns
    ------
    i, j : int
                  Keys for the inputted grid that lead to the lat/lon point
                  the user is seeking.

    Examples
    --------
    >>> import esmtools as et
    >>> import numpy as np
    >>> x = np.linspace(0, 360, 37)
    >>> y = np.linspace(-90, 90, 19)
    >>> xx, yy = np.meshgrid(x, y)
    >>> xp = 20
    >>> yp = -20
    >>> i, j = et.filtering.find_indices(xx, yy, xp, yp)
    >>> print(xx[i, j])
    20.0
    >>> print(yy[i, j])
    -20.0

    """
    dx = xgrid - xpoint
    dy = ygrid - ypoint
    reduced_grid = abs(dx) + abs(dy)
    min_ix = np.nanargmin(reduced_grid)
    i, j = np.unravel_index(min_ix, reduced_grid.shape)
    return i, j

def extract_region(ds, xgrid, ygrid, coords, lat_dim='nlat', lon_dim='nlon'):
    """
    Takes in an array of data, its lon/lat grid, and coordinates pertaining
    to the lat/lon sub-box desired and returns the extracted data. 

    Parameters
    ----------
    ds : array_like
        Data to extract sub-region from. Ideally dataset.
    xgrid, ygrid : array_like
        Longitude and latitude meshgrid.
    coords : vector
        [x0, x1, y0, y1] pertaining to corners of box to extract
    lat_dim, lon_dim : str (optional)
        

    Return
    ------
    subset_data : array_like
        Data subset to domain of interest
    """
    print("NOTE: Make sure your coordinates are in order [x0, x1, y0, y1]")
    x0, x1, y0, y1 = coords
    a, c = _find_indices(lon, lat, x0, y0)
    b, d = _find_indices(lon, lat, x1, y1)
    subset_data = ds.isel(nlat=slice(a, b), nlon=slice(c, d))
    return subset_data

