"""
Objects dealing with any sort of spatial and temporal filtering of output.

Spatial
-------
- `find_indices` -- Find indices for a given lat/lon point on a meshgrid.

"""

import numpy as np

def find_indices(xgrid, ygrid, xpoint, ypoint):
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
    >>> from esmtools import filtering as etf
    >>> 

    """
    dx = xgrid - xpoint
    dy = ygrid - ypoint
    reduced_grid = abs(dx) + abs(dy)
    min_ix = np.nanargmin(reduced_grid)
    i, j = np.unravel_index(min_ix, reduced_grid.shape)
    return i, j
