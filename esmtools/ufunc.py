"""
Objects optimized for using in .apply() functions in xarray datasets. 

To add:
1. Calculate seasonal magnitude
2. Calculate value of linear slope

"""

import numpy as np
import numpy.polynomial.polynomial as poly

def remove_polynomial_fit(ds):
    """
    Removes a 4th order polynomial fit from an xarray dataset.
    Returns the detrended timeseries.

    Parameters
    ----------
    ds : xarray Dataset

    Returns
    -------
    detrended : xarray DataArray

    Examples
    --------
    """
    # Deals with cases where it is a NaN time series
    # which would otherwise break the fitting.
    if ds.min().isnull():
        return xr.DataArray(np.nan)
    else:
        x = np.arange(0, len(ds), 1)
        coefs = poly.polyfit(x, ds, 4)
        line_fit = poly.polyval(x, coefs)
        detrended = (ds - line_fit)
        return xr.DataArray(detrended)

def remove_linear_fit(ds):
    """
    Removes a linear fit from an xarray dataset.
    Returns the detrended timeseries.

    Parameters
    ----------
    ds : xarray Dataset

    Returns:
    -------
    detrended : xarray DataArray

    Examples
    --------
    """
    # Deals with cases where it is a NaN time series
    # which would otherwise break the fitting.
    if ds.min().isnull():
        return xr.DataArray(np.nan)
    else:
        x = np.arange(0, len(ds), 1)
        coefs = poly.polyfit(x, ds, 1)
        line_fit = poly.polyval(x, coefs)
        detrended = (ds - line_fit)
        return xr.DataArray(detrended)
