"""
Objects optimized for using in .apply() functions in xarray datasets. 

Functions
---------
- `remove_polynomial_fit` : Removes a 4th order fit from a time series.
- `remove_linear_fit` : Removes a 1st order fit from a time series.
- `compute_slope` : Computes the slope of a linear regression across a grid.
- `seasonal_magnitude` : Compute the seasonal magnitude of a time series.

"""

import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr

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

def compute_slope(ds):
    """
    Calculates the value of the linear slope over the inputted timeseries.
    Returns a 2D grid of slope values, which can be multiplied to get
    the total change over the period of interest.

    Parameters
    ----------
    ds : xarray Dataset

    Returns
    -------
    m  : DataArray of linear slope value.
     
    """
    # Deals with cases where it is a NaN time series
    # which would otherwise break the fitting.
    if ds.min().isnull():
        return xr.DataArray(np.nan)
    else:
        x = np.arange(0, len(ds), 1)
        coefs = poly.polyfit(x, ds, 1)
        m = coefs[1]
        return xr.DataArray(m)

def seasonal_magnitude(ds):
    """
    Computes the magnitude of the seasonal cycle, given a timeseries with
    a clear seasonal component.

    Parameters
    ----------
    ds : xarray Dataset

    Returns
    -------
    magnitude : DataArray of the value of the seasonal magnitude.

    Method
    ------
    The seasonal varying timeseries is detrended with a 4th order fit, and then
    the monthly climatology of the timeseries is found. The magnitude is
    considered the standard deviation of that monthly climatology.
    """
    if ds.min().isnull():
        return xr.DataArray(np.nan)
    else:
        # Could obviously do this chain of events with the above
        # remove_polynomial_fit ufunc, but I get some errors when
        # passing one .apply() func to another. Safe to do it
        # this way for now.
        x = np.arange(0, len(ds), 1)
        coefs = poly.polyfit(x, ds, 4)
        poly_fit = poly.polyval(x, coefs)
        seasonality = (ds - poly_fit)
        climatology = seasonality.groupby('time.month').mean()
        magnitude = climatology.std()
        return xr.DataArray(magnitude)
