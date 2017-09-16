"""
Objects dealing with timeseries and ensemble statistics. All functions will
auto-check for type DataArray. If it is a DataArray, it will return a type
DataArray to ensure .apply() function from xarray can be applied.

Time Series
-----------
`remove_polynomial_fit` : Returns a time series with some order polynomial
removed.

"""
import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr

def remove_polynomial_fit(data, order):
    """
    Removes any order polynomial fit from a time series (including a linear
    fit). Returns the detrended time series.

    Parameters
    ----------
    data : array_like, can be an xr.DataArray type.
         Unfiltered time series.
    order : int
         Order of polynomial to be removed.

    Returns
    -------
    detrended_ts : array_like
         Time series with declared order polynomial removed.

    Examples
    --------
    import numpy as np
    import esmtools as et
    slope = np.arange(0, 100, 1) * 3
    noise = np.random.randn(100,)
    data = slope + noise
    detrended = es.statistics.remove_polynomial_fit(data, 4)
    """
    x = np.arange(0, len(data), 1)
    coefs = poly.polyfit(x, data, order)
    fit = poly.polyval(x, coefs)
    detrended_ts = (data - fit)
    if isinstance(data, xr.DataArray):
        return xr.DataArray(detrended_ts)
    else:
        return detrended_ts 
