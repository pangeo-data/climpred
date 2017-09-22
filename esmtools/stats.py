"""
Objects dealing with timeseries and ensemble statistics. All functions will
auto-check for type DataArray. If it is a DataArray, it will return a type
DataArray to ensure .apply() function from xarray can be applied.

Time Series
-----------
`remove_polynomial_fit` : Returns a time series with some order polynomial
removed.
`smooth_series` : Returns a smoothed time series.
`linear_regression` : Performs a least-squares linear regression.

"""
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr
from scipy import stats

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
    detrended = es.stats.remove_polynomial_fit(data, 4)
    """
    x = np.arange(0, len(data), 1)
    coefs = poly.polyfit(x, data, order)
    fit = poly.polyval(x, coefs)
    detrended_ts = (data - fit)
    if isinstance(data, xr.DataArray):
        return xr.DataArray(detrended_ts)
    else:
        return detrended_ts 

def smooth_series(x, length, center=False):
    """
    Returns a smoothed version of the input timeseries.

    Parameters
    ----------
    x : array_like, unsmoothed timeseries.
    length : int, number of time steps to smooth over
    center : boolean
        Whether to center the smoothing filter or start from the beginning..

    Returns 
    -------
    smoothed : numpy array, smoothed timeseries

    Examples
    --------
    import numpy as np
    import esmtools as et
    x = np.random.rand(100)
    smoothed = et.stats.smooth_series(x, 12)
    """
    if isinstance(x, xr.DataArray):
        da = True
        x = np.asarray(x)
    x = pd.DataFrame(x)
    smoothed = pd.rolling_mean(x, length, center=center)
    smoothed = smoothed.dropna()
    smoothed = np.asarray(smoothed)
    if da == True:
        return xr.DataArray(smoothed).squeeze()
    else:
        return smoothed.squeeze()
    
def linear_regression(x, y):
    """
    Performs a simple least-squares linear regression.
    
    Parameters
    ----------
    x : array; independent variable
    y : array; predictor variable

    Returns
    -------
    m : slope
    b : intercept
    r : r-value
    p : p-value
    e : standard error

    Examples
    --------
    import numpy as np
    import esmtools as et
    x = np.random.randn(100)
    y = np.random.randn(100)
    m,b,r,p,e = et.stats.linear_regression(x,y)
    """
    m, b, r, p, e = stats.linregress(x, y)
    return m, b, r, p, e
