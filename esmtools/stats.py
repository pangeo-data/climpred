"""
Objects dealing with timeseries and ensemble statistics. All functions will
auto-check for type DataArray. If it is a DataArray, it will return a type
DataArray to ensure .apply() function from xarray can be applied.

Gridded Data
------------
`reg_aw` : Area-weights data on a regular (e.g. 180x360) grid.

Time Series
-----------
`remove_polynomial_fit` : Returns a time series with some order polynomial
removed.
`smooth_series` : Returns a smoothed time series.
`linear_regression` : Performs a least-squares linear regression.
`pearsonr` : Performs a Pearson linear correlation accounting for autocorrelation.

"""
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import xarray as xr
from scipy import stats
import scipy.stats as ss
from scipy.stats.stats import pearsonr as pr

def reg_aw(da, lat_coord='lat', lon_coord='lon', one_dimensional=True):
    """
    Area-weights data on a regular (e.g. 360x180) grid that does not come with
    cell areas. Uses cosine-weighting.

    Parameters
    ----------
    da : DataArray with longitude and latitude
    lat_coord : str (optional)
        Name of latitude coordinate
    lon_coord : str (optional)
        Name of longitude coordinate
    one_dimensional : bool (optional)
        If true, assumes that lat and lon are 1D (i.e. not a meshgrid)
    Returns
    -------
    aw_da : Area-weighted DataArray

    Examples
    --------
    import esmtools as et
    da_aw = et.stats.reg_aw(SST)
    """
    if one_dimensional:
        lon, lat = np.meshgrid(da[lon_coord], da[lat_coord])
    else:
        lat = da[lat_coord]
    # NaN out land to not go into area-weighting
    lat[np.isnan(da)] = np.nan
    cos_lat = np.cos(np.deg2rad(lat))
    aw_da = (da * cos_lat).sum() / np.nansum(np.cos(np.deg2rad(lat)))
    return aw_da


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

def pearsonr(x, y, two_sided=True):
    """
    Computes the Pearson product-moment coefficient of linear correlation. This
    version calculates the effective degrees of freedom, accounting for autocorrelation
    within each time series that could fluff the significance of the correlation.
    
    Parameters
    ----------
    x : array; independent variable
    y : array; predicted variable
    two_sided : boolean (optional); Whether or not the t-test should be two sided.

    Returns
    -------
    r     : r-value of correlation
    p     : p-value for significance
    n_eff : effective degrees of freedom

    References:
    ---------- 
    1. Wilks, Daniel S. Statistical methods in the atmospheric sciences. 
    Vol. 100. Academic press, 2011.
    2. Lovenduski, Nicole S., and Nicolas Gruber. "Impact of the Southern Annular Mode 
    on Southern Ocean circulation and biology." Geophysical Research Letters 32.11 (2005).

    Examples
    --------
    import numpy as np
    import esmtools as et
    x = np.random.randn(100,)
    y = np.random.randn(100,)
    r, p, n = et.stats.pearsonr(x, y)
    """
    r, _ = pr(x, y)
    # Compute effective sample size.
    n = len(x)
    xa, ya = x - np.mean(x), y - np.mean(y)
    xauto, _ = pr(xa[1:], xa[:-1])
    yauto, _ = pr(ya[1:], ya[:-1])
    n_eff = n * (1 - xauto*yauto)/(1 + xauto*yauto)
    n_eff = np.floor(n_eff)
    # Compute t-statistic.
    t = r * np.sqrt((n_eff - 2)/(1 - r**2))
    # Compute p-value.
    if two_sided:
        p = ss.t.sf(np.abs(t), n_eff-1)*2
    else:
        p = ss.t.sf(np.abs(t), n_eff-1)
    return r, p, n_eff
    

    
