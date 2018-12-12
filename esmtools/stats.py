import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import scipy.stats as ss
import xarray as xr
from scipy import stats as ss
from scipy.stats import linregress
from scipy.stats.stats import pearsonr as pr
from scipy.signal import periodogram
from xskillscore import pearson_r


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
`vectorized_regression` : Performs a linear regression on a grid of data.
`remove_polynomial_vectorized` : Returns a time series with some order
polynomial removed. Useful for a grid, since it's vectorized.
`pearsonr` : Performs a Pearson linear correlation accounting for autocorrelation.

"""


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
    m, b, r, p, e = linregress(x, y)
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
    n_eff = n * (1 - xauto * yauto) / (1 + xauto * yauto)
    n_eff = np.floor(n_eff)
    # Compute t-statistic.
    t = r * np.sqrt((n_eff - 2) / (1 - r**2))
    # Compute p-value.
    if two_sided:
        p = ss.t.sf(np.abs(t), n_eff - 1) * 2
    else:
        p = ss.t.sf(np.abs(t), n_eff - 1)
    return r, p, n_eff


def vectorized_regression(x, y):
    """
    Vectorized function for regressing many time series onto a fixed time
    series. Most commonly used for regressing a grid of data onto some
    time series.

    Input
    -----
    x : array_like
      Time series of independent values (time, climate index, etc.)
    y : array_like
      Grid of time series to act as dependent values (SST, FG_CO2, etc.)

    Returns
    -------
    m : array_like
      Grid of slopes from linear regression

    Examples
    --------
    """
    print("Make sure that time is the first dimension in your inputs.")
    try: 
        if np.isnan(x).any():
            raise ValueError("Please supply an independent axis (x) without nans.")
    except TypeError:
        print("It's probably best to pass x as an array of integers.")
    # convert to numpy array if xarray
    if isinstance(y, xr.DataArray):
        XARRAY = True
        dim1 = y.dims[1]
        dim2 = y.dims[2]
        y = np.asarray(y)
    x = np.arange(0, len(x), 1)
    data_shape = y.shape
    y = y.reshape((data_shape[0], -1))
    # NaNs screw up vectorized regression; just fill with zeros.
    y[np.isnan(y)] = 0
    coefs = poly.polyfit(x, y, deg=1)
    m = coefs[1].reshape((data_shape[1], data_shape[2]))
    m[m == 0] = np.nan
    if XARRAY:
        m = xr.DataArray(m, dims=[dim1, dim2])
    return m


def vectorized_rm_poly(y, order=1):
    """
    Vectorized function for removing a order-th order polynomial fit of a time
    series

    Input
    -----
    y : array_like
      Grid of time series to act as dependent values (SST, FG_CO2, etc.)

    Returns
    -------
    detrended_ts : array_like
      Grid of detrended time series
    """
    # print("Make sure that time is the first dimension in your inputs.")
    if np.isnan(y).any():
        raise ValueError("Please supply an independent axis (y) without nans.")
    # convert to numpy array if xarray
    if isinstance(y, xr.DataArray):
        XARRAY = True
        dims = y.dims
        coords = y.coords
        y = np.asarray(y)
    data_shape = y.shape
    y = y.reshape((data_shape[0], -1))
    # NaNs screw up vectorized regression; just fill with zeros.
    y[np.isnan(y)] = 0
    x = np.arange(0, len(y), 1)
    coefs = poly.polyfit(x, y, order)
    fit = poly.polyval(x, coefs)
    detrended_ts = (y - fit.T)
    detrended_ts = detrended_ts.reshape(data_shape)
    if XARRAY:
        detrended_ts = xr.DataArray(detrended_ts, dims=dims, coords=coords)
    return detrended_ts


def vec_linregress(ds, dim='time'):
    """Linear trend of a dataset against dimension.

    Slow on lon,lat data
    Works on High-dim datasets without lon,lat
    """
    return xr.apply_ufunc(linregress, ds[dim], ds,
                          input_core_dims=[[dim], [dim]],
                          output_core_dims=[[], [], [], [], []],
                          vectorize=True)


def vec_rm_trend(ds, dim='year'):
    """Remove linear trend from a high-dim dataset."""
    s, i, _, _, _ = vec_linregress(ds, dim)
    new = ds - (s * (ds[dim] - ds[dim].values[0]))
    return new


def taper(x, p):
    from scipy.signal import tukey
    window = tukey(len(x), p)
    y = x * window
    return y


def create_power_spectrum(s, pLow=0.05):
    """Create power spectrum with CI for a given pd.series.

    Reference
    ---------
    - /ncl-6.4.0-gccsys/lib/ncarg/nclscripts/csm/shea_util.ncl

    Parameters
    ----------
    s : pd.series
        input time series
    pLow : float (default 0.05)
        significance interval for markov red-noise spectrum

    Returns
    -------
    p : np.ndarray
        period
    Pxx_den : np.ndarray
        power spectrum
    markov : np.ndarray
        theoretical markov red noise spectrum
    low_ci : np.ndarray
        lower confidence interval
    high_ci : np.ndarray
        upper confidence interval
    """
    from scipy.stats import chi2
    from scipy.signal import detrend, periodogram
    # input parameters
    # pct: percent of the series to be tapered (0.0 <= pct <= 1.0). If pct =0.0, no tapering will be done. If pct = 1.0, the whole series is affected. A value of 0.10 is common (tapering should always be done).
    pct = 0.1  # 0--1
    jave = 1  # smoothing ### DOESNT WORK HERE FOR VALUES OTHER THAN 1 !!!
    tapcf = 0.5 * (128 - 93 * pct) / (8 - 5 * pct)**2
    wgts = np.linspace(1., 1., jave)
    sdof = 2 / (tapcf * np.sum(wgts**2))
    pHigh = 1 - pLow
    data = s - s.mean()
    # detrend
    data = detrend(data)
    data = taper(data, pct)
    # periodigram
    timestep = 1
    frequency, power_spectrum = periodogram(data, timestep)
    Period = 1 / frequency
    power_spectrum_smoothed = pd.Series(power_spectrum).rolling(jave, 1).mean()
    # markov theo red noise spectrum
    twopi = 2. * np.pi
    r = s.autocorr()
    temp = r * 2. * np.cos(twopi * frequency)  # vector
    mkov = 1. / (1 + r**2 - temp)  # Markov model
    sum1 = np.sum(mkov)
    sum2 = np.sum(power_spectrum_smoothed)
    scale = sum2 / sum1
    xLow = chi2.ppf(pLow, sdof) / sdof
    xHigh = chi2.ppf(pHigh, sdof) / sdof
    # output
    markov = mkov * scale  # theor Markov spectrum
    low_ci = markov * xLow  # confidence
    high_ci = markov * xHigh  # interval
    return Period, power_spectrum_smoothed, markov, low_ci, high_ci


def vec_varweighted_mean_period(ds):
    """Calculate the variance weighted mean period of an xr.DataArray.

    Reference
    ---------
    - Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value Decadal
      Predictability in a CGCM.” Journal of Climate 23, no. 23 (August 27, 2010):
      6292–6311. https://doi.org/10/bwq92h.

    """
    f, Pxx = periodogram(ds, axis=0, scaling='spectrum')
    F = xr.DataArray(f)
    PSD = xr.DataArray(Pxx)
    T = PSD.sum('dim_0') / ((PSD * F).sum('dim_0'))
    coords = ds.isel(year=0).coords
    dims = ds.isel(year=0).dims
    T = xr.DataArray(data=T.values, coords=coords, dims=dims)
    return T

def xr_corr(ds, lag=1, dim='year'):
    """Calculated lagged correlation of a xr.Dataset."""
    first = ds[dim].values[0]
    last = ds[dim].values[-1]
    normal = ds.sel(year=slice(first, last - lag))
    shifted = ds.sel(year=slice(first + lag, last))
    shifted['year'] = normal.year
    return pearson_r(normal, shifted, dim)


def vec_tau_d(da, r=20, dim='year'):
    """Calculate decorrelation time of an xr.DataArray.

    tau_d = 1 + 2 * sum_{k=1}^(infinity)(alpha_k)

    Reference
    ---------
    - Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate
    Research. Cambridge ; New York: Cambridge University Press, 1999., p.373

    """
    one = da.mean(dim) / da.mean(dim)
    return one + 2 * xr.concat([xr_corr(da, lag=i) for i in range(1, r)], 'it').sum('it')
