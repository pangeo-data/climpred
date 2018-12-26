"""
Objects dealing with timeseries and ensemble statistics. All functions will
auto-check for type DataArray. If it is a DataArray, it will return a type
DataArray to ensure .apply() function from xarray can be applied.

Area-weighting
------------
`xr_cos_weight`: Area-weights output or observations without grid cell area
                 information using cosine weighting.
`xr_area_weight`: Area-weights output with grid cell area information.

Time Series
-----------
`xr_smooth_series` : Returns a smoothed time series.
`xr_linregress` : Returns results of linear regression over input dataarray.
`xr_eff_pearsonr` : Computes pearsonr between two time series accounting for autocorrelation.
`xr_rm_poly` : Returns time series with polynomial fit removed.
`xr_rm_trend` : Returns detrended (first order) time series.
`xr_varweighted_mean_period` : Calculates the variance weighted mean period of time series.
`xr_autocorr` : Calculates the autocorrelation of time series over some lag.
`xr_tau_d` : Calculates the decorrelation time of a time series.

Generate Time Series Data
-------------------------
`create_power_spectrum` : Creates power spectrum with CI for a given pd.series.
"""
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import scipy.stats as ss
import xarray as xr
from scipy.stats import linregress
from scipy.stats.stats import pearsonr as pr
from scipy.signal import tukey
from scipy.stats import chi2
from scipy.signal import detrend, periodogram
#--------------------------------------------#
# HELPER FUNCTIONS
# Should only be used internally by esmtools.
#--------------------------------------------#
def _check_xarray(x):
    """
    Check if the object being submitted to a given function is either a 
    Dataset or DataArray. This is important since `esmtools` is built as an
    xarray wrapper.

    TODO: Move this to a generalized util.py module with any other functions
    that are being called in other submodules.
    """
    if not (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset)):
        typecheck = type(x)
        raise IOError(f"""The input data is not an xarray object (an xarray
            DataArray or Dataset). esmtools is built to wrap xarray to make
            use of its awesome features. Please input an xarray object and
            retry the function.

            Your input was of type: {typecheck}""")


def _get_coords(da):
    """
    Simple function to retrieve dimensions from a given dataset/dataarray.

    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(da.coords)


def _get_dims(da):
    """
    Simple function to retrieve dimensions from a given dataset/datarray.

    Currently returns as a list, but can add keyword to select tuple or 
    list if desired for any reason.
    """
    return list(da.dims)


def _get_vars(ds):
    """
    Simple function to retrieve variables from a given dataset.

    Currently returns as a list, but can add keyword to select tuple or
    list if desired for any reason.
    """
    return list(ds.variables)


def _taper(x, p):
    """
    Description needed here.
    """
    window = tukey(len(x), p)
    y = x * window
    return y
#-------------------------------------------------------------------#
# AREA-WEIGHTING
# Functions related to area-weighting on grids with and without area
# information.
#-------------------------------------------------------------------#
def xr_cos_weight(da, lat_coord='lat', lon_coord='lon', one_dimensional=True):
    """
    Area-weights data on a regular (e.g. 360x180) grid that does not come with
    cell areas. Uses cosine-weighting.
    
    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.

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
    _check_xarray(da)
    if one_dimensional:
        lon, lat = np.meshgrid(da[lon_coord], da[lat_coord])
    else:
        lat = da[lat_coord]
    # NaN out land to not go into area-weighting
    lat[np.isnan(da)] = np.nan
    cos_lat = np.cos(np.deg2rad(lat))
    aw_da = (da * cos_lat).sum() / np.nansum(np.cos(np.deg2rad(lat)))
    return aw_da


def xr_area_weight(da, area_coord='area'):
    """
    Returns an area-weighted time series from the input xarray dataarray. This
    automatically figures out spatial dimensions vs. other dimensions. I.e.,
    this function works for just a single realization or for many realizations.
    
    See `reg_aw` if you have a regular (e.g. 360x180) grid that does not 
    contain cell areas.

    NOTE: This currently does not support datasets (of multiple variables)
    The user can alleviate this by using the .apply() function.

    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.

    Parameters
    ----------
    da : DataArray
    area_coord : str (defaults to 'area')
        Name of area coordinate if different from 'area' 

    Returns
    -------
    aw_da : Area-weighted DataArray
    """
    _check_xarray(da)
    area = da[area_coord]
    # Mask the area coordinate in case you've got a bunch of NaNs, e.g. a mask
    # or land.
    dimlist = _get_dims(da)
    # Pull out coordinates that aren't spatial. Time, ensemble members, etc.
    non_spatial = [i for i in dimlist if i not in _get_dims(area)]
    filter_dict = {}
    while len(non_spatial) > 0:
        filter_dict.update({non_spatial[0]: 0})
        non_spatial.pop(0)
    masked_area = area.where(da.isel(filter_dict).notnull())
    # Compute area-weighting.
    dimlist = _get_dims(masked_area)
    aw_da = da * masked_area
    # Sum over arbitrary number of dimensions.
    while len(dimlist) > 0:
        print(f'Summing over {dimlist[0]}')
        aw_da = aw_da.sum(dimlist[0])
        dimlist.pop(0)
    # Finish area-weighting by dividing by sum of area coordinate.
    aw_da = aw_da / masked_area.sum()
    return aw_da


#----------------------------------#
# TIME SERIES 
# Functions related to time series. 
#----------------------------------#
def xr_smooth_series(da, dim, length, center=True):
    """
    Returns a smoothed version of the input timeseries.
    
    NOTE: Currently explicitly writing `xr` as a prefix for xarray-specific
    definitions. Since `esmtools` is supposed to be a wrapper for xarray,
    this might be altered in the future.

    Parameters
    ----------
    da : xarray DataArray
    dim : str
        dimension to smooth over (e.g. 'time')
    length : int
        number of steps to smooth over for the given dim
    center : boolean (default to True)
        whether to center the smoothing filter or start from the beginning

    Returns
    -------
    smoothed : smoothed DataArray object 
    """
    _check_xarray(da)
    return da.rolling({dim: length}, center=center).mean()


def xr_linregress(da, dim='time', compact=True):
    """
    Computes the least-squares linear regression of a dataarray over some
    dimension (typically time).

    Parameters
    ----------
    da : xarray DataArray
    dim : str (default to 'time')
        dimension over which to compute the linear regression.
    compact : boolean (default to True)
        If true, return all results of linregress as a single dataset. 
        If false, return results as five separate DataArrays.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing slope, intercept, rvalue, pvalue, stderr from
        the linear regression. Excludes the dimension the regression was
        computed over. If compact is False, these five parameters are
        returned separately.
    """
    _check_xarray(da)
    results = xr.apply_ufunc(linregress, da[dim], da,
                          input_core_dims=[[dim], [dim]],
                          output_core_dims=[[], [], [], [], []],
                          vectorize=True, dask='parallelized')
    # Force into a cleaner dataset. The above function returns a dataset
    # with no clear labeling.
    ds = xr.Dataset()
    labels = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    for i, l in enumerate(labels):
        results[i].name = l
        ds = xr.merge([ds, results[i]])
    if compact:
        return ds
    else:
        return ds['slope'], ds['intercept'], ds['rvalue'], ds['pvalue'], \
               ds['stderr']


def xr_eff_pearsonr(ds, dim='time', two_sided=True):
    """
    Computes the Pearson product-moment coefficient of linear correlation. This
    version calculates the effective degrees of freedom, accounting for 
    autocorrelation within each time series that could fluff the significance
    of the correlation.

    This function is written to accept a dataset of arbitrary number of
    dimensions (e.g., lat, lon, depth).

    TODO: Add functionality for an ensemble.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing exactly two variables of the time series to be 
        correlated (e.g., ds.x and ds.y). This can contain any arbitrary
        number of dimensions in addition to the correlation dimension.
    dim : str (default 'time')
        The dimension over which to compute the correlation.
    two_sided : boolean (default True)
        Whether or not to do a two-sided t-test.

    Returns
    -------
    result : xarray Dataset
        Results of the linear correlation with r, p, and the effective 
        sample size for each time series being correlated.

    References:
    ----------
    1. Wilks, Daniel S. Statistical methods in the atmospheric sciences.
    Vol. 100. Academic press, 2011.
    2. Lovenduski, Nicole S., and Nicolas Gruber. "Impact of the Southern Annular Mode
    on Southern Ocean circulation and biology." Geophysical Research Letters 32.11 (2005).
    """
    _check_xarray(ds)
    def ufunc_pr(x, y, dim):
        """
        Internal ufunc to compute pearsonr over every grid cell.
        """
        return xr.apply_ufunc(pr, x, y,
                              input_core_dims=[[dim], [dim]],
                              output_core_dims=[[], []],
                              vectorize=True, dask='parallelized')


    varlist = _get_vars(ds)
    x, y = varlist[0], varlist[1]
    if len(varlist) < 2 or len(varlist) > 2:
        """
        The philosophy behind this function is to have a dataset containing
        gridded time series for two elements to be correlated.
        """
        raise ValueError("""Please supply an xarray dataset containing the two
            variables you would like correlated. In other words, it should have
            something like ds.x and ds.y that are either individual time series
            or a grid of time series.""")
    # Find raw pearson r. The effective sample size simply changes the threshold
    # for this to be significant.
    r, _ = ufunc_pr(ds[x], ds[y], dim)
    # Compute effective sample size
    n = len(ds[dim])
    # Find autocorrelation
    xa, ya = ds[x] - ds[x].mean(dim), ds[y] - ds[y].mean(dim)
    xauto, _ = ufunc_pr(xa.isel({dim: slice(1, n)}),
                        xa.isel({dim: slice(0, n-1)}), dim)
    yauto, _ = ufunc_pr(ya.isel({dim: slice(1, n)}),
                        ya.isel({dim: slice(0, n-1)}), dim)
    n_eff = n * (1 - xauto * yauto) / (1 + xauto * yauto)
    n_eff = np.floor(n_eff)
    # constrain n_eff to be at maximum the total number of samples.
    n_eff = n_eff.where(n_eff <= n, n)
    # compute t-statistic
    t = r * np.sqrt((n_eff - 2) / (1 - r**2))
    if two_sided:
        p = xr.DataArray(ss.t.sf(np.abs(t), n_eff - 1) * 2)
    else:
        p = xr.DataArray(ss.t.sf(np.abs(t), n_eff - 1))
    # return as a nice dataset
    # fix p dimension names
    dimlist = _get_dims(r)
    for i in range(len(dimlist)):
        p = p.rename({'dim_' + str(i): dimlist[i]})
    r.name, p.name, n_eff.name = 'r', 'p', 'n_eff'
    result = xr.merge([r, p, n_eff])
    return result 


def xr_rm_poly(da, order, dim='time'):
    """
    Returns xarray object with nth-order fit removed from every time series.

    Input
    -----
    da : xarray DataArray
        Single time series or many gridded time series of object to be 
        detrended
    order : int 
        Order of polynomial fit to be removed. If 1, this is functionally
        the same as calling `xr_rm_trend`
    dim : str (default 'time')
        Dimension over which to remove the polynomial fit.

    Returns
    -------
    detrended_ts : xarray DataArray
        DataArray with detrended time series.
    """
    _check_xarray(da)
    def _get_metadata(da):
        dims = da.dims
        coords = da.coords
        return dims, coords

    def _swap_axes(y):
        """
        Push the independent axis up to the first dimension if needed. E.g.,
        the user submits a DataArray with dimensions ('lat','lon','time'), but
        wants the regression performed over time. This function expects the
        leading dimension to be the independent axis, so this subfunction just
        moves it up front.
        """
        dims, coords = _get_metadata(da)
        if dims[0] != dim:
            idx = dims.index(dim)
            y = np.swapaxes(y, 0, idx)
            dims = list(dims)
            dims[0], dims[idx] = dims[idx], dims[0]
            dims = tuple(dims)
        return y, dims, coords


    y = np.asarray(da)
    # Force independent axis to be leading dimension.
    y, dims, coords = _swap_axes(y)
    data_shape = y.shape
    y = y.reshape((data_shape[0], -1))
    # NaNs screw up vectorized regression; just fill with zeros.
    y[np.isnan(y)] = 0
    x = np.arange(0, len(y), 1)
    coefs = poly.polyfit(x, y, order)
    fit = poly.polyval(x, coefs)
    detrended_ts = (y - fit.T).reshape(data_shape)
    # Replace NaNs.
    detrended_ts[detrended_ts == 0] = np.nan
    return xr.DataArray(detrended_ts, dims=dims, coords=coords)


def xr_rm_trend(da, dim='time'):
    """
    Calls xr_rm_poly with an order 1 argument.
    """
    return xr_rm_poly(da, 1, dim=dim) 


def xr_varweighted_mean_period(ds, time_dim='time'):
    """
    Calculate the variance weighted mean period of an xr.DataArray.

    Reference
    ---------
    - Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value Decadal
      Predictability in a CGCM.” Journal of Climate 23, no. 23 (August 27, 2010):
      6292–6311. https://doi.org/10/bwq92h.
    """
    _check_xarray(ds)
    def _create_dataset(ds, f, Pxx, time_dim):
        """
        Organize results of periodogram into clean dataset.
        """
        dimlist = [i for i in _get_dims(ds) if i not in time_dim]
        PSD = xr.DataArray(Pxx, dims=['freq'] + dimlist)
        PSD.coords['freq'] = f
        return PSD

    f, Pxx = periodogram(ds, axis=0, scaling='spectrum')
    PSD = _create_dataset(ds, f, Pxx, time_dim)
    T = PSD.sum('freq') / ((PSD * PSD.freq).sum('freq'))
    return xr.DataArray(T, coords=ds.isel({time_dim: 0}).coords) 


def xr_autocorr(da, lag=1, dim='time', r_only=False):
    """
    Calculated lagged correlation of a xr.Dataset.

    Parameters
    ----------
    da : xarray dataset/dataarray
    lag : int (default 1)
        number of time steps to lag correlate.
    dim : str (default 'time')
        name of time dimension
    r_only : boolean (default False)
       If true, return just the correlation coefficient.
       If false, return both the correlation coefficient and p-value.

    Returns
    -------
    r : Pearson correlation coefficient
    p : p-value
    """
    _check_xarray(da)
    N = len(da[dim])
    da1 = da.isel({dim: slice(0, N - lag)})
    da2 = da.isel({dim: slice(0 + lag, N)})
    r, p = xr.apply_ufunc(pr, da1, da2,
                          input_core_dims=[[dim], [dim]],
                          output_core_dims=[[], []],
                          vectorize=True,
                          dask='parallelized')
    if r_only:
        return r
    else:
        return r, p 


def xr_tau_d(da, r=20, dim='time'):
    """
    Calculate decorrelation time of an xr.DataArray.
    
    tau_d = 1 + 2 * sum_{k=1}^(infinity)(alpha_k)

    Parameters
    ----------
    da : xarray object 
    r : int (default 20)
        Number of iterations to run of the above formula
    dim : str (default 'time')
        Time dimension for xarray object

    Reference
    ---------
    - Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate
    Research. Cambridge ; New York: Cambridge University Press, 1999., p.373

    """
    _check_xarray(da)
    one = da.mean(dim) / da.mean(dim)
    return one + 2 * xr.concat([xr_autocorr(da, lag=i, r_only=True) for i in range(1, r)], 'it').sum('it')
#--------------------------------------------#
# GENERATE TIME SERIES DATA
# Functions that create time series data
# for testing, etc.
#--------------------------------------------#
def create_power_spectrum(s, pct=0.1, pLow=0.05):
    """
    Create power spectrum with CI for a given pd.series.

    Reference
    ---------
    - /ncl-6.4.0-gccsys/lib/ncarg/nclscripts/csm/shea_util.ncl

    Parameters
    ----------
    s : pd.series
        input time series
    pct : float (default 0.10)
        percent of the time series to be tapered. (0 <= pct <= 1). If pct = 0,
        no tapering will be done. If pct = 1, the whole series is tapered.
        Tapering should always be done.
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
    # A value of 0.10 is common (tapering should always be done).
    jave = 1  # smoothing ### DOESNT WORK HERE FOR VALUES OTHER THAN 1 !!!
    tapcf = 0.5 * (128 - 93 * pct) / (8 - 5 * pct)**2
    wgts = np.linspace(1., 1., jave)
    sdof = 2 / (tapcf * np.sum(wgts**2))
    pHigh = 1 - pLow
    data = s - s.mean()
    # detrend
    data = detrend(data)
    data = _taper(data, pct)
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
