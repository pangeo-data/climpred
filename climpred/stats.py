"""Objects dealing with timeseries and ensemble statistics."""
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as ss
import xarray as xr
from scipy.signal import periodogram
from scipy.stats import norm

from xskillscore import pearson_r, pearson_r_p_value


# --------------------------------------------#
# HELPER FUNCTIONS
# Should only be used internally by esmtools.
# --------------------------------------------#
def _check_xarray(x):
    """Check if the object being submitted is either a Dataset or DataArray."""
    if not (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset)):
        typecheck = type(x)
        raise IOError(f"""The input data is not an xarray object (an xarray
            DataArray or Dataset). esmtools is built to wrap xarray to make
            use of its awesome features. Please input an xarray object and
            retry the function.

            Your input was of type: {typecheck}""")


def _get_coords(da):
    return list(da.coords)


def _get_dims(da):
    return list(da.dims)


def _get_vars(ds):
    return list(ds.data_vars)


# ----------------------------------#
# TIME SERIES
# Functions related to time series.
# ----------------------------------#
def xr_corr(x, y, dim='time', lag=0, two_sided=True, return_p=False):
    """Computes the Pearson product-moment coefficient of linear correlation.

    This version calculates the effective degrees of freedom, accounting
    for autocorrelation within each time series that could fluff the
    significance of the correlation.

    References:
        * Wilks, Daniel S. Statistical methods in the atmospheric sciences.
          Vol. 100. Academic press, 2011.
        * Lovenduski, Nicole S., and Nicolas Gruber. "Impact of the Southern
          Annular Mode on Southern Ocean circulation and biology." Geophysical
          Research Letters 32.11 (2005).

    Todo:
      * Test and adapt for xr.Datasets

    Args:
        x (xarray object): Independent variable time series or grid of time
                           series.
        y (xarray object): Dependent variable time series or grid of time
                           series
        dim (optional str): Correlation dimension
        lag (optional int): Lag to apply to correlaton, with x predicting y.
        two_sided (optonal bool): If True, compute a two-sided t-test.
        return_p (optional bool): If True, return correlation coefficients
                                  as well as p values.
    Returns:
        Pearson correlation coefficients

        If return_p True, associated p values.

    """
    _check_xarray(x)
    _check_xarray(y)
    if lag != 0:
        N = x[dim].size
        normal = x.isel({dim: slice(0, N-lag)})
        shifted = y.isel({dim: slice(0 + lag, N)})
        if dim not in list(x.coords):
            normal[dim] = np.arange(1, N)
        shifted[dim] = normal[dim]
        r = pearson_r(normal, shifted, dim)
    else:
        r = pearson_r(x, y, dim)
    if return_p:
        p = _xr_eff_p_value(x, y, r, dim, two_sided)
        # return with proper dimension labeling. would be easier with
        # apply_ufunc, but having trouble getting it to work here. issue
        # probably has to do with core dims.
        dimlist = _get_dims(r)
        for i in range(len(dimlist)):
            p = p.rename({'dim_' + str(i): dimlist[i]})
        return r, p
    else:
        return r


def _xr_eff_p_value(x, y, r, dim, two_sided):
    """Computes p values accounting for autocorrelation in time series.

    Args:
        x (xarray object): Independent time series.
        y (xarray object): Dependent time series.
        r (xarray object): Pearson correlations between x and y.
        dim (str): Dimension to compute compute p values over.
        two_sided (bool): If True, compute two-sided p value.

    Returns:
        p values accounting for autocorrelation in input time series.
    """
    def _compute_autocorr(v, dim, n):
        """
        Return normal and shifted time series
        with equal dimensions so as not to
        throw an error.
        """
        shifted = v.isel({dim: slice(1, n)})
        normal = v.isel({dim: slice(0, n-1)})
        # see explanation in xr_autocorr for this
        if dim not in list(v.coords):
            normal[dim] = np.arange(1, n)
        shifted[dim] = normal[dim]
        return pearson_r(shifted, normal, dim)

    n = x[dim].size
    # find autocorrelation
    xa, ya = x - x.mean(dim), y - y.mean(dim)
    xauto = _compute_autocorr(xa, dim, n)
    yauto = _compute_autocorr(ya, dim, n)
    # compute effective sample size
    n_eff = n * (1 - xauto * yauto) / (1 + xauto * yauto)
    n_eff = np.floor(n_eff)
    # constrain n_eff to be at maximum the total number of samples
    n_eff = n_eff.where(n_eff <= n, n)
    # compute t-statistic
    t = r * np.sqrt((n_eff - 2) / (1 - r**2))
    if two_sided:
        p = xr.DataArray(ss.t.sf(np.abs(t), n_eff - 1) * 2)
    else:
        p = xr.DataArray(ss.t.sf(np.abs(t), n_eff - 1))
    return p


def xr_rm_poly(ds, order, dim='time'):
    """Returns xarray object with nth-order fit removed.

    Args:
        ds (xarray object): Time series to be detrended.
        order (int): Order of polynomial fit to be removed.
        dim (optional str): Dimension over which to remove the polynomial fit.

    Returns:
        xarray object with polynomial fit removed.
    """
    _check_xarray(ds)

    def _get_metadata(ds):
        # copy is to make sure any edits to the dims or coords doesn't
        # affect the original dataset that is entered.
        dims = ds.copy().dims
        coords = ds.copy().coords
        return dims, coords

    def _swap_axes(y):
        """
        Push the independent axis up to the first dimension if needed. E.g.,
        the user submits a DataArray with dimensions ('lat','lon','time'), but
        wants the regression performed over time. This function expects the
        leading dimension to be the independent axis, so this subfunction just
        moves it up front.
        """
        dims, coords = _get_metadata(y)
        if dims[0] != dim:
            idx = dims.index(dim)
            y = np.swapaxes(y, 0, idx)
            # fixes bug with renaming dimensions that have
            # coordinates with differing shapes.
            del y.coords[dim], y.coords[dims[0]]
            y = y.rename({dims[0]: dim,
                          dims[idx]: dims[0]})
            dims = list(dims)
            dims[0], dims[idx] = dims[idx], dims[0]
            dims = tuple(dims)
        return np.asarray(y), dims, coords

    def _reconstruct_ds(y, store_vars):
        """
        Rebuild the original dataset.
        """
        new_ds = xr.Dataset()
        for i, var in enumerate(store_vars):
            new_ds[var] = xr.DataArray(y.isel(variable=i))
        return new_ds

    if isinstance(ds, xr.Dataset):
        DATASET, store_vars = True, ds.data_vars
        y = []
        for var in ds.data_vars:
            y.append(ds[var])
        y = xr.concat(y, 'variable')
    else:
        DATASET = False
        y = ds
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
    detrended_ds = xr.DataArray(detrended_ts, dims=dims, coords=coords)
    if DATASET:
        detrended_ds = _reconstruct_ds(detrended_ds, store_vars)
    return detrended_ds


def xr_rm_trend(da, dim='time'):
    """Calls ``xr_rm_poly`` with an order 1 argument."""
    return xr_rm_poly(da, 1, dim=dim)


# # TODO: coords lon, lat get lost for curvilinear ds
def xr_varweighted_mean_period(ds, time_dim='time'):
    """Calculate the variance weighted mean period of time series.

    ..math:
        P_x = \sum_k V(f_k,x) / \sum_k f_k V(f_k,x)

    Reference:
      * Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value
        Decadal Predictability in a CGCM." Journal of Climate 23, no. 23
        (August 27, 2010): 6292-6311. https://doi.org/10/bwq92h.

    Args:
        ds (xarray object): Time series.
        time_dim (optional str): Name of time dimension.

    """
    _check_xarray(ds)

    def _create_dataset(ds, f, Pxx, time_dim):
        """
        Organize results of periodogram into clean dataset.
        """
        dimlist = [i for i in _get_dims(ds) if i not in [time_dim]]
        PSD = xr.DataArray(Pxx, dims=['freq'] + dimlist)
        PSD.coords['freq'] = f
        return PSD

    f, Pxx = periodogram(ds, axis=0, scaling='spectrum')
    PSD = _create_dataset(ds, f, Pxx, time_dim)
    T = PSD.sum('freq') / ((PSD * PSD.freq).sum('freq'))
    return T


def xr_autocorr(ds, lag=1, dim='time', return_p=False):
    """Calculate the lagged correlation of time series.

    Args:
        ds (xarray object): Time series or grid of time series.
        lag (optional int): Number of time steps to lag correlate to.
        dim (optional str): Name of dimension to autocorrelate over.
        return_p (optional bool): If True, return correlation coefficients
                                  and p values.

    Returns:
        Pearson correlation coefficients.

        If return_p, also returns their associated p values.
    """
    _check_xarray(ds)
    N = ds[dim].size
    normal = ds.isel({dim: slice(0, N - lag)})
    shifted = ds.isel({dim: slice(0 + lag, N)})
    """
    xskillscore pearson_r looks for the dimensions to be matching, but we
    shifted them so they probably won't be. This solution doesn't work
    if the user provides a dataset without a coordinate for the main
    dimension, so we need to create a dummy dimension in that case.
    """
    if dim not in list(ds.coords):
        normal[dim] = np.arange(1, N)
    shifted[dim] = normal[dim]
    r = pearson_r(normal, shifted, dim)
    if return_p:
        # NOTE: This assumes 2-tailed. Need to update xr_eff_pearsonr
        # to utilize xskillscore's metrics but then compute own effective
        # p-value with option for one-tailed.
        p = pearson_r_p_value(normal, shifted, dim)
        return r, p
    else:
        return r


def xr_decorrelation_time(da, r=20, dim='time'):
    """Calculate the decorrelaton time of a time series.

    .. math::
        tau_{d} = 1 + 2 * \sum_{k=1}^{\inf}(alpha_{k})^{k}

    Reference:
        * Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate
          Research. Cambridge ; New York: Cambridge University Press, 1999.,
          p.373

    Args:
        da (xarray object): Time series.
        r (optional int): Number of iterations to run the above formula.
        dim (optional str): Time dimension for xarray object.

    Returns:
        Decorrelation time of time series.

    """
    _check_xarray(da)
    one = da.mean(dim) / da.mean(dim)
    return one + 2 * xr.concat([xr_autocorr(da, dim=dim, lag=i) ** i for i in
                                range(1, r)], 'it').sum('it')


# --------------------------------------------#
# Diagnostic Potential Predictability (DPP)
# Functions related to DPP from Boer et al.
# --------------------------------------------#
# # TODO: coords lon, lat get lost for curvilinear ds
def DPP(ds, m=10, chunk=True):
    """
    Calculate Diagnostic Potential Predictability (DPP) as potentially
    predictable variance fraction (ppvf) in Boer 2004.

    Note: Resplandy et al. 2015 and Seferian et al. 2018 calculate unbiased DPP
    in a slightly different way. chunk=False

    .. math::

        DPP_{\text{unbiased}}(m)=\frac{\sigma^2_m - 1/m \cdot \sigma^2}{\sigma^2}

    References:
    * Boer, G. J. “Long Time-Scale Potential Predictability in an Ensemble of
        Coupled Climate Models.” Climate Dynamics 23, no. 1 (August 1, 2004):
        29–44. https://doi.org/10/csjjbh.
    * Resplandy, L., R. Séférian, and L. Bopp. “Natural Variability of CO2 and
        O2 Fluxes: What Can We Learn from Centuries-Long Climate Models
        Simulations?” Journal of Geophysical Research: Oceans 120, no. 1
        (January 2015): 384–404. https://doi.org/10/f63c3h.
    * Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing the
        Decadal Predictability of Land and Ocean Carbon Uptake.” Geophysical
        Research Letters, March 15, 2018. https://doi.org/10/gdb424.

    Args:
    ds (xr.DataArray): control simulation with time dimension as years.
    m (optional int): separation time scale in years between predictable
                      low-freq component and high-freq noise.
    chunk (optional boolean): Whether chunking is applied. Default: True.
                    If False, then uses Resplandy 2015 / Seferian 2018 method.

    Returns:
        dpp (xr.DataArray): ds without time dimension.

    """
    # TODO: rename or find xr equiv
    def _chunking(ds, number_chunks=False, chunk_length=False):
        """
        Separate data into chunks and reshapes chunks in a c dimension.

        Specify either the number chunks or the length of chunks.
        Needed for DPP.

        Args:
            ds (xr.DataArray): control simulation with time dimension as years.
            chunk_length (int): see DPP(m)
            number_chunks (int): number of chunks in the return data.

        Returns:
            c (xr.DataArray): chunked ds, but with additional dimension c.

        """
        if number_chunks and not chunk_length:
            chunk_length = np.floor(ds['time'].size / number_chunks)
            cmin = int(ds['time'].min())
        elif not number_chunks and chunk_length:
            cmin = int(ds['time'].min())
            number_chunks = int(np.floor(ds['time'].size / chunk_length))
        else:
            raise ValueError('set number_chunks or chunk_length to True')
        c = ds.sel(time=slice(cmin, cmin + chunk_length - 1))
        c = c.expand_dims('c')
        c['c'] = [0]
        for i in range(1, number_chunks):
            c2 = ds.sel(time=slice(cmin + chunk_length * i,
                                   cmin + (i + 1) * chunk_length - 1))
            c2 = c2.expand_dims('c')
            c2['c'] = [i]
            c2['time'] = c['time']
            c = xr.concat([c, c2], 'c')
        return c

    if not chunk:  # Resplandy 2015, Seferian 2018
        s2v = ds.rolling(time=m).mean().var('time')
        s2 = ds.var('time')

    if chunk:  # Boer 2004 ppvf
        # first chunk
        chunked_means = _chunking(
            ds, chunk_length=m).mean('time')
        # sub means in chunks
        chunked_deviations = _chunking(
            ds, chunk_length=m) - chunked_means
        s2v = chunked_means.var('c')
        s2e = chunked_deviations.var(['time', 'c'])
        s2 = s2v + s2e
    dpp = (s2v - s2 / (m)) / s2
    return dpp


# -------
# Z SCORE
# -------
def _z_score(ci):
    """Returns critical z score given a confidence interval

    Source: https://stackoverflow.com/questions/20864847/
            probability-to-z-score-and-vice-versa-in-python
    """
    diff = (100 - ci) / 2
    return norm.ppf((100 - diff) / 100)


def z_significance(r1, r2, N, ci=90):
    """Computes the z test statistic for two ACC time series, e.g. an
       initialized ensemble ACC and persistence forecast ACC.

    Inputs:
        r1, r2: (xarray objects) time series, grids, etc. of pearson
                correlation coefficients between the two prediction systems
                of interest.
        N: (int) length of original time series being correlated.
        ci: (optional int) confidence level for z-statistic test

    Returns:
        Boolean array of same dimensions as input where True means r1 is
        significantly different from r2 at ci.

    Reference:
        https://www.statisticssolutions.com/comparing-correlation-coefficients/
    """
    def _r_to_z(r):
        """Fisher's r to z transformation"""
        return 0.5 * (np.log(1 + r) - np.log(1 - r))

    z1, z2 = _r_to_z(r1), _r_to_z(r2)
    difference = np.abs(z1 - z2)
    zo = difference / (np.sqrt(2*(1 / (N - 3))))
    confidence = np.zeros_like(zo)
    confidence[:] = _z_score(ci)
    sig = xr.DataArray(zo > confidence)
    return sig
