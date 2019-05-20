"""Objects dealing with timeseries and ensemble statistics."""
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.stats as ss
import xarray as xr
from scipy.signal import periodogram
from scipy.stats import norm
from xskillscore import pearson_r, pearson_r_p_value

from .utils import (get_dims, check_xarray)


# ----------------------------------#
# TIME SERIES
# Functions related to time series.
# ----------------------------------#
def xr_corr(x, y, dim='time', lag=0, return_p=False):
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
        return_p (optional bool): If True, return correlation coefficients
                                  as well as p values.
    Returns:
        Pearson correlation coefficients

        If return_p True, associated p values.

    """
    check_xarray(x)
    check_xarray(y)
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
        p = _xr_eff_p_value(x, y, r, dim)
        # return with proper dimension labeling. would be easier with
        # apply_ufunc, but having trouble getting it to work here. issue
        # probably has to do with core dims.
        dimlist = get_dims(r)
        for i in range(len(dimlist)):
            p = p.rename({'dim_' + str(i): dimlist[i]})
        return r, p
    else:
        return r


def _xr_eff_p_value(x, y, r, dim):
    """Computes p values accounting for autocorrelation in time series.

    Args:
        x (xarray object): Independent time series.
        y (xarray object): Dependent time series.
        r (xarray object): Pearson correlations between x and y.
        dim (str): Dimension to compute compute p values over.

    Returns:
        p values accounting for autocorrelation in input time series.

    References:
        * Wilks, Daniel S. Statistical methods in the atmospheric sciences.
          Vol. 100. Academic press, 2011.
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
    p = xr.DataArray(ss.t.sf(np.abs(t), n_eff - 2) * 2)
    return p


@check_xarray(0)
def xr_rm_poly(ds, order, dim='time'):
    """Returns xarray object with nth-order fit removed.

    Args:
        ds (xarray object): Time series to be detrended.
        order (int): Order of polynomial fit to be removed.
        dim (optional str): Dimension over which to remove the polynomial fit.

    Returns:
        xarray object with polynomial fit removed.
    """
    if dim not in ds.dims:
        raise KeyError(
            f"Input dim, '{dim}', was not found in the ds; "
            f"found only the following dims: {list(ds.dims)}."
        )

    # handle both datasets and dataarray
    if isinstance(ds, xr.Dataset):
        da = ds.to_array()
        return_ds = True
    else:
        da = ds.copy()
        return_ds = False

    da_dims_orig = list(da.dims)  # orig -> original
    if len(da_dims_orig) > 1:
        # want independent axis to be the leading dimension
        da_dims_swap = da_dims_orig.copy()  # copy to prevent contamination

        # https://stackoverflow.com/questions/1014523/
        # simple-syntax-for-bringing-a-list-element-to-the-front-in-python
        da_dims_swap.insert(0, da_dims_swap.pop(da_dims_swap.index(dim)))
        da = da.transpose(*da_dims_swap)

        # hide other dims into a single dim
        da = da.stack({'other_dims': da_dims_swap[1:]})
        dims_swapped = True
    else:
        dims_swapped = False

    # NaNs will make the polyfit fail--interpolate any NaNs in
    # the provided dim to prevent poor fit, while other dims' NaNs
    # will be filled with 0s; however, all NaNs will be replaced
    # in the final output
    nan_locs = np.isnan(da.values)

    # any(nan_locs.sum(axis=0)) fails if not 2D
    if nan_locs.ndim == 1:
        nan_locs = nan_locs.reshape(len(nan_locs), 1)
        nan_reshaped = True
    else:
        nan_reshaped = False

    # check if there's any NaNs in the provided dim because
    # interpolate_na is computationally expensive to run regardless of NaNs
    if any(nan_locs.sum(axis=0)) > 0:
        if any(nan_locs[0, :]):
            # [np.nan, 1, 2], no first value to interpolate from; back fill
            da = da.bfill(dim)
        elif any(nan_locs[-1, :]):
            # [0, 1, np.nan], no last value to interpolate from; forward fill
            da = da.ffill(dim)
        else:  # [0, np.nan, 2], can interpolate
            da = da.interpolate_na(dim)

    # this handles the other axes; doesn't matter since it won't affect the fit
    da = da.fillna(0)

    # the actual operation of detrending
    y = da.values
    x = np.arange(0, len(y), 1)
    coefs = poly.polyfit(x, y, order)
    fit = poly.polyval(x, coefs)
    y_dt = y - fit.transpose()  # dt -> detrended
    da.data[:] = y_dt

    # replace back the filled NaNs (keep values where not NaN)
    if nan_reshaped:
        nan_locs = nan_locs[:, 0]
    da = da.where(~nan_locs)

    if dims_swapped:
        # revert the other dimensions to its original form and ordering
        da = da.unstack('other_dims').transpose(*da_dims_orig)

    if return_ds:
        # revert back into a dataset
        return xr.merge(da.sel(variable=var).rename(var).drop('variable')
                        for var in da['variable'].values)
    else:
        return da


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
    check_xarray(ds)

    def _create_dataset(ds, f, Pxx, time_dim):
        """
        Organize results of periodogram into clean dataset.
        """
        dimlist = [i for i in get_dims(ds) if i not in [time_dim]]
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
    check_xarray(ds)
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
    check_xarray(da)
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
