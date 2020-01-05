import numpy as np
from scipy.stats import distributions
from xskillscore.core.np_deterministic import _pearson_r, _spearman_r


def _match_nans(a, b, weights):
    """
    Considers missing values pairwise. If a value is missing
    in a, the corresponding value in b is turned to nan, and
    vice versa.
    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at
        pairwise locations.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a[idx], b[idx] = np.nan, np.nan
        if weights is not None:
            weights[idx] = np.nan
    return a, b, weights


def _get_numpy_funcs(skipna):
    """
    Returns nansum and nanmean if skipna is True;
    Returns sum and mean if skipna is False.
    """
    if skipna:
        return np.nansum, np.nanmean
    else:
        return np.sum, np.mean


def __compute_anomalies(a, b, weights, axis, skipna):
    sumfunc, meanfunc = _get_numpy_funcs(skipna)
    # Only do weighted sums if there are weights. Cannot have a
    # single generic function with weights of all ones, because
    # the denominator gets inflated when there are masked regions.
    if weights is not None:
        ma = sumfunc(a * weights, axis=axis) / sumfunc(weights, axis=axis)
        mb = sumfunc(b * weights, axis=axis) / sumfunc(weights, axis=axis)
    else:
        ma = meanfunc(a, axis=axis)
        mb = meanfunc(b, axis=axis)
    am, bm = a - ma, b - mb
    return am, bm


def _effective_sample_size(a, b, axis, skipna):
    """Effective sample size for temporally correlated data.
    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.
    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to compute the effective sample size over.
    skipna : bool
        If True, skip NaNs when computing function.
    Returns
    -------
    n_eff : ndarray
        Effective sample size.
    Reference
    ---------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)

    # count total number of samples that are non-nan.
    n = np.count_nonzero(~np.isnan(a), axis=0)

    # compute lag-1 autocorrelation.
    am, bm = __compute_anomalies(a, b, weights=None, axis=0, skipna=skipna)
    a_auto = _pearson_r(am[0:-1], am[1::], weights=None, axis=0, skipna=skipna)
    b_auto = _pearson_r(bm[0:-1], bm[1::], weights=None, axis=0, skipna=skipna)

    # compute effective sample size per Bretherton et al. 1999
    n_eff = n * (1 - a_auto * b_auto) / (1 + a_auto * b_auto)
    n_eff = np.floor(n_eff)
    n_eff = np.clip(n_eff, 0, n)
    return n_eff


def _spearman_r_eff_p_value(a, b, axis, skipna):
    """Spearman rank correlation p value, accounting for autocorrelation.
    .. note::
        This metric should only be applied over the time dimension,
        since it is designed for temporal autocorrelation. Weights
        are not included due to the reliance on temporal
        autocorrelation.
    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    weights : ndarray
        Input array.
    skipna : bool
        If True, skip NaNs when computing function.
    Returns
    -------
    res : ndarray
        2-tailed p-value.
    See Also
    --------
    scipy.stats.spearmanr
    Reference
    ---------
    * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
      freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    * Wilks, Daniel S. Statistical methods in the atmospheric sciences. Vol. 100.
      Academic press, 2011.
    """
    if skipna:
        a, b, _ = _match_nans(a, b, None)
    rs = _spearman_r(a, b, None, axis, skipna)
    dof = _effective_sample_size(a, b, axis, skipna) - 2
    t = rs * np.sqrt((dof / ((rs + 1.0) * (1.0 - rs))).clip(0))
    p = 2 * distributions.t.sf(np.abs(t), dof)
    return p
