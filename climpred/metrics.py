import warnings

import numpy as np
import scipy
import xarray as xr
from scipy.stats import norm
from xskillscore import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    mae,
    mape,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
    threshold_brier_score,
)

# the import of CLIMPRED_DIMS from constants fails. currently fixed manually.
# from .constants import CLIMPRED_DIMS
CLIMPRED_DIMS = ['init', 'member', 'lead', 'time']


def _get_norm_factor(comparison):
    """Get normalization factor for normalizing distance metrics.

    A distance metric is normalized by the standard deviation or variance
    of a reference/control simulation. The goal of a normalized distance
    metric is to get a constant and comparable value of typically 1 (or 0 for
    metrics defined as 1 - metric), when the metric saturates and the predictability
    horizon is reached.

    To directly compare skill between different comparisons used, a factor is
    added in the normalized metric formula, see Seferian et al. 2018.
    For example, NRMSE gets smaller in comparison ``m2e`` than ``m2m`` by design,
    because the ensemble mean is always closer to individual ensemble members
    than ensemble members to each other.

     .. note::
         This is used for PPP, NMSE, NRMSE, MSSS, NMAE.

    Args:
        comparison (class): comparison class.

    Returns:
        fac (int): normalization factor.

    Raises:
        KeyError: if comparison is not matching.

    Example:
        >>> # check skill saturation value of roughly 1 for different comparisons
        >>> metric='nrmse'
        >>> for c in ['m2m', 'm2e', 'm2c', 'e2c']:
                s = compute_perfect_model(ds, control, metric=metric,  comparison=c)
                s.plot(label=' '.join([metric,c]))
        >>> plt.legend()

    Reference:
        * Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing
         the Decadal Predictability of Land and Ocean Carbon Uptake.”
         Geophysical Research Letters, March 15, 2018. https://doi.org/10/gdb424.


    """
    if comparison.name in ['m2e', 'e2c', 'e2r']:
        fac = 1
    elif comparison.name in ['m2c', 'm2m', 'm2r']:
        fac = 2
    else:
        raise KeyError('specify comparison to get normalization factor.')
    return fac


def _preprocess_dims(dim):
    """Convert input argument ``dim`` into a list of dimensions.

    Args:
        dim (str or list): The dimension(s) to apply the function along.

    Returns:
        dim (list): List of dimensions to apply function over.
    """
    if isinstance(dim, str):
        dim = [dim]
    return dim


def _display_metric_metadata(self):
    summary = '----- Metric metadata -----\n'
    summary += f'Name: {self.name}\n'
    summary += f'Alias: {self.aliases}\n'
    # positively oriented
    if self.positive:
        summary += 'Orientation: positive\n'
    else:
        summary += 'Orientation: negative\n'
    # probabilistic or deterministic
    if self.probabilistic:
        summary += 'Kind: probabilistic\n'
    else:
        summary += 'Kind: deterministic\n'
    summary += f'Power to units: {self.unit_power}\n'
    summary += f'long_name: {self.long_name}\n'
    summary += f'Minimum skill: {self.minimum}\n'
    summary += f'Maximum skill: {self.maximum}\n'
    summary += f'Perfect skill: {self.perfect}\n'
    summary += f'Strictly proper score: {self.proper}\n'
    # doc
    summary += f'Function: {self.function.__doc__}\n'
    return summary


class Metric:
    """Master class for all metrics."""

    def __init__(
        self,
        name,
        function,
        positive,
        probabilistic,
        unit_power,
        long_name=None,
        aliases=None,
        minimum=None,
        maximum=None,
        perfect=None,
        proper=None,
    ):
        """Metric initialization.

        Args:
            name (str): name of metric.
            function (function): metric function.
            positive (bool): Is metric positively oriented? Higher metric
             values means higher skill.
            probabilistic (bool): Is metric probabilistic? `False` means
             deterministic.
            unit_power (float, int): Power of the unit of skill based on unit
             of input, e.g. input unit [m]: skill unit [(m)**unit_power]
            long_name (str, optional): long_name of metric. Defaults to None.
            aliases (list of str, optional): Allowed aliases for this metric.
             Defaults to None.
            min (float, optional): Minimum skill for metric. Defaults to None.
            max (float, optional): Maxmimum skill for metric. Defaults to None.
            perfect (float, optional): Perfect skill for metric. Defaults to None.
            proper (bool, optional): Is strictly proper skill score?
             According to Gneitning & Raftery (2012).
             See https://en.wikipedia.org/wiki/Scoring_rule. Defaults to None.

        Returns:
            Metric: metric class Metric.

        """
        self.name = name
        self.function = function
        self.positive = positive
        self.probabilistic = probabilistic
        self.unit_power = unit_power
        self.long_name = long_name
        self.aliases = aliases
        self.minimum = minimum
        self.maximum = maximum
        self.perfect = perfect
        self.proper = proper

    def __repr__(self):
        """Show metadata of metric class."""
        return _display_metric_metadata(self)


#####################
# CORRELATION METRICS
#####################
def _pearson_r(forecast, reference, dim=None, **metric_kwargs):
    """Pearson product-moment correlation coefficient

    .. math::
        corr = \\frac{cov(f, o)}{\\sigma_{f}\\cdot\\sigma_{o}}

    .. note::
        Use metric ``pearson_r_p_value`` to get the corresponding pvalue.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 1
        * min: -1
        * max: 1

    See also:
        * xskillscore.pearson_r
        * xskillscore.pearson_r_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return pearson_r(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__pearson_r = Metric(
    name='pearson_r',
    function=_pearson_r,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name='Pearson product-moment correlation coefficient',
    aliases=['pr', 'acc', 'pacc'],
    minimum=-1.0,
    maximum=1.0,
    perfect=1.0,
)


def _pearson_r_p_value(forecast, reference, dim=None, **metric_kwargs):
    """Probability that forecast and reference are linearly uncorrelated

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    See also:
        * xskillscore.pearson_r_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    # p-value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pearson_r_p_value(
            forecast, reference, dim=dim, weights=weights, skipna=skipna
        )


__pearson_r_p_value = Metric(
    name='pearson_r_p_value',
    function=_pearson_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name='Pearson product-moment correlation coefficient p-value',
    aliases=['p_pval', 'pvalue', 'pval'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _effective_sample_size(forecast, reference, dim=None, **metric_kwargs):
    """Effective sample size for temporally correlated data.

    The effective sample size extracts the number of independent samples
    between two time series being correlated. This is derived by assessing
    the magnitude of lag-1 autocorrelation in each of the time series being
    correlated. A higher autocorrelation induces a lower effective sample
    size which raises the correlation coefficient required to achieve statistical
    significance.

    .. math::
        ESS = N\\left( \\frac{1 - \\rho_{1}\\rho_{2}}{1 + \\rho_{1}\\rho_{2}} \\right)

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.

    Range:
        * min: 1
        * max: np.inf

    References:
        * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
        freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    """

    def _autocorr(v, dim, n):
        """Return autocorrelation over the given lag.

        Args:
            v (xarray obj): DataArray or Dataset to compute autocorrelation with.
            dim (str): Dimension to compute autocorrelation over.
            n (int): Number of lags to compute autocorrelation for.

        Returns:
            Temporal autocorrelation at given lag.

        NOTE: Will switch to ``esmtools`` dependency once stats are moved over to there.
        """
        shifted = v.isel({dim: slice(1, n)})
        normal = v.isel({dim: slice(0, n - 1)})
        # see explanation in autocorr for this
        if dim not in list(v.coords):
            normal[dim] = np.arange(1, n)
        shifted[dim] = normal[dim]
        return pearson_r(shifted, normal, dim)

    # Preprocess same way as in ``xskillscore``
    dim = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        forecast = forecast.stack(**{new_dim: dim})
        reference = reference.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]

    # Need to implement ``skipna`` here. This doesn't check NaNs. Might be worth
    # having a ``np_metrics.py`` similar to ``xskillscore``.
    N = forecast[new_dim].size

    # compute lag-1 autocorrelation.
    fa, ra = forecast - forecast.mean(new_dim), reference - reference.mean(new_dim)
    # TODO: Switch this over to ``esmtools`` dependency once stats are moved over there.
    fauto = _autocorr(fa, new_dim, N)
    rauto = _autocorr(ra, new_dim, N)

    # compute effective sample size.
    n_eff = N * (1 - fauto * rauto) / (1 + fauto * rauto)
    n_eff = np.floor(n_eff)  # forces to integer

    # constrain n_eff to be at the maximum the total number of samples.
    n_eff = n_eff.where(n_eff <= N, N)
    return n_eff


__effective_sample_size = Metric(
    name='effective_sample_size',
    function=_effective_sample_size,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name='Effective sample size for temporally correlated data',
    aliases=['n_eff', 'eff_n'],
    minimum=0.0,
    maximum=np.inf,
)


def _pearson_r_eff_p_value(forecast, reference, dim=None, **metric_kwargs):
    """Probability that forecast and reference are linearly uncorrelated, accounting
    for autocorrelation.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    # TODO:
    # * PPP vs MSSS. Aren't we missing the ensemble normalization from the Pohlmann
    # paper?

    # * Go back through MurCSS/bias slope, etc. to clarify those in docs.
    # * CRPSS, CRPSSES

    # * Get $$ working for math in rst. Not just .. math:: directive.
    # * Double check keywords

    # * Threshold brier score math, etc. from properscoring

    # * Open issue that dim='member' doesn't work for hindcast.
    # * Open issue to include lat/lon etc. for dim on arguments.
    # * Open issue about weights not working.

    """

    def _calculate_p(t, n):
        """Calculates the p-value.

        Args:
            t (ndarray): t-test statistic.
            n (ndarray): number of samples.

        Returns:
            p (ndarray): Two-tailed p-value.
        """
        return scipy.stats.t.sf(np.abs(t), n - 2) * 2

    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)

    # compute effective sample size
    n_eff = _effective_sample_size(forecast, reference, dim)

    # compute t-statistic
    r = pearson_r(forecast, reference, dim=dim, weights=weights, skipna=skipna)
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))

    # compute effective p-value
    p = xr.apply_ufunc(
        _calculate_p, t, n_eff, dask='parallelized', output_dtypes=[float]
    )
    return p


__pearson_r_eff_p_value = Metric(
    name='pearson_r_eff_p_value',
    function=_pearson_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Pearson's Anomaly correlation coefficient "
        'p-value using the effective sample size'
    ),
    aliases=['p_pval_eff', 'pvalue_eff', 'pval_eff'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _spearman_r(forecast, reference, dim=None, **metric_kwargs):
    """Spearman's rank correlation coefficient.

    .. math::
        SACC = ACC(ranked(f), ranked(o))

    .. note::
        Use metric ``spearman_r_p_value`` to get the corresponding pvalue.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 1
        * min: -1

    See also:
        * xskillscore.spearman_r
        * xskillscore.spearman_r_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return spearman_r(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__spearman_r = Metric(
    name='spearman_r',
    function=_spearman_r,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name="Spearman's rank correlation coefficient",
    aliases=['sacc', 'sr'],
    minimum=-1.0,
    maximum=1.0,
    perfect=1.0,
)


def _spearman_r_p_value(forecast, reference, dim=None, **metric_kwargs):
    """Probability that forecast and reference are monotonically uncorrelated.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * max: 1
        * min: 0

    See also:
        * xskillscore.spearman_r_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    # p-value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return spearman_r_p_value(
            forecast, reference, dim=dim, weights=weights, skipna=skipna
        )


__spearman_r_p_value = Metric(
    name='spearman_r_p_value',
    function=_spearman_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name="Spearman's rank correlation coefficient p-value",
    aliases=['s_pval', 'spvalue', 'spval'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _spearman_r_eff_p_value(forecast, reference, dim=None, **metric_kwargs):
    """Probability that forecast and reference are monotonically uncorrelated,
    accounting for autocorrelation.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: 1
    """

    def _calculate_p(t, n):
        """Calculates the p-value.

        Args:
            t (ndarray): t-test statistic.
            n (ndarray): number of samples.

        Returns:
            p (ndarray): Two-tailed p-value.
        """
        return scipy.stats.t.sf(np.abs(t), n - 2) * 2

    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)

    # compute effective sample size
    n_eff = _effective_sample_size(forecast, reference, dim)

    # compute t-statistic
    r = spearman_r(forecast, reference, dim=dim, weights=weights, skipna=skipna)
    t = r * np.sqrt((n_eff - 2) / (1 - r ** 2))

    # compute effective p-value
    p = xr.apply_ufunc(
        _calculate_p, t, n_eff, dask='parallelized', output_dtypes=[float]
    )
    return p


__spearman_r_eff_p_value = Metric(
    name='spearman_r_eff_p_value',
    function=_spearman_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Spearman's Rank correlation coefficient "
        'p-value using the effective sample size'
    ),
    aliases=['s_pval_eff', 'spvalue_eff', 'spval_eff'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


##################
# DISTANCE METRICS
##################
def _mse(forecast, reference, dim=None, **metric_kwargs):
    """Mean Sqaure Error (MSE)

    .. math::
        MSE = \\overline{(f - o)^{2}}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mse
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mse(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__mse = Metric(
    name='mse',
    function=_mse,
    positive=False,
    probabilistic=False,
    unit_power=2,
    long_name='Mean Squared Error',
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _rmse(forecast, reference, dim=None, **metric_kwargs):
    """Root Mean Sqaure Error (RMSE)

    .. math::
        RMSE = \\sqrt{\\overline{(f - o)^{2}}}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.rmse
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return rmse(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__rmse = Metric(
    name='rmse',
    function=_rmse,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name='Root Mean Squared Error',
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _mae(forecast, reference, dim=None, **metric_kwargs):
    """Mean Absolute Error (MAE)

    .. math::
        MAE = \\overline{|f - o|}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mae
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mae(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__mae = Metric(
    name='mae',
    function=_mae,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name='Mean Absolute Error',
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _median_absolute_error(forecast, reference, dim=None, **metric_kwargs):
    """Median Absolute Error

    .. math::
        median(|f - o|)

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.median_absolute_error
    """
    skipna = metric_kwargs.get('skipna', False)
    return median_absolute_error(forecast, reference, dim=dim, skipna=skipna)


__median_absolute_error = Metric(
    name='median_absolute_error',
    function=_median_absolute_error,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name='Median Absolute Error',
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _mape(forecast, reference, dim=None, **metric_kwargs):
    """Mean Absolute Percentage Error (MAPE)

    .. math::
        MAPE = \\frac{1}{n} \\sum \\frac{|f-o|}{|o|}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    See also:
        * xskillscore.mape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mape(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__mape = Metric(
    name='mape',
    function=_mape,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Mean Absolute Percentage Error',
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _smape(forecast, reference, dim=None, **metric_kwargs):
    """Symmetric Mean Absolute Percentage Error (sMAPE)

    .. math::
        sMAPE = \\frac{1}{n} \\sum \\frac{|f-o|}{|f|+|o|}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    See also:
        * xskillscore.smape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return smape(forecast, reference, dim=dim, weights=weights, skipna=skipna)


__smape = Metric(
    name='smape',
    function=_smape,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='symmetric Mean Absolute Percentage Error',
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


#############################
# NORMALIZED DISTANCE METRICS
#############################
def _nmse(forecast, reference, dim=None, **metric_kwargs):
    """Normalized MSE (NMSE), also known as Normalized Ensemble Variance (NEV).

    .. math::
        NMSE = NEV = \\frac{MSE}{\\sigma^2_{o}\\cdot fac}
             = \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{o} \\cdot fac}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13,
          no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    mse_skill = __mse.function(forecast, reference, dim=dim, **metric_kwargs)
    var = reference.var(dim)
    if 'comparison' in metric_kwargs:
        comparison = metric_kwargs['comparison']
    else:
        raise ValueError(
            'Comparison needed to normalize NMSE. Not found in', metric_kwargs
        )
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


__nmse = Metric(
    name='nmse',
    function=_nmse,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Normalized Mean Squared Error',
    aliases=['nev'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _nmae(forecast, reference, dim=None, **metric_kwargs):
    """Normalized Mean Absolute Error (NMAE).

    .. math::
        NMAE = \\frac{MAE}{\\sigma_{o} \\cdot fac}
             = \\frac{\\overline{|f - o|}}{\\sigma_{o} \\cdot fac}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13, no.
          7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

    """
    mae_skill = __mae.function(forecast, reference, dim=dim, **metric_kwargs)
    std = reference.std(dim).mean()
    if 'comparison' in metric_kwargs:
        comparison = metric_kwargs['comparison']
    else:
        raise ValueError(
            'Comparison needed to normalize NMSE. Not found in', metric_kwargs
        )
    fac = _get_norm_factor(comparison)
    nmae_skill = mae_skill / std / fac
    return nmae_skill


__nmae = Metric(
    name='nmae',
    function=_nmae,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Normalized Mean Absolute Error',
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _nrmse(forecast, reference, dim=None, **metric_kwargs):
    """Normalized Root Mean Square Error (NRMSE).

    .. math::

        NRMSE = \\frac{RMSE}{\\sigma_{o}\\cdot\\sqrt{fac}}
              = \\sqrt{\\frac{MSE}{\\sigma^{2}_{o}\\cdot fac}}
              = \\sqrt{ \\frac{\\overline{(f - o)^{2}}}{ \\sigma^2_{o}\\cdot fac}}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Range:
        * 0: perfect forecast
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
      * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.”
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
      * Hawkins, Ed, Steffen Tietsche, Jonathan J. Day, Nathanael Melia, Keith
        Haines, and Sarah Keeley. “Aspects of Designing and Evaluating
        Seasonal-to-Interannual Arctic Sea-Ice Prediction Systems.” Quarterly
        Journal of the Royal Meteorological Society 142, no. 695
        (January 1, 2016): 672–83. https://doi.org/10/gfb3pn.

    """
    rmse_skill = __rmse.function(forecast, reference, dim=dim, **metric_kwargs)
    std = reference.std(dim)
    if 'comparison' in metric_kwargs:
        comparison = metric_kwargs['comparison']
    else:
        raise ValueError(
            'Comparison needed to normalize NRMSE. Not found in', metric_kwargs
        )
    fac = _get_norm_factor(comparison)
    nrmse_skill = rmse_skill / std / np.sqrt(fac)
    return nrmse_skill


__nrmse = Metric(
    name='nrmse',
    function=_nrmse,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Normalized Root Mean Squared Error',
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _ppp(forecast, reference, dim=None, **metric_kwargs):
    """Prognostic Potential Predictability (PPP).

    .. math::
        PPP = 1 - \\frac{MSE}{\\sigma^2_{ref} \\cdot fac} =
              1 - \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{ref} \\cdot fac}

    .. note::
        This is the same as the Mean Squared Skill Score.

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Range:
        * 1: perfect forecast
        * positive: better than climatology forecast
        * negative: worse than climatology forecast

    References:
      * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
        North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
        (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
      * Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.
      * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    """
    mse_skill = __mse.function(forecast, reference, dim=dim, **metric_kwargs)
    var = reference.var(dim)
    if 'comparison' in metric_kwargs:
        comparison = metric_kwargs['comparison']
    else:
        raise ValueError(
            'Comparison needed to normalize PPP. Not found in', metric_kwargs
        )
    fac = _get_norm_factor(comparison)
    ppp_skill = 1 - mse_skill / var / fac
    return ppp_skill


__ppp = Metric(
    name='ppp',
    function=_ppp,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name='Prognostic Potential Predictability',
    aliases=['ppp'],
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


def _uacc(forecast, reference, dim=None, **metric_kwargs):
    """Bushuk's unbiased Anomaly Correlation Coefficient (uACC).

    .. math::
        uACC = \\sqrt{PPP} = \\sqrt{MSSS}
             = \\sqrt{1 - \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{ref} \\cdot fac}}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Range:
        * 1: perfect
        * 0 - 1: better than climatology

    References:
        * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel
          Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional
          Arctic Sea–Ice Prediction: Potential versus Operational Seasonal
          Forecast Skill." Climate Dynamics, June 9, 2018.
          https://doi.org/10/gd7hfq.
    """
    ppp_res = __ppp.function(forecast, reference, dim=dim, **metric_kwargs)
    # ensure no sqrt of neg values
    uacc_res = (ppp_res.where(ppp_res > 0)) ** 0.5
    return uacc_res


__uacc = Metric(
    name='uacc',
    function=_uacc,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Bushuk's unbiased ACC",
    minimum=0.0,
    maximum=1.0,
    perfect=1.0,
)


##############################
# MURPHY DECOMPOSITION METRICS
##############################
def _std_ratio(forecast, reference, dim=None, **metric_kwargs):
    """Ratio of standard deviations of reference over forecast.

    .. math:: \\text{std ratio} = \\frac{\\sigma_o}{\\sigma_f}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    ratio = reference.std(dim) / forecast.std(dim)
    return ratio


__std_ratio = Metric(
    name='std_ratio',
    function=_std_ratio,
    positive=None,
    probabilistic=False,
    unit_power=0,
    long_name='Ratio of standard deviations',
    minimum=-np.inf,
    maximum=np.inf,
    perfect=1.0,
)


def _unconditional_bias(forecast, reference, dim=None, **metric_kwargs):
    """Unconditional bias.

    .. math::
        bias = f - o

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    Range:
        * pos: positive bias
        * neg: negative bias
        * perfect: 0

    References:
        * https://www.cawcr.gov.au/projects/verification/
        * https://www-miklip.dkrz.de/about/murcss/
    """
    bias = (forecast - reference).mean(dim)
    return bias


__unconditional_bias = Metric(
    name='unconditional_bias',
    function=_unconditional_bias,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name='Unconditional bias',
    aliases=['u_b', 'bias'],
    minimum=-np.inf,
    maximum=np.inf,
    perfect=0.0,
)


def _conditional_bias(forecast, reference, dim=None, **metric_kwargs):
    """Conditional bias between forecast and reference.

    .. math::
        \\text{conditional bias} = r_{fo} - \\frac{\\sigma_f}{\\sigma_o}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    acc = __pearson_r.function(forecast, reference, dim=dim, **metric_kwargs)
    conditional_bias = (
        acc - __std_ratio.function(forecast, reference, dim=dim, **metric_kwargs) ** -1
    )
    return conditional_bias


__conditional_bias = Metric(
    name='conditional_bias',
    function=_conditional_bias,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Conditional bias',
    aliases=['c_b', 'cond_bias'],
    minimum=-np.inf,
    maximum=1.0,
    perfect=0.0,
)


def _bias_slope(forecast, reference, dim=None, **metric_kwargs):
    """Bias slope between reference and forecast standard deviations.

    .. math::
        \\text{bias slope}= r_{fo} \\cdot \\text{std ratio}

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    std_ratio = __std_ratio.function(forecast, reference, dim=dim, **metric_kwargs)
    acc = __pearson_r.function(forecast, reference, dim=dim, **metric_kwargs)
    b_s = std_ratio * acc
    return b_s


__bias_slope = Metric(
    name='bias_slope',
    function=_bias_slope,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Bias slope',
    minimum=-np.inf,
    maximum=np.inf,
    perfect=1.0,
)


def _msss_murphy(forecast, reference, dim=None, **metric_kwargs):
    """Murphy's Mean Square Skill Score (MSSS).

    .. math::
        MSSS_{Murphy} = r_{fo}^2 - [\\text{conditional bias}]^2 -\
         [\\frac{\\text{(unconditional) bias}}{\\sigma_o}]^2

    Args:
        forecast (xarray object): Forecast.
        reference (xarray object): Reference.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    acc = __pearson_r.function(forecast, reference, dim=dim, **metric_kwargs)
    conditional_bias = __conditional_bias.function(
        forecast, reference, dim=dim, **metric_kwargs
    )
    uncond_bias = __unconditional_bias.function(
        forecast, reference, dim=dim, **metric_kwargs
    ) / reference.std(dim)
    skill = acc ** 2 - conditional_bias ** 2 - uncond_bias ** 2
    return skill


__msss_murphy = Metric(
    name='msss_murphy',
    function=_msss_murphy,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Murphy's Mean Square Skill Score",
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


#######################
# PROBABILISTIC METRICS
#######################
def _brier_score(forecast, reference, **metric_kwargs):
    """Brier Score.

    .. math::
        BS(f, o) = (f_1 - o)^2,

    where $f_1$ is the forecast probability of $o=1$.

    .. note::
        The Brier Score requires that the reference is binary, i.e., can be described as
        one (a "hit") or zero (a "miss").

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        reference (xr.object): References without ``member`` dim.
        func (function): Function to be applied to reference and forecasts
                         and then ``mean('member')`` to get forecasts and
                         reference in interval [0,1].

    Reference:
        * Brier, Glenn W. Verification of forecasts expressed in terms of
        probability.” Monthly Weather Review 78, no. 1 (1950).
        https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

    Example:
        >>> def pos(x): return x > 0
        >>> compute_perfect_model(ds, control, metric='brier_score', func=pos)

    See also:
        * properscoring.brier_score
        * xskillscore.brier_score
    """
    if 'func' in metric_kwargs:
        func = metric_kwargs['func']
    else:
        raise ValueError(
            'Please provide a function `func` to be applied to comparison and \
             reference to get values in  interval [0,1]; \
             see properscoring.brier_score.'
        )
    return brier_score(func(reference), func(forecast).mean('member'))


__brier_score = Metric(
    name='brier_score',
    function=_brier_score,
    positive=False,
    probabilistic=True,
    unit_power=0,
    long_name='Brier Score',
    aliases=['brier', 'bs'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _threshold_brier_score(forecast, reference, **metric_kwargs):
    """Brier score of an ensemble for exceeding given thresholds.

    .. math::
        CRPS(F, x) = \int_z BS(F(z), H(z - x)) dz

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    Args:
        * forecast (xr.object): Forecast with `member` dim.
        * reference (xr.object): References without `member` dim.
        * threshold (int, float, xr.object): Threshold to check exceedance, see
                                             properscoring.threshold_brier_score.

    References:
        * Brier, Glenn W. “VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF
        PROBABILITY.” Monthly Weather Review 78, no. 1 (1950).
        https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

    Example:
        >>> compute_perfect_model(ds, control,
                                  metric='threshold_brier_score', threshold=.5)

    See also:
        * properscoring.threshold_brier_score
        * xskillscore.threshold_brier_score
    """
    if 'threshold' not in metric_kwargs:
        raise ValueError('Please provide threshold.')
    else:
        threshold = metric_kwargs['threshold']
    # switch args b/c xskillscore.threshold_brier_score(obs, forecasts)
    return threshold_brier_score(reference, forecast, threshold)


__threshold_brier_score = Metric(
    name='threshold_brier_score',
    function=_threshold_brier_score,
    positive=False,
    probabilistic=True,
    unit_power=0,
    long_name='Threshold Brier Score',
    aliases=['tbs'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _crps(forecast, reference, **metric_kwargs):
    """Continuous Ranked Probability Score (CRPS).

    .. math::
        CRPS(F, o) = \\int_{f} (F(f) - H(f - o))^{2} dz,

    where F() is the cumulative distribution function (CDF) of the forecast;
    H() is the Heaviside step function where the value is 1 if the argument is positive
    or zero and is 0 otherwise.

    .. note::
        The CRPS is expressed in the same unit as the observed variable. It generalizes
        the Mean Absolute Error (MAE), and reduces to the MAE if the forecast is
        determinstic.

    Args:
        forecast (xr.object): Forecast with `member` dim.
        reference (xr.object): References without `member` dim.
        metric_kwargs (xr.object): If provided, the CRPS is calculated exactly with the
        assigned probability weights to each forecast. Weights should be positive, but
        do not need to be normalized. By default, each forecast is weighted equally.

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    References:
        * Matheson, James E., and Robert L. Winkler. “Scoring Rules for
          Continuous Probability Distributions.” Management Science 22, no. 10
          (June 1, 1976): 1087–96. https://doi.org/10/cwwt4g.
        * https://www.lokad.com/continuous-ranked-probability-score

    See also:
        * properscoring.crps_ensemble
        * xskillscore.crps_ensemble
    """
    weights = metric_kwargs.get('weights', None)
    # switch positions because xskillscore.crps_ensemble(obs, forecasts)
    return crps_ensemble(reference, forecast, weights=weights)


__crps = Metric(
    name='crps',
    function=_crps,
    positive=False,
    probabilistic=True,
    unit_power=1.0,
    long_name='Continuous Ranked Probability Score',
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _crps_gaussian(forecast, mu, sig, **metric_kwargs):
    """CRPS assuming a gaussian distribution. Helper function for CRPSS.

    Args:
        forecast (xr.object): Forecast with `member` dim.
        mu (xr.object): The mean of the reference.
        sig (xr.object): The standard deviation reference.

    See also:
        * properscoring.crps_gaussian
        * xskillscore.crps_gaussian
    """
    return crps_gaussian(forecast, mu, sig)


def _crps_quadrature(
    forecast, cdf_or_dist, xmin=None, xmax=None, tol=1e-6, **metric_kwargs
):
    """CRPS assuming distribution cdf_or_dist. Helper function for CRPSS.

    Args:
        forecast (xr.object): Forecast with `member` dim.
        cdf_or_dist (callable or scipy.stats.distribution): Function which returns the
        cumulative density of the forecast distribution at value x.
        xmin (float): Lower bounds for integration.
        xmax (float): Upper bounds for integration.
        tol (float, optional): The desired accuracy of the CRPS. Larger values will
                               speed up integration. If ``tol`` is set to ``None``,
                               bounds errors or integration tolerance errors will be
                               ignored.

    See also:
        * properscoring.crps_quadrature
        * xskillscore.crps_quadrature
    """
    return crps_quadrature(forecast, cdf_or_dist, xmin, xmax, tol)


def _crpss(forecast, reference, **metric_kwargs):
    """Continuous Ranked Probability Skill Score.

    .. note::
        When assuming a gaussian distribution of forecasts, use default gaussian=True.
        If not gaussian, you may specify the distribution type, xmin/xmax/tolerance
        for integration (see xskillscore.crps_quadrature).

    .. math::
        CRPSS = 1 - \\frac{CRPS_{init}}{CRPS_{clim}}

    Args:
        forecast (xr.object): Forecast with `member` dim.
        reference (xr.object): References without `member` dim.
        gaussian (bool, optional): If ``True``, assum Gaussian distribution for baseline
                                   skill. Defaults to ``True``.
        cdf_or_dist (scipy.stats): Function which returns the cumulative density of the
                                   forecast at value x. This can also be an object with
                                   a callable ``cdf()`` method such as a
                                   ``scipy.stats.distribution`` object. Defaults to
                                   ``scipy.stats.norm``.
        xmin (float): Lower bounds for integration. Only use if not assuming Gaussian.
        xmax (float) Upper bounds for integration. Only use if not assuming Gaussian.
        tol (float, optional): The desired accuracy of the CRPS. Larger values will
                               speed up integration. If ``tol`` is set to ``None``,
                               bounds errors or integration tolerance errors will be
                               ignored. Only use if not assuming Gaussian.

    Range:
        * perfect: 1
        * pos: better than climatology forecast
        * neg: worse than climatology forecast

    References:
        * Matheson, James E., and Robert L. Winkler. “Scoring Rules for
          Continuous Probability Distributions.” Management Science 22, no. 10
          (June 1, 1976): 1087–96. https://doi.org/10/cwwt4g.
        * Gneiting, Tilmann, and Adrian E Raftery. “Strictly Proper Scoring
          Rules, Prediction, and Estimation.” Journal of the American
          Statistical Association 102, no. 477 (March 1, 2007): 359–78.
          https://doi.org/10/c6758w.

    Example:
        >>> compute_perfect_model(ds, control, metric='crpss')
        >>> compute_perfect_model(ds, control, metric='crpss', gaussian=False,
                                  cdf_or_dist=scipy.stats.norm, xminimum=-10,
                                  xmaximum=10, tol=1e-6)

    See also:
        * properscoring.crps_ensemble
        * xskillscore.crps_ensemble
    """
    # available climpred dimensions to take mean and std over
    rdim = [tdim for tdim in reference.dims if tdim in CLIMPRED_DIMS]
    mu = reference.mean(rdim)
    sig = reference.std(rdim)

    # checking metric_kwargs, if not found use defaults: gaussian, else crps_quadrature
    if 'gaussian' in metric_kwargs:
        gaussian = metric_kwargs['gaussian']
    else:
        gaussian = True
    if gaussian:
        ref_skill = _crps_gaussian(forecast, mu, sig)
    else:
        if 'cdf_or_dist' in metric_kwargs:
            cdf_or_dist = metric_kwargs['cdf_or_dist']
        else:
            cdf_or_dist = norm
        if 'xmin' in metric_kwargs:
            xmin = metric_kwargs['xmin']
        else:
            xmin = None
        if 'xmax' in metric_kwargs:
            xmax = metric_kwargs['xmax']
        else:
            xmax = None
        if 'tol' in metric_kwargs:
            tol = metric_kwargs['tol']
        else:
            tol = 1e-6
        ref_skill = _crps_quadrature(forecast, cdf_or_dist, xmin, xmax, tol)
    forecast_skill = __crps.function(forecast, reference, **metric_kwargs)
    skill_score = 1 - forecast_skill / ref_skill.mean('member')
    return skill_score


__crpss = Metric(
    name='crpss',
    function=_crpss,
    positive=True,
    probabilistic=True,
    unit_power=0,
    long_name='Continuous Ranked Probability Skill Score',
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


def _crpss_es(forecast, reference, **metric_kwargs):
    """Continuous Ranked Probability Skill Score Ensemble Spread.

    .. math::
        CRPSS = 1 - \\frac{CRPS(\\sigma^2_f)}{CRPS(\\sigma^2_o}))

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        reference (xr.object): References without ``member`` dim.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    References:
        * Kadow, Christopher, Sebastian Illing, Oliver Kunst, Henning W. Rust,
          Holger Pohlmann, Wolfgang A. Müller, and Ulrich Cubasch. “Evaluation
          of Forecasts by Accuracy and Spread in the MiKlip Decadal Climate
          Prediction System.” Meteorologische Zeitschrift, December 21, 2016,
          631–43. https://doi.org/10/f9jrhw.

    Range:
        * perfect: 0
        * else: negative
    """
    # helper dim to calc mu
    rdim = [tdim for tdim in reference.dims if tdim in CLIMPRED_DIMS + ['time']]
    # inside compute_perfect_model
    if 'init' in forecast.dims:
        dim2 = 'init'
    # inside compute_hindcast
    elif 'time' in forecast.dims:
        dim2 = 'time'
    else:
        raise ValueError('dim2 not found automatically in ', forecast.dims)

    mu = reference.mean(rdim)
    forecast, ref2 = xr.broadcast(forecast, reference)
    mse_kwargs = metric_kwargs.copy()
    if 'dim' in mse_kwargs:
        del mse_kwargs['dim']
    sig_r = __mse.function(forecast, ref2, dim='member', **mse_kwargs).mean(dim2)
    sig_h = __mse.function(
        forecast.mean(dim2), ref2.mean(dim2), dim='member', **mse_kwargs
    )
    crps_h = _crps_gaussian(forecast, mu, sig_h)
    if 'member' in crps_h.dims:
        crps_h = crps_h.mean('member')
    crps_r = _crps_gaussian(forecast, mu, sig_r)
    if 'member' in crps_r.dims:
        crps_r = crps_r.mean('member')
    return 1 - crps_h / crps_r


__crpss_es = Metric(
    name='crpss_es',
    function=_crpss_es,
    positive=True,
    probabilistic=True,
    unit_power=0,
    long_name='CRPSS Ensemble Spread',
    minimum=-np.inf,
    maximum=0.0,
    perfect=0.0,
)


__ALL_METRICS__ = [
    __pearson_r,
    __spearman_r,
    __pearson_r_p_value,
    __effective_sample_size,
    __pearson_r_eff_p_value,
    __spearman_r_p_value,
    __spearman_r_eff_p_value,
    __mse,
    __mae,
    __rmse,
    __median_absolute_error,
    __mape,
    __smape,
    __msss_murphy,
    __bias_slope,
    __conditional_bias,
    __unconditional_bias,
    __brier_score,
    __threshold_brier_score,
    __crps,
    __crpss,
    __crpss_es,
    __ppp,
    __nmse,
    __nrmse,
    __nmae,
    __uacc,
]
