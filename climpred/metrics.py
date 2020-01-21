import warnings

import numpy as np
import xarray as xr
from scipy import special
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

from .np_metrics import _effective_sample_size as ess, _spearman_r_eff_p_value as srepv

# the import of CLIMPRED_DIMS from constants fails. currently fixed manually.
# from .constants import CLIMPRED_DIMS
CLIMPRED_DIMS = ['init', 'member', 'lead', 'time']


def _get_norm_factor(comparison):
    """Get normalization factor for normalizing distance metrics.

    A distance metric is normalized by the standard deviation or variance
    of the verification product. The goal of a normalized distance
    metric is to get a constant and comparable value of typically 1 (or 0 for
    metrics defined as 1 - metric), when the metric saturates and the predictability
    horizon is reached.

    To directly compare skill between different comparisons used, a factor is
    added in the normalized metric formula, see Seferian et al. 2018.
    For example, NRMSE gets smaller in comparison ``m2e`` than ``m2m`` by design,
    because the ensemble mean is always closer to individual ensemble members
    than ensemble members to each other.

     .. note::
         This is used for NMSE, NRMSE, MSSS, NMAE.

    Args:
        comparison (class): comparison class.

    Returns:
        fac (int): normalization factor.

    Raises:
        KeyError: if comparison is not matching.

    Example:
        >>> # check skill saturation value of roughly 1 for different comparisons
        >>> metric = 'nrmse'
        >>> for c in ['m2m', 'm2e', 'm2c', 'e2c']:
                s = compute_perfect_model(ds, control, metric=metric, comparison=c)
                s.plot(label=' '.join([metric,c]))
        >>> plt.legend()

    Reference:
        * Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing
          the Decadal Predictability of Land and Ocean Carbon Uptake.”
          Geophysical Research Letters, March 15, 2018. https://doi.org/10/gdb424.
    """
    if comparison.name in ['m2e', 'e2c', 'e2o']:
        fac = 1
    elif comparison.name in ['m2c', 'm2m', 'm2o']:
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

    def __repr__(self):
        """Show metadata of metric class."""
        return _display_metric_metadata(self)


#####################
# CORRELATION METRICS
#####################
def _pearson_r(forecast, verif, dim=None, **metric_kwargs):
    """Pearson product-moment correlation coefficient.

    A measure of the linear association between the forecast and verification data that
    is independent of the mean and variance of the individual distributions. This is
    also known as the Anomaly Correlation Coefficient (ACC) when correlating anomalies.

    .. math::
        corr = \\frac{cov(f, o)}{\\sigma_{f}\\cdot\\sigma_{o}},

    where :math:`\\sigma_{f}` and :math:`\\sigma_{o}` represent the standard deviation
    of the forecast and verification data over the experimental period, respectively.

    .. note::
        Use metric ``pearson_r_p_value`` or ``pearson_r_eff_p_value`` to get the
        corresponding p value.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | -1.0      |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | positive  |
        +-----------------+-----------+

    See also:
        * xskillscore.pearson_r
        * xskillscore.pearson_r_p_value
        * climpred.pearson_r_p_value
        * climpred.pearson_r_eff_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return pearson_r(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _pearson_r_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are linearly uncorrelated.

    Two-tailed p value associated with the Pearson product-moment correlation
    coefficient (``pearson_r``), assuming that all samples are independent. Use
    ``pearson_r_eff_p_value`` to account for autocorrelation in the forecast
    and verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.pearson_r
        * xskillscore.pearson_r_p_value
        * climpred.pearson_r
        * climpred.pearson_r_eff_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pearson_r_p_value(
            forecast, verif, dim=dim, weights=weights, skipna=skipna
        )


__pearson_r_p_value = Metric(
    name='pearson_r_p_value',
    function=_pearson_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name='Pearson product-moment correlation coefficient p value',
    aliases=['p_pval', 'pvalue', 'pval'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _effective_sample_size(forecast, verif, dim=None, **metric_kwargs):
    """Effective sample size for temporally correlated data.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    The effective sample size extracts the number of independent samples
    between two time series being correlated. This is derived by assessing
    the magnitude of the lag-1 autocorrelation coefficient in each of the time series
    being correlated. A higher autocorrelation induces a lower effective sample
    size which raises the correlation coefficient for a given p value.

    The effective sample size is used in computing the effective p value. See
    ``pearson_r_eff_p_value`` and ``spearman_r_eff_p_value``.

    .. math::
        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------------+
        | **minimum**     | 0.0             |
        +-----------------+-----------------+
        | **maximum**     | ∞               |
        +-----------------+-----------------+
        | **perfect**     | N/A             |
        +-----------------+-----------------+
        | **orientation** | positive        |
        +-----------------+-----------------+

    Reference:
        * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
          freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    """
    skipna = metric_kwargs.get('skipna', False)
    dim = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        forecast = forecast.stack(**{new_dim: dim})
        verif = verif.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        ess,
        forecast,
        verif,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
    )


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


def _pearson_r_eff_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are linearly uncorrelated, accounting
    for autocorrelation.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    The effective p value is computed by replacing the sample size :math:`N` in the
    t-statistic with the effective sample size, :math:`N_{eff}`. The same Pearson
    product-moment correlation coefficient :math:`r` is used as when computing the
    standard p value.

    .. math::

        t = r\\sqrt{ \\frac{N_{eff} - 2}{1 - r^{2}} },

    where :math:`N_{eff}` is computed via the autocorrelation in the forecast and
    verification data.

    .. math::

        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See Also:
        * climpred.effective_sample_size
        * climpred.spearman_r_eff_p_value

    Reference:
        * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
          freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    """
    skipna = metric_kwargs.get('skipna', False)

    # compute t-statistic
    r = pearson_r(forecast, verif, dim=dim, skipna=skipna)
    dof = _effective_sample_size(forecast, verif, dim, skipna=skipna) - 2
    t_squared = r ** 2 * (dof / ((1.0 - r) * (1.0 + r)))
    _x = dof / (dof + t_squared)
    _x = _x.where(_x < 1.0, 1.0)
    _a = 0.5 * dof
    _b = 0.5
    res = special.betainc(_a, _b, _x)
    # reset masked values to nan
    res = res.where(r.notnull(), np.nan)
    return res


__pearson_r_eff_p_value = Metric(
    name='pearson_r_eff_p_value',
    function=_pearson_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Pearson's Anomaly correlation coefficient "
        'p value using the effective sample size'
    ),
    aliases=['p_pval_eff', 'pvalue_eff', 'pval_eff'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _spearman_r(forecast, verif, dim=None, **metric_kwargs):
    """Spearman's rank correlation coefficient.

    .. math::
        corr = \\mathrm{pearsonr}(ranked(f), ranked(o))

    This correlation coefficient is nonparametric and assesses how well the relationship
    between the forecast and verification data can be described using a monotonic
    function. It is computed by first ranking the forecasts and verification data, and
    then correlating those ranks using the ``pearson_r`` correlation.

    This is also known as the anomaly correlation coefficient (ACC) when comparing
    anomalies, although the Pearson product-moment correlation coefficient
    (``pearson_r``) is typically used when computing the ACC.

    .. note::
        Use metric ``spearman_r_p_value`` or ``spearman_r_eff_p_value`` to get the
        corresponding p value.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | -1.0      |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | positive  |
        +-----------------+-----------+

    See also:
        * xskillscore.spearman_r
        * xskillscore.spearman_r_p_value
        * climpred.spearman_r_p_value
        * climpred.spearman_r_eff_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return spearman_r(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _spearman_r_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are monotonically uncorrelated.

    Two-tailed p value associated with the Spearman's rank correlation
    coefficient (``spearman_r``), assuming that all samples are independent. Use
    ``spearman_r_eff_p_value`` to account for autocorrelation in the forecast
    and verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.spearman_r
        * xskillscore.spearman_r_p_value
        * climpred.spearman_r
        * climpred.spearman_r_eff_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return spearman_r_p_value(
            forecast, verif, dim=dim, weights=weights, skipna=skipna
        )


__spearman_r_p_value = Metric(
    name='spearman_r_p_value',
    function=_spearman_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name="Spearman's rank correlation coefficient p value",
    aliases=['s_pval', 'spvalue', 'spval'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _spearman_r_eff_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are monotonically uncorrelated,
    accounting for autocorrelation.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    The effective p value is computed by replacing the sample size :math:`N` in the
    t-statistic with the effective sample size, :math:`N_{eff}`. The same Spearman's
    rank correlation coefficient :math:`r` is used as when computing the standard p
    value.

    .. math::

        t = r\\sqrt{ \\frac{N_{eff} - 2}{1 - r^{2}} },

    where :math:`N_{eff}` is computed via the autocorrelation in the forecast and
    verification data.

    .. math::

        N_{eff} = N\\left( \\frac{1 -
                   \\rho_{f}\\rho_{o}}{1 + \\rho_{f}\\rho_{o}} \\right),

    where :math:`\\rho_{f}` and :math:`\\rho_{o}` are the lag-1 autocorrelation
    coefficients for the forecast and verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See Also:
        * climpred.effective_sample_size
        * climpred.pearson_r_eff_p_value

    Reference:
        * Bretherton, Christopher S., et al. "The effective number of spatial degrees of
          freedom of a time-varying field." Journal of climate 12.7 (1999): 1990-2009.
    """
    skipna = metric_kwargs.get('skipna', False)
    dim = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        forecast = forecast.stack(**{new_dim: dim})
        verif = verif.stack(**{new_dim: dim})
    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        srepv,
        forecast,
        verif,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
    )


__spearman_r_eff_p_value = Metric(
    name='spearman_r_eff_p_value',
    function=_spearman_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Spearman's Rank correlation coefficient "
        'p value using the effective sample size'
    ),
    aliases=['s_pval_eff', 'spvalue_eff', 'spval_eff'],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


##################
# DISTANCE METRICS
##################
def _mse(forecast, verif, dim=None, **metric_kwargs):
    """Mean Sqaure Error (MSE).

    .. math::
        MSE = \\overline{(f - o)^{2}}

    The average of the squared difference between forecasts and verification data. This
    incorporates both the variance and bias of the estimator. Because the error is
    squared, it is more sensitive to large forecast errors than ``mae``, and thus a
    more conservative metric. For example, a single error of 2°C counts the same as
    two 1°C errors when using ``mae``. On the other hand, the 2°C error counts double
    for ``mse``. See Jolliffe and Stephenson, 2011.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.mse

    Reference:
        * Ian T. Jolliffe and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. John Wiley & Sons, Ltd,
          Chichester, UK, December 2011. ISBN 978-1-119-96000-3 978-0-470-66071-3.
          URL: http://doi.wiley.com/10.1002/9781119960003.
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mse(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _rmse(forecast, verif, dim=None, **metric_kwargs):
    """Root Mean Sqaure Error (RMSE).

    .. math::
        RMSE = \\sqrt{\\overline{(f - o)^{2}}}

    The square root of the average of the squared differences between forecasts and
    verification data.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.rmse
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return rmse(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _mae(forecast, verif, dim=None, **metric_kwargs):
    """Mean Absolute Error (MAE).

    .. math::
        MAE = \\overline{|f - o|}

    The average of the absolute differences between forecasts and verification data.
    A more robust measure of forecast accuracy than ``mse`` which is sensitive to large
    outlier forecast errors.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.mae

    Reference:
        * Ian T. Jolliffe and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. John Wiley & Sons, Ltd,
          Chichester, UK, December 2011. ISBN 978-1-119-96000-3 978-0-470-66071-3.
          URL: http://doi.wiley.com/10.1002/9781119960003.
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mae(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _median_absolute_error(forecast, verif, dim=None, **metric_kwargs):
    """Median Absolute Error.

    .. math::
        median(|f - o|)

    The median of the absolute differences between forecasts and verification data.
    Applying the median function to absolute error makes it more robust to outliers.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.median_absolute_error
    """
    skipna = metric_kwargs.get('skipna', False)
    return median_absolute_error(forecast, verif, dim=dim, skipna=skipna)


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


#############################
# NORMALIZED DISTANCE METRICS
#############################
def _nmse(forecast, verif, dim=None, **metric_kwargs):
    """Normalized MSE (NMSE), also known as Normalized Ensemble Variance (NEV).

    Mean Square Error (``mse``) normalized by the variance of the verification data.

    .. math::
        NMSE = NEV = \\frac{MSE}{\\sigma^2_{o}\\cdot fac}
             = \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{o} \\cdot fac},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    .. note::
        ``climpred`` uses a single-valued internal reference forecast for the
        NMSE, in the terminology of Murphy 1988. I.e., we use a single
        climatological variance of the verification data *within* the experimental
        window for normalizing MSE.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Details:
        +----------------------------+-----------+
        | **minimum**                | 0.0       |
        +----------------------------+-----------+
        | **maximum**                | ∞         |
        +----------------------------+-----------+
        | **perfect**                | 0.0       |
        +----------------------------+-----------+
        | **orientation**            | negative  |
        +----------------------------+-----------+
        | **better than climatology**| 0.0 - 1.0 |
        +----------------------------+-----------+
        | **worse than climatology** | > 1.0     |
        +----------------------------+-----------+

    Reference:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13,
          no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    mse_skill = __mse.function(forecast, verif, dim=dim, **metric_kwargs)
    var = verif.var(dim)
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
    unit_power=0.0,
    long_name='Normalized Mean Squared Error',
    aliases=['nev'],
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _nmae(forecast, verif, dim=None, **metric_kwargs):
    """Normalized Mean Absolute Error (NMAE).

    Mean Absolute Error (``mae``) normalized by the standard deviation of the
    verification data.

    .. math::
        NMAE = \\frac{MAE}{\\sigma_{o} \\cdot fac}
             = \\frac{\\overline{|f - o|}}{\\sigma_{o} \\cdot fac},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    .. note::
        ``climpred`` uses a single-valued internal reference forecast for the
        NMAE, in the terminology of Murphy 1988. I.e., we use a single
        climatological standard deviation of the verification data *within* the
        experimental window for normalizing MAE.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Details:
        +----------------------------+-----------+
        | **minimum**                | 0.0       |
        +----------------------------+-----------+
        | **maximum**                | ∞         |
        +----------------------------+-----------+
        | **perfect**                | 0.0       |
        +----------------------------+-----------+
        | **orientation**            | negative  |
        +----------------------------+-----------+
        | **better than climatology**| 0.0 - 1.0 |
        +----------------------------+-----------+
        | **worse than climatology** | > 1.0     |
        +----------------------------+-----------+

    Reference:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13, no.
          7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    mae_skill = __mae.function(forecast, verif, dim=dim, **metric_kwargs)
    std = verif.std(dim)
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
    maximum=np.inf,
    perfect=0.0,
)


def _nrmse(forecast, verif, dim=None, **metric_kwargs):
    """Normalized Root Mean Square Error (NRMSE).

    Root Mean Square Error (``rmse``) normalized by the standard deviation of the
    verification data.

    .. math::

        NRMSE = \\frac{RMSE}{\\sigma_{o}\\cdot\\sqrt{fac}}
              = \\sqrt{\\frac{MSE}{\\sigma^{2}_{o}\\cdot fac}}
              = \\sqrt{ \\frac{\\overline{(f - o)^{2}}}{ \\sigma^2_{o}\\cdot fac}},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    .. note::
        ``climpred`` uses a single-valued internal reference forecast for the
        NRMSE, in the terminology of Murphy 1988. I.e., we use a single
        climatological variance of the verification data *within* the experimental
        window for normalizing RMSE.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Details:
        +----------------------------+-----------+
        | **minimum**                | 0.0       |
        +----------------------------+-----------+
        | **maximum**                | ∞         |
        +----------------------------+-----------+
        | **perfect**                | 0.0       |
        +----------------------------+-----------+
        | **orientation**            | negative  |
        +----------------------------+-----------+
        | **better than climatology**| 0.0 - 1.0 |
        +----------------------------+-----------+
        | **worse than climatology** | > 1.0     |
        +----------------------------+-----------+

    Reference:
      * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.”
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
      * Hawkins, Ed, Steffen Tietsche, Jonathan J. Day, Nathanael Melia, Keith
        Haines, and Sarah Keeley. “Aspects of Designing and Evaluating
        Seasonal-to-Interannual Arctic Sea-Ice Prediction Systems.” Quarterly
        Journal of the Royal Meteorological Society 142, no. 695
        (January 1, 2016): 672–83. https://doi.org/10/gfb3pn.
      * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
        Their Relationships to the Correlation Coefficient.” Monthly Weather
        Review 116, no. 12 (December 1, 1988): 2417–24.
        https://doi.org/10/fc7mxd.
    """
    rmse_skill = __rmse.function(forecast, verif, dim=dim, **metric_kwargs)
    std = verif.std(dim)
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
    maximum=np.inf,
    perfect=0.0,
)


def _msess(forecast, verif, dim=None, **metric_kwargs):
    """Mean Squared Error Skill Score (MSESS).

    .. math::
        MSESS = 1 - \\frac{MSE}{\\sigma^2_{ref} \\cdot fac} =
               1 - \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{ref} \\cdot fac},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    This skill score can be intepreted as a percentage improvement in accuracy. I.e.,
    it can be multiplied by 100%.

    .. note::
        ``climpred`` uses a single-valued internal reference forecast for the
        MSSS, in the terminology of Murphy 1988. I.e., we use a single
        climatological variance of the verification data *within* the experimental
        window for normalizing MSE.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Details:
        +----------------------------+-----------+
        | **minimum**                | -∞        |
        +----------------------------+-----------+
        | **maximum**                | 1.0       |
        +----------------------------+-----------+
        | **perfect**                | 1.0       |
        +----------------------------+-----------+
        | **orientation**            | positive  |
        +----------------------------+-----------+
        | **better than climatology**| > 0.0     |
        +----------------------------+-----------+
        | **equal to climatology**   | 0.0       |
        +----------------------------+-----------+
        | **worse than climatology** | < 0.0     |
        +----------------------------+-----------+

    Reference:
      * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
        North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
        (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
      * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
        Their Relationships to the Correlation Coefficient.” Monthly Weather
        Review 116, no. 12 (December 1, 1988): 2417–24.
        https://doi.org/10/fc7mxd.
      * Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.
      * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    """
    mse_skill = __mse.function(forecast, verif, dim=dim, **metric_kwargs)
    var = verif.var(dim)
    if 'comparison' in metric_kwargs:
        comparison = metric_kwargs['comparison']
    else:
        raise ValueError(
            'Comparison needed to normalize MSSS. Not found in', metric_kwargs
        )
    fac = _get_norm_factor(comparison)
    msess_skill = 1 - mse_skill / var / fac
    return msess_skill


__msess = Metric(
    name='msess',
    function=_msess,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name='Mean Squared Error Skill Score',
    aliases=['ppp', 'msss'],
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


def _mape(forecast, verif, dim=None, **metric_kwargs):
    """Mean Absolute Percentage Error (MAPE).

    Mean absolute error (``mae``) expressed as a percentage error relative to the
    verification data.

    .. math::
        MAPE = \\frac{1}{n} \\sum \\frac{|f-o|}{|o|}

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.mape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mape(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _smape(forecast, verif, dim=None, **metric_kwargs):
    """Symmetric Mean Absolute Percentage Error (sMAPE).

    Similar to the Mean Absolute Percentage Error (``mape``), but sums the forecast and
    observation mean in the denominator.

    .. math::
        sMAPE = \\frac{1}{n} \\sum \\frac{|f-o|}{|f|+|o|}

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    See also:
        * xskillscore.smape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return smape(forecast, verif, dim=dim, weights=weights, skipna=skipna)


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


def _uacc(forecast, verif, dim=None, **metric_kwargs):
    """Bushuk's unbiased Anomaly Correlation Coefficient (uACC).

    This is typically used in perfect model studies. Because the perfect model Anomaly
    Correlation Coefficient (ACC) is strongly state dependent, a standard ACC (e.g. one
    computed using ``pearson_r``) will be highly sensitive to the set of start dates
    chosen for the perfect model study. The Mean Square Skill Score (``MSSS``) can be
    related directly to the ACC as ``MSSS = ACC^(2)`` (see Murphy 1988 and
    Bushuk et al. 2019), so the unbiased ACC can be derived as ``uACC = sqrt(MSSS)``.

    .. math::
        uACC = \\sqrt{MSSS}
             = \\sqrt{1 - \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{ref} \\cdot fac}},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    .. note::
        Because of the square root involved, any negative ``MSSS`` values are
        automatically converted to NaNs.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.
        comparison (str): Name comparison needed for normalization factor ``fac``, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)

    Details:
        +----------------------------+-----------+
        | **minimum**                | 0.0       |
        +----------------------------+-----------+
        | **maximum**                | 1.0       |
        +----------------------------+-----------+
        | **perfect**                | 1.0       |
        +----------------------------+-----------+
        | **orientation**            | positive  |
        +----------------------------+-----------+
        | **better than climatology**| > 0.0     |
        +----------------------------+-----------+
        | **equal to climatology**   | 0.0       |
        +----------------------------+-----------+

    Reference:
        * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel
          Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional
          Arctic Sea–Ice Prediction: Potential versus Operational Seasonal
          Forecast Skill." Climate Dynamics, June 9, 2018.
          https://doi.org/10/gd7hfq.
        * Allan H. Murphy. Skill Scores Based on the Mean Square Error and Their
          Relationships to the Correlation Coefficient. Monthly Weather Review,
          116(12):2417–2424, December 1988. https://doi.org/10/fc7mxd.
    """
    msss_res = __msess.function(forecast, verif, dim=dim, **metric_kwargs)
    # Negative values are automatically turned into nans from xarray.
    uacc_res = msss_res ** 0.5
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
def _std_ratio(forecast, verif, dim=None, **metric_kwargs):
    """Ratio of standard deviations of the forecast over the verification data.

    .. math:: \\text{std ratio} = \\frac{\\sigma_f}{\\sigma_o},

    where :math:`\\sigma_{f}` and :math:`\\sigma_{o}` are the standard deviations of the
    forecast and the verification data over the experimental period, respectively.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | N/A       |
        +-----------------+-----------+

    Reference:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    ratio = forecast.std(dim) / verif.std(dim)
    return ratio


__std_ratio = Metric(
    name='std_ratio',
    function=_std_ratio,
    positive=None,
    probabilistic=False,
    unit_power=0,
    long_name='Ratio of standard deviations of the forecast and verification data',
    minimum=0.0,
    maximum=np.inf,
    perfect=1.0,
)


def _unconditional_bias(forecast, verif, dim=None, **metric_kwargs):
    """Unconditional bias.

    .. math::
        bias = f - o

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    Details:
        +-----------------+-----------+
        | **minimum**     | -∞        |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * https://www.cawcr.gov.au/projects/verification/
        * https://www-miklip.dkrz.de/about/murcss/
    """
    bias = (forecast - verif).mean(dim)
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


def _conditional_bias(forecast, verif, dim=None, **metric_kwargs):
    """Conditional bias between forecast and verification data.

    .. math::
        \\text{conditional bias} = r_{fo} - \\frac{\\sigma_f}{\\sigma_o},

    where :math:`\\sigma_{f}` and :math:`\\sigma_{o}` are the standard deviations of the
    forecast and verification data over the experimental period, respectively.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    Details:
        +-----------------+-----------+
        | **minimum**     | -∞        |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    acc = __pearson_r.function(forecast, verif, dim=dim, **metric_kwargs)
    conditional_bias = acc - __std_ratio.function(
        forecast, verif, dim=dim, **metric_kwargs
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


def _bias_slope(forecast, verif, dim=None, **metric_kwargs):
    """Bias slope between verification data and forecast standard deviations.

    .. math::
        \\text{bias slope} = \\frac{s_{o}}{s_{f}} \\cdot r_{fo},

    where :math:`r_{fo}` is the Pearson product-moment correlation between the forecast
    and the verification data and :math:`s_{o}` and :math:`s_{f}` are the standard
    deviations of the verification data and forecast over the experimental period,
    respectively.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   functions.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    std_ratio = __std_ratio.function(forecast, verif, dim=dim, **metric_kwargs)
    acc = __pearson_r.function(forecast, verif, dim=dim, **metric_kwargs)
    b_s = std_ratio * acc
    return b_s


__bias_slope = Metric(
    name='bias_slope',
    function=_bias_slope,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name='Bias slope',
    minimum=0.0,
    maximum=np.inf,
    perfect=1.0,
)


def _msess_murphy(forecast, verif, dim=None, **metric_kwargs):
    """Murphy's Mean Square Error Skill Score (MSESS).

    .. math::
        MSESS_{Murphy} = r_{fo}^2 - [\\text{conditional bias}]^2 -\
         [\\frac{\\text{(unconditional) bias}}{\\sigma_o}]^2,

    where :math:`r_{fo}^{2}` represents the Pearson product-moment correlation
    coefficient between the forecast and verification data and :math:`\\sigma_{o}`
    represents the standard deviation of the verification data over the experimental
    period. See ``conditional_bias`` and ``unconditional_bias`` for their respective
    formulations.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over. Automatically set by compute
                   function.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +-----------------+-----------+
        | **minimum**     | -∞        |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 1.0       |
        +-----------------+-----------+
        | **orientation** | positive  |
        +-----------------+-----------+

    See Also:
        * climpred.pearson_r
        * climpred.conditional_bias
        * climpred.unconditional_bias

    Reference:
        * https://www-miklip.dkrz.de/about/murcss/
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    acc = __pearson_r.function(forecast, verif, dim=dim, **metric_kwargs)
    conditional_bias = __conditional_bias.function(
        forecast, verif, dim=dim, **metric_kwargs
    )
    uncond_bias = __unconditional_bias.function(
        forecast, verif, dim=dim, **metric_kwargs
    ) / verif.std(dim)
    skill = acc ** 2 - conditional_bias ** 2 - uncond_bias ** 2
    return skill


__msess_murphy = Metric(
    name='msess_murphy',
    function=_msess_murphy,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Murphy's Mean Square Error Skill Score",
    aliases=['msss_murphy'],
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


#######################
# PROBABILISTIC METRICS
#######################
def _brier_score(forecast, verif, **metric_kwargs):
    """Brier Score.

    The Mean Square Error (``mse``) of probabilistic two-category forecasts where the
    verification data are either 0 (no occurrence) or 1 (occurrence) and forecast
    probability may be arbitrarily distributed between occurrence and non-occurrence.
    The Brier Score equals zero for perfect (single-valued) forecasts and one for
    forecasts that are always incorrect.

    .. math::
        BS(f, o) = (f_1 - o)^2,

    where :math:`f_1` is the forecast probability of :math:`o=1`.

    .. note::
        The Brier Score requires that the observation is binary, i.e., can be described
        as one (a "hit") or zero (a "miss").

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        verif (xr.object): Verification data without ``member`` dim.
        func (function): Function to be applied to verification data and forecasts
                         and then ``mean('member')`` to get forecasts and
                         verification data in interval [0,1].

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * Brier, Glenn W. Verification of forecasts expressed in terms of
          probability.” Monthly Weather Review 78, no. 1 (1950).
          https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.
        * https://www.nws.noaa.gov/oh/rfcdev/docs/
          Glossary_Forecast_Verification_Metrics.pdf

    See also:
        * properscoring.brier_score
        * xskillscore.brier_score

    Example:
        >>> def pos(x): return x > 0
        >>> compute_perfect_model(ds, control, metric='brier_score', func=pos)
    """
    if 'func' in metric_kwargs:
        func = metric_kwargs['func']
    else:
        raise ValueError(
            'Please provide a function `func` to be applied to comparison and \
             verification data to get values in  interval [0,1]; \
             see properscoring.brier_score.'
        )
    return brier_score(func(verif), func(forecast).mean('member'))


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


def _threshold_brier_score(forecast, verif, **metric_kwargs):
    """Brier score of an ensemble for exceeding given thresholds.

    .. math::
        CRPS = \int_f BS(F(f), H(f - o)) df,

    where :math:`F(o) = \int_{f \leq o} p(f) df` is the cumulative distribution
    function (CDF) of the forecast distribution :math:`F`, :math:`o` is a point estimate
    of the true observation (observational error is neglected), :math:`BS` denotes the
    Brier score and :math:`H(x)` denotes the Heaviside step function, which we define
    here as equal to 1 for :math:`x \geq 0` and 0 otherwise.

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        verif (xr.object): Verification data without ``member`` dim.
        threshold (int, float, xr.object): Threshold to check exceedance, see
            properscoring.threshold_brier_score.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | 1.0       |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * Brier, Glenn W. Verification of forecasts expressed in terms of
          probability.” Monthly Weather Review 78, no. 1 (1950).
          https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

    See also:
        * properscoring.threshold_brier_score
        * xskillscore.threshold_brier_score

    Example:
        >>> compute_perfect_model(ds, control,
                                  metric='threshold_brier_score', threshold=.5)
    """
    if 'threshold' not in metric_kwargs:
        raise ValueError('Please provide threshold.')
    else:
        threshold = metric_kwargs['threshold']
    # switch args b/c xskillscore.threshold_brier_score(verif, forecasts)
    return threshold_brier_score(verif, forecast, threshold)


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


def _crps(forecast, verif, **metric_kwargs):
    """Continuous Ranked Probability Score (CRPS).

    The CRPS can also be considered as the probabilistic Mean Absolute Error (``mae``).
    It compares the empirical distribution of an ensemble forecast to a scalar
    observation. Smaller scores indicate better skill.

    .. math::
        CRPS = \\int_{-\\infty}^{\\infty} (F(f) - H(f - o))^{2} df,

    where :math:`F(f)` is the cumulative distribution function (CDF) of the forecast
    (since the verification data are not assigned a probability), and H() is the
    Heaviside step function where the value is 1 if the argument is positive (i.e., the
    forecast overestimates verification data) or zero (i.e., the forecast equals
    verification data) and is 0 otherwise (i.e., the forecast is less than verification
    data).

    .. note::
        The CRPS is expressed in the same unit as the observed variable. It generalizes
        the Mean Absolute Error (MAE), and reduces to the MAE if the forecast is
        determinstic.

    Args:
        forecast (xr.object): Forecast with `member` dim.
        verif (xr.object): Verification data without `member` dim.
        metric_kwargs (xr.object): If provided, the CRPS is calculated exactly with the
            assigned probability weights to each forecast. Weights should be positive,
            but do not need to be normalized. By default, each forecast is weighted
            equally.

    Details:
        +-----------------+-----------+
        | **minimum**     | 0.0       |
        +-----------------+-----------+
        | **maximum**     | ∞         |
        +-----------------+-----------+
        | **perfect**     | 0.0       |
        +-----------------+-----------+
        | **orientation** | negative  |
        +-----------------+-----------+

    Reference:
        * Matheson, James E., and Robert L. Winkler. “Scoring Rules for
          Continuous Probability Distributions.” Management Science 22, no. 10
          (June 1, 1976): 1087–96. https://doi.org/10/cwwt4g.
        * https://www.lokad.com/continuous-ranked-probability-score

    See also:
        * properscoring.crps_ensemble
        * xskillscore.crps_ensemble
    """
    weights = metric_kwargs.get('weights', None)
    # switch positions because xskillscore.crps_ensemble(verif, forecasts)
    return crps_ensemble(verif, forecast, weights=weights)


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
    """Computes the CRPS of verification data ``o`` relative to normally distributed
    forecasts with mean ``mu`` and standard deviation ``sig``.

    .. note::
        This is a helper function for CRPSS and cannot be called directly by a user.

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        mu (xr.object): The mean of the verification data.
        sig (xr.object): The standard deviation verification data.

    See also:
        * properscoring.crps_gaussian
        * xskillscore.crps_gaussian
    """
    return crps_gaussian(forecast, mu, sig)


def _crps_quadrature(
    forecast, cdf_or_dist, xmin=None, xmax=None, tol=1e-6, **metric_kwargs
):
    """Compute the continuously ranked probability score (CPRS) for a given
    forecast distribution (``cdf``) and observation (``o``) using numerical quadrature.

    This implementation allows the computation of CRPSS for arbitrary forecast
    distributions. If gaussianity can be assumed ``crps_gaussian`` is faster.

    .. note::
        This is a helper function for CRPS and cannot be called directly by a user.

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
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


def _crpss(forecast, verif, **metric_kwargs):
    """Continuous Ranked Probability Skill Score.

    This can be used to assess whether the ensemble spread is a useful measure for the
    forecast uncertainty by comparing the CRPS of the ensemble forecast to that of a
    reference forecast with the desired spread.

    .. math::
        CRPSS = 1 - \\frac{CRPS_{initialized}}{CRPS_{clim}}

    .. note::
        When assuming a Gaussian distribution of forecasts, use default
        ``gaussian=True``. If not gaussian, you may specify the distribution type,
        xmin/xmax/tolerance for integration (see xskillscore.crps_quadrature).

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        verif (xr.object): Verification data without ``member`` dim.
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

    Details:
        +----------------------------+-----------+
        | **minimum**                | -∞        |
        +----------------------------+-----------+
        | **maximum**                | 1.0       |
        +----------------------------+-----------+
        | **perfect**                | 1.0       |
        +----------------------------+-----------+
        | **orientation**            | positive  |
        +----------------------------+-----------+
        | **better than climatology**| > 0.0     |
        +----------------------------+-----------+
        | **worse than climatology** | < 0.0     |
        +----------------------------+-----------+

    Reference:
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
    rdim = [tdim for tdim in verif.dims if tdim in CLIMPRED_DIMS]
    mu = verif.mean(rdim)
    sig = verif.std(rdim)

    # checking metric_kwargs, if not found use defaults: gaussian, else crps_quadrature
    if 'gaussian' in metric_kwargs:
        gaussian = metric_kwargs['gaussian']
    else:
        gaussian = True

    if gaussian:
        ref_skill = _crps_gaussian(forecast, mu, sig)
    # TODO: Add tests for this section.
    else:
        if 'cdf_or_dist' in metric_kwargs:
            cdf_or_dist = metric_kwargs['cdf_or_dist']
        else:
            # Imported at top. This is `scipy.stats.norm`
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
    forecast_skill = __crps.function(forecast, verif, **metric_kwargs)
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


def _crpss_es(forecast, verif, **metric_kwargs):
    """Continuous Ranked Probability Skill Score Ensemble Spread.

    If the ensemble variance is smaller than the observed ``mse``, the ensemble is
    said to be under-dispersive (or overconfident). An ensemble with variance larger
    than the verification data indicates one that is over-dispersive (underconfident).

    .. math::
        CRPSS = 1 - \\frac{CRPS(\\sigma^2_f)}{CRPS(\\sigma^2_o)}

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        verif (xr.object): Verification data without ``member`` dim.
        weights (xarray object, optional): Weights to apply over dimension. Defaults to
                                           ``None``.
        skipna (bool, optional): If True, skip NaNs over dimension being applied to.
                                 Defaults to ``False``.

    Details:
        +----------------------------+-----------+
        | **minimum**                | -∞        |
        +----------------------------+-----------+
        | **maximum**                | 0.0       |
        +----------------------------+-----------+
        | **perfect**                | 0.0       |
        +----------------------------+-----------+
        | **orientation**            | positive  |
        +----------------------------+-----------+
        | **under-dispersive**       | > 0.0     |
        +----------------------------+-----------+
        | **over-dispersive**        | < 0.0     |
        +----------------------------+-----------+

    Reference:
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
    rdim = [tdim for tdim in verif.dims if tdim in CLIMPRED_DIMS + ['time']]
    # inside compute_perfect_model
    if 'init' in forecast.dims:
        dim2 = 'init'
    # inside compute_hindcast
    elif 'time' in forecast.dims:
        dim2 = 'time'
    else:
        raise ValueError('dim2 not found automatically in ', forecast.dims)

    mu = verif.mean(rdim)
    forecast, ref2 = xr.broadcast(forecast, verif)
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
    __msess_murphy,
    __bias_slope,
    __conditional_bias,
    __unconditional_bias,
    __brier_score,
    __threshold_brier_score,
    __crps,
    __crpss,
    __crpss_es,
    __msess,
    __nmse,
    __nrmse,
    __nmae,
    __uacc,
    __std_ratio,
]
