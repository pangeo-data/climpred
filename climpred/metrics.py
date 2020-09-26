import warnings

import numpy as np
from scipy.stats import norm
from xskillscore import (
    Contingency,
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    discrimination,
    effective_sample_size,
    mae,
    mape,
    median_absolute_error,
    mse,
    pearson_r,
    pearson_r_eff_p_value,
    pearson_r_p_value,
    rank_histogram,
    reliability,
    rmse,
    rps,
    smape,
    spearman_r,
    spearman_r_eff_p_value,
    spearman_r_p_value,
    threshold_brier_score,
)

from .constants import CLIMPRED_DIMS


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
    if comparison.name in ["m2e", "e2c", "e2o"]:
        fac = 1
    elif comparison.name in ["m2c", "m2m", "m2o"]:
        fac = 2
    else:
        raise KeyError("specify comparison to get normalization factor.")
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


def _rename_dim(dim, forecast, verif):
    """rename `dim` to `time` or `init` if forecast and verif dims require."""
    if "init" in dim and "time" in forecast.dims and "time" in verif.dims:
        dim = dim.copy()
        dim.remove("init")
        dim = dim + ["time"]
    elif "time" in dim and "init" in forecast.dims and "init" in verif.dims:
        dim = dim.copy()
        dim.remove("time")
        dim = dim + ["init"]
    return dim


def _remove_member_from_dim_or_raise(dim):
    """delete `member` from `dim` to not pass to `xskillscore` where expected as
    default `member_dim`."""
    if "member" in dim:
        dim = dim.copy()
        dim.remove("member")
    else:
        raise ValueError(f"Expected to find `member` in `dim`, found {dim}")
    return dim


def _extract_and_apply_logical(forecast, verif, metric_kwargs, dim):
    """Extract callable `logical` from `metric_kwargs` and apply to `forecast` and
    `verif`."""
    if "comparison" in metric_kwargs:
        metric_kwargs = metric_kwargs.copy()
        comparison = metric_kwargs.pop("comparison")
    if "logical" in metric_kwargs:
        logical = metric_kwargs.pop("logical")
        if not callable(logical):
            raise ValueError(f"`logical` must be `callable`, found {type(logical)}")
        dim = _remove_member_from_dim_or_raise(dim)
        if "member" in forecast.dims:  # apply logical function to get
            forecast = logical(forecast).mean("member")  # forecast probability
            verif = logical(verif)  # binary outcome
        else:
            raise ValueError(
                f"Expected dimension `member` in forecast, found {list(forecast.dims)}"
            )
        # rename dim to time if forecast and verif dims allow

        return forecast, verif, metric_kwargs, dim
    elif (
        comparison.name == "e2o"
        and "logical" not in metric_kwargs
        and "member" not in dim
    ):  # allow e2o comparison without logical
        return forecast, verif, metric_kwargs, dim
    elif (
        comparison.name == "m2o" and "logical" not in metric_kwargs and "member" in dim
    ):  # allow m2o and member
        return forecast, verif, metric_kwargs, dim
    else:
        raise ValueError(
            "Please provide a callable `logical` to be applied to comparison and \
             verification data to get values in interval [0,1]."
        )


def _display_metric_metadata(self):
    summary = "----- Metric metadata -----\n"
    summary += f"Name: {self.name}\n"
    summary += f"Alias: {self.aliases}\n"
    # positively oriented
    if self.positive:
        summary += "Orientation: positive\n"
    else:
        summary += "Orientation: negative\n"
    # probabilistic or deterministic
    if self.probabilistic:
        summary += "Kind: probabilistic\n"
    else:
        summary += "Kind: deterministic\n"
    summary += f"Power to units: {self.unit_power}\n"
    summary += f"long_name: {self.long_name}\n"
    summary += f"Minimum skill: {self.minimum}\n"
    summary += f"Maximum skill: {self.maximum}\n"
    summary += f"Perfect skill: {self.perfect}\n"
    summary += f"Normalize: {self.normalize}\n"
    summary += f"Allows logical: {self.allows_logical}\n"
    # doc
    summary += f"Function: {self.function.__doc__}\n"
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
        normalize=False,
        allows_logical=False,
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
            normalize (bool, optional): Will the metric be normalized? Then metric
             function will require to get Comparison passed. Defaults to False.
            allows_logical (bool, optional): Does the metric allow a logical to be
              passed in metric_kwargs? Some probabilistic metrics allow this. Defaults
              to False.

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
        self.normalize = normalize
        self.allows_logical = allows_logical

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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return pearson_r(forecast, verif, dim=dim, **metric_kwargs)


__pearson_r = Metric(
    name="pearson_r",
    function=_pearson_r,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name="Pearson product-moment correlation coefficient",
    aliases=["pr", "acc", "pacc"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r_p_value

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
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(RuntimeWarning, DeprecationWarning))
        return pearson_r_p_value(forecast, verif, dim=dim, **metric_kwargs)


__pearson_r_p_value = Metric(
    name="pearson_r_p_value",
    function=_pearson_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name="Pearson product-moment correlation coefficient p value",
    aliases=["p_pval", "pvalue", "pval"],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _effective_sample_size(forecast, verif, dim=None, **metric_kwargs):
    """Effective sample size for temporally correlated data.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    .. note::
        This metric can only be used for hindcast-type simulations.

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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.effective_sample_size

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return effective_sample_size(forecast, verif, dim=dim, **metric_kwargs)


__effective_sample_size = Metric(
    name="effective_sample_size",
    function=_effective_sample_size,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name="Effective sample size for temporally correlated data",
    aliases=["n_eff", "eff_n"],
    minimum=0.0,
    maximum=np.inf,
)


def _pearson_r_eff_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are linearly uncorrelated, accounting
    for autocorrelation.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    .. note::
        This metric can only be used for hindcast-type simulations.

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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r_eff_p_value

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
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(RuntimeWarning, DeprecationWarning))
        return pearson_r_eff_p_value(forecast, verif, dim=dim, **metric_kwargs)


__pearson_r_eff_p_value = Metric(
    name="pearson_r_eff_p_value",
    function=_pearson_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Pearson's Anomaly correlation coefficient "
        "p value using the effective sample size"
    ),
    aliases=["p_pval_eff", "pvalue_eff", "pval_eff"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.spearman_r

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return spearman_r(forecast, verif, dim=dim, **metric_kwargs)


__spearman_r = Metric(
    name="spearman_r",
    function=_spearman_r,
    positive=True,
    probabilistic=False,
    unit_power=0.0,
    long_name="Spearman's rank correlation coefficient",
    aliases=["sacc", "sr"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.spearman_r_p_value

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
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return spearman_r_p_value(forecast, verif, dim=dim, **metric_kwargs)


__spearman_r_p_value = Metric(
    name="spearman_r_p_value",
    function=_spearman_r_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name="Spearman's rank correlation coefficient p value",
    aliases=["s_pval", "spvalue", "spval"],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _spearman_r_eff_p_value(forecast, verif, dim=None, **metric_kwargs):
    """Probability that forecast and verification data are monotonically uncorrelated,
    accounting for autocorrelation.

    .. note::
        Weights are not included here due to the dependence on temporal autocorrelation.

    .. note::
        This metric can only be used for hindcast-type simulations.

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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.spearman_r_eff_p_value

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
    # p value returns a runtime error when working with NaNs, such as on a climate
    # model grid. We can avoid this annoying output by specifically suppressing
    # warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(RuntimeWarning, DeprecationWarning))
        return spearman_r_eff_p_value(forecast, verif, dim=dim, **metric_kwargs)


__spearman_r_eff_p_value = Metric(
    name="spearman_r_eff_p_value",
    function=_spearman_r_eff_p_value,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name=(
        "Spearman's Rank correlation coefficient "
        "p value using the effective sample size"
    ),
    aliases=["s_pval_eff", "spvalue_eff", "spval_eff"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.mse

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
    return mse(forecast, verif, dim=dim, **metric_kwargs)


__mse = Metric(
    name="mse",
    function=_mse,
    positive=False,
    probabilistic=False,
    unit_power=2,
    long_name="Mean Squared Error",
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.rmse

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
    return rmse(forecast, verif, dim=dim, **metric_kwargs)


__rmse = Metric(
    name="rmse",
    function=_rmse,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name="Root Mean Squared Error",
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.mae

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
    return mae(forecast, verif, dim=dim, **metric_kwargs)


__mae = Metric(
    name="mae",
    function=_mae,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name="Mean Absolute Error",
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.median_absolute_error

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
    return median_absolute_error(forecast, verif, dim=dim, **metric_kwargs)


__median_absolute_error = Metric(
    name="median_absolute_error",
    function=_median_absolute_error,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name="Median Absolute Error",
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
        dim (str): Dimension(s) to perform metric over.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)
        metric_kwargs (dict): see xskillscore.mse

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
    if "comparison" in metric_kwargs:
        comparison = metric_kwargs.pop("comparison")
    else:
        raise ValueError(
            "Comparison needed to normalize NMSE. Not found in", metric_kwargs
        )
    mse_skill = __mse.function(forecast, verif, dim=dim, **metric_kwargs)
    var = verif.var(dim)
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


__nmse = Metric(
    name="nmse",
    function=_nmse,
    positive=False,
    probabilistic=False,
    unit_power=0.0,
    long_name="Normalized Mean Squared Error",
    aliases=["nev"],
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
    normalize=True,
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
        dim (str): Dimension(s) to perform metric over.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)
        metric_kwargs (dict): see xskillscore.mae

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
    if "comparison" in metric_kwargs:
        comparison = metric_kwargs.pop("comparison")
    else:
        raise ValueError(
            "Comparison needed to normalize NMAE. Not found in", metric_kwargs
        )
    mae_skill = __mae.function(forecast, verif, dim=dim, **metric_kwargs)
    std = verif.std(dim)
    fac = _get_norm_factor(comparison)
    nmae_skill = mae_skill / std / fac
    return nmae_skill


__nmae = Metric(
    name="nmae",
    function=_nmae,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="Normalized Mean Absolute Error",
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
    normalize=True,
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
        dim (str): Dimension(s) to perform metric over.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)
        metric_kwargs (dict): see xskillscore.rmse

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
    if "comparison" in metric_kwargs:
        comparison = metric_kwargs.pop("comparison")
    else:
        raise ValueError(
            "Comparison needed to normalize NRMSE. Not found in", metric_kwargs
        )
    rmse_skill = __rmse.function(forecast, verif, dim=dim, **metric_kwargs)
    std = verif.std(dim)
    fac = _get_norm_factor(comparison)
    nrmse_skill = rmse_skill / std / np.sqrt(fac)
    return nrmse_skill


__nrmse = Metric(
    name="nrmse",
    function=_nrmse,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="Normalized Root Mean Squared Error",
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
    normalize=True,
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
        dim (str): Dimension(s) to perform metric over.
        comparison (str): Name comparison needed for normalization factor `fac`, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)
        metric_kwargs (dict): see xskillscore.mse

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
    if "comparison" in metric_kwargs:
        comparison = metric_kwargs.pop("comparison")
    else:
        raise ValueError(
            "Comparison needed to normalize MSSS. Not found in", metric_kwargs
        )
    mse_skill = __mse.function(forecast, verif, dim=dim, **metric_kwargs)
    var = verif.var(dim)
    fac = _get_norm_factor(comparison)
    msess_skill = 1 - mse_skill / var / fac
    return msess_skill


__msess = Metric(
    name="msess",
    function=_msess,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Mean Squared Error Skill Score",
    aliases=["ppp", "msss"],
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
    normalize=True,
)


def _mape(forecast, verif, dim=None, **metric_kwargs):
    """Mean Absolute Percentage Error (MAPE).

    Mean absolute error (``mae``) expressed as the fractional error relative to the
    verification data.

    .. math::
        MAPE = \\frac{1}{n} \\sum \\frac{|f-o|}{|o|}

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.mape

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
        * xskillscore.mape
    """
    return mape(forecast, verif, dim=dim, **metric_kwargs)


__mape = Metric(
    name="mape",
    function=_mape,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="Mean Absolute Percentage Error",
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.smape

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
    return smape(forecast, verif, dim=dim, **metric_kwargs)


__smape = Metric(
    name="smape",
    function=_smape,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="symmetric Mean Absolute Percentage Error",
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _uacc(forecast, verif, dim=None, **metric_kwargs):
    """Bushuk's unbiased Anomaly Correlation Coefficient (uACC).

    This is typically used in perfect model studies. Because the perfect model Anomaly
    Correlation Coefficient (ACC) is strongly state dependent, a standard ACC (e.g. one
    computed using ``pearson_r``) will be highly sensitive to the set of start dates
    chosen for the perfect model study. The Mean Square Skill Score (``MESSS``) can be
    related directly to the ACC as ``MESSS = ACC^(2)`` (see Murphy 1988 and
    Bushuk et al. 2019), so the unbiased ACC can be derived as ``uACC = sqrt(MESSS)``.

    .. math::
        uACC = \\sqrt{MSESS}
             = \\sqrt{1 - \\frac{\\overline{(f - o)^{2}}}{\\sigma^2_{ref} \\cdot fac}},

    where :math:`fac` is 1 when using comparisons involving the ensemble mean (``m2e``,
    ``e2c``, ``e2o``) and 2 when using comparisons involving individual ensemble
    members (``m2c``, ``m2m``, ``m2o``). See
    :py:func:`~climpred.metrics._get_norm_factor`.

    .. note::
        Because of the square root involved, any negative ``MSESS`` values are
        automatically converted to NaNs.

    Args:
        forecast (xarray object): Forecast.
        verif (xarray object): Verification data.
        dim (str): Dimension(s) to perform metric over.
        comparison (str): Name comparison needed for normalization factor ``fac``, see
            :py:func:`~climpred.metrics._get_norm_factor`
            (Handled internally by the compute functions)
        metric_kwargs (dict): see xskillscore.mse

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
    messs_res = __msess.function(forecast, verif, dim=dim, **metric_kwargs)
    # Negative values are automatically turned into nans from xarray.
    uacc_res = messs_res ** 0.5
    return uacc_res


__uacc = Metric(
    name="uacc",
    function=_uacc,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Bushuk's unbiased ACC",
    minimum=0.0,
    maximum=1.0,
    perfect=1.0,
    normalize=True,
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xarray.std

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
    return forecast.std(dim=dim, **metric_kwargs) / verif.std(dim=dim, **metric_kwargs)


__std_ratio = Metric(
    name="std_ratio",
    function=_std_ratio,
    positive=None,
    probabilistic=False,
    unit_power=0,
    long_name="Ratio of standard deviations of the forecast and verification data",
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
        dim (str): Dimension(s) to perform metric over
        metric_kwargs (dict): see xarray.mean

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
    return (forecast - verif).mean(dim=dim, **metric_kwargs)


__unconditional_bias = Metric(
    name="unconditional_bias",
    function=_unconditional_bias,
    positive=False,
    probabilistic=False,
    unit_power=1,
    long_name="Unconditional bias",
    aliases=["u_b", "bias"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r and xarray.std

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
    return acc - __std_ratio.function(forecast, verif, dim=dim, **metric_kwargs)


__conditional_bias = Metric(
    name="conditional_bias",
    function=_conditional_bias,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="Conditional bias",
    aliases=["c_b", "cond_bias"],
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r and xarray.std

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
    return std_ratio * acc


__bias_slope = Metric(
    name="bias_slope",
    function=_bias_slope,
    positive=False,
    probabilistic=False,
    unit_power=0,
    long_name="Bias slope",
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
        dim (str): Dimension(s) to perform metric over.
        metric_kwargs (dict): see xskillscore.pearson_r, xarray.mean and xarray.std

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
    ) / verif.std(dim=dim, **metric_kwargs)
    return acc ** 2 - conditional_bias ** 2 - uncond_bias ** 2


__msess_murphy = Metric(
    name="msess_murphy",
    function=_msess_murphy,
    positive=True,
    probabilistic=False,
    unit_power=0,
    long_name="Murphy's Mean Square Error Skill Score",
    aliases=["msss_murphy"],
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


#######################
# PROBABILISTIC METRICS
#######################


def _brier_score(forecast, verif, dim=None, **metric_kwargs):
    """Brier Score for binary events.

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
        as one (a "hit") or zero (a "miss"). So either provide a function with
        with binary outcomes `logical` in `metric_kwargs` or create binary
        verifs and probability forecasts by
        `hindcast.map(logical).mean('member')`.
        This Brier Score is not the original formula given in Brier's 1950 paper.

    Args:
        forecast (xr.object): Raw forecasts with ``member`` dimension if `logical`
            provided in `metric_kwargs`. Probability forecasts in [0,1] if `logical` is
            not provided.
        verif (xr.object): Verification data without ``member`` dim. Raw verification if
            `logical` provided, else binary verification.
        dim (list or str): Dimensions to aggregate. Requires `member` if `logical`
            provided in `metric_kwargs` to create probability forecasts. If `logical`
            not provided in `metric_kwargs`, should not include `member`.
        metric_kwargs (dict): optional
            logical (callable): Function with bool result to be applied to verification
                data and forecasts and then ``mean('member')`` to get forecasts and
                verification data in interval [0,1].
            see xskillscore.brier_score

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
        * https://www.nws.noaa.gov/oh/rfcdev/docs/
          Glossary_Forecast_Verification_Metrics.pdf
        * https://en.wikipedia.org/wiki/Brier_score

    See also:
        * properscoring.brier_score
        * xskillscore.brier_score

    Example:
        Define a boolean/logical function for binary scoring:

        >>> def pos(x): return x > 0  # checking binary outcomes

        Option 1. Pass with keyword `logical`: (Works also for PerfectModelEnsemble)

        >>> hindcast.verify(metric='brier_score', comparison='m2o',
                dim='member', alignment='same_verifs', logical=pos)

        Option 2. Pre-process to generate a binary forecast and verification product:

        >>> hindcast.map(pos).verify(metric='brier_score',
                comparison='m2o', dim='member', alignment='same_verifs')

        Option 3. Pre-process to generate a probability forecast and binary
        verification product. Because `member` no present in `hindcast`, use
        ``comparison='e2o'`` and ``dim=[]``:

        >>> hindcast.map(pos).mean('member').verify(metric='brier_score',
                comparison='e2o', dim=[], alignment='same_verifs')
    """
    forecast, verif, metric_kwargs, dim = _extract_and_apply_logical(
        forecast, verif, metric_kwargs, dim
    )
    return brier_score(verif, forecast, dim=dim, **metric_kwargs)


__brier_score = Metric(
    name="brier_score",
    function=_brier_score,
    positive=False,
    probabilistic=True,
    unit_power=0,
    long_name="Brier Score",
    aliases=["brier", "bs"],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
    allows_logical=True,
)


def _threshold_brier_score(forecast, verif, dim=None, **metric_kwargs):
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
        dim (list of str): Dimension to apply metric over. Expects at least
            `member`. Other dimensions are passed to `xskillscore` and averaged.
        threshold (int, float, xr.object): Threshold to check exceedance, see
            properscoring.threshold_brier_score.
        metric_kwargs (dict): optional, see xskillscore.threshold_brier_score

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
        >>> hindcast.verify(metric='threshold_brier_score', comparison='m2o',
                dim='member', threshold=.5)
        >>> hindcast.verify(metric='threshold_brier_score', comparison='m2o',
                dim='member', threshold=[.3, .7])
    """
    if "threshold" not in metric_kwargs:
        raise ValueError("Please provide threshold.")
    else:
        threshold = metric_kwargs.pop("threshold")
    dim = _remove_member_from_dim_or_raise(dim)
    # switch args b/c xskillscore.threshold_brier_score(verif, forecasts)
    return threshold_brier_score(verif, forecast, threshold, dim=dim, **metric_kwargs)


__threshold_brier_score = Metric(
    name="threshold_brier_score",
    function=_threshold_brier_score,
    positive=False,
    probabilistic=True,
    unit_power=0,
    long_name="Threshold Brier Score",
    aliases=["tbs"],
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _crps(forecast, verif, dim=None, **metric_kwargs):
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
        dim (list of str): Dimension to apply metric over. Expects at least
            `member`. Other dimensions are passed to `xskillscore` and averaged.
        metric_kwargs (dict): optional, see xskillscore.crps_ensemble

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

    Example:
        >>> hindcast.verify(metric='crps', comparison='m2o', dim='member')
    """
    dim = _remove_member_from_dim_or_raise(dim)
    # switch positions because xskillscore.crps_ensemble(verif, forecasts)
    return crps_ensemble(verif, forecast, dim=dim, **metric_kwargs)


__crps = Metric(
    name="crps",
    function=_crps,
    positive=False,
    probabilistic=True,
    unit_power=1.0,
    long_name="Continuous Ranked Probability Score",
    minimum=0.0,
    maximum=np.inf,
    perfect=0.0,
)


def _crps_gaussian(verification, mu, sig, dim=None, **metric_kwargs):
    """Computes the CRPS of verification data ``o`` relative to normally distributed
    forecasts with mean ``mu`` and standard deviation ``sig``.

    .. note::
        This is a helper function for CRPSS and cannot be called directly by a user.

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        mu (xr.object): The mean of the verification data.
        sig (xr.object): The standard deviation verification data.
        metric_kwargs (dict): optional, see xskillscore.crps_gaussian

    See also:
        * properscoring.crps_gaussian
        * xskillscore.crps_gaussian
    """
    return crps_gaussian(verification, mu, sig, dim=dim, **metric_kwargs)


def _crps_quadrature(verification, cdf_or_dist, dim=None, **metric_kwargs):
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
        metric_kwargs (dict): see xskillscore.crps_quadrature

    See also:
        * properscoring.crps_quadrature
        * xskillscore.crps_quadrature
    """
    return crps_quadrature(verification, cdf_or_dist, dim=dim, **metric_kwargs)


def _crpss(forecast, verif, dim=None, **metric_kwargs):
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
        dim (list of str): Dimension to apply metric over. Expects at least
            `member`. Other dimensions are passed to `xskillscore` and averaged.
        metric_kwargs (dict): optional
            gaussian (bool, optional): If ``True``, assume Gaussian distribution for
                baseline skill. Defaults to ``True``.
            see xskillscore.crps_ensemble, xskillscore.crps_gaussian and
            xskillscore.crps_quadrature

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
        >>> hindcast.verify(metric='crpss', comparison='m2o',
                alignment='same_verifs', dim='member')
        >>> perfect_model.verify(metric='crpss', comparison='m2m', dim='member',
                gaussian=False, cdf_or_dist=scipy.stats.norm, xminimum=-10,
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
    gaussian = metric_kwargs.pop("gaussian", True)
    # dim="member" only for crps_gaussian, remove member for crps_gaussian/quadrature
    dim_for_gaussian = dim.copy()
    dim_for_gaussian.remove("member")
    if gaussian:
        ref_skill = _crps_gaussian(
            verif, mu, sig, dim=dim_for_gaussian, **metric_kwargs
        )
    else:
        cdf_or_dist = metric_kwargs.pop("cdf_or_dist", norm)
        xmin = metric_kwargs.pop("xmin", None)
        xmax = metric_kwargs.pop("xmax", None)
        tol = metric_kwargs.pop("tol", 1e-6)
        ref_skill = _crps_quadrature(
            forecast,
            cdf_or_dist,
            xmin=xmin,
            xmax=xmax,
            tol=tol,
            dim=dim_for_gaussian,
            **metric_kwargs,
        )
    forecast_skill = __crps.function(forecast, verif, dim=dim, **metric_kwargs)
    skill_score = 1 - forecast_skill / ref_skill
    return skill_score


__crpss = Metric(
    name="crpss",
    function=_crpss,
    positive=True,
    probabilistic=True,
    unit_power=0,
    long_name="Continuous Ranked Probability Skill Score",
    minimum=-np.inf,
    maximum=1.0,
    perfect=1.0,
)


def _crpss_es(forecast, verif, dim=None, **metric_kwargs):
    """Continuous Ranked Probability Skill Score Ensemble Spread.

    If the ensemble variance is smaller than the observed ``mse``, the ensemble is
    said to be under-dispersive (or overconfident). An ensemble with variance larger
    than the verification data indicates one that is over-dispersive (underconfident).

    .. math::
        CRPSS = 1 - \\frac{CRPS(\\sigma^2_f)}{CRPS(\\sigma^2_o)}

    Args:
        forecast (xr.object): Forecast with ``member`` dim.
        verif (xr.object): Verification data without ``member`` dim.
        dim (list of str): Dimension to apply metric over. Expects at least
            `member`. Other dimensions are passed to `xskillscore` and averaged.
        metric_kwargs (dict): see xskillscore.crps_ensemble and xskillscore.mse

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

    Example:
        >>> hindcast.verify(metric='crpss_es', comparison='m2o',
                alignment='same_verifs', dim='member')
    """

    # helper dim to calc mu
    rdim = [d for d in verif.dims if d in CLIMPRED_DIMS]
    mu = verif.mean(rdim)
    # forecast, verif_member = xr.broadcast(forecast, verif)
    dim_no_member = [d for d in dim if d != "member"]
    ensemble_spread = forecast.std("member").mean(dim=dim_no_member, **metric_kwargs)
    mse_h = __mse.function(
        forecast.mean("member"), verif, dim=dim_no_member, **metric_kwargs
    )
    crps_h = crps_gaussian(verif, mu, mse_h, dim=dim_no_member, **metric_kwargs)
    crps_r = crps_gaussian(
        verif, mu, ensemble_spread, dim=dim_no_member, **metric_kwargs
    )
    res = 1 - crps_h / crps_r
    if "time" in res.dims:
        res = res.rename({"time": "init"})
    return res


__crpss_es = Metric(
    name="crpss_es",
    function=_crpss_es,
    positive=True,
    probabilistic=True,
    unit_power=0,
    long_name="CRPSS Ensemble Spread",
    minimum=-np.inf,
    maximum=0.0,
    perfect=0.0,
)


def _discrimination(forecast, verif, dim=None, **metric_kwargs):
    """
    Returns the data required to construct the discrimination diagram for an event. The
    histogram of forecasts likelihood when observations indicate an event has occurred
    and has not occurred.

    Args:
        forecast (xr.object): Raw forecasts with ``member`` dimension if `logical`
            provided in `metric_kwargs`. Probability forecasts in [0,1] if `logical` is
            not provided.
        verif (xr.object): Verification data without ``member`` dim. Raw verification if
            `logical` provided, else binary verification.
        dim (list or str): Dimensions to aggregate. Requires `member` if `logical`
            provided in `metric_kwargs` to create probability forecasts. If `logical`
            not provided in `metric_kwargs`, should not include `member`. At least one
            dimension other than `member` is required.
        logical (callable, optional): Function with bool result to be applied to
            verification data and forecasts and then ``mean('member')`` to get
            forecasts and verification data in interval [0,1]. Passed via metric_kwargs.
        probability_bin_edges (array_like, optional): Probability bin edges used to
            compute the histograms. Bins include the left most edge, but not the
            right. Passed via metric_kwargs. Defaults to 6 equally spaced edges between
            0 and 1+1e-8.


    Returns:
        Discrimination (xr.object) with added dimension "event" containing the
        histograms of forecast probabilities when the event was observed and not
        observed

    Details:
        +-----------------+------------------------+
        | **perfect**     | distinct distributions |
        +-----------------+------------------------+

    See also:
        * xskillscore.discrimination

    Example:
        Define a boolean/logical function for binary scoring:

        >>> def pos(x): return x > 0  # checking binary outcomes

        Option 1. Pass with keyword `logical`: (Works also for PerfectModelEnsemble)

        >>> hindcast.verify(metric='discrimination', comparison='m2o',
                dim=['member', 'init'], alignment='same_verifs', logical=pos)

        Option 2. Pre-process to generate a binary forecast and verification product:

        >>> hindcast.map(pos).verify(metric='discrimination',
                comparison='m2o', dim=['member','init'], alignment='same_verifs')

        Option 3. Pre-process to generate a probability forecast and binary
        verification product. Because `member` no present in `hindcast`, use
        ``comparison='e2o'`` and ``dim='init'``:

        >>> hindcast.map(pos).mean('member').verify(metric='discrimination',
                comparison='e2o', dim='init', alignment='same_verifs')
    """
    forecast, verif, metric_kwargs, dim = _extract_and_apply_logical(
        forecast, verif, metric_kwargs, dim
    )
    return discrimination(verif, forecast, dim=dim, **metric_kwargs)


__discrimination = Metric(
    name="discrimination",
    function=_discrimination,
    positive=None,
    probabilistic=True,
    unit_power=0,
    long_name="Discrimination",
    allows_logical=True,
)


def _reliability(forecast, verif, dim=None, **metric_kwargs):
    """
    Returns the data required to construct the reliability diagram for an event. The
    the relative frequencies of occurrence of an event for a range of forecast
    probability bins.


    Args:
        forecast (xr.object): Raw forecasts with ``member`` dimension if `logical`
            provided in `metric_kwargs`. Probability forecasts in [0,1] if `logical` is
            not provided.
        verif (xr.object): Verification data without ``member`` dim. Raw verification if
            `logical` provided, else binary verification.
        dim (list or str): Dimensions to aggregate. Requires `member` if `logical`
            provided in `metric_kwargs` to create probability forecasts. If `logical`
            not provided in `metric_kwargs`, should not include `member`.
        logical (callable, optional): Function with bool result to be applied to
            verification data and forecasts and then ``mean('member')`` to get
            forecasts and verification data in interval [0,1]. Passed via metric_kwargs.
        probability_bin_edges (array_like, optional): Probability bin edges used to
            compute the reliability. Bins include the left most edge, but not the
            right. Passed via metric_kwargs. Defaults to 6 equally spaced edges between
            0 and 1+1e-8.

    Returns:
        reliability (xr.object): The relative frequency of occurrence for each
            probability bin


    Details:
        +-----------------+-------------------+
        | **perfect**     | flat distribution |
        +-----------------+-------------------+

    See also:
        * xskillscore.reliability

    Example:
        Define a boolean/logical function for binary scoring:

        >>> def pos(x): return x > 0  # checking binary outcomes

        Option 1. Pass with keyword `logical`: (Works also for PerfectModelEnsemble)

        >>> hindcast.verify(metric='reliability', comparison='m2o',
                dim=['member','init'], alignment='same_verifs', logical=pos)

        Option 2. Pre-process to generate a binary forecast and verification product:

        >>> hindcast.map(pos).verify(metric='reliability',
                comparison='m2o', dim=['member','init'], alignment='same_verifs')

        Option 3. Pre-process to generate a probability forecast and binary
        verification product. Because `member` no present in `hindcast`, use
        ``comparison='e2o'`` and ``dim='init'``:

        >>> hindcast.map(pos).mean('member').verify(metric='reliability',
                comparison='e2o', dim='init', alignment='same_verifs')
    """
    forecast, verif, metric_kwargs, dim = _extract_and_apply_logical(
        forecast, verif, metric_kwargs, dim
    )
    return reliability(verif, forecast, dim=dim, **metric_kwargs)


__reliability = Metric(
    name="reliability",
    function=_reliability,
    positive=None,
    probabilistic=True,
    unit_power=0,
    long_name="Reliability",
    allows_logical=True,
)


def _rank_histogram(forecast, verif, dim=None, **metric_kwargs):
    """Rank histogram or Talagrand diagram.


    Args:
        forecast (xr.object): Raw forecasts with ``member`` dimension.
        verif (xr.object): Verification data without ``member`` dim.
        dim (list or str): Dimensions to aggregate. Requires to contain `member` and at
            least one additional dimension.

    Details:
        +-----------------+-------------------+
        | **perfect**     | flat distribution |
        +-----------------+-------------------+

    See also:
        * xskillscore.rank_histogram

    Example:
        >>> hindcast.verify(metric='rank_histogram', comparison='m2o',
                dim=['member','init'], alignment='same_verifs')
        >>> perfect_model.verify(metric='rank_histogram', comparison='m2c',
                dim=['member','init'])

    """
    dim = _remove_member_from_dim_or_raise(dim)
    return rank_histogram(verif, forecast, dim=dim, **metric_kwargs)


__rank_histogram = Metric(
    name="rank_histogram",
    function=_rank_histogram,
    positive=None,
    probabilistic=True,
    unit_power=0,
    long_name="rank_histogram",
)


def _rps(forecast, verif, dim=None, **metric_kwargs):
    """Ranked Probability Score.

    .. math::
        RPS(p, k) = 1/M \\sum_{m=1}^{M} [(\\sum_{k=1}^{m} p_k) - (\\sum_{k=1}^{m} \
            o_k)]^{2}

    Args:
        forecast (xr.object): Raw forecasts with ``member`` dimension.
        verif (xr.object): Verification data without ``member`` dim.
        dim (list or str): Dimensions to aggregate. Requires to contain `member`.
        category_edges (array_like): Category bin edges used to compute the CDFs.
            Bins include the left most edge, but not the right. Passed via
            metric_kwargs.

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
        * xskillscore.rps

    Example:
        >>> category_edges = np.array([-.5, 0., .5, 1.])
        >>> hindcast.verify(metric='rps', comparison='m2o', dim='member',
                alignment='same_verifs', category_edges=category_edges)
        >>> perfect_model.verify(metric='rps', comparison='m2c',
                dim='member', category_edges=category_edges)

    """
    dim = _remove_member_from_dim_or_raise(dim)
    if "category_edges" in metric_kwargs:
        category_edges = metric_kwargs.pop("category_edges")
    else:
        raise ValueError("require category_edges")
    return rps(verif, forecast, category_edges, dim=dim, **metric_kwargs)


__rps = Metric(
    name="rps",
    function=_rps,
    positive=False,
    probabilistic=True,
    unit_power=0,
    long_name="rps",
    minimum=0.0,
    maximum=1.0,
    perfect=0.0,
)


def _contingency(forecast, verif, score="table", dim=None, **metric_kwargs):
    """Contingency table.

    Args:
        forecast (xr.object): Raw forecasts.
        verif (xr.object): Verification data.
        dim (list or str): Dimensions to aggregate.
        score (str): Score derived from contingency table. Attribute from
            xskillscore.Contingency. Use ``score=table`` to return a contingency table
            or any other contingency score, e.g. ``score=hit_rate``.
        observation_category_edges (array_like): Category bin edges used to compute
            the observations CDFs. Bins include the left most edge, but not the right.
            Passed via metric_kwargs.
        forecast_category_edges  (array_like): Category bin edges used to compute
            the forecast CDFs. Bins include the left most edge, but not the right.
            Passed via metric_kwargs

    See also:
        * xskillscore.Contingency

    References
    ----------
        * http://www.cawcr.gov.au/projects/verification/
        * https://xskillscore.readthedocs.io/en/stable/api.html#contingency-based-metrics # noqa

    Example:
        >>> category_edges = np.array([-0.5, 0., .5, 1.])
        >>> hindcast.verify(metric='contingency', score='table', comparison='m2o',
                dim=[], alignment='same_verifs',
                observation_category_edges=category_edges,
                forecast_category_edges=category_edges)
        >>> perfect_model.verify(metric='contingency', score='hit_rate',
                comparison='m2c', dim=['member','init'],
                observation_category_edges=category_edges,
                forecast_category_edges=category_edges)

    """
    if score == "table":
        return Contingency(verif, forecast, dim=dim, **metric_kwargs).table
    else:
        return getattr(Contingency(verif, forecast, dim=dim, **metric_kwargs), score)()


__contingency = Metric(
    name="contingency",
    function=_contingency,
    positive=None,
    probabilistic=False,
    unit_power=0,
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
    __contingency,
    __rank_histogram,
    __discrimination,
    __reliability,
    __rps,
]


# To match a metric/comparison for (multiple) keywords.
METRIC_ALIASES = dict()
for m in __ALL_METRICS__:
    if m.aliases is not None:
        for a in m.aliases:
            METRIC_ALIASES[a] = m.name


DETERMINISTIC_METRICS = [m.name for m in __ALL_METRICS__ if not m.probabilistic]
DETERMINISTIC_HINDCAST_METRICS = DETERMINISTIC_METRICS.copy()
# Metrics to be used in compute_perfect_model.
DETERMINISTIC_PM_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()
# Effective sample size does not make much sense in this framework.
DETERMINISTIC_PM_METRICS = [
    e
    for e in DETERMINISTIC_PM_METRICS
    if e
    not in ("effective_sample_size", "pearson_r_eff_p_value", "spearman_r_eff_p_value",)
]
# Used to set attrs['units'] to None.
DIMENSIONLESS_METRICS = [m.name for m in __ALL_METRICS__ if m.unit_power == 1]
# More positive skill is better than more negative.
POSITIVELY_ORIENTED_METRICS = [m.name for m in __ALL_METRICS__ if m.positive]
PROBABILISTIC_METRICS = [m.name for m in __ALL_METRICS__ if m.probabilistic]
# Combined allowed metrics for compute_hindcast and compute_perfect_model
HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS + PROBABILISTIC_METRICS
PM_METRICS = DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS
ALL_METRICS = [m.name for m in __ALL_METRICS__]
