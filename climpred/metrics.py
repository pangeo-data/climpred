import warnings

import numpy as np
import xarray as xr
from scipy.stats import norm
from xskillscore import (
    brier_score,
    crps_ensemble,
    crps_gaussian,
    crps_quadrature,
    mad,
    mae,
    mape,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
    smape,
    spearman_r,
    spearman_r_p_value,
    threshold_brier_score,
)

from .constants import CLIMPRED_DIMS


def _get_norm_factor(comparison):
    """Get normalization factor for PPP, NMSE, NRMSE, MSSS.

    Used in compute_perfect_model. Comparison 'm2e' gets smaller rmse's than
    'm2m' by design, see Seferian et al. 2018. 'm2m', 'm2c' ensemble variance
    is divided by 2 to get control variance.

    Args:
        comparison (function): comparison function.

    Returns:
        fac (int): normalization factor.

    Raises:
        KeyError: if comparison is not matching.

    """
    comparison_name = comparison.__name__
    if comparison_name in ['_m2e', '_e2c', '_e2r']:
        fac = 1
    elif comparison_name in ['_m2c', '_m2m', '_m2r']:
        fac = 2
    else:
        raise KeyError('specify comparison to get normalization factor.')
    return fac


def _pearson_r(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate Person's Anomaly Correlation Coefficient (ACC).

    .. math::
        ACC = \\frac{cov(f, o)}{\\sigma_{f}\\cdot\\sigma_{o}}

    .. note::
        Use metric ``pearson_r_p_value`` to get the corresponding pvalue.

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.pearson_r

    Range:
        * perfect: 1
        * min: -1

    See also:
        * xskillscore.pearson_r
        * xskillscore.pearson_r_p_value
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return pearson_r(forecast, reference, dim=dim, weights=weights, skipna=skipna)


# TODO: all metrics: docs
def _pearson_r_p_value(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the probability associated with Person's Anomaly Correlation
    Coefficient not being random.

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.pearson_r_p_value

    Range:
        * perfect: 0
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


def _spearman_r(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate Spearman's Anomaly Correlation Coefficient (SACC).

    .. math::
        SACC = ACC(ranked(f),ranked(o))

    .. note::
        Use metric ``spearman_r_p_value`` to get the corresponding pvalue.

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.spearman_r

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


def _spearman_r_p_value(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the probability associated with the ACC not being random.

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.spearman_r_p_value

    Range:
        * perfect: 0
        * max: 1

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


def _mse(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the Mean Sqaure Error (MSE).

    .. math::
        MSE = \\overline{(f - o)^{2}}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.mse

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


def _rmse(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the Root Mean Sqaure Error (RMSE).

    .. math::
        RMSE = \\sqrt{\\overline{(f - o)^{2}}}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.rmse

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


def _mae(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the Mean Absolute Error (MAE).

    .. math::
        MAE = \\overline{|f - o|}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.mae

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


def _mad(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the Median Absolute Deviation (MAD).

    .. math::
        MAD = median(|f - o|)

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.mad

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mad
    """
    skipna = metric_kwargs.get('skipna', False)
    return mad(forecast, reference, dim=dim, skipna=skipna)


def _mape(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    .. math::
        MAPE = MAPE = 1/n \sum \frac{|f-o|}{|o|}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.mape

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return mape(forecast, reference, dim=dim, weights=weights, skipna=skipna)


def _smape(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate the symmetric Mean Absolute Percentage Error (sMAPE).

    .. math::
        sMAPE = 1/n \sum \frac{|f-o|}{|f|+|o|}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna, see xskillscore.smape

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.smape
    """
    weights = metric_kwargs.get('weights', None)
    skipna = metric_kwargs.get('skipna', False)
    return smape(forecast, reference, dim=dim, weights=weights, skipna=skipna)


def _brier_score(forecast, reference, **metric_kwargs):
    """Calculate Brier score for forecasts on binary reference.

    ..math:
        BS(f, o) = (f - o)^2

    Args:
        * forecast (xr.object): forecast with `member` dim
        * reference (xr.object): references without `member` dim
        * func (function): function to be applied to reference and forecasts
                           and then mean('member') to get forecasts and
                           reference in interval [0,1].
                           (required to be added via **metric_kwargs)

    Reference:
        * Brier, Glenn W. “VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF
        PROBABILITY.” Monthly Weather Review 78, no. 1 (1950).
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


def _threshold_brier_score(forecast, reference, **metric_kwargs):
    """
    Calculate the Brier scores of an ensemble for exceeding given thresholds.
    Provide threshold via metric_kwargs.

    .. math::
        CRPS(F, x) = \int_z BS(F(z), H(z - x)) dz

    Range:
        * perfect: 0
        * min: 0
        * max: 1

    Args:
        * forecast (xr.object): forecast with `member` dim
        * reference (xr.object): references without `member` dim
        * threshold (int, float, xr.object): Threshold to check exceedance,
            see properscoring.threshold_brier_score
            (required to be added via **metric_kwargs)

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


def _crps(forecast, reference, **metric_kwargs):
    """
    Continuous Ranked Probability Score (CRPS) is the probabilistic MSE.

    Args:
        * forecast (xr.object): forecast with `member` dim
        * reference (xr.object): references without `member` dim
        * metric_kwargs (optional dict): weights, see properscoring.crps_ensemble

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    References:
        * Matheson, James E., and Robert L. Winkler. “Scoring Rules for
          Continuous Probability Distributions.” Management Science 22, no. 10
          (June 1, 1976): 1087–96. https://doi.org/10/cwwt4g.

    See also:
        * properscoring.crps_ensemble
        * xskillscore.crps_ensemble
    """
    # switch positions because xskillscore.crps_ensemble(obs, forecasts)
    weights = metric_kwargs.get('weights', None)
    return crps_ensemble(reference, forecast, weights=weights)


def _crps_gaussian(forecast, mu, sig, **metric_kwargs):
    """CRPS assuming a gaussian distribution. Helper function for CRPSS.

    Args:
        * forecast (xr.object): forecast with `member` dim
        * mu (xr.object): mean reference
        * sig (xr.object): standard deviation reference

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
        * forecast (xr.object): forecast with `member` dim
        * see properscoring.crps_quadrature

    See also:
        * properscoring.crps_quadrature
        * xskillscore.crps_quadrature
    """
    return crps_quadrature(forecast, cdf_or_dist, xmin, xmax, tol)


def _crpss(forecast, reference, **metric_kwargs):
    """Continuous Ranked Probability Skill Score

    .. note::
        When assuming a gaussian distribution of forecasts, use default gaussian=True.
        If not gaussian, you may specify the distribution type, xmin/xmax/tolerance
        for integration (see xskillscore.crps_quadrature).

    .. math::
        CRPSS = 1 - \\frac{CRPS_{init}}{CRPS_{clim}}

    Args:
        * forecast (xr.object): forecast with `member` dim
        * reference (xr.object): references without `member` dim
        * gaussian (bool): Assuming gaussian distribution for baseline skill.
                           Default: True (optional)
        * cdf_or_dist (scipy.stats): distribution to assume if not gaussian.
                                     default: scipy.stats.norm
        * xmin, xmax, tol: only relevant if not gaussian
                           (see xskillscore.crps_quadrature)

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
                                  cdf_or_dist=scipy.stats.norm, xmin=-10,
                                  xmax=10, tol=1e-6)

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
    forecast_skill = _crps(forecast, reference, **metric_kwargs)
    skill_score = 1 - forecast_skill / ref_skill.mean('member')
    return skill_score


def _crpss_es(forecast, reference, **metric_kwargs):
    """CRPSS Ensemble Spread.

    .. math:: CRPSS = 1 - \\frac{CRPS(\\sigma^2_f)}{CRPS(\\sigma^2_o}))

    Args:
        * forecast (xr.object): forecast with `member` dim
        * reference (xr.object): references without `member` dim
        * metric_kwargs (optional): weights, skipna used for mse

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
    sig_r = _mse(forecast, ref2, dim='member', **mse_kwargs).mean(dim2)
    sig_h = _mse(forecast.mean(dim2), ref2.mean(dim2), dim='member', **mse_kwargs)
    crps_h = _crps_gaussian(forecast, mu, sig_h)
    if 'member' in crps_h.dims:
        crps_h = crps_h.mean('member')
    crps_r = _crps_gaussian(forecast, mu, sig_r)
    if 'member' in crps_r.dims:
        crps_r = crps_r.mean('member')
    return 1 - crps_h / crps_r


def _bias(forecast, reference, dim=None, **metric_kwargs):
    """Calculate unconditional bias.

    .. math::
        bias = f - o

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.

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


def _msss_murphy(forecast, reference, dim=None, **metric_kwargs):
    """Calculate Murphy's Mean Square Skill Score (MSSS).

    .. math::
        MSSS_{Murphy} = r_{fo}^2 - [\\text{conditional bias}]^2 -\
         [\\frac{\\text{(unconditional) bias}}{\\sigma_o}]^2

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        metric_kwargs: (optional) weights, skipna

    References:
        * https://www-miklip.dkrz.de/about/murcss/
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    acc = _pearson_r(forecast, reference, dim=dim, **metric_kwargs)
    conditional_bias = _conditional_bias(forecast, reference, dim=dim, **metric_kwargs)
    uncond_bias = _bias(forecast, reference, dim=dim, **metric_kwargs) / reference.std(
        dim
    )
    skill = acc ** 2 - conditional_bias ** 2 - uncond_bias ** 2
    return skill


def _conditional_bias(forecast, reference, dim=None, **metric_kwargs):
    """Calculate the conditional bias between forecast and reference.

    .. math:: \\text{conditional bias} = r_{fo} - \\frac{\\sigma_f}{\\sigma_o}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    acc = _pearson_r(forecast, reference, dim=dim, **metric_kwargs)
    conditional_bias = (
        acc - _std_ratio(forecast, reference, dim=dim, **metric_kwargs) ** -1
    )
    return conditional_bias


def _std_ratio(forecast, reference, dim=None, **metric_kwargs):
    """Calculate the ratio of standard deviations of reference over forecast.

    .. math:: \\text{std ratio} = \\frac{\\sigma_o}{\\sigma_f}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    ratio = reference.std(dim) / forecast.std(dim)
    return ratio


def _bias_slope(forecast, reference, dim=None, **metric_kwargs):
    """Calculate bias slope between reference and forecast standard deviations.

    .. math:: \\text{bias slope}= r_{fo} \\cdot \\text{std ratio}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    std_ratio = _std_ratio(forecast, reference, dim=dim, **metric_kwargs)
    acc = _pearson_r(forecast, reference, dim=dim, **metric_kwargs)
    b_s = std_ratio * acc
    return b_s


def _ppp(forecast, reference, dim=None, **metric_kwargs):
    """Prognostic Potential Predictability (PPP) metric.

    .. math:: PPP = 1 - \\frac{MSE}{ \\sigma^2_{ref} \\cdot fac}

    Args:
        forecast (xarray object): forecast
        reference (xarray object): reference
        dim (str): dimension(s) to perform metric over.
                   Automatically set by compute_.
        comparison (str): name comparison needed for normalization factor
                           (required to be added via **metric_kwargs)
        metric_kwargs: (optional) weights, skipna, see xskillscore.mse

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
    mse_skill = _mse(forecast, reference, dim=dim, **metric_kwargs)
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


def _nrmse(forecast, reference, dim=None, **metric_kwargs):
    """Normalized Root Mean Square Error (NRMSE) metric.

    .. math:: NRMSE = \\frac{RMSE}{\\sigma_{o} \\cdot \\sqrt{fac} }
                    = \\sqrt{ \\frac{MSE}{ \\sigma^2_{o} \\cdot fac} }

    Args:
        * forecast (xr.object)
        * reference (xr.object)
        * dim (str): dimension to apply metric to
        * comparison (str): name comparison needed for normalization factor
                           (required to be added via **metric_kwargs)

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
    rmse_skill = _rmse(forecast, reference, dim=dim, **metric_kwargs)
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


def _nmse(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate Normalized MSE (NMSE) = Normalized Ensemble Variance (NEV).

    .. math:: NMSE = NEV = \\frac{MSE}{\\sigma^2_{o} \\cdot fac}

    Args:
        * forecast (xr.object)
        * reference (xr.object)
        * dim (str): dimension to apply metric to
        * comparison (str): name comparison needed for normalization factor
                           (required to be added via **metric_kwargs)

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13,
          no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    mse_skill = _mse(forecast, reference, dim=dim, **metric_kwargs)
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


def _nmae(forecast, reference, dim=None, **metric_kwargs):
    """
    Normalized Ensemble Mean Absolute Error metric.

    .. math:: NMAE = \\frac{MAE}{\\sigma_{o} \\cdot fac}

    Args:
        * forecast (xr.object)
        * reference (xr.object)
        * dim (str): dimension to apply metric to
        * comparison (str): name comparison needed for normalization factor
                           (required to be added via **metric_kwargs)

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13, no.
          7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

    """
    mae_skill = _mae(forecast, reference, dim=dim, **metric_kwargs)
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


def _uacc(forecast, reference, dim=None, **metric_kwargs):
    """
    Calculate Bushuk's unbiased ACC (uACC).

    .. math:: uACC = \\sqrt{PPP} = \\sqrt{MSSS}

    Args:
        * forecast (xr.object)
        * reference (xr.object)
        * dim (str): dimension to apply metric to
        * metric_kwargs: (optional) weights, skipna, see xskillscore.mse


    Range:
        * 1: perfect
        * 0 - 1: better than climatology

    References:
        * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel
          Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional
          Arctic Sea–Ice Prediction: Potential versus Operational Seasonal
          Forecast Skill. Climate Dynamics, June 9, 2018.
          https://doi.org/10/gd7hfq.
    """
    ppp_res = _ppp(forecast, reference, dim=dim, **metric_kwargs)
    # ensure no sqrt of neg values
    uacc_res = (ppp_res.where(ppp_res > 0)) ** 0.5
    return uacc_res
