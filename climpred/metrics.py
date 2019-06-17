import numpy as np

from xskillscore import (
    crps_ensemble,
    crps_gaussian,
    mae,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse,
)


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


def _pearson_r(forecast, reference, dim='svd', comparison=None):
    """
    Calculate the Anomaly Correlation Coefficient (ACC).

    .. math::
        ACC = \\frac{cov(f, o)}{\\sigma_{f}\\cdot\\sigma_{o}}

    .. note::
        Use metric ``pearson_r_p_value`` to get the corresponding pvalue.

    Range:
        * perfect: 1
        * min: -1

    See also:
        * xskillscore.pearson_r
        * xskillscore.pearson_r_p_value
    """
    return pearson_r(forecast, reference, dim=dim)


def _pearson_r_p_value(forecast, reference, dim='svd', comparison=None):
    """
    Calculate the probability associated with the ACC not being random.
    """
    return pearson_r_p_value(forecast, reference, dim=dim)


def _mse(forecast, reference, dim='svd', comparison=None):
    """
    Calculate the Mean Sqaure Error (MSE).

    .. math::
        MSE = \\overline{(f - o)^{2}}

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mse
    """
    return mse(forecast, reference, dim=dim)


def _rmse(forecast, reference, dim='svd', comparison=None):
    """
    Calculate the Root Mean Sqaure Error (RMSE).

    .. math::
        RMSE = \\sqrt{\\overline{(f - o)^{2}}}

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.rmse
    """
    return rmse(forecast, reference, dim=dim)


def _mae(forecast, reference, dim='svd', comparison=None):
    """
    Calculate the Mean Absolute Error (MAE).

    .. math::
        MSE = \\overline{(f - o)^{2}}

    Range:
        * perfect: 0
        * min: 0
        * max: ∞

    See also:
        * xskillscore.mae
    """
    return mae(forecast, reference, dim=dim)


def _crps(forecast, reference, dim='svd', comparison=None):
    """
    Continuous Ranked Probability Score (CRPS) is the probabilistic MSE.

    Range:
        * perfect: 0
        * max: 0
        * else: negative

    References:
        * Matheson, James E., and Robert L. Winkler. “Scoring Rules for
          Continuous Probability Distributions.” Management Science 22, no. 10
          (June 1, 1976): 1087–96. https://doi.org/10/cwwt4g.

    See also:
        * properscoring.crps_ensemble
    """
    return crps_ensemble(forecast, reference).mean(dim)


def _crps_gaussian(forecast, mu, sig, dim='svd', comparison=None):
    return crps_gaussian(forecast, mu, sig).mean(dim)


def _crpss(forecast, reference, dim='svd', comparison=None):
    """
    Continuous Ranked Probability Skill Score is strictly proper.

    .. math::
        CRPSS = \\frac{CRPS_{clim}-CRPS_{init}}{CRPS_{clim}}

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

    See also:
        * properscoring.crps_ensemble
    """
    mu = reference.mean(dim)
    sig = reference.std(dim)
    ref_skill = _crps_gaussian(forecast, mu, sig, dim=dim)
    forecast_skill = _crps(forecast, reference, dim=dim)
    skill_score = (ref_skill - forecast_skill) / ref_skill
    return skill_score


def _less(forecast, reference, dim='svd', comparison=None):
    """
    Logarithmic Ensemble Spread Score.

    .. math:: LESS = ln(\\frac{\\sigma^2_f}{\\sigma^2_o})

    References:
        * Kadow, Christopher, Sebastian Illing, Oliver Kunst, Henning W. Rust,
          Holger Pohlmann, Wolfgang A. Müller, and Ulrich Cubasch. “Evaluation
          of Forecasts by Accuracy and Spread in the MiKlip Decadal Climate
          Prediction System.” Meteorologische Zeitschrift, December 21, 2016,
          631–43. https://doi.org/10/f9jrhw.

    Range:
        * pos: under-disperive
        * neg: over-disperive
        * perfect: 0
    """
    if comparison.__name__ != '_m2r':
        raise KeyError(
            'LESS requires member dimension and therefore '
            "compute_hindcast(comparison='m2r')"
        )
    numerator = _mse(forecast, reference, dim='member').mean(dim)
    # not corrected for conditional bias yet
    denominator = _mse(forecast.mean('member'), reference.mean('member'), dim=dim)
    less = np.log(numerator / denominator)
    return less


def _bias(forecast, reference, dim='svd', comparison=None):
    """Calculate unconditional bias.

    .. math::
        bias = f - o

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


def _msss_murphy(forecast, reference, dim='svd', comparison=None):
    """Calculate Murphy's Mean Square Skill Score (MSSS).

    .. math::
        MSSS_{Murphy} = r_{fo}^2 - [\\text{conditional bias}]^2 -\
         [\\frac{\\text{(unconditional) bias}}{\\sigma_o}]^2

    References:
        * https://www-miklip.dkrz.de/about/murcss/
        * Murphy, Allan H. “Skill Scores Based on the Mean Square Error and
          Their Relationships to the Correlation Coefficient.” Monthly Weather
          Review 116, no. 12 (December 1, 1988): 2417–24.
          https://doi.org/10/fc7mxd.
    """
    acc = _pearson_r(forecast, reference, dim=dim)
    conditional_bias = _conditional_bias(forecast, reference, dim=dim)
    uncond_bias = _bias(forecast, reference, dim=dim) / reference.std(dim)
    skill = acc ** 2 - conditional_bias ** 2 - uncond_bias ** 2
    return skill


def _conditional_bias(forecast, reference, dim='svd', comparison=None):
    """Calculate the conditional bias between forecast and reference.

    .. math:: \\text{conditional bias} = r_{fo} - \\frac{\\sigma_f}{\\sigma_o}

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    acc = _pearson_r(forecast, reference, dim=dim)
    conditional_bias = acc - _std_ratio(forecast, reference, dim=dim) ** -1
    return conditional_bias


def _std_ratio(forecast, reference, dim='svd', comparison=None):
    """Calculate the ratio of standard deviations of reference over forecast.

    .. math:: \\text{std ratio} = \\frac{\\sigma_o}{\\sigma_f}

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    ratio = reference.std(dim) / forecast.std(dim)
    return ratio


def _bias_slope(forecast, reference, dim='svd', comparison=None):
    """Calculate bias slope between reference and forecast standard deviations.

    .. math:: \\text{bias slope}= r_{fo} \\cdot \\text{std ratio}

    References:
        * https://www-miklip.dkrz.de/about/murcss/
    """
    std_ratio = _std_ratio(forecast, reference, dim=dim)
    acc = _pearson_r(forecast, reference, dim=dim)
    b_s = std_ratio * acc
    return b_s


def _ppp(forecast, reference, dim='svd', comparison=None):
    """Prognostic Potential Predictability (PPP) metric.

    .. math:: PPP = 1 - \\frac{MSE}{ \\sigma_{ref} \\cdot fac}

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
    mse_skill = _mse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    ppp_skill = 1 - mse_skill / var / fac
    return ppp_skill


def _nrmse(forecast, reference, dim='svd', comparison=None):
    """Normalized Root Mean Square Error (NRMSE) metric.

    .. math:: NRMSE = \\frac{RMSE}{\\sigma_{o} \\cdot \\sqrt{fac} }
                    = \\sqrt{ \\frac{MSE}{ \\sigma^2_{o} \\cdot fac} }

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
    rmse_skill = _rmse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nrmse_skill = rmse_skill / np.sqrt(var) / np.sqrt(fac)
    return nrmse_skill


def _nmse(forecast, reference, dim='svd', comparison=None):
    """
    Calculate Normalized MSE (NMSE) = Normalized Ensemble Variance (NEV).

    .. math:: NMSE = NEV = \\frac{MSE}{\\sigma^2_{o} \\cdot fac}

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13,
          no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    mse_skill = _mse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


def _nmae(forecast, reference, dim='svd', comparison=None):
    """
    Normalized Ensemble Mean Absolute Error metric.

    .. math:: NMAE = \\frac{MAE}{\\sigma^2_{o} \\cdot fac}

    Range:
        * 0: perfect forecast: 0
        * 0 - 1: better than climatology forecast
        * > 1: worse than climatology forecast

    References:
        * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
          North Atlantic Multidecadal Variability.” Climate Dynamics 13, no.
          7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

    """
    mae_skill = _mae(forecast, reference, dim=dim)
    # TODO: check if this is the expected normalization
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nmse_skill = mae_skill / var / fac
    return nmse_skill


def _uacc(forecast, reference, dim='svd', comparison=None):
    """
    Calculate Bushuk's unbiased ACC (uACC).

    .. math:: uACC = \\sqrt{PPP} = \\sqrt{MSSS}

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
    return _ppp(forecast, reference, dim=dim, comparison=comparison) ** 0.5
