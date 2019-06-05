import numpy as np
from xskillscore import (
    crps_ensemble,
    crps_gaussian,
    mae,
    mse,
    pearson_r,
    pearson_r_p_value,
    rmse
)


def _get_norm_factor(comparison):
    """Get normalization factor for PPP, nvar, nRMSE.

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


# wrap xskillscore metrics to work with comparison argument
def _pearson_r(forecast, reference, dim='svd', comparison=None):
    return pearson_r(forecast, reference, dim=dim)


def _pearson_r_p_value(forecast, reference, dim='svd', comparison=None):
    return pearson_r_p_value(forecast, reference, dim=dim)


def _mse(forecast, reference, dim='svd', comparison=None):
    return mse(forecast, reference, dim=dim)


def _rmse(forecast, reference, dim='svd', comparison=None):
    return rmse(forecast, reference, dim=dim)


def _mae(forecast, reference, dim='svd', comparison=None):
    return mae(forecast, reference, dim=dim)


def _crps(forecast, reference, dim='svd', comparison=None):
    return crps_ensemble(forecast, reference).mean(dim)


def _crps_gaussian(forecast, mu, sig, dim='svd', comparison=None):
    return crps_gaussian(forecast, mu, sig).mean(dim)


def _crpss(forecast, reference, dim='svd', comparison=None):
    """
    Continuous Ranked Probability Skill Score.
    Reference
    ---------
    * Matheson, James E., and Robert L. Winkler. “Scoring Rules for Continuous
      Probability Distributions.” Management Science 22, no. 10 (June 1, 1976):
      1087–96. https://doi.org/10/cwwt4g.
    Range
    -----
    perfect: 0
    max: 0
    else: negative
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

    Formula
    -------
    .. math:: LESS = ln(\frac{\sigma^2_F}{\sigma^2_R})

    Reference
    ---------
    * Kadow, Christopher, Sebastian Illing, Oliver Kunst, Henning W. Rust,
      Holger Pohlmann, Wolfgang A. Müller, and Ulrich Cubasch. “Evaluation of
      Forecasts by Accuracy and Spread in the MiKlip Decadal Climate Prediction
      System.” Meteorologische Zeitschrift, December 21, 2016, 631–43.
      https://doi.org/10/f9jrhw.

    Range
    -----
    pos: under-disperive
    neg: over-disperive
    perfect: 0
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
    """(unconditional) bias: https://www.cawcr.gov.au/projects/verification/"""
    bias = (forecast - reference).mean(dim)
    return bias


def _msss_murphy(forecast, reference, dim='svd', comparison=None):
    """msss_murphy: https://www-miklip.dkrz.de/about/murcss/"""
    acc = _pearson_r(forecast, reference, dim=dim)
    conditional_bias = acc - _std_ratio(forecast, reference, dim=dim)
    bias = _bias(forecast, reference, dim=dim) / reference.std(dim)
    skill = acc ** 2 - conditional_bias ** 2 - bias ** 2
    return skill


def _conditional_bias(forecast, reference, dim='svd', comparison=None):
    """conditional_bias: https://www-miklip.dkrz.de/about/murcss/"""
    acc = _pearson_r(forecast, reference, dim=dim)
    conditional_bias = acc - _std_ratio(forecast, reference, dim=dim)
    return conditional_bias


def _std_ratio(forecast, reference, dim='svd', comparison=None):
    """std ratio: https://www-miklip.dkrz.de/about/murcss/"""
    ratio = forecast.std(dim) / reference.std(dim)
    return ratio


def _bias_slope(forecast, reference, dim='svd', comparison=None):
    """bias slope: https://www-miklip.dkrz.de/about/murcss/"""
    std_ratio = _std_ratio(forecast, reference, dim=dim)
    acc = _pearson_r(forecast, reference, dim=dim)
    b_s = std_ratio * acc
    return b_s


def _ppp(forecast, reference, dim='svd', comparison=None):
    """Prognostic Potential Predictability (PPP) metric.

    .. math:: PPP = 1 - \frac{MSE}{ \sigma_{control} \cdot fac}

    Perfect forecast: 1
    Climatology forecast: 0

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

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension.
        control (xarray object): xr.Dataset/xr.DataArray of control simulation.
        comparison (function): comparison function.
        running (int): smoothing of control. Default: None (no smoothing).
        reference_period (str): see _control_for_reference_period.

    Returns:
        ppp_skill (xarray object): skill of PPP.

    """
    mse_skill = _mse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    ppp_skill = 1 - mse_skill / var / fac
    return ppp_skill


def _nrmse(forecast, reference, dim='svd', comparison=None):
    """Normalized Root Mean Square Error (NRMSE) metric.

    .. math:: NRMSE = \frac{RMSE}{\sigma_{control} \cdot \sqrt{fac}
                    = sqrt{ \frac{MSE}{ \sigma^2_{control} \cdot fac} }

    Perfect forecast: 0
    Climatology forecast: 1


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

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension.
        control (xarray object): xr.Dataset/xr.DataArray of control simulation.
        comparison (function): comparison function.
        running (int): smoothing of control. Default: None (no smoothing).
        reference_period (str): see _control_for_reference_period.

    Returns:
        nrmse_skill (xarray object): skill of NRMSE.

    """
    rmse_skill = _rmse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nrmse_skill = rmse_skill / np.sqrt(var) / np.sqrt(fac)
    return nrmse_skill


def _nmse(forecast, reference, dim='svd', comparison=None):
    """
    Normalized MSE (NMSE) = Normalized Ensemble Variance (NEV) metric.


    .. math:: NMSE = NEV = frac{MSE}{\sigma^2_{control} \cdot fac}

    Perfect forecast: 0
    Climatology forecast: 1

    Reference:
    * Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension.
        control (xarray object): xr.Dataset/xr.DataArray of control simulation.
        comparison (function): comparison function.
        running (int): smoothing of control. Default: None (no smoothing).
        reference_period (str): see _control_for_reference_period.

    Returns:
        nmse_skill (xarray object): skill of NMSE.
    """
    mse_skill = _mse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


def _nmae(forecast, reference, dim='svd', comparison=None):
    """
    Normalized Ensemble Mean Absolute Error metric.

    Formula
    -------
    NMAE-SS = mse / var

    Perfect forecast: 0
    Climatology forecast: 1

    Reference
    ---------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

    """
    mse_skill = _mse(forecast, reference, dim=dim)
    var = reference.std(dim)
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


def _uacc(forecast, reference, dim='svd', comparison=None):
    """
    Unbiased ACC (uACC) metric.

    .. math::
        uACC = \sqrt{PPP} = \sqrt{MSSS}

    Reference
    * Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
      Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
      Prediction: Potential versus Operational Seasonal Forecast Skill.
      Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension.
        control (xarray object): xr.Dataset/xr.DataArray of control simulation.
        comparison (function): comparison function.
        running (int): smoothing of control. Default: None (no smoothing).
        reference_period (str): see _control_for_reference_period.

    Returns:
        uacc_skill (xarray object): skill of uACC
    """
    return _ppp(forecast, reference, dim=dim, comparison=comparison) ** 0.5
