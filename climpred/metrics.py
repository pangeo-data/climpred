import numpy as np
import types
from xskillscore import mae as _mae
from xskillscore import mse as _mse
from xskillscore import pearson_r as _pearson_r
from xskillscore import pearson_r_p_value
from xskillscore import rmse as _rmse


def _control_for_reference_period(control, reference_period='MK',
                                  obs_years=40):
    """Modifies control according to knowledge approach.

    Args:
        reference_period (str):
            'MK' : maximum knowledge
            'OP' : operational
            'OP_full_length' : operational observational record length but keep
                               full length of record.
        obs_years (int): length of observational record.

    Returns:
        Control with modifications applied.

    Reference:
        * Hawkins, Ed, Steffen Tietsche, Jonathan J. Day, Nathanael Melia,Keith
          Haines, and Sarah Keeley. “Aspects of Designing and Evaluating
          Seasonal-to-Interannual Arctic Sea-Ice Prediction Systems.” Quarterly
          Journal of the Royal Meteorological Society 142, no. 695
          (January 1, 2016): 672–83. https://doi.org/10/gfb3pn.
    """
    if reference_period is 'MK':
        control = control
    elif reference_period is 'OP_full_length':
        control = control - \
            control.rolling(time=obs_years, min_periods=1,
                            center=True).mean() + control.mean('time')
    elif reference_period is 'OP':
        raise ValueError('not yet implemented')
    else:
        raise ValueError("choose a reference period")
    return control


def _get_variance(control, reference_period=None, time_length=None):
    """Get variance to normalize skill score.

    Args:
        control (xarray object): Control simulation.
        reference_period (str): See _control_for_reference_period.
        time_length (int): Number of time steps to smooth control by before
                           taking variance.

    """
    if reference_period is not None and isinstance(time_length, int):
        control = _control_for_reference_period(
            control, reference_period=reference_period, obs_years=time_length)
        return control.var('time')
    else:
        return control.var('time')


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
        ValueError: if comparison is not matching.

    """
    comparison_name = comparison.__name__
    if comparison_name in ['_m2e', '_e2c']:
        fac = 1
        return fac
    elif comparison_name in ['_m2c', '_m2m']:
        fac = 2
        return fac
    else:
        raise ValueError('specify comparison to get normalization factor.')


def get_metric_function(metric):
    """
    This allows the user to submit a string representing the desired function
    to anything that takes a metric.

    Currently compatable with functions:
    * compute_persistence()
    * compute_perfect_model()
    * compute_reference()

    Currently compatable with metrics:
    * pearson_r
    * rmse
    * mae
    * mse
    * nrmse
    * nmae
    * nmse
    * msss
    * uacc

    Args:
        metric (str): name of metric.

    Returns:
        metric (function): function object of the metric.

    Raises:
        ValueError: if metric not implemented.

    """
    # catches issues with wrappers, etc. that actually submit the
    # proper underscore function
    if isinstance(metric, types.FunctionType):
        return metric
    else:
        pearson = ['pr', 'pearsonr', 'pearson_r']
        if metric.lower() in pearson:
            metric = '_pearson_r'
        elif metric.lower() == 'rmse':
            metric = '_rmse'
        elif metric.lower() == 'mae':
            metric = '_mae'
        elif metric.lower() == 'mse':
            metric = '_mse'
        elif metric.lower() == 'nrmse':
            metric = '_nrmse'
        elif metric.lower() in ['nev', 'nmse']:
            metric = '_nmse'
        elif metric.lower() in ['ppp', 'msss']:
            metric = '_ppp'
        elif metric.lower() == 'nmae':
            metric = '_nmae'
        elif metric.lower() == 'uacc':
            metric = '_uacc'
        else:
            raise ValueError("""Please supply a metric from the following list:
                'pearson_r'
                'rmse'
                'mae'
                'mse'
                'nrmse'
                'nev'
                'nmse'
                'ppp'
                'msss'
                'nmae'
                'uacc'
                """)
        return eval(metric)


# TODO: Do we need wrappers or should we rather create wrappers for skill score
#       as used in a specific paper: def Seferian2018(ds, control):
#       return PM_compute(ds, control, metric=_ppp, comparison=_m2e)
def _ppp(ds, control, comparison, running=None, reference_period=None):
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
    supervector_dim = 'svd'
    forecast, reference = comparison(ds, supervector_dim)
    mse_skill = _mse(forecast, reference, dim=supervector_dim)
    var = _get_variance(
        control, time_length=running, reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    ppp_skill = 1 - mse_skill / var / fac
    return ppp_skill


def _nrmse(ds, control, comparison, running=None, reference_period=None):
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
    supervector_dim = 'svd'
    forecast, reference = comparison(ds, supervector_dim)
    rmse_skill = _rmse(forecast, reference, dim=supervector_dim)
    var = _get_variance(
        control, time_length=running, reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    nrmse_skill = 1 - rmse_skill / np.sqrt(var) / np.sqrt(fac)
    return nrmse_skill


def _nmse(ds, control, comparison, running=None, reference_period=None):
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
    supervector_dim = 'svd'
    forecast, reference = comparison(ds, supervector_dim)
    mse_skill = _mse(forecast, reference, dim=supervector_dim)
    var = _get_variance(
        control, time_length=running, reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    nmse_skill = 1 - mse_skill / var / fac
    return nmse_skill


def _nmae(ds, control, comparison, running=None, reference_period=None):
    """
    Normalized Ensemble Mean Absolute Error metric.

    Formula
    -------
    NMAE-SS = 1 - mse / var

    Reference
    ---------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.

      NOTE: NMSE = - 1 - NEV
    """
    supervector_dim = 'svd'
    fct, truth = comparison(ds, supervector_dim)
    mse_skill = _mse(fct, truth, dim=supervector_dim)
    var = _get_variance(
        control, time_length=running, reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    nmse_skill = 1 - mse_skill / var / fac
    return nmse_skill


def _uacc(forecast, reference, control, running=None, reference_period=None):
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
    return np.sqrt(
        _ppp(forecast, reference, control, running, reference_period))
