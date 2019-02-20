"""Objects dealing with decadal prediction metrics."""
import types
import warnings

import numpy as np
import xarray as xr

from xskillscore import mae as _mae
from xskillscore import mse as _mse
from xskillscore import pearson_r as _pearson_r
from xskillscore import pearson_r_p_value
from xskillscore import rmse as _rmse

from .stats import _check_xarray, _get_dims, z_significance


# -------------------------------------------- #
# HELPER FUNCTIONS
# Should only be used internally by esmtools
# -------------------------------------------- #
def _shift(a, b, lag, dim='time'):
    """
    Helper function to return two shifted time series for applying statistics
    to lags. This shifts them, and then forces them to have a common dimension
    so as not to break the metric functions.

    This function is usually applied in a loop. So, one loops over (1, nlags)
    applying the shift, then the metric, then concatenates all results into
    one xarray object.
    """
    if a[dim].size != b[dim].size:
        raise IOError("Please provide time series of equal lengths.")
    N = a[dim].size
    a = a.isel({dim: slice(0, N - lag)})
    b = b.isel({dim: slice(0 + lag, N)})
    b[dim] = a[dim]
    return a, b


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


def _drop_ensembles(ds, rmd_ensemble=[0]):
    """Drop ensembles by name selection .sel(member=) from ds.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with ensemble dimension.
        rmd_ensemble (list): list of ensemble names to be dropped. Default: [0]

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with less ensembles.

    Raises:
        ValueError: if list items are not all in ds.ensemble.

    """
    if all(ens in ds.initialization.values for ens in rmd_ensemble):
        ensemble_list = list(ds.initialization.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select available ensembles only', rmd_ensemble)
    return ds.sel(initialization=ensemble_list)


def _drop_members(ds, rmd_member=[0]):
    """Drop members by name selection .sel(member=) from ds.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension
        rmd_ensemble (list): list of members to be dropped. Default: [0]

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with less members.

    Raises:
        ValueError: if list items are not all in ds.member

    """
    if all(m in ds.member.values for m in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise ValueError('select available members only', rmd_member)
    return ds.sel(member=member_list)


def _select_members_ensembles(ds, m=None, i=None):
    """Subselect ensembles and members from ds.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        m (list): list of members to select. Default: None
        i (list): list of members to select. Default: None

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with less members or
                            ensembles.

    """
    if m is None:
        m = ds.member.values
    if i is None:
        i = ds.initialization.values
    return ds.sel(member=m, initialization=i)


def _stack_to_supervector(ds,
                          new_dim='svd',
                          stacked_dims=('initialization', 'member')):
    """Stack all stacked_dims (likely initialization and member) dimensions into one
    supervector dimension to perform metric over.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        new_dim (str): name of new supervector dimension. Default: 'svd'
        stacked_dims (set): dimensions to be stacked.

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with stacked new_dim
                            dimension.
    """
    return ds.stack({new_dim: stacked_dims})


# --------------------------------------------#
# COMPARISONS
# --------------------------------------------#
def _get_comparison_function(comparison):
    """Converts a string comparison entry from the user into an actual
       function for the package to interpret.

    PERFECT MODEL:
    m2m: Compare all members to all other members.
    m2c: Compare all members to the control.
    m2e: Compare all members to the ensemble mean.
    e2c: Compare the ensemble mean to the control.

    REFERENCE:
    e2r: Compare the ensemble mean to the reference.
    m2r: Compare each ensemble member to the reference.

    Args:
        comparison (str): name of comparison.

    Returns:
        comparison (function): comparison function.

    """
    if comparison == 'm2m':
        comparison = '_m2m'
    elif comparison == 'm2c':
        comparison = '_m2c'
    elif comparison == 'm2e':
        comparison = '_m2e'
    elif comparison == 'e2c':
        comparison = '_e2c'
    elif comparison == 'e2r':
        comparison = '_e2r'
    elif comparison == 'm2r':
        comparison = '_m2r'
    else:
        raise ValueError("""Please supply a comparison from the following list:
            'm2m'
            'm2c'
            'm2e'
            'e2c'
            'e2r'
            'm2r'
            """)
    return eval(comparison)


def _m2m(ds, supervector_dim='svd'):
    """
    Create two supervectors to compare all members to all other members in turn.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension. Default: 'svd'

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        # drop the member being reference
        ds_reduced = _drop_members(ds, rmd_member=[m])
        reference = ds.sel(member=m)
        for m2 in ds_reduced.member:
            for i in ds.initialization:
                reference_list.append(reference.sel(initialization=i))
                forecast_list.append(
                    ds_reduced.sel(member=m2, initialization=i))
    reference = xr.concat(reference_list, supervector_dim)
    forecast = xr.concat(forecast_list, supervector_dim)
    return forecast, reference


def _m2e(ds, supervector_dim='svd'):
    """
    Create two supervectors to compare all members to ensemble mean.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension. Default: 'svd'

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    reference = ds.mean('member')
    forecast, reference = xr.broadcast(ds, reference)
    forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
    reference = _stack_to_supervector(reference, new_dim=supervector_dim)
    return forecast, reference


def _m2c(ds, supervector_dim='svd', control_member=[0]):
    """
    Create two supervectors to compare all members to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension. Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    forecast, reference = xr.broadcast(ds_dropped, reference)
    forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
    reference = _stack_to_supervector(reference, new_dim=supervector_dim)
    return forecast, reference

    return forecast, reference


def _e2c(ds, supervector_dim='svd', control_member=[0]):
    """
    Create two supervectors to compare ensemble mean to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension. Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.
    """
    reference = ds.isel(member=control_member).squeeze()
    reference = reference.rename({'initialization': supervector_dim})
    # drop the member being reference
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    forecast = ds.mean('member')
    forecast = forecast.rename({'initialization': supervector_dim})
    return forecast, reference


def _e2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares the
    ensemble mean prediction to the reference (hindcast, simulation,
    observations).

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    if 'member' in _get_dims(ds):
        print("Taking ensemble mean...")
        forecast = ds.mean('member')
    else:
        forecast = ds
    return forecast, reference


def _m2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares each
    member individually to the reference (hindcast, simulation,
    observations).

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    # check that this contains more than one member
    if ('member' not in _get_dims(ds)) or (ds.member.size == 1):
        raise ValueError("""Please supply a decadal prediction ensemble with
            more than one member. You might have input the ensemble mean here
            although asking for a member-to-reference comparison.""")
    else:
        forecast = ds
    reference = reference.expand_dims('member')
    nMember = forecast.member.size
    reference = reference.isel(member=[0] * nMember)
    reference['member'] = forecast['member']
    return forecast, reference


# --------------------------------------------#
# METRICS
# Metrics for computing predictability.
# --------------------------------------------#
def _get_metric_function(metric):
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
    if type(metric) == types.FunctionType:
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


# --------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
# --------------------------------------------#
def compute_perfect_model(ds,
                          control,
                          metric='pearson_r',
                          comparison='m2m',
                          running=None,
                          reference_period=None):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Args:
        ds (xarray object): ensemble with dimensions time and member.
        control (xarray object): control with dimensions time.
        metric (str): metric name see _get_metric_function.
        comparison (str): comparison name see _get_comparison_function.
        running (optional int): size of the running window for variance
                                smoothing. Default: None (no smoothing)
        reference_period (optional str): choice of reference period of control.
                                Default: None (corresponds to MK approach)

    Returns:
        res (xarray object): skill score.

    Raises:
        ValueError: if comarison not implemented.
                    if metric not implemented.
    """
    supervector_dim = 'svd'
    comparison = _get_comparison_function(comparison)
    if comparison not in [_m2m, _m2c, _m2e, _e2c]:
        raise ValueError('specify comparison argument')

    metric = _get_metric_function(metric)
    if metric in [_pearson_r, _rmse, _mse, _mae]:
        forecast, reference = comparison(ds, supervector_dim)
        res = metric(forecast, reference, dim=supervector_dim)
        return res
    # perfect-model only metrics
    elif metric in [_nmae, _nrmse, _nmse, _ppp, _uacc]:
        res = metric(ds, control, comparison, running, reference_period)
        return res
    else:
        raise ValueError('specify metric argument')


def compute_reference(ds,
                      reference,
                      metric='pearson_r',
                      comparison='e2r',
                      nlags=None,
                      return_p=False):
    """
    Compute a predictability skill score against some reference (hindcast,
    assimilation, reconstruction, observations).

    Note that if reference is the reconstruction, the output correlation
    coefficients are for potential predictability. If the reference is
    observations, the output correlation coefficients are actual skill.

    Parameters
    ----------
    ds (xarray object):
        Expected to follow package conventions:
        `initialization` : dim of initialization dates
        `time` : dim of lead years from those initializations
        Additional dims can be lat, lon, depth.
    reference (xarray object):
        reference output/data over same time period.
    metric (str):
        Metric used in comparing the decadal prediction ensemble with the
        reference.
        * pearson_r (Default)
        * rmse
        * mae
        * mse
    comparison (str):
        How to compare the decadal prediction ensemble to the reference.
        * e2r : ensemble mean to reference (Default)
        * m2r : each member to the reference
    nlags (int): How many lags to compute skill/potential predictability out
                 to. Default: length of `time` dim
    return_p (bool): If True, return p values associated with pearson r.

    Returns:
        skill (xarray object): Predictability with main dimension `lag`.
        p_value (xarray object): If `return_p`, p values associated with
                                 pearson r correlations.
    """
    _check_xarray(ds)
    _check_xarray(reference)
    comparison = _get_comparison_function(comparison)
    if comparison not in [_e2r, _m2r]:
        raise ValueError("""Please input either 'e2r' or 'm2r' for your
            comparison.""")
    forecast, reference = comparison(ds, reference)
    if nlags is None:
        nlags = forecast.time.size
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please input 'pearson_r', 'rmse', 'mse', or
            'mae' for your metric.""")
    plag = []
    for i in range(0, nlags):
        a, b = _shift(
            forecast.isel(time=i), reference, i, dim='initialization')
        plag.append(metric(a, b, dim='initialization'))
    skill = xr.concat(plag, 'time')
    skill['time'] = np.arange(1, 1 + nlags)
    if (return_p) & (metric != _pearson_r):
        raise ValueError("""You can only return p values if the metric is
            pearson_r.""")
    elif (return_p) & (metric == _pearson_r):
        # NaN values throw warning for p-value comparison, so just
        # suppress that here.
        p_value = []
        for i in range(0, nlags):
            a, b = _shift(
                forecast.isel(time=i), reference, i, dim='initialization')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value.append(pearson_r_p_value(a, b, dim='initialization'))
        p_value = xr.concat(p_value, 'time')
        p_value['time'] = np.arange(1, 1 + nlags)
        return skill, p_value
    else:
        return skill


def compute_persistence(ds, reference, nlags, metric='pearson_r', dim='time'):
    """
    Computes the skill of  a persistence forecast from a reference
    (e.g., hindcast/assimilation) or control run.

    This simply applies some metric on the input out to some lag. The user
    should avoid computing persistence with prebuilt ACF functions in e.g.,
    python, MATLAB, R as they tend to use FFT methods for speed but incorporate
    error due to this.

    Currently supported metrics for persistence:
    * pearson_r
    * rmse
    * mse
    * mae

    Reference:
    * Chapter 8 (Short-Term Climate Prediction) in
        Van den Dool, Huug. Empirical methods in short-term climate prediction.
        Oxford University Press, 2007.


    Args:
        ds (xarray object): The initialization years to get persistence from.
        reference (xarray object): The reference time series.
        nlags (int): Number of lags to compute persistence to.
        metric (str): Metric name to apply at each lag for the persistence
                      computation. Default: 'pearson_r'
        dim (str): Dimension over which to compute persistence forecast.
                   Default: 'ensemble'

    Returns:
        pers (xarray object): Results of persistence forecast with the input
                              metric applied.
    """
    _check_xarray(reference)
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please select between the following metrics:
            'pearson_r',
            'rmse',
            'mse',
            'mae'""")
    plag = []  # holds results of persistence for each lag
    inits = ds.initialization.values
    reference = reference.isel({dim: slice(0, -nlags)})
    for lag in range(1, 1 + nlags):
        ref = reference.sel({dim: inits + lag})
        fct = reference.sel({dim: inits})
        ref[dim] = fct[dim]
        plag.append(metric(ref, fct, dim=dim))
    pers = xr.concat(plag, 'time')
    pers['time'] = np.arange(1, 1 + nlags)
    return pers


# --------------------------------------------#
# BOOTSTRAPPING
# Functions for sampling an ensemble
# --------------------------------------------#
def _pseudo_ens(ds, control):
    """
    Create a pseudo-ensemble from control run.

    Needed for block bootstrapping confidence intervals of a metric in perfect
    model framework. Takes randomly segments of length of ensemble dataset from
    control and rearranges them into ensemble and member dimensions.

    Args:
        ds (xarray object): ensemble simulation.
        control (xarray object): control simulation.

    Returns:
        ds_e (xarray object): pseudo-ensemble generated from control run.
    """
    nens = ds.initialization.size
    nmember = ds.member.size
    length = ds.time.size
    c_start = 0
    c_end = control['time'].size
    time = ds['time']

    def isel_years(control, year_s, m=None, length=length):
        new = control.isel(time=slice(year_s, year_s + length))
        new['time'] = time
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nmember)
        return xr.concat([isel_years(control, start) for start in startlist],
                         'member')

    return xr.concat([create_pseudo_members(control) for _ in range(nens)],
                     'initialization')


def bootstrap_perfect_model(ds,
                            control,
                            metric='pearson_r',
                            comparison='m2e',
                            sig=95,
                            pers_sig=50,
                            bootstrap=500,
                            compute_uninitized_skill=True,
                            compute_persistence_skill=True,
                            nlags=None,
                            running=None,
                            reference_period='MK'):
    """Bootstrap perfect-model ensemble simulations with replacement.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.Dataset): prediction ensemble.
        control (xr.Dataset): control simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'm2e'.
        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence forecast.
                        Defaults to 50.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        compute_uninitized_skill (bool): Defaults to True.
        compute_persistence_skill (bool): Defaults to True.
        nlags (type): number of lags persistence forecast skill.
                      Defaults to ds.time.size.

    Returns:
        init_skill (xr.Dataset): skill of initialized
        init_ci (xr.Dataset): confidence levels of init_skill
        uninit_skill (xr.Dataset): skill of uninitialized
        uninit_ci (xr.Dataset): confidence levels of uninit_skill
        p_uninit_over_init (xr.Dataset): p-value of the hypothesis
                                         that the difference of
                                         correlations between the
                                         initialized and uninitialized
                                         simulations is smaller or
                                         equal to zero based on
                                         bootstrapping with
                                         replacement.
                                         Defaults to None.
        pers_skill (xr.Dataset): skill of persistence
        pers_ci (xr.Dataset): confidence levels of pers_skill
        p_pers_over_init (xr.Dataset): p-value of the hypothesis
                                       that the difference of
                                       correlations between the
                                       initialized and persistence
                                       simulations is smaller or
                                       equal to zero based on
                                       bootstrapping with
                                       replacement.
                                       Defaults to None.

    """
    if nlags is None:
        nlags = ds.time.size
    p = (100 - sig) / 100  # 0.05
    ci_low = p / 2  # 0.025
    ci_high = 1 - p / 2  # 0.975
    p_pers = (100 - pers_sig) / 100  # 0.5
    ci_low_pers = p_pers / 2
    ci_high_pers = 1 - p_pers / 2

    inits = ds.initialization.values
    init = []
    uninit = []
    pers = []
    for _ in range(bootstrap):  # resample with replacement
        smp = np.random.choice(inits, len(inits))
        smp_ds = ds.sel(initialization=smp)
        # compute init skill
        init.append(
            compute_perfect_model(
                smp_ds,
                control,
                metric=metric,
                comparison=comparison,
                running=running,
                reference_period=reference_period))
        if compute_uninitized_skill:
            # generate uninitialized ensemble from control
            uninit_ds = _pseudo_ens(ds, control).isel(time=0)
            # compute uninit skill
            uninit.append(
                compute_perfect_model(
                    uninit_ds,
                    control,
                    metric=metric,
                    comparison=comparison,
                    running=running,
                    reference_period=reference_period))
        # compute persistence skill
        if compute_persistence_skill:
            pers.append(
                compute_persistence(
                    smp_ds, control, nlags=nlags, dim='time', metric=metric))
    init = xr.concat(init, dim='bootstrap')
    if compute_uninitized_skill:
        uninit = xr.concat(uninit, dim='bootstrap')
    if compute_persistence_skill:
        pers = xr.concat(pers, dim='bootstrap')

    def _distribution_to_signal_ci(ds, ci_low, ci_high, dim='bootstrap'):
        ds_ci = ds.quantile(q=[ci_low, ci_high], dim=dim)
        ds_skill = ds.mean(dim)
        return ds_skill, ds_ci

    init_skill, init_ci = _distribution_to_signal_ci(init, ci_low, ci_high)
    if compute_uninitized_skill:
        uninit_skill, uninit_ci = _distribution_to_signal_ci(
            uninit, ci_low, ci_high)
    if compute_persistence_skill:
        pers_skill, pers_ci = _distribution_to_signal_ci(
            pers, ci_low_pers, ci_high_pers)

    def _pvalue_from_distributions(simple_fct, init):
        """Get probability that simple_fct is larger than init."""
        return ((simple_fct - init) > 0).sum('bootstrap') / init.bootstrap.size

    if compute_uninitized_skill:
        p_uninit_over_init = _pvalue_from_distributions(uninit, init)
    else:
        p_uninit_over_init, uninit_skill, uninit_ci = None, None, None

    if compute_persistence_skill:
        p_pers_over_init = _pvalue_from_distributions(uninit, init)
    else:
        p_pers_over_init, pers_skill, pers_ci = None, None, None

    return init_skill, init_ci, uninit_skill, uninit_ci, p_uninit_over_init, pers_skill, pers_ci, p_pers_over_init


# --------------------------------------------#
# PREDICTABILITY HORIZON
# --------------------------------------------#
def xr_predictability_horizon(skill,
                              threshold,
                              limit='upper',
                              perfect_model=False,
                              p_values=None,
                              N=None,
                              alpha=0.05,
                              ci=90):
    """
    Get predictability horizons for skill better than threshold.

    Args:
        skill (xarray object): skill.
        threshold (xarray object): threshold.
        limit (str): bounds for comparison. Default: 'upper'.
        perfect_model: (optional bool) If True, do not consider p values, N,
                       etc.
        p_values: (optional xarray object) If using 'upper' limit, input
                  a DataArray/Dataset of the same dimensions as skill that
                  contains p-values for the skill correlatons.

    Returns:
        ph (xarray object)
    """
    if (limit is 'upper') and (not perfect_model):
        if (p_values is None):
            raise ValueError("""Please submit p values associated with the
                correlation coefficients.""")
        if (p_values.dims != skill.dims):
            raise ValueError("""Please submit an xarray object of the same
                dimensions as `skill` that contains p-values for the skill
                correlatons.""")
        if N is None:
            raise ValueError("""Please submit N, the length of the original
                time series being correlated.""")
        sig = z_significance(skill, threshold, N, ci)
        ph = ((p_values < alpha) & (sig)).argmin('time')
        # where ph not reached, set max time
        ph_not_reached = ((p_values < alpha) & (sig)).all('time')
    elif (limit is 'upper') and (perfect_model):
        ph = (skill > threshold).argmin('time')
        ph_not_reached = (skill > threshold).all('time')
    elif limit is 'lower':
        ph = (skill < threshold).argmin('time')
        # where ph not reached, set max time
        ph_not_reached = (skill < threshold).all('time')
    else:
        raise ValueError("""Please either submit 'upper' or 'lower' for the
            limit keyword.""")
    ph = ph.where(~ph_not_reached, other=skill['time'].max())
    # mask out any initial NaNs (land, masked out regions, etc.)
    mask = np.asarray(skill.isel({'time': 0}))
    mask = np.isnan(mask)
    ph = ph.where(~mask, np.nan)
    return ph
