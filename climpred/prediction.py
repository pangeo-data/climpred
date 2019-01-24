"""
Objects dealing with decadal prediction metrics.

Concept of calculating predictability skill
-------------------------------------------
- metric: how is skill calculated, e.g. pearson_r, rmse
- comparison: how forecasts and observation/truth are compared, e.g., m2m, e2r

High-level functions
-------------------
- compute_perfect_model: computes the perfect-model predictability skill
according to metric and comparison
- compute_reference: computes predictability/skill relative to some reference
- compute_persistence: computes a persistence forecast from some simulation
- bootstrap from uninitialized ensemble:
    - PM_sig(ds, control, metric=rmse, comparison=_m2m, bootstrap=500, sig=99)
    - threshold to determine predictability horizon

# TODO: make metrics non-dependent of prediction framework used
Metrics (submit to functions as strings)
-------
- mae: Mean Absolute Error
- mse: Mean Square Error (perfect-model only)
- nev: Normalized Ensemble Variance (perfect-model only)
- msss: Mean Square Skill Score (perfect-model only)
- ppp: Prognostic Potential Predictability (perfect-model only)
- rmse:  Root-Mean Square Error
- rmse_v: Root-Mean Square Error (perfect-model only)
- nrmse: Normalized Root-Mean Square Error (perfect-model only)
- pearson_r: Anomaly correlation coefficient
- uACC: unbiased ACC (perfect-model only)

Comparisons (submit to functions as strings)
-----------
Perfect Model:
- m2c: many forecasts vs. control truth
- m2e: many forecasts vs. ensemble mean truth
- m2m: many forecasts vs. many truths in turn
- e2c: ensemble mean forecast vs. control truth

Reference:
- e2r: ensemble mean vs. reference
- m2r: individual ensemble members vs. reference

Additional Functions
--------
- Diagnostic Potential Predictability (DPP)
(Boer 2004, Resplandy 2015/Seferian 2018)
- predictability horizon:
    - bootstrapping limit


Data Structure
--------------
This module works on xr.Datasets with the following dimensions and coordinates:
- 1D (Predictability of timelines of preprocessed regions):
    - ensemble : initialization months/years
    - area : pre-processed region strings
    - time : lead months/years from the initialization
    - period : time averaging: yearmean, seasonal mean

Example ds via load_dataset('PM_MPI-ESM-LR_ds'):
<xarray.Dataset>
Dimensions:                  (area: 14, ensemble: 12, member: 10, period: 5,
                              time: 20)
Coordinates:
  * ensemble                 (ensemble) int64 3014 3023 3045 3061 3124 3139 ...
  * area                     (area) object 'global' 'North_Atlantic_SPG' ...
  * time                     (time) int64 1 2 3 4 5 6 ...
  * period                   (period) object 'DJF' 'JJA' 'MAM' 'SON' 'ym'
Dimensions without coordinates: member
Data variables:
    tos                   (period, time, area, ensemble, member) float32 ...
...

- 3D (Predictability maps):
    - ensemble
    - lon(y, x), lat(y, x)
    - time (as in lead time)
    - period (time averaging: yearmean, seasonal mean)

Example via load_dataset('PM_MPI-ESM-LR_ds3d'):
<xarray.Dataset>
Dimensions:      (bnds: 2, ensemble: 11, member: 9, x: 256, y: 220, time: 21)
Coordinates:
    lon          (y, x) float64 -47.25 -47.69 -48.12 ... 131.3 132.5 133.8
    lat          (y, x) float64 76.36 76.3 76.24 76.17 ... -77.25 -77.39 -77.54
  * ensemble     (ensemble) int64 3061 3124 3178 3023 ... 3228 3175 3144 3139
  * time         (time) int64 1 2 3 4 5 ... 19 20
Dimensions without coordinates: bnds, member, x, y
Data variables:
    tos          (time, ensemble, member, y, x) float32
    dask.array<shape=(21, 11, 9, 220, 256), chunksize=(1, 1, 1, 220, 256)>
...

This 3D example data is from curivlinear grid MPIOM (MPI Ocean Model)
NetCDF output.
The time dimensions is called 'time' and is in integer, not datetime[ns]
"""
import numpy as np
import xarray as xr

from xskillscore import mse as _mse
from xskillscore import pearson_r as _pearson_r
from xskillscore import rmse as _rmse
from xskillscore import mae as _mae
from xskillscore import pearson_r_p_value

from .stats import _check_xarray, _get_dims, z_significance
import warnings
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
    """
    Modifies control according to knowledge approach.

    Reference
    ---------
    - Hawkins, Ed, Steffen Tietsche, Jonathan J. Day, Nathanael Melia, Keith
        Haines, and Sarah Keeley. “Aspects of Designing and Evaluating
        Seasonal-to-Interannual Arctic Sea-Ice Prediction Systems.” Quarterly
        Journal of the Royal Meteorological Society 142, no. 695
        (January 1, 2016): 672–83. https://doi.org/10/gfb3pn.

    args:
    reference_period : str
        'MK' : maximum knowledge
        'OP' : operational
        'OP_full_length' : operational observational record length but keep
                           full length of record
    obs_years : int
        length of observational record

    return:
        control
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
    """
    Get variance to normalize skill score.

    Args:
    control
    reference_period : str see _control_for_reference_period
    time_length : int
        smooth control by time_length before taking variance
    """
    if reference_period is not None and isinstance(time_length, int):
        control = _control_for_reference_period(control,
                                                reference_period=reference_period,
                                                obs_years=time_length)
        return control.var('time')
    else:
        return control.var('time')


def _get_norm_factor(comparison):
    """
    Get normalization factor for ppp, nvar, nrmse.
    Used in compute_perfect_model.

    m2e gets smaller rmse's than m2m by design, see Seferian 2018 et al.
    m2m, m2c-ensemble variance should be divided by 2 to get var(control)
    """
    comparison_name = comparison.__name__
    if comparison_name in ['_m2e', '_e2c']:
        return 1
    elif comparison_name in ['_m2c', '_m2m']:
        return 2
    else:
        raise ValueError('specify comparison to get normalization factor.')


def _drop_ensembles(ds, rmd_ensemble=[0]):
    """Drop ensembles from ds."""
    if all(ens in ds.ensemble.values for ens in rmd_ensemble):
        ensemble_list = list(ds.ensemble.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select available ensemble starting years',
                         rmd_ensemble)
    return ds.sel(ensemble=ensemble_list)


def _drop_members(ds, rmd_member=[0]):
    """Drop members by name selection .sel(member=) from ds."""
    if all(m in ds.member.values for m in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise ValueError('select available members', rmd_member)
    return ds.sel(member=member_list)


def _select_members_ensembles(ds, m=None, e=None):
    """Subselect ensembles and members from ds."""
    if m is None:
        m = ds.member.values
    if e is None:
        e = ds.ensemble.values
    return ds.sel(member=m, ensemble=e)


def _stack_to_supervector(ds, new_dim='svd',
                          stacked_dims=('ensemble', 'member')):
    """
    Stack all stacked_dims (likely ensemble and member) dimensions into one
    supervector dimension to perform metric over.
    """
    return ds.stack({new_dim: stacked_dims})


# --------------------------------------------#
# COMPARISONS
# --------------------------------------------#
def _get_comparison_function(comparison):
    """
    Similar to _get_metric_function. This converts a string comparison entry
    from the user into an actual function for the package to interpret.

    PERFECT MODEL:
    m2m : Compare all members to all other members.
    m2c : Compare all members to the control.
    m2e : Compare all members to the ensemble mean.
    e2c : Compare the ensemble mean to the control.

    REFERENCE:
    e2r : Compare the ensemble mean to the reference.
    m2r : Compare each ensemble member to the reference.
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
    Create two supervectors to compare all members to all other members in turn
    """
    truth_list = []
    fct_list = []
    for m in ds.member.values:
        # drop the member being truth
        ds_reduced = _drop_members(ds, rmd_member=[m])
        truth = ds.sel(member=m)
        for m2 in ds_reduced.member:
            for e in ds.ensemble:
                truth_list.append(truth.sel(ensemble=e))
                fct_list.append(ds_reduced.sel(member=m2, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)
    return fct, truth


def _m2e(ds, supervector_dim='svd'):
    """
    Create two supervectors to compare all members to ensemble mean.
    """
    truth = ds.mean('member')
    fct, truth = xr.broadcast(ds, truth)
    fct = _stack_to_supervector(fct, new_dim=supervector_dim)
    truth = _stack_to_supervector(truth, new_dim=supervector_dim)
    return fct, truth


def _m2c(ds, supervector_dim='svd', control_member=[0]):
    """
    Create two supervectors to compare all members to control.

    control_member: list of one integer
        index to be removed, default 0
    """
    truth = ds.isel(member=control_member).squeeze()
    # drop the member being truth
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    fct, truth = xr.broadcast(ds_dropped, truth)
    fct = _stack_to_supervector(fct, new_dim=supervector_dim)
    truth = _stack_to_supervector(truth, new_dim=supervector_dim)
    return fct, truth

    return fct, truth


def _e2c(ds, supervector_dim='svd', control_member=[0]):
    """
    Create two supervectors to compare ensemble mean to control.
    """
    truth = ds.isel(member=control_member).squeeze()
    truth = truth.rename({'ensemble': supervector_dim})
    # drop the member being truth
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    fct = ds.mean('member')
    fct = fct.rename({'ensemble': supervector_dim})
    return fct, truth


def _e2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares the
    ensemble mean prediction to the reference (hindcast, simulation,
    observations).
    """
    if 'member' in _get_dims(ds):
        print("Taking ensemble mean...")
        fct = ds.mean('member')
    else:
        fct = ds
    return fct, reference


def _m2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares each
    member individually to the reference (hindcast, simulation,
    observations).
    """
    # check that this contains more than one member
    if ('member' not in _get_dims(ds)) or (ds.member.size == 1):
        raise ValueError("""Please supply a decadal prediction ensemble with
            more than one member. You might have input the ensemble mean here
            although asking for a member-to-reference comparison.""")
    else:
        fct = ds
    reference = reference.expand_dims('member')
    nMember = fct.member.size
    reference = reference.isel(member=[0] * nMember)
    reference['member'] = fct['member']
    return fct, reference


# --------------------------------------------#
# METRICS
# Metrics for computing predictability.
# --------------------------------------------#
def _get_metric_function(metric):
    """
    This allows the user to submit a string representing the desired function
    to anything that takes a metric. The old format forced the user to import
    an underscore function (which by definition we don't want).

    Currently compatable with functions:
    * compute_persistence()
    * compute_perfect_model()
    * compute_reference()

    Currently compatable with metrics:
    * pearson_r
    * rmse
    * mae
    * mse
    * rmse_v
    * nrmse
    * nev
    * ppp
    * msss
    * uACC

    Metrics
    --------
    pearson_r : 'pearson_r', 'pearsonr', 'pr'
    rmse: 'rmse'
    mae: 'mae'
    mse: 'mse'
    nrmse: 'mrmse'
    nmse: 'nmse','nev'
    ppp: 'ppp','msss'
    uACC: 'uacc'

    Returns
    --------
    metric : function object of the metric.
    """
    pearson = ['pr', 'pearsonr', 'pearson_r']
    if metric in pearson:
        metric = '_pearson_r'
    elif metric == 'rmse':
        metric = '_rmse'
    elif metric == 'mae':
        metric = '_mae'
    elif metric.lower() == 'mse':
        metric = '_mse'
    elif metric.lower() == 'nrmse':
        metric = '_nrmse'
    elif metric.lower() in ['nev', 'nmse']:
        metric = '_nmse'
    elif metric.lower() in ['ppp', 'msss']:
        metric = '_ppp'
    elif metric.lower() == 'uacc':
        metric = '_uacc'
    else:
        raise ValueError("""Please supply a metric from the following list:
            'pearson_r'
            'rmse'
            'mse'
            'nrmse'
            'nev'
            'nmse'
            'ppp'
            'msss'
            'uacc'
            """)
    return eval(metric)


# TODO: Do we need wrappers or should we rather create wrappers for skill score
#       as used in a specific paper: def Seferian2018(ds, control):
#       return PM_compute(ds, control, metric=_ppp, comparison=_m2e)
def _ppp(ds, control, comparison, running=None, reference_period=None):
    """
    Prognostic Potential Predictability (PPP) metric.

    Formula
    -------
    PPP = 1 - MSE / std_control

    References
    ----------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
        North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
        (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    - Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    """
    supervector_dim = 'svd'
    fct, truth = comparison(ds, supervector_dim)
    mse_skill = _mse(fct, truth, dim=supervector_dim)
    var = _get_variance(control, time_length=running,
                        reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    ppp_skill = 1 - mse_skill / var / fac
    return ppp_skill


def _nrmse(ds, control, comparison, running=None, reference_period=None):
    """
    Normalized Root Mean Square Error (NRMSE) metric.

    Formula
    -------
    NRMSE = 1 - RMSE_ens / std_control = 1 - (var_ens / var_control ) ** .5

    References
    ----------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
        Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
        Prediction: Potential versus Operational Seasonal Forecast Skill.”
        Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    - Hawkins, Ed, Steffen Tietsche, Jonathan J. Day, Nathanael Melia, Keith
        Haines, and Sarah Keeley. “Aspects of Designing and Evaluating
        Seasonal-to-Interannual Arctic Sea-Ice Prediction Systems.” Quarterly
        Journal of the Royal Meteorological Society 142, no. 695
        (January 1, 2016): 672–83. https://doi.org/10/gfb3pn.

    """
    supervector_dim = 'svd'
    fct, truth = comparison(ds, supervector_dim)
    rmse_skill = _rmse(fct, truth, dim=supervector_dim)
    var = _get_variance(control, time_length=running,
                        reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    nrmse_skill = rmse_skill / np.sqrt(var) / np.sqrt(fac)
    return nrmse_skill


def _nmse(ds, control, comparison, running=None, reference_period=None):
    """
    Normalized Ensemble Variance (NEV) metric.

    Reference
    ---------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    supervector_dim = 'svd'
    fct, truth = comparison(ds, supervector_dim)
    mse_skill = _mse(fct, truth, dim=supervector_dim)
    var = _get_variance(control, time_length=running,
                        reference_period=reference_period)
    fac = _get_norm_factor(comparison)
    nmse_skill = mse_skill / var / fac
    return nmse_skill


def _uacc(fct, truth, control, running=None, reference_period=None):
    """
    Unbiased ACC (uACC) metric.

    Formula
    -------
    - uACC = PPP ** .5 = MSSS ** .5

    Reference
    ---------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
      Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
      Prediction: Potential versus Operational Seasonal Forecast Skill.
      Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    """
    return np.sqrt(_ppp(fct, truth, control, running, reference_period))


# --------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
# --------------------------------------------#
def compute_perfect_model(ds, control, metric='pearson_r', comparison='m2m',
                          running=None, reference_period=None):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Parameters
    ----------
    ds, control : xr.DataArray or xr.Dataset with 'time' dimension (optional
                  spatial coordinates)
        input data
    metric : function
        metric from ['rmse', 'mae', 'pearson_r', 'mse', 'ppp', 'nev', 'nmse',
                     'uACC', 'MSSS']
    comparison : function
        comparison from ['m2m', 'm2e', 'm2c', 'e2c']
    running : int
        Size of the running window for variance smoothing
        (only optionally applicable to perfect-model metrics)
    reference_period : str


    Returns
    -------
    res : xr.DataArray or xr.Dataset
        skill score
    """
    supervector_dim = 'svd'
    comparison = _get_comparison_function(comparison)
    if comparison not in [_m2m, _m2c, _m2e, _e2c]:
        raise ValueError('specify comparison argument')

    metric = _get_metric_function(metric)
    if metric in [_pearson_r, _rmse, _mse, _mae]:
        fct, truth = comparison(ds, supervector_dim)
        res = metric(fct, truth, dim=supervector_dim)
        return res
    elif metric in [_nrmse, _nmse, _ppp, _uacc]:  # perfect-model only metrics
        res = metric(ds, control, comparison, running, reference_period)
        return res
    else:
        raise ValueError('specify metric argument')


def compute_reference(ds, reference, metric='pearson_r', comparison='e2r',
                      nlags=None, horizon=False, alpha=0.05, ci=90):
    """
    Compute a predictability skill score against some reference (hindcast,
    assimilation, reconstruction, observations)

    Note that if reference is the reconstruction, the output correlation
    coefficients are for potential predictability. If the reference is
    observations, the output correlation coefficients are actual skill.

    Parameters
    ----------
    ds : xarray object
        Expected to follow package conventions (and should be an ensemble mean)
        `ensemble` : dim of initialization dates
        `time` : dim of lead years from those initializations
        Additional dims can be lat, lon, depth, etc. but should not be
        individual members.
    reference : xarray object
        reference output/data over same time period
    metric : str (default 'pearson_r')
        Metric used in comparing the decadal prediction ensemble with the
        reference.
        * pearson_r
        * rmse
        * mae
        * mse
    comparison : str (default 'e2r')
        How to compare the decadal prediction ensemble to the reference.
        * e2r : ensemble mean to reference
        * m2r : each member to the reference
    nlags : int (default length of `time` dim)
        How many lags to compute skill/potential predictability out to
    horizon : (optional bool) If true, compute and return the predictability
              horizon. This checks that (1) the initialized ensemble skill
              correlations to the reference simulation are statistically
              significant, and (2) that the resulting r-values are
              significantly different from the persistence r-values.
    alpha: (optional double) p-value significance to check for correlations
           between initialized ensemble and reference simulation.
    ci: (optional int) confidence level in comparing initialized skill to
        persistence skill.

    Returns
    -------
    skill : xarray object
        Predictability with main dimension `lag`
    persistence : xarray object (if horizon is True)
    horizon : xarray object (if horizon is True)
    """
    _check_xarray(ds)
    _check_xarray(reference)
    comparison = _get_comparison_function(comparison)
    if comparison not in [_e2r, _m2r]:
        raise ValueError("""Please input either 'e2r' or 'm2r' for your
            comparison.""")
    fct, reference = comparison(ds, reference)
    if nlags is None:
        nlags = fct.time.size
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please input 'pearson_r', 'rmse', 'mse', or
            'mae' for your metric.""")
    plag = []
    if horizon:
        p_value = []
    for i in range(0, nlags):
        a, b = _shift(fct.isel(time=i), reference, i, dim='ensemble')
        plag.append(metric(a, b, dim='ensemble'))
    skill = xr.concat(plag, 'time')
    skill['time'] = np.arange(1, 1 + nlags)
    if (horizon) & (metric == _pearson_r):
        # NaN values throw warnings for p-value comparison, so just
        # suppress that here.
        for i in range(0, nlags):
            a, b = _shift(fct.isel(time=i), reference, i, dim='ensemble')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value.append(pearson_r_p_value(a, b, dim='ensemble'))
        p_value = xr.concat(p_value, 'time')
        p_value['time'] = np.arange(1, 1 + nlags)
    if horizon:
        persistence = compute_persistence(reference, nlags)
        if metric == _pearson_r:
            horizon = xr_predictability_horizon(skill, persistence,
                                                limit='upper',
                                                p_values=p_value,
                                                N=reference.ensemble.size,
                                                alpha=alpha, ci=ci)
        else:
            horizon = xr_predictability_horizon(skill, persistence,
                                                limit='lower')
        return skill, persistence, horizon
    else:
        return skill


def compute_persistence(reference, nlags, metric='pearson_r', dim='ensemble'):
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

    Parameters
    ---------
    reference : xarray object
        The reference time series over which to compute persistence.
    nlags : int
        Number of lags to compute persistence to.
    metric : str (default 'pearson_r')
        Metric to apply at each lag for the persistence computation. Choose
        from 'pearson_r' or 'rmse'.
    dim : str (default 'ensemble')
        Dimension over which to compute persistence forecast.

    Returns
    -------
    pers : xarray object
        Results of persistence forecast with the input metric applied.


    References
    ----------
    Chapter 8 (Short-Term Climate Prediction) in
    Van den Dool, Huug. Empirical methods in short-term climate prediction.
    Oxford University Press, 2007.
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
    for i in range(1, 1 + nlags):
        a, b = _shift(reference, reference, i, dim=dim)
        plag.append(metric(a, b, dim=dim))
    pers = xr.concat(plag, 'time')
    pers['time'] = np.arange(1, 1 + nlags)
    return pers


# --------------------------------------------#
# Diagnostic Potential Predictability (DPP)
# Functions related to DPP from Boer et al.
# --------------------------------------------#
def DPP(ds, m=10, chunk=True, var_all_e=False):
    """
    Calculate Diagnostic Potential Predictability (DPP) as potentially
    predictable variance fraction (ppvf) in Boer 2004.

    Note: Different way of calculating it than in Seferian 2018 or
    Resplandy 2015, but quite similar results.

    References
    ----------
    - Boer, G. J. “Long Time-Scale Potential Predictability in an Ensemble of
        Coupled Climate Models.” Climate Dynamics 23, no. 1 (August 1, 2004):
        29–44. https://doi.org/10/csjjbh.
    - Resplandy, L., R. Séférian, and L. Bopp. “Natural Variability of CO2 and
        O2 Fluxes: What Can We Learn from Centuries-Long Climate Models
        Simulations?” Journal of Geophysical Research: Oceans 120, no. 1
        (January 2015): 384–404. https://doi.org/10/f63c3h.
    - Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing the
        Decadal Predictability of Land and Ocean Carbon Uptake.” Geophysical
        Research Letters, March 15, 2018. https://doi.org/10/gdb424.

    Parameters
    ----------
    ds : DataArray with time dimension (optional spatial coordinates)
    m : int
        separation time scale in years between predictable low-freq
        component and high-freq noise
    chunk : boolean
        Whether chunking is applied. Default: True.
        If False, then uses Resplandy 2015 / Seferian 2018 method.

    Returns
    -------
    DPP : DataArray as ds without time dimension

    Example 1D
    ----------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ds_DPPm10 = et.prediction.DPP(ds,m=10,chunk=True)

    """
    # TODO: rename or find xr equiv
    def _chunking(ds, number_chunks=False, chunk_length=False):
        """
        Separate data into chunks and reshapes chunks in a c dimension.

        Specify either the number chunks or the length of chunks.
        Needed for DPP.

        Parameters
        ----------
        ds : DataArray with time dimension (optional spatial coordinates)
            Input data
        number_chunks : boolean
            Number of chunks in the return data
        chunk_length : boolean
            Length of chunks

        Returns
        -------
        c : DataArray
            Output data as ds, but with additional dimension c and
            all same time coordinates
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

    if not chunk:
        s2v = ds.rolling(time=m).mean().var('time')
        s2e = (ds - ds.rolling(time=m).mean()).var('time')
        s2 = s2v + s2e
    if chunk:
        # first chunk
        chunked_means = _chunking(
            ds, chunk_length=m).mean('time')
        # sub means in chunks
        chunked_deviations = _chunking(
            ds, chunk_length=m) - chunked_means
        s2v = chunked_means.var('c')
        if var_all_e:
            s2e = chunked_deviations.var(['time', 'c'])
        else:
            s2e = chunked_deviations.var('time').mean('c')
        s2 = s2v + s2e
    DPP = (s2v - s2 / (m)) / (s2)
    return DPP


# --------------------------------------------#
# BOOTSTRAPPING
# Functions for sampling an ensemble
# --------------------------------------------#
def _pseudo_ens(ds, control):
    """
    Create a pseudo-ensemble from control run.

    Needed for bootstrapping confidence intervals of a metric.
    Takes randomly 20yr segments from control and rearranges them into ensemble
    and member dimensions.

    Parameters
    ----------
    control : xr.DataArray with 'time' dimension
        Input ensemble data

    Returns
    -------
    ds_e : xr.DataArray with time, ensemble, member dimension
        pseudo-ensemble generated from control run

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ds_e = et.prediction.pseudo_ens(control,ds)
    """
    nens = ds.ensemble.size
    nmember = ds.member.size
    length = ds.time.size
    c_start = control['time'].min()
    c_end = control['time'].max()
    time = ds['time']

    def sel_years(control, year_s, m=None, length=length):
        new = control.sel(time=slice(year_s, year_s + length - 1))
        new['time'] = time
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nmember)
        return xr.concat([sel_years(control, start)
                          for start in startlist], 'member')
    return xr.concat([create_pseudo_members(control) for _ in range(nens)],
                     'ensemble')


def bootstrap_perfect_model(ds, control, metric='rmse', comparison='m2m',
                            reference_period='MK', sig=95, bootstrap=30):
    """
    Return sig-th percentile of function to be choosen from pseudo ensemble
    generated from control.

    Parameters
    ----------
    control : xr.DataArray/Dataset with time dimension
        input control data
    ds : xr.DataArray/Dataset with time, ensemble and member dimensions
        input ensemble data
    sig: int or list
        Significance level for bootstrapping from pseudo ensemble
    bootstrap: int
        number of iterations

    Returns
    -------
    sig_level : xr.DataArray/Dataset as inputs
        significance level without time, ensemble and member dimensions
        as many sig_level as listitems in sig

    """
    _check_xarray(ds)
    _check_xarray(control)
    x = []
    _control = _control_for_reference_period(
        control, reference_period=reference_period)
    for _ in range(1 + int(bootstrap / ds['time'].size)):
        ds_pseudo = _pseudo_ens(ds, _control)
        ds_pseudo_metric = compute_perfect_model(
            ds_pseudo, _control, metric=metric, comparison=comparison)
        x.append(ds_pseudo_metric)
    ds_pseudo_metric = xr.concat(x, dim='it')
    if isinstance(sig, list):
        qsig = [x / 100 for x in sig]
    else:
        qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(q=qsig, dim=['time', 'it'])
    return sig_level


# --------------------------------------------#
# PREDICTABILITY HORIZON
# --------------------------------------------#
def xr_predictability_horizon(skill, threshold, limit='upper',
                              p_values=None, N=None, alpha=0.05, ci=90):
    """
    Get predictability horizons of dataset from skill and
    threshold dataset.

    Inputs:
        skill: (xarray object) skill (e.g., ACC) at different lead times.
        threshold: (xarray object) skill for persistence or uninitialized
                   ensemble.
        limit: (optional str) If 'upper' check horizon for correlation
               coefficients by testing lead time to which the skill
               beats out the threshold. If 'lower', check horizon for which
               error (e.g., MAE) is lower than threshold.
        p_values: (optional xarray object) If using 'upper' limit, input
                  a DataArray/Dataset of the same dimensions as skill that
                  contains p-values for the skill correlatons.

    Returns:
        predictability horizon reduced by the lead time dimension.
    """
    if limit is 'upper':
        if (p_values is None) | (p_values.dims != skill.dims):
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
