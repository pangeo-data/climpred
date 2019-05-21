import numpy as np
import xarray as xr

from .comparisons import (ALL_HINDCAST_COMPARISONS_DICT,
                          ALL_PM_COMPARISONS_DICT, _drop_members, _e2c,
                          get_comparison_function)
from .metrics import (ALL_HINDCAST_METRICS_DICT, ALL_PM_METRICS_DICT,
                      get_metric_function)
from .stats import z_significance
from .utils import check_xarray


# -------------------------------------------- #
# HELPER FUNCTIONS
# Should only be used internally by climpred
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


def _intersection(lst1, lst2):
    """
    Custom intersection, since `set.intersection()` changes type of list.
    """
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def _validate_PM_comparison(comparison):
    """Validate if comparison is PM comparison."""
    if comparison not in ALL_PM_COMPARISONS_DICT.values():
        raise ValueError(f'specify comparison from',
                         f'{ALL_PM_COMPARISONS_DICT.keys()}')


def _validate_hindcast_comparison(comparison):
    """Validate if comparison is hindcast comparison."""
    if comparison not in ALL_HINDCAST_COMPARISONS_DICT.values():
        raise ValueError(f'specify comparison from',
                         f'{ALL_HINDCAST_COMPARISONS_DICT.keys()}')


def _validate_PM_metric(metric):
    """Validate if metric is PM metric."""
    if metric not in ALL_PM_METRICS_DICT.values():
        raise ValueError(f'specify metric argument from',
                         f'{ALL_PM_METRICS_DICT.keys()}')


def _validate_hindcast_metric(metric):
    """Validate if metric is hindcast metric."""
    if metric not in ALL_HINDCAST_METRICS_DICT.values():
        raise ValueError(f'specify metric argument from',
                         f'{ALL_HINDCAST_METRICS_DICT.keys()}')


# --------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
# --------------------------------------------#
def compute_perfect_model(ds, control, metric='rmse', comparison='m2e'):
    """
    Compute a predictability skill score for a perfect-model framework
    simulation dataset.

    Args:
        ds (xarray object): ensemble with dimensions time and member.
        control (xarray object): control with dimensions time.
        metric (str): metric name see get_metric_function.
        comparison (str): comparison name see get_comparison_function.

    Returns:
        res (xarray object): skill score.

    Raises:
        ValueError: if comarison not implemented.
                    if metric not implemented.
    """
    supervector_dim = 'svd'
    comparison = get_comparison_function(comparison)
    _validate_PM_comparison(comparison)
    metric = get_metric_function(metric)
    _validate_PM_metric(metric)

    forecast, reference = comparison(ds, supervector_dim)

    res = metric(forecast,
                 reference,
                 dim=supervector_dim,
                 comparison=comparison)
    return res


check_xarray([0, 1])
def compute_hindcast(hind, reference, metric='pearson_r', comparison='e2r'):
    """
    Compute a predictability skill score against some reference (hindcast,
    assimilation, reconstruction, observations).

    Note that if reference is the reconstruction, the output correlation
    coefficients are for potential predictability. If the reference is
    observations, the output correlation coefficients are actual skill.

    Parameters
    ----------
    hind (xarray object):
        Expected to follow package conventions:
        `time` : dim of initialization dates
        `lead` : dim of lead time from those initializations
        Additional dims can be lat, lon, depth.
    reference (xarray object):
        reference output/data over same time period.
    metric (str):
        Metric used in comparing the decadal prediction ensemble with the
        reference.
    comparison (str):
        How to compare the decadal prediction ensemble to the reference.
        * e2r : ensemble mean to reference (Default)
        * m2r : each member to the reference
    nlags (int): How many lags to compute skill/potential predictability out
                 to. Default: length of `lead` dim

    Returns:
        skill (xarray object): Predictability with main dimension `lag`.
    """
    nlags = hind.lead.size

    comparison = get_comparison_function(comparison)
    _validate_hindcast_comparison(comparison)
    metric = get_metric_function(metric)
    _validate_hindcast_metric(metric)

    forecast, reference = comparison(hind, reference)
    # think in real time dimension: real time = init + lag
    forecast = forecast.rename({'init': 'time'})
    # take only inits for which we have references at all leahind
    imin = max(forecast.time.min(), reference.time.min())
    imax = min(forecast.time.max(), reference.time.max() - nlags)
    forecast = forecast.where(forecast.time <= imax, drop=True)
    forecast = forecast.where(forecast.time >= imin, drop=True)
    reference = reference.where(reference.time >= imin, drop=True)

    plag = []
    # iterate over all leads (accounts for lead.min() in [0,1])
    for i in forecast.lead.values:
        # take lead year i timeseries and convert to real time
        a = forecast.sel(lead=i).drop('lead')
        a['time'] = [t + i for t in a.time.values]
        # take real time reference of real time forecast years
        b = reference.sel(time=a.time.values)
        plag.append(metric(a, b, dim='time', comparison=comparison))
    skill = xr.concat(plag, 'lead')
    skill['lead'] = forecast.lead.values
    return skill


check_xarray([0, 1])
def compute_persistence(hind, reference, metric='pearson_r'):
    """
    Computes the skill of  a persistence forecast from a reference
    (e.g., hindcast/assimilation) or a control run.

    Reference:
    * Chapter 8 (Short-Term Climate Prediction) in
        Van den Dool, Huug. Empirical methods in short-term climate prediction.
        Oxford University Press, 2007.


    Args:
        hind (xarray object): The initialized ensemble.
        reference (xarray object): The reference time series.
        metric (str): Metric name to apply at each lag for the persistence
                      computation. Default: 'pearson_r'

    Returns:
        pers (xarray object): Results of persistence forecast with the input
                              metric applied.
    """
    metric = get_metric_function(metric)
    _validate_hindcast_metric(metric)

    plag = []  # holhind results of persistence for each lag
    for lag in hind.lead.values:
        inits = hind['init'].values
        ctrl_inits = reference.isel(time=slice(0, -lag))['time'].values
        inits = _intersection(inits, ctrl_inits)
        ref = reference.sel(time=inits + lag)
        fct = reference.sel(time=inits)
        ref['time'] = fct['time']
        plag.append(metric(ref, fct, dim='time', comparison=_e2c))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    return pers


# ToDo: do we really need a function here
# or cannot we somehow use compute_hindcast for that?
check_xarray([0, 1])
def compute_uninitialized(uninit,
                          reference,
                          metric='pearson_r',
                          comparison='e2r'):
    """
    Compute a predictability skill score between an uninitialized ensemble
    and some reference (hindcast, assimilation, reconstruction, observations).

    Based on Decadal Prediction protocol, this should only be computed for the
    first lag and then projected out to any further lags being analyzed.

    Parameters
    ----------
    uninit (xarray object):
        uninitialized ensemble.
    reference (xarray object):
        reference output/data over same time period.
    metric (str):
        Metric used in comparing the decadal prediction ensemble with the
        reference.
    comparison (str):
        How to compare the decadal prediction ensemble to the reference.
        * e2r : ensemble mean to reference (Default)
        * m2r : each member to the reference

    Returns:
        u (xarray object): Results from comparison at the first lag.
    """
    comparison = get_comparison_function(comparison)
    _validate_hindcast_comparison(comparison)
    metric = get_metric_function(metric)
    _validate_hindcast_metric(metric)
    uninit, reference = comparison(uninit, reference)
    u = metric(uninit, reference, dim='time', comparison=comparison)
    return u


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
        ph = ((p_values < alpha) & (sig)).argmin('lead')
        # where ph not reached, set max time
        ph_not_reached = ((p_values < alpha) & (sig)).all('lead')
    elif (limit is 'upper') and (perfect_model):
        ph = (skill > threshold).argmin('lead')
        ph_not_reached = (skill > threshold).all('lead')
    elif limit is 'lower':
        ph = (skill < threshold).argmin('lead')
        # where ph not reached, set max time
        ph_not_reached = (skill < threshold).all('lead')
    else:
        raise ValueError("""Please either submit 'upper' or 'lower' for the
            limit keyword.""")
    ph = ph.where(~ph_not_reached, other=skill['lead'].max())
    # mask out any initial NaNs (land, masked out regions, etc.)
    mask = np.asarray(skill.isel({'lead': 0}))
    mask = np.isnan(mask)
    ph = ph.where(~mask, np.nan)
    return ph
