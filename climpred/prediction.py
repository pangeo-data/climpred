import xarray as xr

from .comparisons import _e2c
from .constants import (
    ALL_PM_METRICS_DICT,
    ALL_PM_COMPARISONS_DICT,
    ALL_HINDCAST_METRICS_DICT,
    ALL_HINDCAST_COMPARISONS_DICT
)
from .utils import get_metric_function, get_comparison_function, intersect
from .checks import is_xarray

# --------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
# --------------------------------------------#
@is_xarray([0, 1])
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
        KeyError: if comarison not implemented.
                  if metric not implemented.
    """
    supervector_dim = 'svd'
    metric = get_metric_function(metric, ALL_PM_METRICS_DICT)
    comparison = get_comparison_function(comparison, ALL_PM_COMPARISONS_DICT)

    forecast, reference = comparison(ds, supervector_dim)

    res = metric(forecast, reference, dim=supervector_dim, comparison=comparison)
    return res


@is_xarray([0, 1])
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
    comparison = get_comparison_function(comparison, ALL_HINDCAST_COMPARISONS_DICT)
    metric = get_metric_function(metric, ALL_HINDCAST_METRICS_DICT)

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


@is_xarray([0, 1])
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
    metric = get_metric_function(metric, ALL_HINDCAST_METRICS_DICT)

    plag = []  # holhind results of persistence for each lag
    for lag in hind.lead.values:
        inits = hind['init'].values
        ctrl_inits = reference.isel(time=slice(0, -lag))['time'].values
        inits = intersect(inits, ctrl_inits)
        ref = reference.sel(time=inits + lag)
        fct = reference.sel(time=inits)
        ref['time'] = fct['time']
        plag.append(metric(ref, fct, dim='time', comparison=_e2c))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = hind.lead.values
    return pers


# ToDo: do we really need a function here
# or cannot we somehow use compute_hindcast for that?
@is_xarray([0, 1])
def compute_uninitialized(uninit, reference, metric='pearson_r', comparison='e2r'):
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
    comparison = get_comparison_function(comparison, ALL_HINDCAST_COMPARISONS_DICT)
    metric = get_metric_function(metric, ALL_HINDCAST_METRICS_DICT)
    uninit, reference = comparison(uninit, reference)
    u = metric(uninit, reference, dim='time', comparison=comparison)
    return u
