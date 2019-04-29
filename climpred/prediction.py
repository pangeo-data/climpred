"""Objects dealing with decadal prediction metrics."""
import warnings

import cftime
import numpy as np
import xarray as xr

from xskillscore import mae as _mae
from xskillscore import mse as _mse
from xskillscore import pearson_r as _pearson_r
from xskillscore import pearson_r_p_value
from xskillscore import rmse as _rmse

from .stats import _check_xarray, z_significance
from .comparisons import (_get_comparison_function, _m2m, _m2c,
                          _m2e, _e2c, _e2r, _m2r)
from .metrics import (_get_metric_function, _nmae, _nrmse,
                      _nmse, _ppp, _uacc)


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
    if all(ens in ds.time.values for ens in rmd_ensemble):
        ensemble_list = list(ds.time.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select available ensembles only', rmd_ensemble)
    return ds.sel(time=ensemble_list)


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
        i = ds.time.values
    return ds.sel(member=m, time=i)


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
    # perfect-model only metrics
    elif metric in [_nmae, _nrmse, _nmse, _ppp, _uacc]:
        res = metric(ds, control, comparison, running, reference_period)
    else:
        raise ValueError('specify metric argument')
    # Note: Aaron implemented this in PR #87. They break when
    # compute_perfect_model is called from `bootstrap_perfect_model`. So need
    # to debug why that is the case and see if these lines are even
    # necessary.
#    time_size = ds.time.size
#    del res['time']
#    res['time'] = np.arange(1, 1 + time_size)
    return res


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
        `time` : dim of initialization dates
        `lead` : dim of lead time from those initializations
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
                 to. Default: length of `lead` dim
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
        nlags = forecast.lead.size
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please input 'pearson_r', 'rmse', 'mse', or
            'mae' for your metric.""")
    plag = []
    for i in range(0, nlags):
        # Temporary rename of init dimension to time to allow the
        # metric to run properly.
        a, b = _shift(
            forecast.isel(lead=i).rename({'init': 'time'}), reference, i,
            dim='time')
        plag.append(metric(a, b, dim='time'))
    skill = xr.concat(plag, 'lead')
    skill['lead'] = np.arange(1, 1 + nlags)
    if (return_p) & (metric != _pearson_r):
        raise ValueError("""You can only return p values if the metric is
            pearson_r.""")
    elif (return_p) & (metric == _pearson_r):
        # NaN values throw warning for p-value comparison, so just
        # suppress that here.
        p_value = []
        for i in range(0, nlags):
            a, b = _shift(
                forecast.isel(lead=i).rename({'init': 'time'}), reference, i,
                dim='time')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value.append(pearson_r_p_value(a, b, dim='time'))
        p_value = xr.concat(p_value, 'lead')
        p_value['lead'] = np.arange(1, 1 + nlags)
        return skill, p_value
    else:
        return skill


def compute_persistence_pm(ds, control, nlags, metric='pearson_r',
                           dim='time', init_month_index=0):
    """
    Computes the skill of  a persistence forecast from a control run.

    This simply applies some metric on the input out to some lag. The user
    should avoid computing persistence with prebuilt ACF functions in e.g.,
    python, MATLAB, R as they tend to use FFT methods for speed but incorporate
    error due to this.

    TODO: Merge this and `compute_persistence` into one function. These two
    functions employ different philosophies on how to compute persistence.

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
                   Default: 'time'

    Returns:
        pers (xarray object): Results of persistence forecast with the input
                              metric applied.
    """
    _check_xarray(control)
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please select between the following metrics:
            'pearson_r',
            'rmse',
            'mse',
            'mae'""")

    init_years = ds['init'].values
    if isinstance(ds.init.values[0],
                  cftime._cftime.DatetimeProlepticGregorian) \
            or isinstance(ds.init.values[0], np.datetime64):
        init_cftimes = []
        for year in init_years:
            init_cftimes.append(control.sel(
                time=str(year)).isel(time=init_month_index).time)
        init_cftimes = xr.concat(init_cftimes, 'time')
    elif isinstance(ds.init.values[0], np.int64):
        init_cftimes = []
        for year in init_years:
            init_cftimes.append(control.sel(
                time=year).time)
        init_cftimes = xr.concat(init_cftimes, 'time')
    else:
        raise ValueError(
            'Set time axis to xr.cftime_range, pd.date_range or np.int64.')

    inits_index = []
    control_time_list = list(control.time.values)
    for i, inits in enumerate(init_cftimes.time.values):
        inits_index.append(control_time_list.index(init_cftimes[i]))

    plag = []  # holds results of persistence for each lag
    control = control.isel({dim: slice(0, -nlags)})
    for lag in range(1, 1 + nlags):
        inits_index_plus_lag = [x + lag for x in inits_index]
        ref = control.isel({dim: inits_index_plus_lag})
        fct = control.isel({dim: inits_index})
        ref[dim] = fct[dim]
        plag.append(metric(ref, fct, dim=dim))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = np.arange(1, 1 + nlags)
    return pers


def compute_persistence(ds, reference, nlags, metric='pearson_r',
                        dim='time'):
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
        ds (xarray object): The initialized ensemble.
        reference (xarray object): The reference time series.
        nlags (int): Number of lags to compute persistence to.
        metric (str): Metric name to apply at each lag for the persistence
                      computation. Default: 'pearson_r'
        dim (str): Dimension over which to compute persistence forecast.
                   Default: 'time'

    Returns:
        pers (xarray object): Results of persistence forecast with the input
                              metric applied.
    """
    def _intersection(lst1, lst2):
        """
        Custom intersection, since `set.intersection()` changes type of list.
        """
        lst3 = [value for value in lst1 if value in lst2]
        return np.array(lst3)

    _check_xarray(reference)
    metric = _get_metric_function(metric)
    if metric not in [_pearson_r, _rmse, _mse, _mae]:
        raise ValueError("""Please select between the following metrics:
            'pearson_r',
            'rmse',
            'mse',
            'mae'""")
    plag = []  # holds results of persistence for each lag
    for lag in range(1, 1 + nlags):
        inits = ds['init'].values
        ctrl_inits = reference.isel({dim: slice(0, -lag)})[dim].values
        inits = _intersection(inits, ctrl_inits)
        ref = reference.sel({dim: inits + lag})
        fct = reference.sel({dim: inits})
        ref[dim] = fct[dim]
        plag.append(metric(ref, fct, dim=dim))
    pers = xr.concat(plag, 'lead')
    pers['lead'] = np.arange(1, 1 + nlags)
    return pers


def compute_uninitialized(uninit, reference, metric='pearson_r',
                          comparison='e2r', return_p=False,
                          dim='time'):
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
        * pearson_r (Default)
        * rmse
        * mae
        * mse
    comparison (str):
        How to compare the decadal prediction ensemble to the reference.
        * e2r : ensemble mean to reference (Default)
        * m2r : each member to the reference
    return_p (bool): If True, return p values associated with pearson r.

    Returns:
        u (xarray object): Results from comparison at the first lag.
        p (xarray object): If `return_p`, p values associated with
                                 pearson r correlations.
    """
    _check_xarray(uninit)
    _check_xarray(reference)
    comparison = _get_comparison_function(comparison)
    if comparison not in [_e2r, _m2r]:
        raise KeyError("""Please input either 'e2r' or 'm2r' for your
            comparison. This will be implemented for the perfect model setup
            in the future.""")
    uninit, reference = comparison(uninit, reference)
    metric = _get_metric_function(metric)
    u = metric(uninit, reference, dim=dim)
    if (return_p) & (metric != _pearson_r):
        raise KeyError("""You can only return p values if the metric is
            'pearson_r'.""")
    elif (return_p) & (metric == _pearson_r):
        p = pearson_r_p_value(uninit, reference, dim=dim)
        return u, p
    else:
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
