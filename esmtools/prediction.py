"""Objects dealing with prediction metrics. In particular, these objects are specific to decadal prediction -- skill, persistence forecasting, etc and perfect-model predictability --  etc.

Concept of calculating predictability skill
-------------------------------------------
- metric: how is skill calculated, eg. rmse
- comparison: how forecasts and observation/truth are compared, eg. _m2m
- PM_compute: computes the perfect-model predictability skill according to metric and comparison
    - PM_compute(ds, control, metric=rmse, comparison=_m2m)
- bootstrap from uninitialized ensemble:
    - PM_sig(ds, control, metric=rmse, comparison=_m2m, bootstrap=500, sig=99)
    - threshold to determine predictability horizon

Metrics
-------
- _mse: Mean Square Error
- _nev: Normalized Ensemble Variance
- _msss: Mean Square Skill Score = _ppp: Prognostic Potential Predictability
- _rmse and _rmse_v: Root-Mean Square Error
- _nrmse: Normalized Root-Mean Square Error
- _pearson_r: Anomaly correlation coefficient
- _uACC: unbiased ACC

Comparisons
-----------
- _m2c: many forecasts vs. control truth
- _m2e: many forecasts vs. ensemble mean truth
- _m2m: many forecasts vs. many truths in turn
- _e2c: ensemble mean forecast vs. control truth

Missing
-------
- Relative Entropy (Kleeman 2002; Branstator and Teng 2010)
- Mutual information (DelSole)
- Average Predictability Time (APT) (DelSole)
- persistence forecast

Also
----
- Diagnostic Potential Predictability (DPP) (Boer 2004, Resplandy 2015/Seferian 2018)
- predictability horizon:
    - linear breakpoint fit (Seferian 2018) (missing)
    - f-test significant test (Pohlmann 2004, Griffies 1997) (missing)
    - bootstrapping limit

- Persistence Forecasts
    - persistence (missing)
    - damped persistence


Data Structure
--------------
This module works on xr.Datasets with the following dimensions and coordinates:
- 1D (Predictability of timelines of preprocessed regions):
    - ensemble
    - area
    - time (as in Lead Year)
    - period (time averaging: yearmean, seasonal mean)

Example ds via load_dataset('PM_MPI-ESM-LR_ds'):
<xarray.Dataset>
Dimensions:                  (area: 14, ensemble: 12, member: 10, period: 5, 'time': 20)
Coordinates:
  * ensemble                 (ensemble) int64 3014 3023 3045 3061 3124 3139 ...
  * area                     (area) object 'global' 'North_Atlantic_SPG' ...
  * 'time'                     (time) int64 1 2 3 4 5 6 ...
  * period                   (period) object 'DJF' 'JJA' 'MAM' 'SON' 'ym'
Dimensions without coordinates: member
Data variables:
    tos                   (period, 'time', area, ensemble, member) float32 ...
...

- 3D (Predictability maps):
    - ensemble
    - lon(y, x), lat(y, x)
    - time (as in Lead Year)
    - period (time averaging: yearmean, seasonal mean)

Example via load_dataset('PM_MPI-ESM-LR_ds3d'):
<xarray.Dataset>
Dimensions:      (bnds: 2, ensemble: 11, member: 9, vertices: 4, x: 256, y: 220, time: 21)
Coordinates:
    lon          (y, x) float64 -47.25 -47.69 -48.12 ... 131.3 132.5 133.8
    lat          (y, x) float64 76.36 76.3 76.24 76.17 ... -77.25 -77.39 -77.54
  * ensemble     (ensemble) int64 3061 3124 3178 3023 ... 3228 3175 3144 3139
  * time         (time) int64 1 2 3 4 5 ... 19 20
Dimensions without coordinates: bnds, member, vertices, x, y
Data variables:
    tos          (time, ensemble, member, y, x) float32 dask.array<shape=(21, 11, 9, 220, 256), chunksize=(1, 1, 1, 220, 256)>
...

This 3D example data is from curivlinear grid MPIOM (MPI Ocean Model) netcdf output.
The time dimensions is called 'time' and is in integer, not datetime[ns]

"""
import os
from random import randint

import numpy as np
import pandas as pd
import xarray as xr
from bs4 import BeautifulSoup
from six.moves.urllib.request import urlopen, urlretrieve
from xskillscore import pearson_r, rmse

from .stats import xr_corr, xr_linregress

# standard setup for load dataset and examples
varname = 'tos'
period = 'ym'
area = 'North_Atlantic'

#--------------------------------------------#
# HELPER FUNCTIONS
# Should only be used internally by esmtools
#--------------------------------------------#


def _get_variance(control, running=None):
    """
    Get running variance.

    Needed for compute and metrics PPP and NEV to normalize a variance.
    """
    if isinstance(running, int):
        var = control.rolling(time=running).var().mean('time')
    else:
        var = control.var('time')
    return var


def _ensemble_variance(ds, comparison):
    """
    Calculate ensemble variance for selected comparison.
    """
    comparison_name = comparison.__name__
    if comparison_name is '_m2e':
        return _ens_var_against_mean(ds)
    if comparison_name is '_m2c':
        return _ens_var_against_control(ds)
    if comparison_name is '_m2m':
        return _ens_var_against_every(ds)
    if comparison_name is '_e2c':
        return _ensmean_against_control(ds)


def _get_norm_factor(comparison):
    """
    Get normalization factor for ppp, nvar, nrmse. Used in PM_compute.

    m2e gets smaller rmse's than m2m by design, see Seferian 2018 et al.
    # m2m-ensemble variance should be divided by 2 to get var(control)
    """
    comparison_name = comparison.__name__
    if comparison_name is '_m2e':
        return 1
    elif comparison_name in ['_m2c', '_m2m', '_e2c']:
        return 1  # 2
    else:
        raise ValueError('specify comparison to get normalization factor.')


def _control_for_reference_period(control, reference_period='MK', obs_time=40):
    """
    Modifies control according to knowledge approach, see Hawkins 2016.

    Used in PM_compute(metric=[_mse, _rmse_v]
    """
    if reference_period == 'MK':
        _control = control
    elif reference_period == 'OP_full_length':
        _control = control - \
            control.rolling(time=obs_time, min_periods=1,
                            center=True).mean() + control.mean('time')
    elif reference_period == 'OP':
        raise ValueError('not yet implemented')
    else:
        raise ValueError("choose a reference period")
    return _control


def _drop_ensembles(ds, rmd_ensemble=[0]):
    """Drop ensembles from ds."""
    if all(ens in ds.ensemble.values for ens in rmd_ensemble):
        ensemble_list = list(ds.ensemble.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select available ensemble starting years', rmd_ensemble)
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


#--------------------------------------------#
# SAMPLE DATA
# Definitions related to loading sample
# datasets.
#--------------------------------------------#
def _get_data_home(data_home=None):
    """
    Return the path of the data directory.

    This is used by the ``load_dataset`` function.
    If the ``data_home`` argument is not specified, the default location
    is ``~/seaborn-data``.

    """
    if data_home is None:
        data_home = os.environ.get('HOME', '~')
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def get_dataset_names():
    """
    Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    # copied from seaborn
    http = urlopen(
        'https://github.com/bradyrx/esmtools/tree/master/sample_data/prediction')
    gh_list = BeautifulSoup(http, features='lxml')
    return [l.text.replace('.nc', '')
            for l in gh_list.find_all("a", {"class": "js-navigation-open"})
            if l.text.endswith('.nc')]


def load_dataset(name, cache=True, data_home=None, **kws):
    """
    Load a datasets ds and control from the online repository (requires internet).

    Parameters
    ----------
    name : str
        Name of the dataset (`ds`.nc on
        https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction).
        You can obtain list of available datasets using :func:`get_dataset_names`
    cache : boolean, optional
        If True, then cache data locally and use the cache on subsequent calls
    data_home : string, optional
        The directory in which to cache data. By default, uses ~/.
    kws : dict, optional
        Passed to pandas.read_csv

    """
    path = (
        'https://github.com/bradyrx/esmtools/tree/master/sample_data/prediction/{}.nc')
    full_path = path.format(name)
    # print('Load from URL:', full_path)
    if cache:
        cache_path = os.path.join(_get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path
    df = xr.open_dataset(full_path, **kws)
    return df


#--------------------------------------------#
# COMPARISONS
# Ways to calculate ensemble spread.
# Generally from Griffies & Bryan 1997
# Two different approaches here:
# - np vectorized from xskillscore (_rmse, _pearson_r) but manually 'stacked'
#   (_m2m, m2e, ...); supervector is stacked vector of all ensembles and members
# - xarray vectorized (_mse, _rmse_v, ...) from ensemble variance (_ens_var_against_mean, _..control)
# Leads to the same results: (metric=_rmse, comparison=c) equals (metric=_rmse_v, comparison=c) for all c in comparisons
#--------------------------------------------#


def _m2m(ds, supervector_dim):
    """
    Create two supervectors to compare all members to all other members in turn.
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


def _ens_var_against_every(ds):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance is evaluated against each ensemble
    member and then averaged.
    """
    var_list = []
    for m in ds.member.values:
        var_list.append(
            ((ds - ds.sel(member=m))**2).sum(dim='member') / (m - 1))
    var = xr.concat(var_list, 'member').mean('member')
    return var.mean('ensemble')


def _m2e(ds, supervector_dim):
    """
    Create two supervectors to compare all members to ensemble mean.
    """
    truth_list = []
    fct_list = []
    mean = ds.mean('member')
    for m in ds.member.values:
        for e in ds.ensemble.values:
            truth_list.append(mean.sel(ensemble=e))
            fct_list.append(ds.sel(member=m, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)
    return fct, truth


def _ens_var_against_mean(ds):
    """
    Calculate the ensemble spread (ensemble variance (squared difference between each ensemble member and the ensemble mean) as a function of time).

    Parameters
    ----------
    ds : DataArray with time dimension (optional spatial coordinates)
        Input data

    Returns
    -------
    c : DataArray as ds reduced by member dimension
        Output data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    # display as dataframe
    ens_var_against_mean.to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)

    """
    return ds.var('member').mean('ensemble')


def _m2c(ds, supervector_dim, control_member=[0]):
    """
    Create two supervectors to compare all members to control.

    control_member: list or int??
        index to be removed, default 0
    """
    truth_list = []
    fct_list = []
    truth = ds.isel(member=control_member).squeeze()
    # drop the member being truth
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    for m in ds_dropped.member.values:
        for e in ds_dropped.ensemble.values:
            fct_list.append(truth.sel(ensemble=e))
            truth_list.append(ds_dropped.sel(member=m, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)

    return fct, truth


def _ens_var_against_control(ds, control_member=0):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance is evaluated against member=0 which is
    the control run.
    """
    var = ds.copy()
    var = ((ds - ds.sel(member=ds.member.values[control_member]))**2).sum('member') / (ds.member.size - 1)
    return var.mean('ensemble')


def _e2c(ds, supervector_dim, control_member=[0]):
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


def _ensmean_against_control(ds, control_member=0):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance is evaluated between ensemble mean and
    control.
    """
    # drop the member being truth
    truth = ds.sel(member=ds.member.values[control_member])
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    return ((ds.mean('member') - truth)**2).mean('ensemble')


#--------------------------------------------#
# METRICS
# Metrics for computing predictability.
#--------------------------------------------#
# importing to _metric
def _pearson_r(a, b, dim):
    """
    Compute anomaly correlation coefficient (ACC) of two xr objects. See xskillscore.pearson_r.
    """
    return pearson_r(a, b, dim)


def _rmse(a, b, dim):
    """
    Compute root-mean-square-error (RMSE) of two xr objects. See xskillscore.rmse.
    """
    return rmse(a, b, dim)


def _mse(ds, control, comparison, running):
    """
    Mean Square Error (MSE) metric.
    """
    return _ensemble_variance(ds, comparison)


def _rmse_v(ds, control, comparison, running):
    """
    Root Mean Square Error (RMSE) metric.
    """
    return _ensemble_variance(ds, comparison) ** .5


def _nrmse(ds, control, comparison, running):
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
    var = _get_variance(control, running=running)
    ens = _ensemble_variance(ds, comparison)
    fac = _get_norm_factor(comparison)
    return (ens / var / fac) ** .5


def _nev(ds, control, comparison, running):
    """
    Normalized Ensemble Variance (NEV) metric.

    Reference
    ---------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    var = _get_variance(control, running=running)
    ens = _ensemble_variance(ds, comparison)
    fac = _get_norm_factor(comparison)
    return ens / var / fac


def _ppp(ds, control, comparison, running):
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
    """
    var = _get_variance(control, running=running)
    ens = _ensemble_variance(ds, comparison)
    fac = _get_norm_factor(comparison)
    return 1 - ens / var / fac


# TODO: Do we need wrappers or should we rather create wrappers for skill scores
#       as used in a specific paper: def Seferian2018(ds, control): return PM_compute(ds, control, metric=_ppp, comparison=_m2e)
def _PPP(ds, control, comparison, running):
    """
    Wraps ppp.
    """
    return _ppp(ds, control, comparison, running)


def _uACC(ds, control, comparison, running):
    """
    Unbiased ACC (uACC) metric.

    Formula
    -------
    - uACC = PPP ** .5 = MSSS ** .5

    Reference
    ---------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong
      Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice
      Prediction: Potential versus Operational Seasonal Forecast Skill.” Climate
      Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    """
    return _ppp(ds, control, comparison, running) ** .5


def _msss(ds, control, comparison, running):
    """
    Wraps ppp.

    Formula
    -------
    MSSS = 1 - MSE_ens / var_control

    References
    ----------
    - Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.
    """
    return _ppp(ds, control, comparison, running)


#--------------------------------------------#
# COMPUTE PREDICTABILITY/FORECASTS
# Highest-level features for computing
# predictability.
#--------------------------------------------#
def PM_compute(ds, control, metric=_pearson_r, comparison=_m2m, anomaly=False,
               detrend=False, running=None):
    """
    Compute a predictability skill score for a perfect-model framework simulation dataset.

    Relies on two concepts yielding equal results (see comparisons):
    - np vectorized from xskillscore (_rmse, _pearson_r) but manually 'stacked' (_m2m, m2e, ...)
    - xarray vectorized (_mse, _rmse_v, ...) from ensemble variance (_ens_var_against_mean, _..control)

    Parameters
    ----------
    ds, control : xr.DataArray or xr.Dataset with 'time' dimension (optional spatial coordinates)
        input data
    metric : function
        metric from [_rmse, _pearson_r, _mse, _rmse_r, _ppp, _nev, _uACC, _MSSS]
    comparison : function
        comparison from [_m2m, _m2e, _m2c, _e2c]
    running : int
        Size of the running window for variance smoothing ( only used for PPP, NEV)

    Returns
    -------
    res : xr.DataArray or xr.Dataset
        skill score
    """
    supervector_dim = 'svd'
    if comparison.__name__ not in ['_m2m', '_m2c', '_m2e', '_e2c']:
        raise ValueError('specify comparison argument')

    if metric.__name__ in ['_pearson_r', '_rmse']:
        fct, truth = comparison(ds, supervector_dim)
        res = metric(fct, truth, dim=supervector_dim)
        return res
    elif metric.__name__ in ['_mse', '_rmse_v', '_nrmse', '_nev', '_ppp', '_PPP', '_MSSS', '_uACC']:
        res = metric(ds, control, comparison, running)
        return res
    else:
        raise ValueError('specify metric argument')


#--------------------------------------------#
# PERSISTANCE FORECASTS
#--------------------------------------------#
# TODO: adapt for maps
def generate_damped_persistence_forecast(control, startyear, length=20):
    """
    Generate damped persistence forecast mean and range.

    Reference
    ---------
    - missing: got a script from a collegue

    Parameters
    ----------
    control : pandas.series
        input timeseries from control run
    startyear : int
        year damped persistence forecast should start from

    Returns
    -------
    ar1 : pandas.series
        mean damped persistence
    ar50 : pandas.series
        50% damped persistence range
    ar90 : pandas.series
        90% damped persistence range

    Example
    -------
    import esmtools as et
    ar1, ar50, ar90 = et.prediction.generate_damped_persistence_forecast(control_,3014)
    ar1.plot(label='damped persistence forecast')
    plt.fill_between(ar1.index,ar1-ar50,ar1+ar50,alpha=.2,
                     color='gray',label='50% forecast range')
    plt.fill_between(ar1.index,ar1-ar90,ar1+ar90,alpha=.1,
                     color='gray',label='90% forecast range')
    control_.sel(time=slice(3014,3034)).plot(label='control')
    plt.legend()

    """
    anom = (control.sel(time=startyear) - control.mean('time')).values
    t = np.arange(0., length + 1, 1)
    alpha = xr_corr(control).values
    exp = anom * np.exp(-alpha * t)  # exp. decay towards mean
    ar1 = exp + control.mean('time').values
    ar50 = 0.7 * control.std('time').values * np.sqrt(1 - np.exp(-2 * alpha * t))
    ar90 = 1.7 * control.std('time').values * np.sqrt(1 - np.exp(-2 * alpha * t))

    index = control.sel(time=slice(startyear, startyear + length))['time']
    ar1 = pd.Series(ar1, index=index)
    ar50 = pd.Series(ar50, index=index)
    ar90 = pd.Series(ar90, index=index)
    return ar1, ar50, ar90

# TODO: needs complete redo
def generate_predictability_damped_persistence(s, kind='PPP', percentile=True, length=20):
    """
    Calculate the PPP (or NEV) damped persistence mean and range in PPP plot.

    Lag1 autocorrelation coefficient (alpha) is bootstrapped. Range can be
    indicated as +- std or 5-95-percentile.

    Reference
    ---------
    Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4. Appendix

    Parameters
    ----------
    s : pandas.series
        input timeseries from control run
    kind : str
        determine kind of damped persistence. 'PPP' or 'NEV' (normalized ensemble variance)
    percentile : bool
        use percentiles for alpha range
    length : int
        length of the output timeseries

    Returns
    -------
    PPP_persistence_0 : pandas.series
        mean damped persistence
    PPP_persistence_minus : pandas.series
        lower range damped persistence
    PP_persistence_plus : pandas.series
        upper range damped persistence

    Example
    -------
    import esmtools as et
    # s = pd.Series(np.sin(range(1000)+np.cos(range(1000))+np.random.randint(.1,1000)))
    s = control.sel(area=area,period=period).to_dataframe()[varname]
    PPP_persistence_0, PPP_persistence_minus, PPP_persistence_plus = et.prediction.generate_predictability_persistence(
        s)
    t = np.arange(0,20+1,1.)
    plt.plot(PPP_persistence_0,color='black',
             linestyle='--',label='persistence mean')
    plt.fill_between(t,PPP_persistence_minus,PPP_persistence_plus,
                     color='gray',alpha=.3,label='persistence range')
    plt.axhline(y=0,color='black')

    """
    # bootstrapping persistence
    iterations = 50  # iterations
    chunk_length = 100  # length of chunks of control run to take lag1 autocorr
    data = np.zeros(iterations)
    for i in range(iterations):
        random_start_year = randint(s.index.min(), s.index.max() - chunk_length)
        data[i] = s.loc[str(random_start_year):str(
            random_start_year + chunk_length)].autocorr()

    alpha_0 = np.mean(data)
    alpha_minus = np.mean(data) - np.std(data)
    alpha_plus = np.mean(data) + np.std(data)
    if percentile:
        alpha_minus = np.percentile(data, 5)
        alpha_plus = np.percentile(data, 95)

    # persistence function
    def generate_PPP_persistence(alpha, t):
        values = np.exp(-2 * alpha * t)  # Griffies 1997
        s = pd.Series(values, index=t)
        return s

    t = np.arange(0, length + 1, 1.)
    PPP_persistence_0 = generate_PPP_persistence(alpha_0, t)
    PPP_persistence_minus = generate_PPP_persistence(alpha_plus, t)
    PPP_persistence_plus = generate_PPP_persistence(alpha_minus, t)

    if kind in ['nvar', 'NEV']:
        PPP_persistence_0 = 1 - PPP_persistence_0
        PPP_persistence_minus = 1 - PPP_persistence_minus
        PPP_persistence_plus = 1 - PPP_persistence_plus

    return PPP_persistence_0, PPP_persistence_minus, PPP_persistence_plus


# TODO: Adjust for 3d fields
def damped_persistence_forecast(ds, control, varname=varname, area=area, period=period,
                                comparison=_m2e):
    """
    Generate damped persistence forecast timeseries.
    """
    starting_years = ds.ensemble.values
    anom = (control.sel(time=starting_years) - control.mean('time'))
    t = ds['time']
    alpha = control.to_series().autocorr()
    persistence_forecast_list = []
    for ens in anom['time']:
        ar1 = anom.sel(time=ens).values * \
            np.exp(-alpha * t) + control.mean('time').values
        pf = xr.DataArray(data=ar1, coords=[t], dims='time_dim')
        pf = pf.expand_dims('ensemble')
        pf['ensemble'] = [ens]
        persistence_forecast_list.append(pf)
    return xr.concat(persistence_forecast_list, dim='ensemble')


def PM_compute_damped_persistence(ds, control, metric=_rmse, comparison=_m2e):
    """
    Compute skill for persistence forecast. See PM_compute().
    """
    persistence_forecasts = damped_persistence_forecast(ds, control)
    if comparison.__name__ == '_m2e':
        result = metric(persistence_forecasts, ds.mean('member'), 'ensemble')
    elif comparison.__name__ == '_m2m':
        persistence_forecasts = persistence_forecasts.expand_dims('member')
        all_persistence_forecasts = persistence_forecasts.sel(
            member=[0] * ds.member.size)
        fct = _m2e(all_persistence_forecasts, 'svd')[0]
        truth = _m2e(ds, 'svd')[0]
        result = metric(fct, truth, 'svd')
    else:
        raise ValueError('not defined')
    return result


#--------------------------------------------#
# Diagnostic Potential Predictability (DPP)
# Functions related to DPP from Boer et al.
#--------------------------------------------#
def DPP(ds, m=10, chunk=True, var_all_e=False):
    """
    Calculate Diagnostic Potential Predictability (DPP) as potentially predictable variance fraction (ppvf) in Boer 2004.

    Note: Different way of calculating it than in Seferian 2018 or Resplandy 2015,
    but quite similar results.

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


#--------------------------------------------#
# BOOTSTRAPPING
# Functions for sampling an ensemble
#--------------------------------------------#
def pseudo_ens(ds, control):
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
    varname='tos'
    period='ym'
    area='North_Atlantic'
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
    return xr.concat([create_pseudo_members(control) for _ in range(nens)], 'ensemble')


def PM_sig(ds, control, metric=_rmse, comparison=_m2m, reference_period='MK', sig=95, bootstrap=30):
    """
    Return sig-th percentile of function to be choosen from pseudo ensemble generated from control.

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
    x = []
    _control = _control_for_reference_period(
        control, reference_period=reference_period)
    for _ in range(1 + int(bootstrap / ds['time'].size)):
        ds_pseudo = pseudo_ens(ds, _control)
        ds_pseudo_metric = PM_compute(
            ds_pseudo, _control, metric=metric, comparison=comparison)
        x.append(ds_pseudo_metric)
    ds_pseudo_metric = xr.concat(x, dim='it')
    if isinstance(sig, list):
        qsig = [x / 100 for x in sig]
    else:
        qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(q=qsig, dim=['time', 'it'])
    return sig_level


#--------------------------------------------#
# PREDICTABILITY HORIZON
#--------------------------------------------#
def xr_predictability_horizon(skill, threshold, limit='upper'):
    """Get predictability horizons of dataset from skill and threshold dataset."""
    if limit is 'upper':
        ph = (skill > threshold).argmin('time')
        # where ph not reached, set max time
        ph_not_reached = (skill > threshold).all('time')
    elif limit is 'lower':
        ph = (skill < threshold).argmin('time')
        # where ph not reached, set max time
        ph_not_reached = (skill < threshold).all('time')
    ph = ph.where(~ph_not_reached, other=skill['time'].max())
    return ph
