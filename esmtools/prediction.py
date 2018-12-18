"""Objects dealing with prediction metrics. In particular, these objects are specific to decadal prediction -- skill, persistence forecasting, etc and perfect-model predictability --  etc.

ToDos
-----
- how to treat for 1D timeseries and 3D maps?
    - different functions
    - arguments
    - generic
- variable names and file setup:
    - fgco2 (cmorized) or co2_flux (MPI) or (CESM)
    - config file?
    - global keywords?


Concept of calculating predictability skill
-------------------------------------------
- metric: how is skill calculated, eg. rmse
- comparison: how forecasts and observation/truth are compared, eg. m2m
- compute: computes the predictability skill according to metric and comparison
    - compute(ds, control, metric=rmse, comparison=m2m)
- bootstrap from uninitialized ensemble:
    - PM-sig(ds, control, metric=rmse, comparison=m2m, bootstrap=500, sig=99)
    - threshold to determine predictability horizon

Metrics
-------
- MSE: Mean Square Error
- NEV: Normalized Ensemble Variance
- MSSS: Mean Square Skill Score = PPP: Prognostic Potential Predictability
- RMSE: Root-Mean Square Error
- NRMSE: Normalized Root-Mean Square Error
- ACC: Anomaly correlation coefficient
- uACC: unbiased ACC (Bushuk et al. 2018)

Comparisons
-----------
- m2c: many forecasts vs. control truth
- m2e: many forecasts vs. ensemble mean truth
- m2m: many forecasts vs. many truths in turn
- e2c: ensemble mean forecast vs. control truth

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
    - year (as in Lead Year)
    - period (time averaging: yearmean, seasonal mean)

Example ds via load_dataset('PM_MPI-ESM-LR_ds'):
<xarray.Dataset>
Dimensions:                  (area: 14, ensemble: 12, member: 10, period: 5, year: 20)
Coordinates:
  * ensemble                 (ensemble) int64 3014 3023 3045 3061 3124 3139 ...
  * area                     (area) object 'global' 'North_Atlantic_SPG' ...
  * year                     (year) int64 1900 1901 1902 1903 1904 1905 1906 ...
  * period                   (period) object 'DJF' 'JJA' 'MAM' 'SON' 'ym'
Dimensions without coordinates: member
Data variables:
    atmco2                   (period, year, area, ensemble, member) float32 ...
...

- 3D (Predictability maps):
    - ensemble
    - lon, lat
    - year (as in Lead Year)
    - period (time averaging: yearmean, seasonal mean)

Example via load_dataset('PM_MPI-ESM-LR_ds3d'):
<xarray.Dataset>
Dimensions:      (bnds: 2, ensemble: 11, member: 9, vertices: 4, x: 256, y: 220, year: 21)
Coordinates:
    lon          (y, x) float64 -47.25 -47.69 -48.12 ... 131.3 132.5 133.8
    lat          (y, x) float64 76.36 76.3 76.24 76.17 ... -77.25 -77.39 -77.54
  * ensemble     (ensemble) int64 3061 3124 3178 3023 ... 3228 3175 3144 3139
  * year         (year) int64 1900 1901 1902 1903 1904 ... 1917 1918 1919 1920
Dimensions without coordinates: bnds, member, vertices, x, y
Data variables:
    lon_bnds     (year, ensemble, member, y, x, vertices) float64 dask.array<shape=(21, 11, 9, 220, 256, 4), chunksize=(1, 1, 1, 220, 256, 4)>
    lat_bnds     (year, ensemble, member, y, x, vertices) float64 dask.array<shape=(21, 11, 9, 220, 256, 4), chunksize=(1, 1, 1, 220, 256, 4)>
    time_bnds    (year, ensemble, member, bnds) float64 dask.array<shape=(21, 11, 9, 2), chunksize=(1, 1, 1, 2)>
    atmco2          (year, ensemble, member, y, x) float32 dask.array<shape=(21, 11, 9, 220, 256), chunksize=(1, 1, 1, 220, 256)>
...

This 3D example data is from curivlinear grid MPIOM (MPI Ocean Model) netcdf output.
Time dimensions is called Years and is integer. (Original data was year
3000-3300, now 3600. Had problems with datetime[ns] limits and xr.to_netcdf()!)

"""
import os
import esmtools as et
import numpy as np
import pandas as pd
import xarray as xr
from pyfinance import ols
from six.moves.urllib.request import urlopen, urlretrieve
from xskillscore import pearson_r, rmse
from bs4 import BeautifulSoup
from random import randint

# standard setup for load dataset and examples
varname = 'tos'
period = 'ym'
area = 'North_Atlantic'


def get_data_home(data_home=None):
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
        'https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction/')
    # print('Load from URL:', http)
    gh_list = BeautifulSoup(http)
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
        "https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction/{}.nc")
    full_path = path.format(name)
    # print('Load from URL:', full_path)
    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path
    df = xr.open_dataset(full_path, **kws)
    return df


# Diagnostic Potential Predictability (DPP)
def chunking(ds, number_chunks=False, chunk_length=False, output=False, time_dim='year'):
    """
    Separate data into chunks and reshapes chunks in a c dimension.

    Specify either the number chunks or the length of chunks.
    Needed for DPP.

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
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

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    control_chunked_into_30yr_chunks = et.prediction.chunking(
        control,chunk_length=30)
    control_chunked_into_30_chunks = et.prediction.chunking(
        control,number_chunks=30)

    """
    if number_chunks and not chunk_length:
        chunk_length = np.floor(ds.year.size / number_chunks)
        cmin = int(ds.year.min())
    elif not number_chunks and chunk_length:
        cmin = int(ds.year.min())
        number_chunks = int(np.floor(ds.year.size / chunk_length))
    else:
        raise ValueError('set number_chunks or chunk_length to True')
    c = ds.sel(year=slice(cmin, cmin + chunk_length - 1))
    c = c.expand_dims('c')
    c['c'] = [0]
    for i in range(1, number_chunks):
        if output:
            print(i, cmin + chunk_length * i,
                  cmin + (i + 1) * chunk_length - 1)
        c2 = ds.sel(year=slice(cmin + chunk_length * i,
                               cmin + (i + 1) * chunk_length - 1))
        c2 = c2.expand_dims('c')
        c2['c'] = [i]
        c2[time_dim] = c[time_dim]
        c = xr.concat([c, c2], 'c')
    return c


def DPP(ds, m=10, chunk=True, var_all_e=False, time_dim='year'):
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
    ds : DataArray with year dimension (optional spatial coordinates)
    m : int
        separation time scale in years between predictable low-freq
        component and high-freq noise
    chunk : boolean
        Whether chunking is applied. Default: True.
        If False, then uses Resplandy 2015 / Seferian 2018 method.

    Returns
    -------
    DPP : DataArray as ds without time/year dimension

    Example 1D
    ----------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ds_DPPm10 = et.prediction.DPP(ds,m=10,chunk=True)

    """
    if not chunk:
        s2v = ds.rolling(year=m, min_periods=1, center=True).mean().var(time_dim)
        s2e = (ds - ds.rolling(year=m, min_periods=1,
                               center=True).mean()).var(time_dim)
        s2 = s2v + s2e
    if chunk:
        # first chunk
        chunked_means = chunking(ds, chunk_length=m).mean(time_dim)
        # sub means in chunks
        chunked_deviations = chunking(ds, chunk_length=m) - chunked_means
        s2v = chunked_means.var('c')
        if var_all_e:
            s2e = chunked_deviations.var([time_dim, 'c'])
        else:
            s2e = chunked_deviations.var(time_dim).mean('c')
        s2 = s2v + s2e
    DPP = (s2v - s2 / (m)) / (s2)
    return DPP


# Prognostic Potential Predictability Griffies & Bryan 1997
# 3 different ways of calculation ensemble spread:
def ens_var_against_mean(ds):
    """
    Calculate the ensemble spread (ensemble variance (squared difference between each ensemble member and the ensemble mean) as a function of time).

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
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


def ens_var_against_control(ds):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance is
    evaluated against member=0 which is the control run.
    """
    var = ds.copy()
    var = ((ds - ds.sel(member=0))**2).sum('member') / (ds.member.size - 0)
    return var.mean('ensemble')


def ens_var_against_every(ds):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance
    is evaluated against each ensemble member and then averaged.
    """
    var = ds.copy()
    m = ds.member.size
    for i in range(0, m):
        var_a = ((ds - ds.sel(member=i))**2).sum(dim='member') / m
        var = xr.concat([var, var_a], 'member')
    var = var.sel(member=slice(m, 2 * m)).mean('member')
    return var.mean('ensemble')


def pseudo_ens(ds, control, time_dim='year'):
    """
    Create a pseudo-ensemble from control run.

    Needed for bootstrapping confidence intervals of a metric.
    Takes randomly 20yr segments from control and rearranges them into ensemble
    and member dimensions.

    Parameters
    ----------
    control : xr.DataArray with year dimension
        Input ensemble data

    Returns
    -------
    ds_e : xr.DataArray with year, ensemble, member dimension
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
    nm = ds.member.size
    length = ds.year.size
    c_start = control.year[0]
    c_end = control.year[-1]
    year = ds[time_dim]

    def sel_years(control, year_s, m=None, length=length, time_dim=time_dim):
        new = control.sel(year=slice(year_s, year_s + length - 1))
        new[time_dim] = year
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nm)
        return xr.concat([sel_years(control, start)
                          for start in startlist], 'member')
    return xr.concat([create_pseudo_members(control) for _ in range(nens)], 'ensemble')


def pseudo_ens_fast(ds3d, control3d, varname=None, shuffle=True, bootstrap=None):
    """
    Create a pseudo-ensemble from control run in the form of ensemble ds.

    Needed for bootstrapping confidence intervals of a metric.

    Parameters
    ----------
    control : xr.DataArray with year dimension
        Input ensemble data

    Returns
    -------
    ds_e : xr.DataArray with year, ensemble, member dimension
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
    control3d2 = control3d.copy()
    if varname is not None:
        control3d2 = control3d2[varname]

    ensembles = ds3d.ensemble
    members = ds3d.member
    length = int(control3d.year.size / ensembles.size / members.size)

    if bootstrap is not None:
        control_year_values = np.copy(control3d2.year.values)
        needed_repetitions = int(bootstrap / length)
        new_control_years = np.concatenate(
            (control_year_values,) * needed_repetitions)
        control3d2 = control3d2.sel(year=new_control_years)
        control3d2['year'] = np.arange(1900, 1900 + control3d2.year.size)
    else:
        bootstrap = length

    if shuffle:
        time_before = np.copy(control3d2.year.values)
        np.random.shuffle(time_before)
        control3d2 = control3d2.sel(year=time_before)

    length = bootstrap
    input_time = control3d2.year[:int(length * ensembles.size * members.size)]
    new_time = control3d.year[:length]
    # sel fewer years for dimsizes to match
    control3d2 = control3d2.sel(year=input_time)

    coords = [new_time, ensembles, members, ds3d.y.values, ds3d.x.values]
    dims = ['year', 'ensemble', 'member', 'y', 'x']

    new_order = (length, ensembles.size, members.size,
                 ds3d.y.size, ds3d.x.size)

    reshaped = np.reshape(control3d2.values, new_order)
    new = xr.DataArray(reshaped, dims=dims, coords=coords)
    return new


def m2m(ds, supervector_dim):
    """
    Create two supervectors to compare members to all other members.
    """
    truth_list = []
    fct_list = []
    for m in ds.member.values:
        # drop the member being truth
        ds_reduced = drop_members(ds, rmd_member=[m])
        truth = ds.sel(member=m)
        for m2 in ds_reduced.member:
            for e in ds.ensemble:
                truth_list.append(truth.sel(ensemble=e))
                fct_list.append(ds_reduced.sel(member=m2, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)
    return fct, truth


def m2e(ds3d, supervector_dim):
    """
    Create two supervectors to compare members to ensemble mean.
    """
    truth_list = []
    fct_list = []
    mean = ds3d.mean('member')
    for m in range(ds3d.member.size):
        for e in ds3d.ensemble:
            truth_list.append(mean.sel(ensemble=e))
            fct_list.append(ds3d.sel(member=m, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)
    return fct, truth


def m2c(ds3d, supervector_dim, control_member=0):
    """
    Create two supervectors to compare members to control.
    """
    truth_list = []
    fct_list = []
    truth = ds3d.sel(member=control_member)
    # drop the member being truth
    ds3d_dropped = drop_members(ds3d, rmd_member=[control_member])
    for m in ds3d_dropped.member:
        for e in ds3d_dropped.ensemble:
            fct_list.append(truth.sel(ensemble=e))
            truth_list.append(ds3d_dropped.sel(member=m, ensemble=e))
    truth = xr.concat(truth_list, supervector_dim)
    fct = xr.concat(fct_list, supervector_dim)

    return fct, truth


def e2c(ds, supervector_dim, control_member=0):
    """
    Create two supervectors to compare ensemble mean to control.
    """
    truth = ds.sel(member=control_member)
    truth = truth.rename({'ensemble': supervector_dim})
    # drop the member being truth
    ds = drop_members(ds, rmd_member=[control_member])
    fct = ds.mean('member')
    fct = fct.rename({'ensemble': supervector_dim})
    return fct, truth


def ensmean_against_control(ds, control_member=0):
    """
    Add description here.
    """
    # drop the member being truth
    truth = ds.sel(member=control_member)
    ds = drop_members(ds, rmd_member=[control_member])
    return ((ds.mean('member') - truth)**2).mean('ensemble')


def compute(ds, control, metric=pearson_r, comparison=m2m, anomaly=False, 
            detrend=False, running=None, varname=None, time_dim='year'):
    """
    Add description here.
    """
    supervector_dim = 'svd'
    if anomaly:
        _ds = ds - control.mean(time_dim)
        _control = control - control.mean(time_dim)
    else:
        _ds = ds
        _control = control
    if detrend:
        s, i, _, _, _ = et.stats.xr_linregress(_control, time_dim)
        _control = _control - \
            (s * _control[time_dim] - _control[time_dim].values[0])
        _ds = _ds - (s * _ds[time_dim] - _ds[time_dim].values[0])

    if comparison.__name__ not in ['m2m', 'm2c', 'm2e', 'e2c']:
        raise ValueError('specify comparison argument')

    if metric.__name__ in ['pearson_r', 'rmse']:
        fct, truth = comparison(_ds, supervector_dim)
        res = metric(fct, truth, dim=supervector_dim)
        res[time_dim]=np.arange(1,res[time_dim].size+1)
        return res
    elif metric.__name__ in ['mse', 'rmse_v', 'nrmse', 'nev', 'ppp', 'PPP', 'MSSS', 'uACC']:
        res = metric(ds, control, comparison, running)
        res[time_dim]=np.arange(1,res[time_dim].size+1)
        return res
    else:
        raise ValueError('specify metric argument')


def get_variance(control, running=None, time_dim='year'):
    """
    Get running variance.
    """
    if isinstance(running, int):
        var = control.rolling(year=running).var().mean(time_dim)
    else:
        var = control.var(time_dim)
    return var


def choose_comparison(ds, comparison):
    """
    Choose comparison for any mse-style metric.
    """
    comparison_name = comparison.__name__
    if comparison_name is 'm2e':
        return ens_var_against_mean(ds)
    if comparison_name is 'm2c':
        return ens_var_against_control(ds)
    if comparison_name is 'm2m':
        return ens_var_against_every(ds)
    if comparison_name is 'e2c':
        return ensmean_against_control(ds)


def get_norm_factor(comparison):
    """
    Get normalization factor for ppp, nvar, nrmse.
    """
    comparison_name = comparison.__name__
    if comparison_name is 'm2e':
        return 1
    if comparison_name in ['m2c', 'm2m', 'e2c']:
        return 2


def mse(ds, control, comparison, running):
    """
    Mean Square Error (MSE) metric.
    """
    return choose_comparison(ds, comparison)


def rmse_v(ds, control, comparison, running):
    """
    Root Mean Square Error (RMSE) metric.
    """
    return choose_comparison(ds, comparison) ** .5


def nrmse(ds, control, comparison, running):
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
    var = get_variance(control, running=running)
    ens = choose_comparison(ds, comparison)
    fac = get_norm_factor(comparison)
    return (ens / var / fac) ** .5


def nev(ds, control, comparison, running):
    """
    Normalized Ensemble Variance (NEV) metric.

    Reference
    ---------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North
      Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
      (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    """
    var = get_variance(control, running=running)
    ens = choose_comparison(ds, comparison)
    fac = get_norm_factor(comparison)
    return ens / var / fac


def ppp(ds, control, comparison, running):
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
    var = get_variance(control, running=running)
    ens = choose_comparison(ds, comparison)
    fac = get_norm_factor(comparison)
    return 1 - ens / var / fac


def PPP(ds, control, comparison, running):
    """
    Wraps ppp.
    """
    return ppp(ds, control, comparison, running)


def uACC(ds, control, comparison, running):
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
    return ppp(ds, control, comparison, running) ** .5


def MSSS(ds, control, comparison, running):
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
    return ppp(ds, control, comparison, running)


def PM_sig_fast(ds, control, metric=rmse, comparison=m2m, sig=95, bootstrap=10, time_dim='year'):
    """
    Return sig-th percentile of function to be choosen from pseudo ensemble generated from control.

    Parameters
    ----------
    control : xr.DataArray/Dataset with year dimension
        input control data
    ds : xr.DataArray/Dataset with year, ensemble and member dimensions
        input ensemble data
    sig: int or list
        Significance level for bootstrapping from pseudo ensemble
    bootstrap: int
        number of iterations

    Returns
    -------
    sig_level : xr.DataArray/Dataset as inputs
        significance level without year, ensemble and member dimensions
        as many sig_level as listitems in sig

    """
    ds_pseudo = pseudo_ens_fast(ds, control, bootstrap=bootstrap)
    ds_pseudo_metric = compute(
        ds_pseudo, control, metric=metric, comparison=comparison)
    if isinstance(sig, list):
        qsig = [x / 100 for x in sig]
    else:
        qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(q=qsig / 100, dim=time_dim)
    return sig_level


def control_for_reference_period(control, reference_period='MK', obs_years=40, time_dim='year'):
    """
    Modifies control according to knowledge approach, see Hawkins 2016.
    """
    if reference_period == 'MK':
        _control = control
    elif reference_period == 'OP_full_length':
        _control = control - \
            control.rolling(year=obs_years, min_periods=1,
                             center=True).mean() + control.mean(time_dim)
    elif reference_period == 'OP':
        raise ValueError('not yet implemented')
    else:
        raise ValueError("choose a reference period")
    return _control


def PM_sig(ds, control, metric=rmse, comparison=m2m, reference_period='MK', sig=95, bootstrap=30, time_dim='year'):
    """
    Return sig-th percentile of function to be choosen from pseudo ensemble generated from control.

    Parameters
    ----------
    control : xr.DataArray/Dataset with year dimension
        input control data
    ds : xr.DataArray/Dataset with year, ensemble and member dimensions
        input ensemble data
    sig: int or list
        Significance level for bootstrapping from pseudo ensemble
    bootstrap: int
        number of iterations

    Returns
    -------
    sig_level : xr.DataArray/Dataset as inputs
        significance level without year, ensemble and member dimensions
        as many sig_level as listitems in sig

    """
    x = []
    _control = control_for_reference_period(
        control, reference_period=reference_period)
    for _ in range(1 + int(bootstrap / ds.year.size)):
        ds_pseudo = pseudo_ens(ds, _control)
        ds_pseudo_metric = compute(
            ds_pseudo, _control, metric=metric, comparison=comparison)
        x.append(ds_pseudo_metric)
    ds_pseudo_metric = xr.concat(x, dim='it')
    if isinstance(sig, list):
        qsig = [x / 100 for x in sig]
    else:
        qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(q=qsig, dim=[time_dim, 'it'])
    return sig_level


def get_predictability_horizon(s, threshold):
    """
    Get predictability horizon of series from threshold value.
    """
    first_index = s.index[0]
    ph = (s > threshold).idxmin() - first_index
    return ph


def vectorized_predictability_horizon(ds, threshold, limit='upper', time_dim='year'):
    """Get predictability horizons of dataset form threshold dataset."""
    if limit is 'upper':
        return (ds > threshold).argmin(time_dim)
    if limit is 'lower':
        return (ds < threshold).argmin(time_dim)


def running_trend(df, varname, dim=None, window=10, timestamp_location='middle', rename=True):
    """
    Calculate running trend of a pd.DataFrame.

    Requires pyfinance (NOTE: We should remove this dependency)
    """
    if dim is None:
        x = df.index
    result = ols.PandasRollingOLS(y=df[varname], x=x, window=10).beta
    if timestamp_location is 'middle':
        result.index = result.index - int(window / 2)
    if timestamp_location is 'first':
        result.index = result.index - int(window - 1)
    if timestamp_location is 'last':
        pass
    if rename:
        result = result.rename(
            columns={'feature1': '_'.join((varname, 'trend', str(window)))})
    else:
        result = result.rename(columns={'feature1': varname})
    return result


def trend_over_numeric_varnames(df, **kwargs):
    """
    Calc running trend over numeric variables.
    """
    list = []
    for col in df.columns:
        if np.issubdtype(df[col], np.number):
            list.append(running_trend(df, col, **kwargs))
    all_column_trends = pd.concat(list, axis=1)
    return all_column_trends


def normalize(ds, dim=None):
    """
    Add description.
    """
    return (ds - ds.mean(dim)) / ds.std(dim)


def get_anomalous_states(ds, control, threshold=1, varname=varname, area=area, 
                         period=period):
    """
    Get ensemble starting years with anomalous states.
    """
    s = control.sel(area=area, period=period).to_dataframe()[varname]
    snorm = normalize(s)
    ensemble_starting_years = ds.ensemble.values
    shift = 1101  # index shift 1900 to 3000
    starting_states = snorm.loc[ensemble_starting_years - shift]
    anomalous_states = starting_states[starting_states > threshold].index
    return anomalous_states + shift


def compute_anomalous(ds, control, varname='sos', area='North_Atlantic', 
                      period='ym', threshold=1, bootstrap=50, sig=99, 
                      metric=rmse, comparison=m2e):
    """
    Compute skill score for anomalous start years.
    """
    anomalous_years = get_anomalous_states(
        ds, control, varname=varname, area=area, threshold=threshold)
    ds_anomalous = select_members_ensembles(
        ds, e=anomalous_years)
    threshold_anomalous = PM_sig(
        control, ds_anomalous, metric=metric, bootstrap=bootstrap, sig=sig)
    skill_anomalous = compute(ds_anomalous, control, metric=metric, comparison=comparison)
    return skill_anomalous, threshold_anomalous


# Persistence
def persistence_forecast(ds, control, varname=varname, area=area, period=period, 
                         comparison=m2e, time_dim='year'):
    """
    Generate persistence forecast timeseries.
    """
    starting_years = [x - 1100 - 1 for x in ds.ensemble.values]
    anom = (control.sel(year=starting_years) - control.mean())
    t = np.arange(1, ds.year.size + 1)
    tds = np.arange(1900, 1900 + ds.year.size)
    alpha = control.to_series().autocorr()
    persistence_forecast_list = []
    for year in anom.year:
        ar1 = anom.sel(year=year).values * \
            np.exp(-alpha * t) + control.mean().values
        pf = xr.DataArray(data=ar1, coords=[tds], dims=time_dim)
        pf = pf.expand_dims('ensemble')
        pf['ensemble'] = [year + 1100 + 1]
        persistence_forecast_list.append(pf)
    return xr.concat(persistence_forecast_list, dim='ensemble')


def compute_persistence(ds, control, metric=rmse, comparison=m2e, time_dim='year'):
    """
    Compute skill for persistence forecast.
    """
    persistence_forecasts = persistence_forecast(ds, control)
    if comparison.__name__ == 'm2e':
        result = metric(persistence_forecasts, ds.mean('member'), 'ensemble')
    elif comparison.__name__ == 'm2m':
        persistence_forecasts = persistence_forecasts.expand_dims('member')
        all_persistence_forecasts = persistence_forecasts.sel(
            member=[0] * ds.member.size)
        fct = m2e(all_persistence_forecasts, 'svd')[0]
        truth = m2e(ds, 'svd')[0]
        result = metric(fct, truth, 'svd')
    else:
        raise ValueError('not defined')
    result[time_dim] = np.arange(1, ds[time_dim].size + 1)
    return result


def generate_predictability_persistence(s, kind='PPP', percentile=True, length=20):
    """
    Calculate the PPP (or NEV) damped persistence mean and range.

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
        determine kind of damped persistence. 'PPP' or 'NEV' (normalized ensemlbe variance)
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
        random_start_year = randint(1900, 2200 - chunk_length)
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


def generate_damped_persistence_forecast(control, startyear, length=20):
    """
    Generate damped persistence forecast.

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
    ar1, ar50, ar90 = et.prediction.generate_damped_persistence_forecast(
        s,1919)
    ar1.plot(label='damped persistence forecast')
    plt.fill_between(ar1.index,ar1-ar50,ar1+ar50,alpha=.2,
                     color='gray',label='50% forecast range')
    plt.fill_between(ar1.index,ar1-ar90,ar1+ar90,alpha=.1,
                     color='gray',label='90% forecast range')
    s.loc[1919:1939].plot(label='control')
    plt.legend()

    """
    anom = (control.loc[startyear] - control.mean())
    t = np.arange(0., length + 1, 1)
    alpha = control.autocorr()
    exp = anom * np.exp(-alpha * t)  # exp. decay towards mean
    ar1 = exp + control.mean()
    ar50 = 0.7 * control.std() * np.sqrt(1 - np.exp(-2 * alpha * t))
    ar90 = 1.7 * control.std() * np.sqrt(1 - np.exp(-2 * alpha * t))

    index = control.loc[startyear:startyear + length].index
    ar1 = pd.Series(ar1, index=index)
    ar50 = pd.Series(ar50, index=index)
    ar90 = pd.Series(ar90, index=index)
    return ar1, ar50, ar90


# utils for xr.Datasets
def drop_ensembles(ds, rmd_ensemble=[0]):
    """Drop ensembles from ds."""
    if all(ens in ds.ensemble.values for ens in rmd_ensemble):
        ensemble_list = list(ds.ensemble.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select from ensemble starting years', rmd_ensemble)
    return ds.sel(ensemble=ensemble_list)


def drop_members(ds, rmd_member=[0]):
    """Drop members from ds."""
    if all(m in ds.member.values for m in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise ValueError('select availbale from members', rmd_member)
    return ds.sel(member=member_list)


def select_members_ensembles(ds, m=None, e=None):
    """Subselect ensembles and members from ds."""
    if m is None:
        m = ds.member.values
    if e is None:
        e = ds.ensemble.values
    return ds.sel(member=m, ensemble=e)
