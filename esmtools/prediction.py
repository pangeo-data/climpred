import os

import esmtools as et
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator
from six.moves.urllib.request import urlopen, urlretrieve

"""Objects dealing with prediction metrics. In particular, these objects are specific to decadal prediction -- skill, persistence forecasting, etc and perfect-model predictability --  etc.

ToDos
-----
- add missing functions
- how to treat for 1D timeseries and 3D maps?
    - different functions
    - arguments
    - generic
- variable names and file setup:
    - fgco2 (cmorized) or co2_flux (MPI) or (CESM)
    - config file?
    - global keywords?


Structure
---------
- General metrics
    - Mean Square Error (MSE) = Mean Square Difference (MSD) (missing)
    - Relative Entropy (Kleeman 2002; Branstator and Teng 2010) (missing)
    - anomlay correlation coefficient (ACC) (missing)

- Decadal prediction metrics
    - anomlay correlation coefficient (ACC) (missing)
    - requires: hindcast simulations: ensembles initialised each year

- Perfect-model (PM) predictability metrics
    - ensemble variance against: (=MSE)
        - ensemble mean
        - control
        - each member
    - normalized ensemble variance (NEV) (Griffies) (missing)
    - prognostic potential predictability (PPP) (Pohlmann 2004)
    - PM anomlay correlation coefficient (ACC) (Bushuk 2018)
    - PM mean square skill score (MSSS)
    - unbiased ACC (Bushuk 2018)
    - root mean square error (RMSE) (=MSE^0.5) (missing)
    - Diagnostic Potential Predictability (DPP) (Boer 2004, Resplandy 2015/Seferian 2018)
    - predictability horizon:
        - linear breakpoint fit (Seferian 2018) (missing)
        - f-test significant test (Pohlmann 2004, Griffies 1997) (missing)
        - bootstrapping limit
    - root mean square error (RMSE) (=MSE^0.5) (missing)
    - normalized root mean square error (NRMSE) (=1-MSE^0.5/RMSE_control) (missing)
    - Diagnostic Potential Predictability (DPP)
    - Relative Entropy (Kleeman 2002; Branstator and Teng 2010) (missing)
    - Mutual information (DelSole) (missing)
    - Average Predictability Time (APT) (DelSole) (missing)
    - requires: ensembles at different start years from control run

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

Example ds via et.prediction.load_dataset('PM_MPI-ESM-LR_ds'):
ds
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

Example:
ds
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
    """Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    # copied from seaborn
    from bs4 import BeautifulSoup
    http = urlopen(
        'https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction/')
    #print('Load from URL:', http)
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
    #print('Load from URL:', full_path)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = xr.open_dataset(full_path, **kws)

    return df


def ds2df(ds, area=area, varname=varname, period=period):
    """
    Take a dataset, selects wanted variable, area, period and transforms it into a dataframe.

    Parameters
    ----------
    ds : Dataset
        Input data

    Returns
    -------
    c : DataFrame
        Output data as df

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    PM_MSSS = et.prediction.PM_MSSS(ds,control)
    et.prediction.ds2df(PM_MSSS).plot()

    """
    df = ds.sel(area=area, period=period).to_dataframe()[varname].unstack().T
    return df


# Diagnostic Potential Predictability (DPP)

def chunking(ds, number_chunks=False, chunk_length=False, output=False):
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
    output : boolean (optional)
        Debugging prints

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
    control_chunked_into_30yr_chunks = et.prediction.chunking(control,chunk_length=30)
    control_chunked_into_30_chunks = et.prediction.chunking(control,number_chunks=30)

    """
    if number_chunks and not chunk_length:
        chunk_length = np.floor(ds.year.size / number_chunks)
        cmin = int(ds.year.min())
        cmax = int(ds.year.max())
    elif not number_chunks and chunk_length:
        cmin = int(ds.year.min())
        cmax = int(ds.year.max())
        number_chunks = int(np.floor(ds.year.size / chunk_length))
    else:
        raise ValueError('set number_chunks or chunk_length to True')

    if output:
        print(number_chunks, 'chunks of length',
              chunk_length, 'from', cmin, 'to', cmax)
        print('0', cmin, cmin + chunk_length - 1)
    c = ds.sel(year=slice(cmin, cmin + chunk_length - 1))
    c = c.expand_dims('c')
    c['c'] = [0]
    year = c.year
    for i in range(1, number_chunks):
        if output:
            print(i, cmin + chunk_length * i,
                  cmin + (i + 1) * chunk_length - 1)
        c2 = ds.sel(year=slice(cmin + chunk_length * i,
                               cmin + (i + 1) * chunk_length - 1))
        c2 = c2.expand_dims('c')
        c2['c'] = [i]
        c2['year'] = year
        c = xr.concat([c, c2], 'c')
    return c


def DPP(ds, m=10, chunk=True, var_all_e=False, return_s=False, output=False):
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
    return_s : boolean (optional)
        decide whether to return also intermediate results
    output : boolean (optional)
        Debugging prints

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
    if ds.size > 5000:  # dirty way of figuring out which data
        data3D = True
        print('3D data')
    else:
        data3D = False
    if output:
        print(m, ds.dims, chunk)

    if not chunk:
        s2v = ds.rolling(year=m, min_periods=1, center=True).mean().var('year')
        s2e = (ds - ds.rolling(year=m, min_periods=1,
                               center=True).mean()).var('year')
        s2 = s2v + s2e

    if chunk:
        # first chunk
        chunked_means = chunking(ds, chunk_length=m).mean('year')
        # sub means in chunks
        chunked_deviations = chunking(ds, chunk_length=m) - chunked_means

        s2v = chunked_means.var('c')
        if var_all_e:
            s2e = chunked_deviations.var(['year', 'c'])
        else:
            s2e = chunked_deviations.var('year').mean('c')
        s2 = s2v + s2e

    DPP = (s2v - s2 / (m)) / (s2)

    if output:
        print(DPP, s2v, s2e, s2)

    if data3D:
        return DPP
    if not return_s:
        return DPP
    if return_s:
        return DPP, s2v, s2e, s2


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
    return ds.var('member')


def ens_var_against_control(ds):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance is
    evaluated against member=0 which is the control run.
    """
    var = ds.copy()
    var = ((ds - ds.sel(member=0))**2).sum('member') / (ds.member.size - 0)
    return var


def ens_var_against_every(ds):
    """
    See ens_var_against_mean(ds).

    Only difference is that now distance
    is evaluated against each ensemble member and then averaged.
    """
    var = ds.copy()
    for i in range(0, ds.member.size):
        var_a = ((ds - ds.sel(member=i))**2).sum(dim='member') / ds.member.size
        var = xr.concat([var, var_a], 'member')
    var = var.sel(member=slice(ds.member.size, 2
                               * ds.member.size)).mean('member')
    return var


def rmse(ds, kind='mean'):
    """Calculate root-mean-square-error (RMSE)."""
    if kind == 'mean':
        ens_var = ens_var_against_mean(ds)
    elif kind == 'control':
        ens_var = ens_var_against_control(ds)
    elif kind == 'every':
        ens_var = ens_var_against_every(ds)
    else:
        raise ValueError('Select kind from ["mean","control","every"].')
    return ens_var**.5


def normalize_var(var, control, fac=1, running=True, m=20):
    """
    Normalize the ensemble spread with the temporal spread of the control run.

    Note 1: Ensemble spread against ensemble mean is half the ensemble spread any member.
    Note 2: Which variance should be normalized against?
            running=False evaluates against the variance of the whole temporal domain whereas
            running=True evaluates against a running variance

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
        Input data
    fac : int
        factor for ensemble spread (2 for ensemble variance against every/control, 1 for mean)
    running : boolean
    m : int
        if running, then this marks the time window in years for the variance calc

    Returns
    -------
    c : DataArray as ds
        Output data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    nens_var_against_mean = et.prediction.normalize_var(ens_var_against_mean,control)

    ens_var_against_control = et.prediction.ens_var_against_control(ds)
    nens_var_against_mean = et.prediction.normalize_var(ens_var_against_control,control,fac=2)

    # dataframe # works nice for many variances in dataframe view
    def normalize_var(var,fac=1,running=True):
        if running:
            return (var.stack(level=0).stack(level=0).stack(level=1).to_xarray()/control_var_running/fac).to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)
        else:
            return (var.stack(level=0).stack(level=0).stack(level=1).to_xarray()/control_var/fac).to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)
    var_mean = et.prediction.ens_var_against_mean(ds)..to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)
    nvar_mean = normalize_var(var_mean)

    """
    if not running:
        control_var = control.var('year')  # .mean('year')
        var2 = var / control_var / fac
        return var2
    if running:
        control_var_running = control.rolling(year=m).var().mean('year')
        var2 = var / control_var_running / fac
        return var2


def PPP_from_nvar(nvar):
    """
    Calculate Prognostic Potential Predictability (PPP) as in Pohlmann 2004 or Griffies 1997.

    References
    ----------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated
        North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8
        (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    - Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
        Input data

    Returns
    -------
    c : DataArray
        Output data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    nens_var_against_mean = et.prediction.normalize_var(ens_var_against_mean,control)
    PPP_mean = et.prediction.PPP_from_nvar(nens_var_against_mean)

    """
    return 1 - nvar


# Perfect-model (PM) predictability scores from Bushuk 2018

def PM_MSSS(ds, control, kind='', running=True, m=20):
    """
    Calculate the perfect-model (PM) mean square skill score (MSSS). It is identical to Prognostic Potential Predictability (PPP) in Pohlmann et al. (2004).

    Formula
    -------
    MSSS_{PM} = 1 - MSE/sigma_c

    References
    ----------
    - Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin
        Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a
        Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004):
        4463–72. https://doi.org/10/d2qf62.

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
        Input ensemble data
    control : DataArray with year dimension (optional spatial coordinates)
        Input control run data
    kind : string
        how to calculate ensemble variance: against ["mean","control","every"]
    running : boolean
        if true against running m-yr variance
    m : int
        see running

    Returns
    -------
    msss : DataArray
        Output data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    pm_msss = PM_MSSS(ds,control,kind='control',running=True,m=30)

    """
    if kind == 'mean':
        ens_var = ens_var_against_mean(ds)
        fac = 1
    elif kind == 'control':
        ens_var = ens_var_against_control(ds)
        fac = 2
    elif kind == 'every':
        ens_var = ens_var_against_every(ds)
        fac = 2
    else:
        raise ValueError('Select kind from ["mean","control","every"].')
    nens_var_against_mean = normalize_var(
        ens_var, control, running=running, m=m, fac=fac)
    msss = PPP_from_nvar(nens_var_against_mean)
    return msss


def PM_NRMSE(ds, control, kind='', running=True, m=20):
    """
    Calculate the perfect-model (PM) normalised root mean square error as in Hawkins et al. (2016) or NRMSE+1 in Bushuk et al. (2018).

    Formula
    -------
    NRMSE = 1 - RMSE_ens/std_c

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

    Parameters
    ----------
    ds : DataArray with year dimension (optional spatial coordinates)
        Input ensemble data
    control : DataArray with year dimension (optional spatial coordinates)
        Input control run data
    kind : string
        how to calculate ensemble variance: against ["mean","control","every"]
    running : boolean
        if true against running m-yr variance
    m : int
        see running

    Returns
    -------
    nrmse : DataArray
        Output data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    pm_nrmse = et.prediction.PM_NRMSE(ds,control,kind='control',running=True,m=30)

    """
    if kind == 'mean':
        ens_var = ens_var_against_mean(ds)
        fac = 1
    elif kind == 'control':
        ens_var = ens_var_against_control(ds)
        fac = 2
    elif kind == 'every':
        ens_var = ens_var_against_every(ds)
        fac = 2
    else:
        raise ValueError('Select kind from ["mean","control","every"].')
    nens_var_against_mean = normalize_var(
        ens_var, control, running=running, m=m, fac=fac)
    nrmse = PPP_from_nvar(nens_var_against_mean**.5)
    return nrmse


def PM_ACC_U(msss):
    """
    Calculate the perfect-model (PM) unbiased anomaly correlation coefficient as in Bushuk et al. 2018.

    Formula
    -------
    ACC_{U} = sqrt{MSSS} = sqrt{PPP} or sqrt{NRMSE}

    References
    ----------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice Prediction: Potential versus Operational Seasonal Forecast Skill.” Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.

    Parameters
    ----------
    (TODO: Test whether this works for 3D data)
    msss : DataArray with year dimension (optional spatial coordinates)
        Input msss data

    Returns
    -------
    ACC_U : DataArray
        Output ACC_U data

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    pm_msss = PM_MSSS(ds,control,kind='mean')
    pm_acc_u = PM_ACC_U(pm_msss)

    """
    return msss ** .5


def PM_ACC(ds, control, anomaly=True, varname=varname, area=area, period=period,
           ens=False, control_member=0, m=False, against='mean'):
    """
    Calculate the perfect-model (PM) anomaly correlation coefficient as in Bushuk et al. 2018.

    Create a supervectors (dims=(N*M,length)) for ensemble and observations
    (each member at the turn becomes obs). Returns M ACC timeseries.

    Formula
    -------
    ACC = pd.corrwith()

    References
    ----------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice Prediction: Potential versus Operational Seasonal Forecast Skill.” Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.

    Parameters
    ----------
    (TODO: Test whether this works for 3D data)
    ds : xr.DataArray with year dimension (optional spatial coordinates)
        Input ensemble data
    anomaly: bool
        create anomaly

    Returns
    -------
    ACC : pd.DataArray
        ACC

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    varname='tos'
    period='ym'
    area='North_Atlantic'
    pm_acc = et.prediction.PM_ACC(ds.sel(area=area,period=period),control)
    pm_acc.plot()

    """
    if anomaly:
        ds = ds - control.mean('year')
    if (m is not False) and control_member in m:  # if control_member is in m combination
        return pd.Series([np.nan] * 12)
    # if (ens != False) and len(ens)==1: # if single ens, somehow gives near 0 ACC
    #    return pd.Series([np.nan])
    else:
        sv = ds.sel(area=area, period=period).to_dataframe()[varname].unstack()
        obs = ds.sel(area=area, period=period).to_dataframe()[
            varname].unstack()
        if against not in ['every', 'mean_every']:
            sv = sv.T.reorder_levels([1, 0], axis=1).drop(
                columns=control_member).reorder_levels([1, 0], axis=1).T
            obs = obs.T.reorder_levels([1, 0], axis=1)[control_member].T

        # subselections
        if ens and not m:
            sv = sv.T[ens].T  # fewer ensembles
            obs = obs.T[ens].T  # fewer ensembles
        elif m and not ens:
            sv = sv.T.reorder_levels([1, 0], axis=1)[m].reorder_levels(
                [1, 0], axis=1).sortlevel(axis=1).T  # fewer members
            obs = obs
        elif not m and not ens:
            sv = sv
            obs = obs
        elif m and ens:
            sv = sv.T[ens].reorder_levels([1, 0], axis=1)[m].reorder_levels([
                1, 0], axis=1).sortlevel(axis=1).T
            obs = obs.T[ens].T

        # how to compute ACC: compare forecast against ...
        if against == 'mean':  # correlation control member against ensemble mean
            sv = sv.mean(axis=0, level=0)
            return sv.corrwith(obs)
        elif against == 'every':  # correlation of every member against every member
            obsl = []  # create larger supervector with all members being once truth
            svl = []
            member = list(sv.index.get_level_values(level=1).unique().values)
            for i in member:
                d = sv.T.reorder_levels([1, 0], axis=1)[i].T
                members_left = list(
                    sv.index.get_level_values(level=1).unique().values)
                members_left.remove(i)
                obsl.append(
                    pd.concat([d] * len(members_left), keys=members_left))
                svl.append(sv.T.reorder_levels([1, 0], axis=1).drop(
                    columns=i).reorder_levels([1, 0], axis=1).T)
            SV = pd.concat(svl).sort_index()
            OBS = pd.concat(obsl).reorder_levels([1, 0], axis=0).sort_index()
            ACC = SV.corrwith(OBS)
            return ACC
        elif against == 'mean_every':  # ensemble mean against ACC_every
            obsl = []  # create larger supervector with all members being once truth
            svl = []
            member = list(sv.index.get_level_values(level=1).unique().values)
            for i in member:
                d = sv.T.reorder_levels([1, 0], axis=1)[i].T
                obsl.append(d)
                svll = sv.T.reorder_levels([1, 0], axis=1).drop(
                    columns=i).reorder_levels([1, 0], axis=1).T
                svl.append(svll.mean(axis=0, level=0))
            SV = pd.concat(svl)
            OBS = pd.concat(obsl)
            ACC = SV.corrwith(OBS)
            return ACC
        elif against == 'control':  # correlation each member against control member
            ACC = sv.corrwith(obs)
            return ACC
        else:
            raise ValueError('Specify ["mean","every","control"]')

# T test Bushuk


def pseudo_ens(control, ds, varname=varname, period=period, area=area, nens=12, nm=10):
    """
    Create a pseudo-ensemble from control run.

    Needed for bootstrapping confidence intervals of a metric.
    Takes randomly 20yr segments from control and rearranges them into ensemble
    and member dimensions.

    Parameters
    ----------
    control : xr.DataArray with year dimension
        Input ensemble data
    nens: int
        Number of start dates for pseudo ensemble
    nm: int
        Number of ensemble members per start date for pseudo ensemble
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
    ds_c = control.copy()
    length = ds.year.size
    c_start = control.year[0]
    c_end = control.year[-1]
    elist = []
    year = ds.year
    for j in range(nens):
        mlist = []
        for i in range(nm):
            start = np.random.randint(c_start, c_end - length - 1)
            random_ds = ds_c.sel(year=slice(start, start + length - 1))
            random_ds['year'] = year
            random_ds.expand_dims('member')
            random_ds['member'] = i
            mlist.append(random_ds)
        ds_m = xr.concat(mlist, 'member')
        ds_m.expand_dims('ensemble')
        ds_m['ensemble'] = j
        elist.append(ds_m)
    ds_e = xr.concat(elist, 'ensemble')
    return ds_e


def PM_sig(control, ds, func='ACC', sig=95, it=50, **kwargs):
    """
    Return sig-th percentile of function to be choosen from pseudo ensemble generated from control.

    Parameters
    ----------
    control : xr.DataArray with year dimension
        Input ensemble data
    sig: int
        Significance level for bootstrapping from pseudo ensemble
    it: int
        number of iterations for ACC(pseudo_ens)

    Returns
    -------
    sig_level : float
        significance level value

    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    varname='tos'
    period='ym'
    area='North_Atlantic'
    print(sig,'% significance level at',et.prediction.PM_ACC_sig(control,ds,func='ACC',sig=sig))

    """
    from tqdm import trange
    x = []
    for i in trange(it):
        ds_e = pseudo_ens(control, ds, nens=ds.ensemble.size,
                          nm=ds.member.size)

        if func == 'ACC_control':
            x.append(PM_ACC(ds_e, control, against='control').values)

        elif func == 'ACC_mean':
            x.append(PM_ACC(ds_e, control, against='mean').values)

        elif func == 'ACC_every':
            x.append(PM_ACC(ds_e, control, against='every').values)

        elif func == 'ACC_mean_every':
            x.append(PM_ACC(ds_e, control, against='mean_every').values)

        elif func == 'ACC_U_control':
            ps_ACC_U = PM_ACC_U(PM_NRMSE(ds_e, control, kind='control'))
            x.append(ds2df(ps_ACC_U).values)

        elif func == 'ACC_U_mean':
            ps_ACC_U = PM_ACC_U(PM_NRMSE(ds_e, control, kind='mean'))
            x.append(ds2df(ps_ACC_U).values)

        elif func == 'PPP_mean':
            ps_PPP = PM_MSSS(ds_e, control, kind='mean', **kwargs)
            # x.append(ds2df(ps_PPP).values)
            x.append(ps_PPP)

        elif func == 'PPP_control':
            ps_PPP = PM_MSSS(ds_e, control, kind='control', **kwargs)
            x.append(ds2df(ps_PPP).values)

        elif func == 'NRMSE_mean':
            ps_PPP = PM_NRMSE(ds_e, control, kind='mean', **kwargs)
            x.append(ds2df(ps_PPP).values)

        elif func == 'NRMSE_control':
            ps_PPP = PM_NRMSE(ds_e, control, kind='control', **kwargs)
            x.append(ds2df(ps_PPP).values)

        else:
            raise ValueError('please select proper func string')
    iteration_dim_name = 'it'
    x2 = xr.concat(x, dim=iteration_dim_name)
    sig_level = x2.quantile(
        q=sig / 100, dim=[iteration_dim_name, 'year', 'ensemble'], keep_attrs=True)
    return sig_level


def set_integer_xaxis(ax=False):
    if not ax:
        ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def qplot(test, threshold, period=period, area=area):
    fig, ax = plt.subplots(figsize=(8, 5))
    test['year'] = np.arange(1, 21)
    test = test.sel(area=area, period=period)[varname]
    test.to_dataframe()[
        varname].unstack().plot(ax=ax, title=period + ' ' + area, label='nolegend')
    test.mean('ensemble').to_dataframe()[varname].plot(ax=ax, c='k', lw=3)
    ax.legend(ncol=3)
    ax.set_ylim([-.5, 1])
    threshold_here = threshold.sel(area=area, period=period)[varname]
    ax.axhline(y=threshold_here, c='k', ls='--', alpha=.2)
    ax.axvline(x=vectorized_predictability_horizon(
        test.mean('ensemble'), threshold_here), c='gray', lw=3, ls='-.')
    set_integer_xaxis(ax)
    ax.set_xticks(test.year.values)


#qplot(test, threshold)


# vectorized_predictability_horizon(test, threshold).to_dataframe()[
#    varname].unstack()[period].unstack().T
# _.mean(axis=0)


def get_predictability_horizon(s, threshold):
    """Get predictability horizon of series from threshold value."""
    first_index = s.index[0]
    ph = (s > threshold).idxmin() - first_index - 1
    return ph


def vectorized_predictability_horizon(ds, threshold, dim='year'):
    """Get predictability horizons of dataset form threshold dataset."""
    return (ds > threshold).argmin('year')

# Persistence

# damped persistence forecast based on lag1 autocorrelation


def df_autocorr(df, lag=1, axis=0):
    """Compute full-sample column-wise autocorrelation for a pandas DataFrame."""
    return df.apply(lambda col: col.autocorr(lag), axis=axis)


def calc_tau(alpha):
    """
    Calculate the decorrelation time tau.

    Reference
    ---------
    Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate Research. Cambridge ; New York: Cambridge University Press, 1999., page 373

    Parameters
    ----------
    alpha : float
        lag1 autocorrelation coefficient
    Returns
    -------
    tau : float
        decorrelation time

    Example
    -------
    import esmtools as et
    alpha = .8
    tau = et.prediction.calc_tau(alpha)

    """
    return (1 + alpha) / (1 - alpha)


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
    #s = pd.Series(np.sin(range(1000)+np.cos(range(1000))+np.random.randint(.1,1000)))
    s = control.sel(area=area,period=period).to_dataframe()[varname]
    PPP_persistence_0, PPP_persistence_minus, PPP_persistence_plus = et.prediction.generate_predictability_persistence(s)
    t = np.arange(0,20+1,1.)
    plt.plot(PPP_persistence_0,color='black',linestyle='--',label='persistence mean')
    plt.fill_between(t,PPP_persistence_minus,PPP_persistence_plus,color='gray',alpha=.3,label='persistence range')
    plt.axhline(y=0,color='black')

    """
    # bootstrapping persistence
    iterations = 50  # iterations
    chunk_length = 100  # length of chunks of control run to take lag1 autocorr
    data = np.zeros(iterations)
    from random import randint
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
        return np.exp(-2 * alpha * t)  # Griffies 1997
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
    ar1, ar50, ar90 = et.prediction.generate_damped_persistence_forecast(s,1919)
    ar1.plot(label='damped persistence forecast')
    plt.fill_between(ar1.index,ar1-ar50,ar1+ar50,alpha=.2,color='gray',label='50% forecast range')
    plt.fill_between(ar1.index,ar1-ar90,ar1+ar90,alpha=.1,color='gray',label='90% forecast range')
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
    if all(ens in ds.ensemble.values for ens in rmd_ensemble):
        ensemble_list = list(ds.ensemble.values)
        for ens in rmd_ensemble:
            ensemble_list.remove(ens)
    else:
        raise ValueError('select from ensemble starting years', rmd_ensemble)
    return ds.sel(ensemble=ensemble_list)


def drop_members(ds, rmd_member=[0]):
    if all(ens in ds.member.values for ens in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise ValueError('select availbale from members', rmd_member)
    return ds.sel(member=member_list)


def select_members_ensembles(ds, m=None, e=None):
    if m is None:
        m = ds.member.values
    if e is None:
        e = ds.ensemble.values
    return ds.sel(member=m, ensemble=e)
