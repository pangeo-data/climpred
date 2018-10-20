"""
Objects dealing with prediction metrics. In particular, these objects are specific to decadal prediction -- skill, persistence forecasting, etc and perfect-model predictability --  etc.

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
    - normalized ensemble variance (NEV) (Griffies)
    - prognostic potential predictability (PPP) (Pohlmann 2004)
    - PM anomlay correlation coefficient (ACC) (Bushuk 2018) (missing)
    - PM  mean square skill score (MSSS) (Bushuk 2018)
    - unbiased ACC (Bushuk 2018)
    - root mean square error (RMSE) (=MSE^0.5) (missing)
    - Diagnostic Potential Predictability (DPP) (Boer 2004, Resplandy 2015/Seferian 2018)
    - predictability horizon: (missing)
        - linear breakpoint fit (Seferian 2018)
        - f-test significant test (Pohlmann 2004, Griffies 1997)
        - bootstrapping limit
    - normalized ensemble variance (NEV)
    - prognostic potential predictability (PPP)
    - anomlay correlation coefficient (ACC) (missing)
        - intra-ensemble
        - against mean
    - root mean square error (RMSE) (=MSE^0.5) (missing)
    - normalized root mean square error (NRMSE) (=1-MSE^0.5/RMSE_control) (missing)
    - Diagnostic Potential Predictability (DPP)
    - Relative Entropy (Kleeman 2002; Branstator and Teng 2010) (missing)
    - Mutual information (DelSole)
    - Average Predictability Time (APT) (DelSole)
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

Example:
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
Time dimensions is called Years and is integer. (Original data was year 3000-3300, now 3600. Had problems with datetime[ns] limits and xr.to_netcdf()!)

"""


### general imports
import xarray as xr
import pandas as pd
import numpy as np
import os
from six.moves.urllib.request import urlopen, urlretrieve
from six.moves.http_client import HTTPException


def get_data_home(data_home=None):
    """Return the path of the data directory.
    This is used by the ``load_dataset`` function.
    If the ``data_home`` argument is not specified, the default location
    is ``~/seaborn-data``.
    """
    if data_home is None:
        data_home = os.environ.get('HOME','~')

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home

def get_dataset_names():
    """Report available example datasets, useful for reporting issues."""
    # delayed import to not demand bs4 unless this function is actually used
    # copied from seaborn 
    from bs4 import BeautifulSoup
    http = urlopen('https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction/')
    print('Load from URL:',http)
    gh_list = BeautifulSoup(http)
    
    return [l.text.replace('.nc', '')
            for l in gh_list.find_all("a", {"class": "js-navigation-open"})
            if l.text.endswith('.nc')]


def load_dataset(name, cache=True, data_home=None, **kws):
    """Load a datasets ds and control from the online repository (requires internet).
    Parameters
    ----------
    name : str
        Name of the dataset (`ds`.nc on
        https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction).  You can obtain list of
        available datasets using :func:`get_dataset_names`
    cache : boolean, optional
        If True, then cache data locally and use the cache on subsequent calls
    data_home : string, optional
        The directory in which to cache data. By default, uses ~/.
    kws : dict, optional
        Passed to pandas.read_csv
    """
    path = ("https://github.com/aaronspring/esmtools/raw/develop/sample_data/prediction/{}.nc")
    
    full_path = path.format(name)
    print('Load from URL:',full_path)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                                  os.path.basename(full_path))
        if not os.path.exists(cache_path):
            urlretrieve(full_path, cache_path)
        full_path = cache_path

    df = xr.open_dataset(full_path, **kws)
    
    return df



### Diagnostic Potential Predictability (DPP) 

def chunking(ds, number_chunks=False, chunk_length=False, output=False):
    """
    Separates data into chunks and reshapes chunks in a c dimension. 
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
        Output data as ds, but with additional dimension c and all same time coordinates
    
    Example
    -------
    import esmtools as et
    ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
    control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')
    ds_chunked_into_30yr_chunks = et.prediction.chunking(ds,chunk_length=30)
    ds_chunked_into_30_chunks = et.prediction.chunking(ds,number_chunks=30)
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
        print(number_chunks, 'chunks of length', chunk_length, 'from', cmin, 'to', cmax)
        print('0', cmin, cmin+chunk_length-1)
    c = ds.sel(year=slice(cmin, cmin+chunk_length-1))
    c = c.expand_dims('c')
    c['c'] = [0]
    year = c.year
    for i in range(1, number_chunks):
        if output:
            print(i, cmin+chunk_length*i, cmin+(i+1)*chunk_length-1)
        c2 = ds.sel(year=slice(cmin+chunk_length*i, cmin+(i+1)*chunk_length-1))
        c2 = c2.expand_dims('c')
        c2['c'] = [i]
        c2['year'] = year
        c = xr.concat([c, c2], 'c')
    return c

#DDP_boer_b
def DPP(ds, m=10, chunk=True, return_s=True, output=False):
    """
    Calculate Diagnostic Potential Predictability (DPP) as potentially predictable 
    variance fraction (ppvf) in Boer 2004.
    Note: Different way of calculating it than in Seferian 2018 or Resplandy 2015,
    but quite similar results.
    
    References
    ----------
    - Boer, G. J. “Long Time-Scale Potential Predictability in an Ensemble of Coupled Climate Models.” Climate Dynamics 23, no. 1 (August 1, 2004): 29–44. https://doi.org/10/csjjbh.
    - Resplandy, L., R. Séférian, and L. Bopp. “Natural Variability of CO2 and O2 Fluxes: What Can We Learn from Centuries-Long Climate Models Simulations?” Journal of Geophysical Research: Oceans 120, no. 1 (January 2015): 384–404. https://doi.org/10/f63c3h.
    - Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing the Decadal Predictability of Land and Ocean Carbon Uptake.” Geophysical Research Letters, March 15, 2018. https://doi.org/10/gdb424.
    
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
    
    Example
    -------
    import esmtools as et
    ds_DPPm10 = et.prediction.chunking(ds,m=10,chunk=True)
    """
    
    if ds.size > 5000: #dirty way of figuring out which data
        data3D = True
        print('3D data')
    else:
        data3D = False
    if output:
        print(m,ds.dims,chunk)
        
    if not chunk:
        s2v = ds.rolling(year=m, min_periods=1, center=True).mean().var('year')
        s2e = (ds - ds.rolling(year=m, min_periods=1,
                               center=True).mean()).var('year')
        s2 = s2v + s2e
    
    if chunk:
        # first chunk
        chunked_means = chunking(ds,chunk_length=m).mean('year')
        # sub means in chunks
        chunked_deviations = chunking(ds,chunk_length=m) - chunked_means
        
        s2v = chunked_means.var('c')
        if var_all_e:
            s2e = chunked_deviations.var(['year','c'])
        else:
            s2e = chunked_deviations.var('year').mean('c')
        s2 = s2v + s2e
    
    DPP = (s2v - s2/(m))/(s2)
    
    if output:
        print(DPP,s2v, s2e, s2)
    
    if data3D:
        return DPP
    if not return_s:
        return DPP
    if return_s: 
        return DPP, s2v, s2e, s2


### Prognostic Potential Predictability Griffies & Bryan 1997

# 3 different ways of calculation ensemble spread:
def ens_var_against_mean(ds):
    """
    Calculated the ensemble spread (ensemble variance (sqaured difference between each ensemble member and the ensemble mean) as a function of time).
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
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    # display as dataframe 
    ens_var_against_mean.to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)
    """
    return ds.var('member')

def ens_var_against_control(ds):
    """
    See ens_var_against_mean(ds). Only difference is that now distance is evaluated against member=0 which is the control run.
    """
    var=ds.copy()
    var = ((ds - ds.sel(member=0))**2).sum('member')/(ds.member.size-2) 
    return var

def ens_var_against_every(ds):
    """    See ens_var_against_mean(ds). Only difference is that now distance is evaluated against each ensemble member and then averaged.
    """
    var=ds.copy()
    for i in range(0,ds.member.size):
        var_a = ((ds-ds.sel(member=i))**2).sum(dim='member')/ds.member.size
        var = xr.concat([var,var_a],'member')
    var=var.sel(member=slice(ds.member.size,2*ds.member.size)).mean('member')
    return var

    
def normalize_var(var,control,fac=1,running=True,m=20):
    """
    Normalizes the ensemble spread with the temporal spread of the control run.
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
        control_var = control.var('year')#.mean('year')
        var2 = var/control_var/fac
        return var2
    if running:
        control_var_running = control.rolling(year=m).var().mean('year')
        var2 = var/control_var_running/fac
        return var2

def PPP_from_nvar(nvar):
    """
    Calculated Prognostic Potential Predictability (PPP) as in Pohlmann 2004 
    or Griffies 1997. 
    
    References
    ----------
    - Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4.
    - Pohlmann, Holger, Michael Botzet, Mojib Latif, Andreas Roesch, Martin Wild, and Peter Tschuck. “Estimating the Decadal Predictability of a Coupled AOGCM.” Journal of Climate 17, no. 22 (November 1, 2004): 4463–72. https://doi.org/10/d2qf62.
    
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
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    nens_var_against_mean = et.prediction.normalize_var(ens_var_against_mean)
    PPP_mean = et.prediction.PPP_from_nvar(nens_var_against_mean)
    """
    return 1-nvar


### Perfect-model (PM) predictability scores from Bushuk 2018

def PM_MSSS(ds,control):
    """
    Calculated the perfect-model (PM) mean square skill score (MSSS) as in Bushuk et al. 2018. This is the same as PPP from Pohlmann et al. (2004).
    
    Formula
    -------
    MSSS_{PM} = 1 - MSE/sigma_c  
    
    References
    ----------
    - Bushuk, Mitchell, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. “Regional Arctic Sea–Ice Prediction: Potential versus Operational Seasonal Forecast Skill.” Climate Dynamics, June 9, 2018. https://doi.org/10/gd7hfq.
    
    Parameters
    ---------- 
    (TODO: Test whether this works for 3D data)
    msss : DataArray with year dimension (optional spatial coordinates)
        Input ensemble data
    control : DataArray with year dimension (optional spatial coordinates)
        Input control run data
    Returns
    -------
    msss : DataArray
        Output data
    
    Example
    -------
    import esmtools as et
    ds = 
    control = 
    pm_msss = PM_MSSS(ds,control)
    """
    import esmtools as et
    ens_var_against_mean = et.prediction.ens_var_against_mean(ds)
    nens_var_against_mean = et.prediction.normalize_var(ens_var_against_mean)
    PPP_mean = et.prediction.PPP_from_nvar(nens_var_against_mean)
    return PPP_mean

def PM_ACC_U(msss):
    """
    Calculated the perfect-model (PM) unbiased anomaly correlation coefficient as in Bushuk et al. 2018. 
    
    Formula
    -------
    ACC_{U} = sqrt{MSSS}  
    
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
    ds = 
    control = 
    pm_msss = PM_MSSS(ds,control)
    pm_acc_u = PM_ACC_U(pm_msss)
    """
    return msss.sqrt()

def PM_ACC(ds):
    """
    Calculated the perfect-model (PM) anomaly correlation coefficient as in Bushuk et al. 2018.
    Create a supervectors (dims=(N*M,length)) for ensemble and observations (each member at the turn becomes obs). Returns M ACC timeseries.
    
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
    Returns
    -------
    ACC : pd.DataArray
        ACC
    
    Example
    -------
    import esmtools as et
    ds = 
    control = 
    pm_acc = PM_ACC(ds.sel(area=area,period=period))
    pm_acc.plot()
    """
    sv = ds.to_dataframe()[varname].unstack() # super vector (sv) should be one variable,region,period
    acc_l = pd.DataFrame(index=ds.year)
    for j in range(ds.member.size): #let every member be the observation vector
        svobs = sv.copy()
        for i in range(10): #fill the obs super vector with the jth member timeseries #dirty
            for t in starting_years: #create observations vector
                svobs.loc[t].loc[i] = svobs.loc[t].loc[j]

        ACC = sv.corrwith(svobs)
        acc_l[j] = ACC
    return acc_l


#TODO: Significance for ACC
# T test Bushuk
# pseudo-ensemble (uninit) ACC


### Persistence

# damped persistence forecast based on lag1 autocorrelation
def df_autocorr(df, lag=1, axis=0):
    """Compute full-sample column-wise autocorrelation for a pandas DataFrame."""
    return df.apply(lambda col: col.autocorr(lag), axis=axis)

def calc_tau(alpha):
    """
    Calculates the decorrelation time tau. 
    
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
    return (1+alpha)/(1-alpha)


def generate_predictability_persistence(s,kind='PPP',percentile=True,length=20):
    """
    Calculates the PPP (or NEV) damped persistence mean and range. Lag1 autocorrelation coefficient (alpha) is bootstrapped. Range can be indicated as +- std or 5-95-percentile.
    
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
    s = pd.Series(np.sin(range(1000)+np.cos(range(1000))+np.random.randint(.1,1000)))
    PPP_persistence_0, PPP_persistence_minus, PPP_persistence_plus = et.prediction.generate_predictability_persistence(s)
    t = np.arange(0,20+1,1.)
    plt.plot(PPP_persistence_0,color='black',linestyle='--',label='persistence mean')
    plt.fill_between(PPP_persistence_minus,PPP_persistence_plus,color='gray',alpha=.3,label='persistence range')
    plt.axhline(y=0,color='black')
    """
    #bootstrapping persistence
    it = 50 # iterations
    l = 100 # length of chunks of control run to take lag1 autocorr
    data = np.zeros(it)
    from random import randint
    for i in range(it):
        #np.random.shuffle(control)
        ran = randint(1900,2200-l)
        data[i] = s.loc[str(ran):str(ran+l)].autocorr()

    alpha_0 = np.mean(data)
    alpha_minus = np.mean(data)-np.std(data)
    alpha_plus = np.mean(data)+np.std(data)
    if percentile:
        alpha_minus = np.percentile(data,5)
        alpha_plus = np.percentile(data,95)

    #persistence function
    def generate_PPP_persistence(alpha,t):
        return np.exp(-2*alpha*t) #Griffies 1997
    t = np.arange(0,length+1,1.)
    PPP_persistence_0 = generate_PPP_persistence(alpha_0,t)
    PPP_persistence_minus = generate_PPP_persistence(alpha_plus,t)
    PPP_persistence_plus = generate_PPP_persistence(alpha_minus,t)
    
    if kind in ['nvar','NEV']:
        PPP_persistence_0 = 1-PPP_persistence_0
        PPP_persistence_minus = 1-PPP_persistence_minus
        PPP_persistence_plus = 1-PPP_persistence_plus
    
    return PPP_persistence_0, PPP_persistence_minus, PPP_persistence_plus


def generate_damped_persistence_forecast(control,startyear,length=20):
    """
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
    s = pd.Series(np.sin(range(1000)+np.cos(range(1000))+np.random.randint(.1,1000)))
    ar1, ar50, ar90 = et.prediction.generate_damped_persistence_forecast(s,22)
    ar1.plot()
    plt.fill_between(ar1.index,ar1-ar50,ar1+ar50,alpha=.2,color='gray')
    plt.fill_between(ar1.index,ar1-ar90,ar1+ar90,alpha=.1,color='gray')
    
    """
    
    anom=(control.loc[startyear]-control.mean())
    t=np.arange(0.,length+1,1)
    alpha = control.autocorr()
    exp=anom*np.exp(-alpha*t) # exp. decay towards mean
    ar1=exp+control.mean()
    ar50=0.7*control.std()*np.sqrt(1-np.exp(-2*alpha*t))
    ar90=1.7*control.std()*np.sqrt(1-np.exp(-2*alpha*t))
    
    index = control.loc[startyear:startyear+length].index
    ar1 = pd.Series(ar1, index=index)
    ar50 = pd.Series(ar50, index=index)
    ar90 = pd.Series(ar90, index=index)
    return ar1, ar50, ar90
