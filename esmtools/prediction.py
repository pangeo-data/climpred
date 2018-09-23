"""
Objects dealing with prediction metrics. In particular, these objects are 
specific to decadal prediction -- skill, persistence forecasting, etc.
"""


# damped persistence forecast based on lag1 autocorrelation
def df_autocorr(df, lag=1, axis=0):
    """Compute full-sample column-wise autocorrelation for a DataFrame."""
    return df.apply(lambda col: col.autocorr(lag), axis=axis)

from random import randint
def get_autocorr(x):
    return x.autocorr()
def calc_tau(alpha):
    return (1-alpha)/(1+alpha)
def generate_PPP_persistence(alpha,t):
    return np.exp(-2*alpha*t)


# Diagnostic Potential Predictability based on Boer 2004
def chunking(ds, number_chunks=False, chunk_length=False, output=False):
    '''
    takes a dataarray/dataset with one time dimensions
    return a dataarray/dataset with 2 dimensions: length length [years] and c (chunks) 
    '''
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
def DPP(ds, m=10, chunk=True, var_all_e=True,return_s=True,output=False):
    if ds.size > 5000:
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
        
    if output:
        print('did not chunk, start chunk')
    
    if chunk:
        # first chunk
        chunked_means = chunking(control[varname],chunk_length=m).mean('year')
        # sub means in chunks
        chunked_deviations = chunking(control[varname],chunk_length=m) - chunked_means
        
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

# Prognostic Potential Predictability Griffies & Bryan 1997

def ens_var_against_mean(ds):
    return ds.var('member')

def ens_var_against_control(ds):
    var=ds.copy()
    var = ((ds - ds.sel(member=0))**2).sum('member')/(ds.member.size-2) 
    return var/2

# calculates the deviations against all other ensemble members
def ens_var_against_every(ds):
    var=ds.copy()
    for i in range(0,ds.member.size):
        var_a = ((ds-ds.sel(member=i))**2).sum(dim='member')/ds.member.size
        var = xr.concat([var,var_a],'member')
    var=var.sel(member=slice(ds.member.size,2*ds.member.size)).mean('member')
    return var

def normalize_var(var,fac=1,running=True):
    if running:
        #print('running var')
        return (var.stack(level=0).stack(level=0).stack(level=1).to_xarray()/control_var_running/fac).to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)
    else:
        #print('just var')
        return (var.stack(level=0).stack(level=0).stack(level=1).to_xarray()/control_var/fac).to_dataframe().unstack(level=0).unstack(level=0).unstack(level=0).reorder_levels([3,1,0,2],axis=1)

def PPP_from_nvar(nvar):
    return 1-nvar


