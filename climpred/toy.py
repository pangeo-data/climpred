import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from climpred.bootstrap import bootstrap_compute
from climpred.graphics import plot_bootstrapped_skill_over_leadyear
from climpred.prediction import compute_hindcast, compute_perfect_model
from climpred.tutorial import load_dataset
from tqdm import tqdm

t = xr.DataArray(np.arange(0, 20, 1), dims='time')
t['time'] = t.values


control = load_dataset('MPI-control-1D')['tos'].isel(period=-1, area=1)
# amplitudes of the signal and noise
noise_amplitude = control.std().values * 2.5
signal_amplitude = control.std().values
# period of potentially predictable variable
P = 8


def ramp(t, a=0.2, A_tot=0.5, t_opt=0.1, r=1.8):
    """A weighting function that starts at 0 and approaches 1."""
    A = A_tot * (
        1.0 / (1 + a * np.exp((t_opt - t) / r)) - 1.0 / (1 + a * np.exp((t_opt) / r))
    )
    A[0] = 0
    A = A / A[-1]
    return A


ramp(t).plot()


def create_noise(n=3, m=3, noise_amplitude=noise_amplitude, ramp=ramp):
    """Create gaussian noise."""
    noise = (
        noise_amplitude
        * xr.DataArray(np.random.rand(t.size, n, m), dims={'time', 'init', 'member'})
        - noise_amplitude / 2
    )
    noise['time'] = t.time
    noise = noise * ramp(t)
    noise['member'] = np.arange(1, 1 + m)
    noise['init'] = np.arange(1, 1 + n)
    return noise


create_noise().to_dataframe('s').unstack().unstack()['s'].plot(legend=False)


def signal(signal_amplitude=signal_amplitude, P=P, t_offset=0):
    """The signal to be predicted."""
    return signal_amplitude * xr.DataArray(np.sin((t - t_offset) * np.pi * 2 / P))


def create_initialized(ninits=10, nmember=10, init_range=150):
    """Create initialized ensemble."""
    # span range of initial conditions
    to = xr.DataArray(np.random.rand(ninits) * init_range - init_range, dims='init')
    # create initialized
    init = (signal(t_offset=to) + create_noise(n=ninits, m=nmember)).rename(
        {'time': 'lead'}
    )
    return init


i = create_initialized()
i.to_dataframe('s').unstack().unstack().plot(alpha=0.5, legend=False)
i.mean('member').to_dataframe('mean').unstack()['mean'].plot(ax=plt.gca(), lw=3)


def run(nmember=5, ninits=5, metric='rmse', bootstrap=10):
    s = []
    # for i in tqdm(range(bootstrap),desc='bootstrap'):
    for i in range(bootstrap):
        ds = create_initialized(nmember=nmember, ninits=ninits)
        s.append(compute_perfect_model(ds, ds, metric=metric))
    s = xr.concat(s, 'bootstrap')
    s['lead'] = t.values
    ss = s.to_dataframe('skill').unstack()['skill']
    ss.plot.box()
    plt.title(f'metric: {metric}, nmember: {nmember} ninits:{ninits}')


run(nmember=3, ninits=3)
plt.savefig('m3i3')

run(nmember=10, ninits=12)
plt.savefig('m10i12')


def shuffle(ds, dim='initialization'):
    """Shuffle ensemble members to uninitialize the data."""
    old_dim_range = ds[dim]
    shuffled = ds.sel({dim: np.random.permutation(ds[dim])})
    if isinstance(ds, xr.DataArray):
        shuffled[dim] = old_dim_range
    elif isinstance(ds, xr.Dataset):
        shuffled = shuffled.assign({dim: old_dim_range})
    shuffled = shuffled.sortby(dim)
    return shuffled


def uninit_ensemble(ds, ds2, dim='init'):
    """
    Shuffle initializations to uninitialize the data.
    """

    shuffledlist = []
    for ds_m in list(ds.member.values):
        shuffledlist.append(shuffle(ds.sel(member=ds_m), dim=dim))

    shuffled = xr.concat(shuffledlist, dim='member')
    return shuffled


nmember = 3
ninits = 100
ds = create_initialized(nmember=nmember, ninits=ninits)

uninit = uninit_ensemble(ds, ds)  # .isel(lead=slice(None, 6))
uninit.to_dataframe('s').unstack().unstack(0).plot(legend=False)
uninit.mean('member').to_dataframe('mean').unstack()['mean'].plot(ax=plt.gca(), lw=3)


def uninit_ensemble_ori(ds, ds2, dim='init', only_first=True):
    """
    Shuffle ensemble members to uninitialize the data.
    """
    memberlist = []
    for m in ds.member.values:
        memberlist.append(ds.sel(member=m))

    shuffledlist = []
    for ds_m in memberlist:
        shuffledlist.append(shuffle(ds_m, dim=dim))

    shuffled = xr.concat(shuffledlist, dim='member')
    if only_first:
        shuffled = shuffled.isel(lead=[0] * (shuffled.lead.size))
        shuffled['lead'] = ds.lead.values
    else:
        pass
    return shuffled


def bootstrap_perfect_model_toy(
    ds,
    control,
    metric='pearson_r',
    comparison='m2e',
    dim=None,
    sig=95,
    bootstrap=5,
    pers_sig=None,
):
    """Use bootstrap_compute with different resampling function for large ensembles."""
    if dim is None:
        dim = ['init', 'member']
    return bootstrap_compute(
        ds,
        control,
        hist=None,
        metric=metric,
        comparison=comparison,
        dim=dim,
        sig=sig,
        bootstrap=bootstrap,
        pers_sig=pers_sig,
        compute=compute_perfect_model,
        resample_uninit=uninit_ensemble_ori,
    )


bootstrap = 100
fref = ds.rename({'init': 'time'}).isel(lead=0, member=0, drop=True)

bs = bootstrap_perfect_model_toy(ds, fref, metric='rmse', bootstrap=bootstrap)


plot_bootstrapped_skill_over_leadyear(bs, sig=95, plot_persistence=False)


# Sienz 2016
# y ̃t = α0 + α1xCO2,t + α2 sin(2πt/P) + α3 cos(2πt/P)
a0 = 16.478  # [°C]
a1 = 0.006495  # [°Cppm−1]
a2 = 0.07824  # [°C]
a3 = 0.1691  # [°C]
sigma = 0.1367  # [°C2]


simulation_start = 1871
simulation_end = 2010
CO2 = (
    xr.open_dataset('~/PhD_Thesis/PhD_scripts/160701_Grand_Ensemble/co2atm.nc')[
        'co2atm'
    ]
    .sel(ext='rcp45')
    .sel(year=slice(simulation_start, simulation_end))
    .rename({'year': 'time'})
)
del CO2['member']
del CO2['ext']
CO2_forcing = CO2 * a1
t2 = np.arange(1, CO2_forcing.time.size + 1, 1)
CO2_forcing = xr.DataArray(CO2_forcing, dims={'time': t2})


def forcing(ds, init=1880):
    t2 = np.arange(simulation_start, simulation_end + 1, 1)
    CO2_forcing = CO2 * a1
    CO2_forcing = xr.DataArray(CO2_forcing, dims={'time': t2})
    CO2_forcing['time'] = t2
    nleads = ds.lead.size
    r = CO2_forcing.sel(time=slice(init, init + nleads - 1)).rename({'time': 'lead'})
    r['lead'] = ds.lead
    return r


ds = create_initialized(nmember=10, ninits=2010 - 1880 + 1)
ds['init'] = np.arange(1880, 1880 + ds.init.size)


ds_f = xr.concat(
    [forcing(ds, init=i) + ds.sel(init=i) for i in ds.init.values if i < 1990], 'init'
)

ds_f.mean('member').to_dataframe('forced').unstack().plot(legend=False)
CO2_forcing.plot(c='k', lw=3)

# useless skill of CO2_forcing with synthetic initialized dataset
compute_hindcast(ds_f, CO2_forcing).plot()

# hindcast as a perfect_model
compute_perfect_model(ds, ds.sel(member=1), comparison='m2c', metric='rmse').plot()
