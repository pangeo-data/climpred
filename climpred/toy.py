import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from climpred.bootstrap import bootstrap_compute
from climpred.prediction import compute_perfect_model
from climpred.tutorial import load_dataset

lead = xr.DataArray(np.arange(0, 20, 1), dims='lead')
lead['lead'] = lead.values


control = load_dataset('MPI-control-1D')['tos'].isel(period=-1, area=1)
# amplitudes of the signal and noise
noise_amplitude = control.std().values * 2.5
signal_amplitude = control.std().values
# period of potentially predictable variable
P = 8


def ramp(lead, a=0.2, A_tot=0.5, t_opt=0.1, r=1.8):
    """A weighting function that starts at 0 and approaches 1."""
    A = A_tot * (
        1.0 / (1 + a * np.exp((t_opt - lead) / r)) - 1.0 / (1 + a * np.exp((t_opt) / r))
    )
    A[0] = 0
    A = A / A[-1]
    return A


def create_noise(lead=lead, n=3, m=3, noise_amplitude=noise_amplitude, ramp=ramp):
    """Create gaussian noise."""
    noise = (
        noise_amplitude
        * xr.DataArray(np.random.rand(lead.size, n, m), dims=['lead', 'init', 'member'])
        - noise_amplitude / 2
    )
    noise['lead'] = lead.values
    noise = noise * ramp(lead)
    noise['member'] = np.arange(1, 1 + m)
    noise['init'] = np.arange(1, 1 + n)
    return noise


def signal(lead=lead, signal_amplitude=signal_amplitude, P=P, lead_offset=0):
    """The signal to be predicted."""
    return signal_amplitude * xr.DataArray(np.sin((lead - lead_offset) * np.pi * 2 / P))


def create_initialized(
    lead=lead,
    ninits=10,
    nmember=10,
    init_range=150,
    signal_amplitude=signal_amplitude,
    noise_amplitude=noise_amplitude,
):
    """Create initialized ensemble."""
    # span range of initial conditions
    to = xr.DataArray(np.random.rand(ninits) * init_range - init_range, dims='init')
    # create initialized
    init = signal(
        lead=lead, lead_offset=to, signal_amplitude=signal_amplitude
    ) + create_noise(lead=lead, n=ninits, m=nmember, noise_amplitude=noise_amplitude)
    return init


def run_skill_for_ensemble(
    nmember=5,
    ninits=5,
    metric='rmse',
    bootstrap=10,
    plot=True,
    ax=None,
    label=None,
    color='k',
    **metric_kwargs,
):
    s = []
    # for i in tqdm(range(bootstrap),desc='bootstrap'):
    for i in range(bootstrap):
        ds = create_initialized(nmember=nmember, ninits=ninits)
        s.append(compute_perfect_model(ds, ds))
    s = xr.concat(s, 'bootstrap')
    s['lead'] = lead.values
    if plot:
        if ax is None:
            _, ax = plt.subplots()
        ss = s.to_dataframe('skill').unstack()['skill']
        bp = ss.plot.box(ax=ax, color=color, return_type='dict', label=label)
        ax.set_title(f'metric: {metric}, nmember: {nmember} ninits:{ninits}')
        return bp['whiskers'][0]
    else:
        return s


# uninit_skill


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
uninit = uninit_ensemble(ds, ds)
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
