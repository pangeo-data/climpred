import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from climpred.bootstrap import bootstrap_compute
from climpred.prediction import compute_perfect_model
from climpred.tutorial import load_dataset

# standard setup
lead = xr.DataArray(np.arange(0, 20, 1), dims='lead')
lead['lead'] = lead.values

control = load_dataset('MPI-control-1D')['tos'].isel(period=-1, area=1)
# amplitudes of the signal and noise
noise_amplitude = control.std().values * 2.5
signal_amplitude = control.std().values
# period of potentially predictable variable
signal_P = 8


def ramp(lead, a=0.2, t_opt=0.1, r=1.8):
    """A weighting function that starts at 0 and approaches 1 to mimick increasing noise.

    Args:
        lead (xr.DataArray): lead times to create synthetic data for.
        a (type): shape of ramping up. Defaults to 0.2.
        t_opt (type): lead time near to saturation. Defaults to 0.1.
        r (type): intensity of ramping up. Defaults to 1.8.

    Returns:
        np.array: weighting ramping up to 1.

    """
    A = 1.0 / (1 + a * np.exp((t_opt - lead) / r)) - 1.0 / (1 + a * np.exp((t_opt) / r))
    A[0] = 0
    A = A / A[-1]
    return A


def create_noise(
    lead=lead, ninit=3, nmember=3, noise_amplitude=noise_amplitude, ramp=ramp
):
    """Create gaussian noise.

    Args:
        lead (xr.DataArray): lead times to create synthetic data for.
        ninit (type): number of initialzations. Defaults to 3.
        nmember (type): number of members. Defaults to 3.
        noise_amplitude (type): amplitude of the gaussian noise.
        ramp (type): Ramping up function.

    Returns:
        xr.DataArray: ramped up noise with initializations and members

    """
    noise = (
        noise_amplitude
        * xr.DataArray(
            np.random.rand(lead.size, ninit, nmember), dims=['lead', 'init', 'member']
        )
        - noise_amplitude / 2
    )
    noise['lead'] = lead.values
    noise = noise * ramp(lead)
    noise['member'] = np.arange(1, 1 + nmember)
    noise['init'] = np.arange(1, 1 + ninit)
    return noise


def signal(
    lead=lead, signal_amplitude=signal_amplitude, signal_P=signal_P, lead_offset=0
):
    """The low-frequency signal to be predicted.

    Args:
        lead (xr.DataArray): lead times to create synthetic data for.
        signal_amplitude (float): amplitude of the signal.
        signal_P (float): period of the signal.
        lead_offset (float): phase of the signal. Defaults to 0.

    Returns:
        xr.DataArray: signal to be predicted

    """
    return signal_amplitude * xr.DataArray(
        np.sin((lead - lead_offset) * np.pi * 2 / signal_P)
    )


def create_initialized(
    lead=lead,
    ninit=10,
    nmember=10,
    signal_amplitude=signal_amplitude,
    noise_amplitude=noise_amplitude,
    signal_P=signal_P,
    ramp=ramp,
):
    """Create initialized ensemble.

    Args:
        lead (xr.DataArray): lead times to create synthetic data for.
        ninit (type): number of initialzations. Defaults to 3.
        nmember (type): number of members. Defaults to 3.
        noise_amplitude (type): amplitude of the gaussian noise.
        ramp (np.array): Ramping up function.
        signal_amplitude (float): amplitude of the signal.
        signal_P (float): period of the signal.

    Returns:
        xr.DataArray: initialized ensemble with potentially predictable variation.

    """
    # span range of initial conditions
    init_range = 2 * signal_P
    to = xr.DataArray(np.random.rand(ninit) * init_range - init_range, dims='init')
    # create initialized
    init = signal(
        lead=lead, lead_offset=to, signal_amplitude=signal_amplitude, signal_P=signal_P
    ) + create_noise(
        lead=lead,
        ninit=ninit,
        nmember=nmember,
        noise_amplitude=noise_amplitude,
        ramp=ramp,
    )
    return init


def run_skill_for_ensemble(
    lead=lead,
    nmember=5,
    ninit=5,
    signal_P=signal_P,
    signal_amplitude=signal_amplitude,
    noise_amplitude=noise_amplitude,
    ramp=ramp,
    bootstrap=10,
    plot=True,
    ax=None,
    label=None,
    color='k',
    compute=compute_perfect_model,
    **metric_kwargs,
):
    """Short summary.

    Args:
        lead (xr.DataArray): lead times to create synthetic data for.
        ninit (type): number of initialzations. Defaults to 3.
        nmember (type): number of members. Defaults to 3.
        noise_amplitude (type): amplitude of the gaussian noise.
        ramp (np.array): Ramping up function.
        signal_amplitude (float): amplitude of the signal.
        signal_P (float): period of the signal.
        bootstrap (int): bootstrap iterations to calc synthetic skill. Defaults to 10.
        plot (bool): plot as boxplot. Defaults to True.
        ax (plt.axes): using existing ax. Defaults to None.
        label (str): label in legend. Defaults to None.
        color (str): color of boxplot. Defaults to 'k'.
        compute (function): compute_perfect_model or compute_hindcast
        **metric_kwargs (type): Argument to be passed to compute `**metric_kwargs`.

    Returns:
        whisker_line: to use in legend, if plot.
        xr.DataArray: skill, else.

    """
    s = []
    # for i in tqdm(range(bootstrap),desc='bootstrap'):
    for i in range(bootstrap):
        ds = create_initialized(
            nmember=nmember,
            ninit=ninit,
            signal_P=signal_P,
            signal_amplitude=signal_amplitude,
            noise_amplitude=noise_amplitude,
        )
        s.append(compute_perfect_model(ds, ds, **metric_kwargs))
    s = xr.concat(s, 'bootstrap')
    s['lead'] = lead.values
    if plot:
        if ax is None:
            _, ax = plt.subplots()
        ss = s.to_dataframe('skill').unstack()['skill']
        bp = ss.plot.box(
            ax=ax,
            color=color,
            return_type='dict',
            label=label,
            flierprops={'markeredgecolor': color},
        )
        metric = metric_kwargs['metric'] if 'metric' in metric_kwargs else ''
        ax.set_title(f'metric: {metric}, nmember: {nmember} ninit:{ninit}')
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
    # broadcast lead0 to all leads
    if only_first:
        shuffled = shuffled.isel(lead=[0] * (shuffled.lead.size))
        shuffled['lead'] = ds.lead.values
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
    """bootstrap_compute for large ensembles."""
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
