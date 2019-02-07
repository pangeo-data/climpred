import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eofs.xarray import Eof


def _relative_entropy_formula(sigma_b, sigma_x, mu_x, mu_b, neofs):
    """
    Computes the relative entropy formula given in Branstator and Teng, (2010).

    References:
        - Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value
            Decadal Predictability in a CGCM.” Journal of Climate 23, no. 23
            (August 27, 2010): 6292–6311. https://doi.org/10/bwq92h.
        - Kleeman, Richard. “Measuring Dynamical Prediction Utility Using
            Relative Entropy.” Journal of the Atmospheric Sciences 59, no. 13
            (July 1, 2002): 2057–72. https://doi.org/10/fqwxpk.

    Args:
        sigma_b (xr.DataArray): covariance matrix of baseline distribution
        sigma_x (xr.DataArray): covariance matrix of forecast distribution
        mu_b (xr.DataArray): mean state vector of the baseline distribution
        mu_x (xr.DataArray): mean state vector of the forecast distribution
        neofs (int): number of EOFs used

    Returns:
        R (float): relative entropy
        dispersion (float): dispersion component
        signal (float): signal component
    """
    fac = 0.5
    dispersion = fac * (np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_x))
                        + np.trace(sigma_x / sigma_b) - neofs)

    # https://stackoverflow.com/questions/7160162/left-matrix-division-and-numpy-solve
    # (A.T\B)*A
    x, resid, rank, s = np.linalg.lstsq(
        sigma_b, mu_x - mu_b)  # sigma_b \ (mu_x - mu_b)
    signal = fac * np.matmul((mu_x.values - mu_b.values), x)
    R = dispersion + signal
    return R, dispersion, signal


def _bootstrap_dim(control, lead_years, time_dim='initialization', dim='member', dim_label=list(np.arange(10))):
    """
    Add a `len(dim_label)` dimension `dim` to uninitialized control with time_dim by bootstrapping.

    """
    c_start = 0
    c_end = control[time_dim].size
    time = np.arange(1, 1 + lead_years)

    def isel_years(control, year_s, m=None, length=lead_years):
        new = control.isel({time_dim: slice(year_s, year_s + length - 0)})
        new = new.rename({time_dim: 'time'})
        if isinstance(new, xr.DataArray):
            new['time'] = time
        elif isinstance(new, xr.Dataset):
            new = new.assign(time=time)
        return new

    def create_pseudo_members(control):
        # print(c_start,c_end,lead_years,dim_label)
        startlist = np.random.randint(
            c_start, c_end - lead_years - 1, len(dim_label))
        return xr.concat([isel_years(control, start)
                          for start in startlist], dim)

    control_uninitialized = create_pseudo_members(
        control).assign({dim: dim_label})
    return control_uninitialized


def compute_relative_entropy(initialized, control,
                             anomaly_data=False, neofs=None, curv=True,
                             ntime=None, detrend_by_control_unitialized=True,
                             nmember_control=10):
    """
    Compute relative entropy.

    Calculates EOFs from anomalies. Projects fields on EOFs to receive
    pseudo-Principle Components per initialization and lead year. Calculate
    relative entropy based on _relative_entropy_formula.

    Args:
        initialized (xr.Dataset): anomaly ensemble data with dimensions
                                    initialization, member, time and spatial
                                    [lon (x), lat(y)]. DPLE or PM_ds
        control (xr.Dataset): anomaly control distribution with
                                              non-spatial dimensions:
                                              spatial [lon (x), lat(y)].
                                              - LENS: member, time
                                              - PM_control: initialization
        anomaly_data (bool): Input data is anomaly alread. Default: False.
        neofs (int): number of EOFs to use. Default: initialized.member.size.
        curv (bool): if curvilinear grids are provided disables EOF weights.
        ntime (int): number of timesteps calculated.
        detrend_by_control_uninitialized (bool): Default: True
        nmember_control (int): number of members created from bootstrapping from control

    Returns:
        rel_ent (pd.DataFrame): relative entropy

    """
    # Defaults
    if neofs is None:
        neofs = initialized.member.size
    if ntime is None:
        ntime = initialized.time.size

    # case if you only submit control with dim time, PM case
    if ('ensemble' not in control.dims) and ('member' not in control.dims):
        control_uninitialized = xr.concat([_bootstrap_dim(control, initialized.time.size, dim='member', dim_label=np.arange(
            nmember_control)) for _ in range(initialized.initialization.size)], dim='initialization')
        if isinstance(control_uninitialized, xr.DataArray):
            control_uninitialized['initialization'] = initialized.initialization.values
        elif isinstance(control_uninitialized, xr.Dataset):
            control_uninitialized = control_uninitialized.assign(
                initialization=initialized.initialization.values)

    # case if you submit control with dim time and member, LENS case
    elif 'member' in control.dims:
        control_uninitialized = _bootstrap_dim(
            control, initialized.time.size, time_dim='time', dim='initialization', dim_label=list(initialized.initialization.values))
        #print('added member')

    # initialized and control_uninitialized are allowed to have different dims
    # as I need more members to sample my control distribution properly
    # ToDo: understand this more
    if initialized.dims != control_uninitialized.dims:
        print(initialized.dims)
        print(control_uninitialized.dims)
        # raise ValueError('init and uninit must have same dims')
        warnings.warn(
            "Warning: initialized and control_uninitialized have different coords.")

    if isinstance(control_uninitialized, xr.Dataset):
        control_uninitialized = control_uninitialized.to_array().squeeze()
    if isinstance(initialized, xr.Dataset):
        initialized = initialized.to_array().squeeze()

    # detrend
    #non_spatial_dims = set(control_uninitialized.dims).intersection(['time', 'initialization','member'])
    non_spatial_dims = set(control_uninitialized.dims).intersection(
        ['initialization', 'member'])
    non_spatial_dims = list(non_spatial_dims)
    if not anomaly_data:  # if ds, control are raw values
        if detrend_by_control_unitialized:
            anom_x = initialized - control_uninitialized.mean(non_spatial_dims)
            anom_b = control_uninitialized - \
                control_uninitialized.mean(non_spatial_dims)
    else:  # leave as is when already anomalies
        anom_x = initialized
        anom_b = control_uninitialized

    # prepare for EOF
    if curv:  # if curvilinear lon(x,y), lat(x,y) data inputs
        wgts = None
    else:
        coslat = np.cos(np.deg2rad(anom_x.coords['lat'].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]

    if isinstance(control, xr.Dataset):  # EOF requires xr.dataArray
        # print('convert control to xr.dataarray')
        control = control.to_array().squeeze()

    if 'member' in control.dims:  # LENS
        # stack all dimensions member and initialization into time, make time first
        # print('member is in control')
        non_spatial_control_dims = list(
            set(control.dims).intersection(['time', 'member']))
        # print('non_spatial',non_spatial_control_dims)
        transpose_dims = list(control.dims)
        transpose_dims.remove('member')
        dims = tuple(transpose_dims)
        # print('transpose(*dims)',*dims,'time should be first')
        base_to_calc_eofs = control.stack(
            new=tuple(non_spatial_control_dims)).rename({'new': 'time'}).set_index({'time': 'time'}).transpose(*dims)
    elif 'initialization' in control.dims and 'member' not in control.dims:  # PM_control
        base_to_calc_eofs = control.rename({'initialization': 'time'})
    else:
        raise ValueError('adapt you inputs to PM- or DPLE/LENS-style')

    # print('calc EOF base',base_to_calc_eofs.dims,'time should be first')
    solver = Eof(base_to_calc_eofs, weights=wgts)

    re_leadtime_list = []
    lead_times = initialized.time.values[:ntime]
    initializations = initialized.initialization.values
    # DoTo: parallelize this double loop
    for init in initializations:  # loop over initializations
        rl, sl, dl = ([] for _ in range(3))  # lists to store results in
        for t in lead_times:  # loop over lead time
            # P_b base distribution
            pc_b = solver.projectField(anom_b.sel(initialization=init, time=t)
                                             .drop('time')
                                             .rename({'member': 'time'}),
                                       neofs=neofs, eofscaling=0,
                                       weighted=False)

            mu_b = pc_b.mean('time')
            sigma_b = xr.DataArray(np.cov(pc_b.T))

            # P_x initialization distribution
            pc_x = solver.projectField(anom_x.sel(initialization=init, time=t)
                                             .drop('time')
                                             .rename({'member': 'time'}),
                                       neofs=neofs, eofscaling=0,
                                       weighted=False)

            mu_x = pc_x.mean('time')
            sigma_x = xr.DataArray(np.cov(pc_x.T))

            r, d, s = _relative_entropy_formula(sigma_b, sigma_x, mu_x, mu_b,
                                                neofs)

            rl.append(r)
            sl.append(s)
            dl.append(d)

        re_leadtime_list.append(xr.Dataset({'R': ('time', rl),
                                            'S': ('time', sl),
                                            'D': ('time', dl)}))

    re = xr.concat(re_leadtime_list, dim='initialization').assign(
        initialization=initializations, time=lead_times)

    return re


def _shuffle(ds, dim='initialization'):
    """Shuffle ensemble members to uninitialize the data."""
    old_dim_range = ds[dim]
    shuffled = ds.sel({dim: np.random.permutation(ds[dim])})
    if isinstance(ds, xr.DataArray):
        shuffled[dim] = old_dim_range
    elif isinstance(ds, xr.Dataset):
        shuffled = shuffled.assign({dim: old_dim_range})
    shuffled = shuffled.sortby(dim)
    return shuffled


def create_uninitialized_ensemble_from_control(ds, control, member_label=[1, 2, 3, 4, 5]):
    """Create uninitialized ensemble from control."""
    control_uninitialized = xr.concat([_bootstrap_dim(control, ds.time.size, dim='member',
                                                      dim_label=member_label) for _ in range(ds.initialization.size)], dim='initialization')
    if isinstance(control_uninitialized, xr.DataArray):
        control_uninitialized['initialization'] = ds.initialization.values
    elif isinstance(control_uninitialized, xr.Dataset):
        control_uninitialized = control_uninitialized.assign(
            initialization=ds.initialization.values)
    return control_uninitialized


def bootstrap_relative_entropy(initialized, control, sig=95,
                               bootstrap=100, curv=True, neofs=None,
                               ntime=None, anomaly_data=False,
                               detrend_by_control_unitialized=True,
                               nmember_control=15):
    """
    Bootstrap relative entropy threshold.

    Generates a random uninitialized initialization and calculates the relative
    entropy. sig-th percentile determines threshold level.

    Args:
        initialized (xr.DataArray): initialized ensemble with dimensions
                                    initialization, member, time, lon (x),
                                    lat(y).
        control_uninitialized (xr.DataArray): control distribution with
                                              dimensions initialization,
                                              lon (x), lat(y).
        sig (int): significance level for threshold.
        bootstrap (int): number of bootstrapping iterations.
        neofs (int): number of EOFs to use. Default: initialized.member.size
        ntime (int): number of timestep to calculate.
                     Default: initialized.time.size.
        curv (bool): if curvilinear grids are provided disables EOF weights.

    Returns:
        rel_ent (pd.DataFrame): relative entropy sig-th percentile threshold.

    """
    if neofs is None:
        neofs = initialized.member.size
    if ntime is None:
        ntime = initialized.time.size

    x = []
    for _ in range(min(1, int(bootstrap / initialized.time.size))):
        uninitialized_initialized = create_uninitialized_ensemble_from_control(
            initialized, control, member_label=list(initialized.member.values))
        print(uninitialized_initialized.dims)
        print(control.dims)
        ds_pseudo_rel_ent = compute_relative_entropy(
            uninitialized_initialized, control, neofs=neofs,
            curv=curv, ntime=ntime, anomaly_data=anomaly_data,
            detrend_by_control_unitialized=detrend_by_control_unitialized,
            nmember_control=nmember_control)
        x.append(ds_pseudo_rel_ent)
    ds_pseudo_metric = xr.concat(x, dim='it')
    qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(
        q=qsig, dim=['it', 'time', 'initialization'])
    return sig_level


def plot_relative_entropy(rel_ent, rel_ent_threshold=None, **kwargs):
    """
    Plot relative entropy results.

    Args:
        rel_ent (pd.DataFrame): relative entropy from compute_relative_entropy
        rel_ent_threshold (pd.DataFrame): threshold from
                                          bootstrap_relative_entropy

    """
    colors = ['royalblue', 'indianred', 'goldenrod']
    fig, ax = plt.subplots(ncols=3, **kwargs)
    std = rel_ent.std('initialization')
    for i, dim in enumerate(['R', 'S', 'D']):
        m = rel_ent[dim].median('initialization')
        std = rel_ent[dim].std('initialization')
        ax[i].plot(rel_ent.time, rel_ent[dim].to_dataframe().unstack(0), c='gray',
                   label='individual initializations', linewidth=.5, alpha=.5)
        ax[i].plot(rel_ent.time, m, c=colors[i], label=dim, linewidth=2.5)
        ax[i].plot(rel_ent.time, (m - std), c=colors[i], label=dim + ' median +/- std',
                   linewidth=2.5, ls='--')
        ax[i].plot(rel_ent.time, (m + std), c=colors[i],
                   label='', linewidth=2.5, ls='--')
        if rel_ent_threshold is not None:
            ax[i].axhline(y=rel_ent_threshold[dim].values,
                          label='bootstrapped threshold', c='gray', ls='--')
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax[i].legend(by_label.values(), by_label.keys(), frameon=False)
    ax[0].set_title('Relative Entropy')
    ax[1].set_title('Signal')
    ax[2].set_title('Dispersion')
    ax[0].set_ylim(bottom=0)
