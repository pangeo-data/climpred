from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from eofs.xarray import Eof

from climpred.prediction import _pseudo_ens


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
        neofs (int): number of EOFs to use

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
    signal = fac * np.matmul((mu_x - mu_b), x)
    R = dispersion + signal
    return R, dispersion, signal


def _gen_control_ensemble(ds, control, it=10):
    """
    Generate a large control ensemble of the control PDF.

    Args:
        ds (xr.DataArray): ensemble data with dimensions ensemble, member, time,
                                lon (x), lat(y)
        control (xr.DataArray): control data with dimensions time, lon (x),
                                lat(y)
        it (int): multiplying factor for ds.member

    Returns:
        control_ensemble (xr.DataArray): data with dimensions ensemble, member,
                                         time, lon (x), lat(y)

    """
    ds_list = []
    for _ in range(it):
        control_ensemble = _pseudo_ens(ds, control)
        control_ensemble['ensemble'] = ds.ensemble.values
        ds_list.append(control_ensemble)
    control_ensemble = xr.concat(ds_list, 'member')
    control_ensemble['member'] = np.arange(control_ensemble.member.size)
    print('control_ensemble', control_ensemble.nbytes / 1e6, 'MB', 'member',
          control_ensemble.member.size)
    return control_ensemble


def compute_relative_entropy(ds, control, control_ensemble, neofs=5,
                             curv=True, ntime=10):
    """
    Compute relative entropy.

    Create anomalies. Calculates EOFs. Projects fields on EOFs to receive
    pseudo-Principle Components per ensemble and lead year. Calculate relative
    entropy based on _relative_entropy_formula.

    Args:
        ds (xr.DataArray): ensemble data with dimensions ensemble, member, time,
                           lon (x), lat(y)
        control (xr.DataArray): control data with dimensions time, lon (x),
                                lat(y)
        control_ensemble (xr.DataArray): control distribution with dimensions
                                         ensemble, member, time, lon (x), lat(y)
        neofs (int): number of EOFs to use
        curv (bool): if curvilinear grids are provided disables EOF weights
        ntime (int): number of timesteps calculated

    Returns:
        rel_ent (pd.DataFrame): relative entropy

    """
    anom_x = ds - control.mean('time')
    anom_b = control_ensemble - control.mean('time')
    anom = control - control.mean('time')
    ensembles = ds.ensemble.values
    length = ds.time.size
    iterables = [['R', 'S', 'D'], ensembles]
    mindex = pd.MultiIndex.from_product(
        iterables, names=['component', 'ensemble'])
    tt = np.arange(1, length)
    rel_ent = pd.DataFrame(index=tt, columns=mindex)
    rel_ent.index.name = 'Lead Year'

    coslat = np.cos(np.deg2rad(anom_x.coords['lat'].values))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    if curv:
        wgts = None
    solver = Eof(anom, weights=wgts)

    for ens in ensembles:
        for t in ds.time.values[:ntime]:
            # P_b base distribution
            pc_b = solver.projectField(anom_b.sel(ensemble=ens, time=t)
                                             .drop('time')
                                             .rename({'member': 'time'}),
                                       neofs=neofs, eofscaling=0, weighted=False)

            mu_b = pc_b.mean('time')
            sigma_b = xr.DataArray(np.cov(pc_b.T))

            # P_x ensemble distribution
            pc_x = solver.projectField(anom_x.sel(ensemble=ens, time=t)
                                             .drop('time')
                                             .rename({'member': 'time'}),
                                       neofs=neofs, eofscaling=0, weighted=False)

            mu_x = pc_x.mean('time')
            sigma_x = xr.DataArray(np.cov(pc_x.T))

            r, d, s = _relative_entropy_formula(sigma_b, sigma_x, mu_x, mu_b,
                                                neofs)

            rel_ent.T.loc['R', ens][t] = r
            rel_ent.T.loc['D', ens][t] = d
            rel_ent.T.loc['S', ens][t] = s
    return rel_ent


def bootstrap_relative_entropy(ds, control, control_ensemble, sig=95,
                               bootstrap=100, curv=True, neofs=5):
    """
    Bootstrap relative entropy threshold.

    Generates a random uninitialized ensemble and calculates the relative
    entropy. sig-th percentile determines threshold level.

    Args:
        ds (xr.DataArray): ensemble data with dimensions ensemble, member, time,
                           lon (x), lat(y)
        control (xr.DataArray): control data with dimensions time, lon (x),
                                lat(y)
        control_ensemble (xr.DataArray): control distribution with dimensions
                                         ensemble, member, time, lon (x), lat(y)
        sig (int): significance level for threshold
        bootstrap (int): number of bootstrapping iterations
        neofs (int): number of EOFs to use
        curv (bool): if curvilinear grids are provided disables EOF weights

    Returns:
        rel_ent (pd.DataFrame): relative entropy sig-th percentile threshold

    """
    x = []
    for _ in range(min(1, int(bootstrap / ds['time'].size))):
        ds_control = _pseudo_ens(ds, control)
        ds_control['ensemble'] = ds.ensemble.values
        ds_control['member'] = np.arange(ds_control.member.size)
        ds_pseudo_rel_ent = compute_relative_entropy(
            ds_control, control, control_ensemble, neofs=neofs, curv=curv,
            ntime=ds.time.size)
        x.append(ds_pseudo_rel_ent)
    ds_pseudo_metric = pd.concat(x, ignore_index=True)
    qsig = sig / 100
    sig_level = ds_pseudo_metric.stack().apply(lambda g: np.quantile(g, q=qsig))
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
    for i, dim in enumerate(['R', 'S', 'D']):
        m = rel_ent[dim].median(axis=1)
        std = rel_ent[dim].std(axis=1)
        ax[i].plot(rel_ent[dim], c='gray',
                   label='individual ensembles', linewidth=.5, alpha=.5)
        ax[i].plot(m, c=colors[i], label=dim, linewidth=2.5)
        ax[i].plot((m - std), c=colors[i], label=dim + ' median +/- std',
                   linewidth=2.5, ls='--')
        ax[i].plot((m + std), c=colors[i], label='', linewidth=2.5, ls='--')
        if rel_ent_threshold is not None:
            ax[i].axhline(y=rel_ent_threshold[dim],
                          label='bootstrapped threshold', c='gray', ls='--')
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax[i].legend(by_label.values(), by_label.keys(), frameon=False)
    ax[0].set_title('Relative Entropy')
    ax[1].set_title('Signal')
    ax[2].set_title('Dispersion')
    ax[0].set_ylim(bottom=0)
