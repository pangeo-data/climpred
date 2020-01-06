from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from .constants import PROBABILISTIC_METRICS


def plot_relative_entropy(rel_ent, rel_ent_threshold=None, **kwargs):
    """
    Plot relative entropy results.

    Args:
        rel_ent (xr.Dataset): relative entropy from compute_relative_entropy
        rel_ent_threshold (xr.Dataset): threshold from
                                          bootstrap_relative_entropy
        **kwargs: for plt.subplots( **kwargs)

    """
    colors = ['royalblue', 'indianred', 'goldenrod']
    fig, ax = plt.subplots(ncols=3, **kwargs)

    for i, dim in enumerate(['R', 'S', 'D']):
        m = rel_ent[dim].median('init')
        std = rel_ent[dim].std('init')
        ax[i].plot(
            rel_ent.lead,
            rel_ent[dim].to_dataframe().unstack(0),
            c='gray',
            label='individual initializations',
            linewidth=0.5,
            alpha=0.5,
        )
        ax[i].plot(rel_ent.lead, m, c=colors[i], label=dim, linewidth=2.5)
        ax[i].plot(
            rel_ent.lead,
            (m - std),
            c=colors[i],
            label=dim + ' median +/- std',
            linewidth=2.5,
            ls='--',
        )
        ax[i].plot(
            rel_ent.lead, (m + std), c=colors[i], label='', linewidth=2.5, ls='--'
        )
        if rel_ent_threshold is not None:
            ax[i].axhline(
                y=rel_ent_threshold[dim].values,
                label='bootstrapped threshold',
                c='gray',
                ls='--',
            )
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax[i].legend(by_label.values(), by_label.keys(), frameon=False)
        ax[i].set_xlabel('Lead')
    ax[0].set_title('Relative Entropy')
    ax[1].set_title('Signal')
    ax[2].set_title('Dispersion')
    ax[0].set_ylabel('Relative Entropy [ ]')
    ax[0].set_ylim(bottom=0)
    return ax


def plot_bootstrapped_skill_over_leadyear(
    bootstrapped, sig, plot_persistence=True, ax=None
):
    """
    Plot Ensemble Prediction skill as in Li et al. 2016 Fig.3a-c.

    Args:
        bootstrapped (xr.Dataset): from bootstrap_perfect_model or
                                   bootstrap_hindcast
        contains:
        init_skill (xr.Dataset): skill of initialized
        init_ci (xr.Dataset): confidence levels of init_skill
        uninit_skill (xr.Dataset): skill of uninitialized
        uninit_ci (xr.Dataset): confidence levels of uninit_skill
        sig (int): Significance level for uninitialized and
                   initialized skill.
        p_uninit_over_init (xr.Dataset): p value of the hypothesis that the
                                         difference of skill between the
                                         initialized and uninitialized
                                         simulations is smaller or equal to
                                         zero based on bootstrapping with
                                         replacement. Defaults to None.
        pers_skill (xr.Dataset): skill of persistence
        pers_ci (xr.Dataset): confidence levels of pers_skill
        pers_sig (int): Significance level for persistence forecast.
        p_pers_over_init (xr.Dataset): p value of the hypothesis that the
                                       difference of skill between the
                                       initialized and persistence simulations
                                       is smaller or equal to zero based on
                                       bootstrapping with replacement.
        ax (plt.axes): plot on ax. Defaults to None.

    Returns:
        None

    Reference:
      * Li, Hongmei, Tatiana Ilyina, Wolfgang A. Müller, and Frank
            Sienz. “Decadal Predictions of the North Atlantic CO2
            Uptake.” Nature Communications 7 (March 30, 2016): 11076.
            https://doi.org/10/f8wkrs.

    """
    pers_sig = sig

    if 'metric' in bootstrapped.attrs:
        if bootstrapped.attrs['metric'] in PROBABILISTIC_METRICS:
            plot_persistence = False

    init_skill = bootstrapped.sel(kind='init', results='skill')
    init_ci = bootstrapped.sel(kind='init', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    uninit_skill = bootstrapped.sel(kind='uninit', results='skill').isel(lead=0)
    uninit_ci = (
        bootstrapped.sel(kind='uninit', results=['low_ci', 'high_ci'])
        .rename({'results': 'quantile'})
        .isel(lead=0)
    )
    pers_skill = bootstrapped.sel(kind='pers', results='skill')
    pers_ci = bootstrapped.sel(kind='pers', results=['low_ci', 'high_ci']).rename(
        {'results': 'quantile'}
    )
    p_uninit_over_init = bootstrapped.sel(kind='uninit', results='p')
    p_pers_over_init = bootstrapped.sel(kind='pers', results='p')

    fontsize = 8
    c_uninit = 'indianred'
    c_init = 'steelblue'
    c_pers = 'gray'
    capsize = 4

    if pers_sig != sig:
        raise NotImplementedError('pers_sig != sig not implemented yet.')

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        init_skill.lead,
        init_skill,
        yerr=[
            init_skill - init_ci.isel(quantile=0),
            init_ci.isel(quantile=1) - init_skill,
        ],
        fmt='--o',
        capsize=capsize,
        c=c_uninit,
        label=(' ').join(['initialized with', str(sig) + '%', 'confidence interval']),
    )
    # uninit
    if p_uninit_over_init is not None:
        # add p values
        for t in init_skill.lead.values:
            ax.text(
                init_skill.lead.sel(lead=t),
                init_ci.isel(quantile=1).sel(lead=t).values,
                '%.2f' % float(p_uninit_over_init.sel(lead=t).values),
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=fontsize,
                color=c_uninit,
            )
        ax.errorbar(
            0,
            uninit_skill,
            yerr=[
                [uninit_skill - uninit_ci.isel(quantile=0)],
                [uninit_ci.isel(quantile=1) - uninit_skill],
            ],
            fmt='--o',
            capsize=capsize,
            c=c_init,
            label=(' ').join(
                ['uninitialized with', str(sig) + '%', 'confidence interval']
            ),
        )
        ax.axhline(y=uninit_skill, c='steelblue', ls=':')
    # persistence
    if plot_persistence:
        if pers_skill is not None and pers_ci is not None:
            ax.errorbar(
                pers_skill.lead,
                pers_skill,
                yerr=[
                    pers_skill - pers_ci.isel(quantile=0),
                    pers_ci.isel(quantile=1) - pers_skill,
                ],
                fmt='--o',
                capsize=capsize,
                c=c_pers,
                label=(' ').join(
                    ['persistence with', str(pers_sig) + '%', 'confidence interval']
                ),
            )
        for t in pers_skill.lead.values:
            ax.text(
                pers_skill.lead.sel(lead=t),
                pers_ci.isel(quantile=0).sel(lead=t).values,
                '%.2f' % float(p_pers_over_init.sel(lead=t).values),
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=fontsize,
                color=c_pers,
            )

    ax.xaxis.set_ticks(np.arange(init_skill.lead.size + 1))
    ax.legend(frameon=False)
    ax.set_xlabel('Lead time [years]')
    return ax
