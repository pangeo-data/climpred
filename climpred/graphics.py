from collections import OrderedDict

import matplotlib.pyplot as plt


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
    std = rel_ent.std('initialization')
    for i, dim in enumerate(['R', 'S', 'D']):
        m = rel_ent[dim].median('initialization')
        std = rel_ent[dim].std('initialization')
        ax[i].plot(rel_ent.time, rel_ent[dim].to_dataframe().unstack(0),
                   c='gray', label='individual initializations',
                   linewidth=.5, alpha=.5)
        ax[i].plot(rel_ent.time, m, c=colors[i], label=dim, linewidth=2.5)
        ax[i].plot(rel_ent.time, (m - std), c=colors[i],
                   label=dim + ' median +/- std', linewidth=2.5, ls='--')
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
