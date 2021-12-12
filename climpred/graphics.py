import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union

import cftime
import numpy as np
import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import DimensionError
from .classes import HindcastEnsemble, PerfectModelEnsemble
from .constants import CLIMPRED_DIMS
from .metrics import ALL_METRICS
from .utils import get_metric_class

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass


def plot_relative_entropy(rel_ent, rel_ent_threshold=None, **kwargs):
    """
    Plot relative entropy results.

    Args:
        rel_ent (xr.Dataset): relative entropy from compute_relative_entropy
        rel_ent_threshold (xr.Dataset): threshold from
                                          bootstrap_relative_entropy
        **kwargs: for plt.subplots( **kwargs)

    """
    colors = ["royalblue", "indianred", "goldenrod"]
    _, ax = plt.subplots(ncols=3, **kwargs)

    for i, dim in enumerate(["R", "S", "D"]):
        m = rel_ent[dim].median("init")
        std = rel_ent[dim].std("init")
        ax[i].plot(
            rel_ent.lead,
            rel_ent[dim].to_dataframe().unstack(0),
            c="gray",
            label="individual initializations",
            linewidth=0.5,
            alpha=0.5,
        )
        ax[i].plot(rel_ent.lead, m, c=colors[i], label=dim, linewidth=2.5)
        ax[i].plot(
            rel_ent.lead,
            (m - std),
            c=colors[i],
            label=dim + " median +/- std",
            linewidth=2.5,
            ls="--",
        )
        ax[i].plot(
            rel_ent.lead,
            (m + std),
            c=colors[i],
            label="",
            linewidth=2.5,
            ls="--",
        )
        if rel_ent_threshold is not None:
            ax[i].axhline(
                y=rel_ent_threshold[dim].values,
                label="bootstrapped threshold",
                c="gray",
                ls="--",
            )
        handles, labels = ax[i].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax[i].legend(by_label.values(), by_label.keys(), frameon=False)
        ax[i].set_xlabel("Lead")
    ax[0].set_title("Relative Entropy")
    ax[1].set_title("Signal")
    ax[2].set_title("Dispersion")
    ax[0].set_ylabel("Relative Entropy [ ]")
    ax[0].set_ylim(bottom=0)
    return ax


def plot_bootstrapped_skill_over_leadyear(
    bootstrapped: xr.Dataset,
    ax: Optional["plt.Axes"] = None,
    color_initialized: str = "indianred",
    color_uninitialized: str = "steelblue",
    color_persistence: str = "gray",
    color_climatology: str = "tan",
    capsize: Union[int, float] = 4,
    fontsize: Union[int, float] = 8,
    figsize: Tuple = (10, 4),
    fmt: str = "--o",
) -> "plt.Axes":
    """
    Plot Ensemble Prediction skill as in Li et al. 2016 Fig.3a-c.

    Args:
        bootstrapped (xr.DataArray or xr.Dataset with one variable):
            from PredictionEnsembleEnsemble.bootstrap() or HindcastEnsemble.bootstrap()

        ax ("plt.Axes"): plot on ax. Defaults to None.

    Returns:
        ax

    Reference:
      * Li, Hongmei, Tatiana Ilyina, Wolfgang A. Müller, and Frank
            Sienz. “Decadal Predictions of the North Atlantic CO2
            Uptake.” Nature Communications 7 (March 30, 2016): 11076.
            https://doi.org/10/f8wkrs.

    """
    if isinstance(bootstrapped, xr.Dataset):
        var = list(bootstrapped.data_vars)
        if len(var) > 1:
            raise ValueError(
                "Please provide only xr.Dataset with one variable or xr.DataArray."
            )
        # copy attributes to xr.DataArray
        elif len(var) == 1:
            var = var[0]
            attrs = bootstrapped.attrs
            bootstrapped = bootstrapped[var]
            bootstrapped.attrs = attrs

    assert isinstance(bootstrapped, xr.DataArray)
    reference = list(bootstrapped.drop_sel(skill="initialized").coords["skill"].values)

    sig = bootstrapped.attrs["confidence_interval_levels"].split("-")
    sig = int(100 * (float(sig[0]) - float(sig[1])))
    pers_sig = sig

    init_skill = bootstrapped.sel(skill="initialized", results="verify skill")
    init_ci = bootstrapped.sel(
        skill="initialized", results=["low_ci", "high_ci"]
    ).rename({"results": "quantile"})

    if pers_sig != sig:
        raise NotImplementedError("pers_sig != sig not implemented yet.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    # plot init
    ax.errorbar(
        init_skill.lead,
        init_skill,
        yerr=[
            init_skill - init_ci.isel(quantile=0),
            init_ci.isel(quantile=1) - init_skill,
        ],
        fmt=fmt,
        capsize=capsize,
        c=color_initialized,
        label="initialized",
    )
    # plot references
    for r in reference:
        r_skill = bootstrapped.sel(skill=r, results="verify skill")
        if (r_skill == np.nan).all():
            warnings.warn(f"Found only NaNs in {r} verify skill and skipped.")
            continue
        p_r_over_init = bootstrapped.sel(skill=r, results="p")
        r_ci = bootstrapped.sel(skill=r, results=["low_ci", "high_ci"]).rename(
            {"results": "quantile"}
        )
        c = eval(f"color_{r}")
        # add p values over all reference skills
        for t in init_skill.lead.values:
            ax.text(
                r_skill.lead.sel(lead=t),
                r_ci.isel(quantile=0).sel(lead=t).values,
                "%.2f" % float(p_r_over_init.sel(lead=t).values),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=fontsize,
                color=c,
            )
        yerr = [
            r_skill - r_ci.isel(quantile=0),
            r_ci.isel(quantile=1) - r_skill,
        ]
        x = r_skill.lead
        ax.errorbar(
            x,
            r_skill,
            yerr=yerr,
            fmt=fmt,
            capsize=capsize,
            c=c,
            label=r,
        )

    ax.xaxis.set_ticks(bootstrapped.lead.values)
    ax.legend(frameon=False, title=f"skill with {sig}% confidence interval:")
    ax.set_xlabel(f"Lead time [{bootstrapped.lead.attrs['units']}]")
    ax.set_ylabel(get_metric_class(bootstrapped.attrs["metric"], ALL_METRICS).long_name)
    return ax


def _check_only_climpred_dims(pe):
    """Warns if dimensions other than `CLIMPRED_DIMS` are in `PredictionEnsemble`."""
    additional_dims = set(pe.get_initialized().dims) - set(CLIMPRED_DIMS)
    if len(additional_dims) != 0:
        raise DimensionError(
            f"{type(pe.__name__)}.plot() does not allow dimensions other "
            f"than {CLIMPRED_DIMS}, found {additional_dims}. "
            f"Please use .mean({additional_dims}) "
            f"or .isel() before plot."
        )


def plot_lead_timeseries_hindcast(
    he: HindcastEnsemble,
    variable: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show_members: bool = False,
    cmap: Optional[str] = "viridis",
    x: str = "time",
) -> "plt.Axes":
    """Plot datasets from HindcastEnsemble.

    Args:
        he (HindcastEnsemble): HindcastEnsemble.
        variable (str or None): `variable` to plot. Defaults to the first in data_vars.
        ax (plt.axes): Axis to use in plotting. By default, creates a new axis.
        show_members (bool): whether to display all members individually.
            Defaults to False.
        cmap (str): Name of matplotlib-recognized colorbar. Defaults to 'viridis'.

    Returns:
        ax: plt.axes

    """
    if x == "time":
        x = "valid_time"
    _check_only_climpred_dims(he)
    if variable is None:
        variable = list(he.get_initialized().data_vars)[0]
    hind = he.get_initialized()[variable]
    hist = he.get_uninitialized()
    if isinstance(hist, xr.Dataset):
        hist = hist[variable]
    obs = he.get_observations()
    if isinstance(obs, xr.Dataset):
        obs = obs[variable]

    _cmap = mpl.cm.get_cmap(cmap, hind.lead.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    if isinstance(hist, xr.DataArray) and x == "valid_time":
        if "member" in hist.dims and not show_members:
            hist = hist.mean("member")
            member_alpha = 1.0
            lw = 2
        else:
            member_alpha = 0.4
            lw = 1
        hist.plot(
            ax=ax,
            lw=lw,
            hue="member",
            color="gray",
            alpha=member_alpha,
            label="uninitialized",
            zorder=hind.lead.size + 1,
        )

    for i, lead in enumerate(hind.lead.values):
        h = hind.sel(lead=lead)
        if not show_members and "member" in h.dims:
            h = h.mean("member")
            lead_alpha = 1.0
        else:
            lead_alpha = 0.5
        h.plot(
            ax=ax,
            x=x,
            hue="member",
            color=_cmap(i),
            label=f"initialized: lead={lead} {hind.lead.attrs['units'][:-1]}",
            alpha=lead_alpha,
            zorder=hind.lead.size - i,
        )

    if isinstance(obs, xr.DataArray) and x == "valid_time":
        obs.plot(
            ax=ax,
            x="time",
            color="k",
            lw=3,
            ls="-",
            label="observations",
            zorder=hind.lead.size + 2,
        )

    # show only one item per label in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1, 0.5)
    )
    ax.set_title("")
    ax.set_xlabel(he.coords[x].attrs["long_name"])
    return ax


def plot_ensemble_perfect_model(
    pm: PerfectModelEnsemble,
    variable: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show_members: bool = False,
    cmap: Optional[str] = "tab10",
    x: str = "time",
) -> "plt.Axes":
    """Plot datasets from PerfectModelEnsemble.

    Args:
        pm (PerfectModelEnsemble): PerfectModelEnsemble.
        variable (str or None): `variable` to plot. Defaults to the first in data_vars.
        ax (plt.axes): Axis to use in plotting. By default, creates a new axis.
        show_members (bool): whether to display all members individually.
            Defaults to False.
        cmap (str): Name of matplotlib-recognized colorbar. Defaults to 'tab10'.

    Returns:
        ax: plt.axes

    """
    x = "valid_time"
    _check_only_climpred_dims(pm)
    if variable is None:
        variable = list(pm.get_initialized().data_vars)[0]
    initialized = pm.get_initialized()[variable]
    uninitialized = pm.get_uninitialized()
    if isinstance(uninitialized, xr.Dataset):
        uninitialized = uninitialized[variable]
        uninitialized_present = True
    else:
        uninitialized_present = False
    control = pm.get_control()
    if isinstance(control, xr.Dataset):
        control = control[variable]

    control_color = "gray"

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    _cmap = mpl.cm.get_cmap(cmap, initialized.init.size)

    for ii, i in enumerate(initialized.init.values):
        dsi = initialized.sel(init=i)
        if uninitialized_present:
            dsu = uninitialized.sel(init=i)
        if not show_members:
            dsi = dsi.mean("member")
            if uninitialized_present:
                dsu = dsu.mean("member")
            member_alpha = 1.0
            lw = 2
            labelstr = "ensemble mean"
        else:
            member_alpha = 0.5
            lw = 1
            labelstr = "members"
            # plot ensemble mean, first white then color to highlight ensemble mean
            if uninitialized_present:
                dsu.mean("member").plot(
                    ax=ax, x=x, color="white", lw=3, zorder=8, alpha=0.6
                )
                dsu.mean("member").plot(
                    ax=ax, x=x, color=control_color, lw=2, zorder=9, alpha=0.6
                )
            # plot ensemble mean, first white then color to highlight ensemble mean
            dsi.mean("member").plot(ax=ax, x=x, color="white", lw=3, zorder=10)
            dsi.mean("member").plot(ax=ax, x=x, color=_cmap(ii), lw=2, zorder=11)
        dsi.plot(
            ax=ax,
            x=x,
            hue="member",
            color=_cmap(ii),
            alpha=member_alpha,
            lw=lw,
            label=labelstr,
        )
        if uninitialized_present:
            dsu.plot(
                ax=ax,
                x=x,
                hue="member",
                color=control_color,
                alpha=member_alpha / 2,
                lw=lw,
                label="uninitialized " + labelstr,
            )

    if isinstance(control, xr.DataArray):
        control.plot(ax=ax, color=control_color, label="control")

    # show only one item per label in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(" ")
    ax.set_xlabel(pm.coords[x].attrs["long_name"])
    return ax


def _verif_dates_xr(hindcast, alignment, reference, date2num_units):
    """Create ``valid_time`` ``xr.DataArray`` with dims lead and init in units passed to
    cftime.date2num."""
    inits, verif_dates = return_inits_and_verif_dates(
        hindcast.get_initialized().rename({"init": "time"}),
        hindcast.get_observations(),
        alignment,
        reference=reference,
        hist=hindcast.get_uninitialized()
        if isinstance(hindcast.get_uninitialized(), xr.Dataset)
        else None,
    )

    verif_dates_xr = xr.concat(
        [
            xr.DataArray(
                cftime.date2num(verif_dates[k], date2num_units),
                dims="init",
                coords={"init": v.rename({"time": "init"}).to_index()},
                name="valid_time",
                attrs=dict(units=date2num_units),
            )
            for k, v in inits.items()
        ],
        dim="lead",
    ).assign_coords(lead=hindcast.get_initialized().lead)
    return verif_dates_xr
