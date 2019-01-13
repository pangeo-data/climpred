"""Quick plots using `proplot`

These are not meant to be final publishible plots."""
import numpy as np
import proplot as plot
import matplotlib.pyplot as plt
# from . import prediction


def _set_aeshetics(ax, cb=None, cb_label=None):
    """Sets rcParams for `proplot`

    Cannot find any rcParam for colorbar-related
    axes. Have to go about it this way.
    """
    rc_kw = {'axes.labelsize': 14,
             'figure.titlesize': 20,
             'figure.facecolor': 'w',
             'fontname': 'Helvetica Neue',
             }
    if cb is not None:
        cb.ax.tick_params(labelsize=12)
    if cb_label is not None:
        cb.set_label(cb_label, fontsize=12)
    ax.format(rc_kw=rc_kw)


def _check_dp_dims(dp):
    """Make sure that the dp being plotted has appropriate dimensions."""
    # required dimensions for this package.
    dple_dims = ['ensemble', 'member', 'time']
    if not (set(dp.dims) < set(dple_dims)) | (set(dp.dims) == set(dple_dims)):
        raise IOError("""Please rename your decadal prediction dataset to have
            the following dimensions:
            'ensemble': initialization year/month
            'member': ensemble member
            'time': lead time since intiialization""")


def _check_ref_dims(ref):
    """Check that reference simulation is just dimension 'ensemble'"""
    if len(ref.dims) > 1:
        raise IOError("""Please provide a reference simulation with only the
            singular dimension 'ensemble' that coincides with initialization
            years for the decadal prediction ensemble.""")
    if not (ref.dims[0] == 'ensemble'):
        raise IOError("""Please provide a reference simulation with the
            dimension 'ensemble' that coincides with initialization years for
            the decadal prediction ensemble.""")


def forecasts(dp, ref, init_years=None, raw_values=False, **kws):
    """Plots all initialized forecasts for specified years relative to the
    reference simulating it.

    Args:
        dp: (xarray object) Decadal Prediction simulation
        ref: (xarray object) Reference simulation (e.g., control, hindcast)
        init_years: (list of ints) Initialization years to plot ensemble
                    forecasts and forecast mean for.
        raw_values: (optional boolean) If true, add reference mean to decadal
                    prediction output (assuming decadal prediction output is
                    in anomaly space and reference is not)
        **kws: Keywords to pass to axis, e.g. "ylabel = 'SST Anomaly'"
    Returns:
        Plot of reference run with fan plots of all forecasts and forecast
        mean.
    Raises:
        IOError: If decadal prediction ensemble doesn't have exactly dimensions
                 'ensemble', 'member', and 'time'.
                 If reference run does not have a singular dimension
                 'ensemble'.
        ValueError: If 'init_years' is not defined or is not a list of ints.
    """
    _check_dp_dims(dp)
    _check_ref_dims(ref)
    if (init_years is None):
        raise ValueError("""Please input a list of initialization years.""")
    if not all(isinstance(n, int) for n in init_years):
        raise ValueError("""Please provie init_years as a list of ints.""")
    if raw_values:
        dp = dp + ref.mean()

    f, ax = plot.subplots(axwidth=6, aspect=4, bottomlegend=True)
    r = ax.plot(ref.ensemble, ref, linewidth=1.5, color='k',
                label='reference')
    for iy in init_years:
        case = dp.sel(ensemble=iy)
        case['time'] = np.arange(iy, iy + dp.time.size)
        f = ax.plot(case.time, case, color='orchid', linewidth=0.5, alpha=0.75,
                    label='individual forecasts')
        fm = ax.plot(case.time, case.mean('member'), linewidth=2, color='plum',
                     label='forecast mean', zorder=4)
        ax.plot(iy, case.isel(time=0).mean('member'), 'o', markersize=6,
                color='plum')
    _set_aeshetics(ax)
    ax.format(**kws)
    plt.legend([r[0], f[0], fm[0]])
