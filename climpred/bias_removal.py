import logging
import warnings

import pandas as pd
import xarray as xr

from .metrics import Metric
from .options import OPTIONS
from .utils import convert_cftime_to_datetime_coords, convert_time_index


def _mean_bias_removal_quick(hind, bias, dim):
    """Quick removal of mean bias over all initializations.

    Args:
        hind (xr.object): hindcast.
        bias (xr.object): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.object: bias removed hind

    """
    seasonality_str = OPTIONS["seasonality"]
    with xr.set_options(keep_attrs=True):
        if seasonality_str == "weekofyear":
            # convert to datetime for weekofyear operations, now isocalendar().week
            hind = convert_cftime_to_datetime_coords(hind, dim)
            bias = convert_cftime_to_datetime_coords(bias, dim)
            bias_removed_hind = (
                hind.groupby(hind[dim].dt.isocalendar().week)
                - bias.groupby(bias[dim].dt.isocalendar().week).mean()
            )
        else:  # dayofyear month season
            bias_removed_hind = (
                hind.groupby(f"{dim}.{seasonality_str}")
                - bias.groupby(f"{dim}.{seasonality_str}").mean()
            )
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind[dim].to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, dim, "hindcast")
    return bias_removed_hind


def _mean_bias_removal_cross_validate(hind, bias, dim):
    """Remove mean bias from all but the given initialization (cross-validation).

    .. note::
        This method follows Jolliffe 2011. For a given initialization, bias is computed
        over all other initializations, excluding the one in question. This calculated
        bias is removed from the given initialization, and then the process proceeds to
        the following one.

    Args:
        hind (xr.object): hindcast.
        bias (xr.object): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.object: bias removed hind

    Reference:
        * Jolliffe, Ian T., and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley &
          Sons, Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80
    """
    seasonality_str = OPTIONS["seasonality"]
    bias = bias.rename({dim: "init"})
    bias_removed_hind = []
    logging.info("mean bias removal:")
    if seasonality_str == "weekofyear":
        # convert to datetime for weekofyear operations, now isocalendar().week
        hind = convert_cftime_to_datetime_coords(hind, "init")
        bias = convert_cftime_to_datetime_coords(bias, "init")

    for init in hind.init.data:
        hind_drop_init = hind.drop_sel(init=init).init
        hind_drop_init_where_bias = hind_drop_init.where(bias.init)
        logging.info(
            f"initialization {init}: remove bias from"
            f"{hind_drop_init_where_bias.min().values}-"
            f"{hind_drop_init_where_bias.max().values}"
        )
        with xr.set_options(keep_attrs=True):
            if seasonality_str == "weekofyear":
                init_bias_removed = (
                    hind.sel(init=[init])
                    - bias.sel(init=hind_drop_init_where_bias)
                    .groupby(
                        bias.sel(init=hind_drop_init_where_bias)
                        .init.dt.isocalendar()
                        .week
                    )
                    .mean()
                )
            else:  # dayofyear month
                init_bias_removed = (
                    hind.sel(init=init)
                    - bias.sel(init=hind_drop_init_where_bias)
                    .groupby(f"init.{seasonality_str}")
                    .mean()
                )
        bias_removed_hind.append(init_bias_removed)
    bias_removed_hind = xr.concat(bias_removed_hind, "init")
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind.init.to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, "init", "hindcast")
    return bias_removed_hind


def mean_bias_removal(hindcast, alignment, cross_validate=True, **metric_kwargs):
    """Calc and remove bias from py:class:`~climpred.classes.HindcastEnsemble`.

    Args:
        hindcast (HindcastEnsemble): hindcast.
        alignment (str): which inits or verification times should be aligned?
            - maximize/None: maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.
            - same_inits: slice to a common init frame prior to computing
            metric. This philosophy follows the thought that each lead should be
            based on the same set of initializations.
            - same_verif: slice to a common/consistent verification time frame prior
            to computing metric. This philosophy follows the thought that each lead
            should be based on the same set of verification dates.
        cross_validate (bool): Use properly defined mean bias removal function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    """
    if hindcast.get_initialized().lead.attrs["units"] != "years":
        warnings.warn(
            "HindcastEnsemble.remove_bias() is still experimental and is only tested "
            "for annual leads. Please consider contributing to "
            "https://github.com/pangeo-data/climpred/issues/605"
        )

    def bias_func(a, b, **kwargs):
        return a - b

    bias_metric = Metric("bias", bias_func, True, False, 1)

    # calculate bias lead-time dependent
    bias = hindcast.verify(
        metric=bias_metric,
        comparison="e2o",
        dim=[],  # not used by bias func, therefore best to add [] here
        alignment=alignment,
        **metric_kwargs,
    ).squeeze()

    # how to remove bias
    if cross_validate:  # more correct
        mean_bias_func = _mean_bias_removal_cross_validate
    else:  # faster
        mean_bias_func = _mean_bias_removal_quick

    bias_removed_hind = mean_bias_func(hindcast._datasets["initialized"], bias, "init")
    bias_removed_hind = bias_removed_hind.squeeze()
    # remove groupby label from coords
    for c in ["dayofyear", "skill", "week", "month"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind
    return hindcast_bias_removed
