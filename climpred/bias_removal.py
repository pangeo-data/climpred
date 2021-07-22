import logging
import warnings

import pandas as pd
import xarray as xr

from .constants import EXTERNAL_BIAS_CORRECTION_METHODS
from .metrics import Metric
from .options import OPTIONS
from .utils import convert_cftime_to_datetime_coords, convert_time_index


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def _mean_additive_bias_removal_func(hind, bias, dim, how):
    """Quick removal of mean bias over all initializations without cross validation.

    Args:
        hind (xr.object): hindcast.
        bias (xr.object): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.object: bias removed hind

    """
    how_operator = sub if how == "additive" else div
    seasonality_str = OPTIONS["seasonality"]
    with xr.set_options(keep_attrs=True):
        if seasonality_str == "weekofyear":
            # convert to datetime for weekofyear operations, now isocalendar().week
            hind = convert_cftime_to_datetime_coords(hind, dim)
            bias = convert_cftime_to_datetime_coords(bias, dim)
            bias_removed_hind = how_operator(
                hind.groupby(hind[dim].dt.isocalendar().week),
                bias.groupby(bias[dim].dt.isocalendar().week).mean(),
            )
        else:  # dayofyear month season
            bias_removed_hind = how_operator(
                hind.groupby(f"{dim}.{seasonality_str}"),
                bias.groupby(f"{dim}.{seasonality_str}").mean(),
            )
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind[dim].to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, dim, "hindcast")
    return bias_removed_hind


def _multiplicative_std_correction_quick(hind, spread, dim, obs=None):
    """Quick removal of std bias over all initializations without cross validation.

    Args:
        hind (xr.object): hindcast.
        spread (xr.object): model spread.
        dim (str): Time dimension name in bias.
        obs (xr.object): observations

    Returns:
        xr.object: bias removed hind

    """
    seasonality_str = OPTIONS["seasonality"]
    with xr.set_options(keep_attrs=True):
        model_spread = spread.groupby(f"init.{seasonality_str}").mean()
        model_member_mean = (
            hind.mean("member").groupby(f"init.{seasonality_str}").mean()
        )
        # assume that no trend here
        obs_spread = obs.groupby(f"time.{seasonality_str}").std()

        # z distr
        init_z = (hind.groupby(f"init.{seasonality_str}") - model_member_mean).groupby(
            f"init.{seasonality_str}"
        ) / model_spread
        # scale with obs_spread and model mean
        init_std_corrected = (
            init_z.groupby(f"init.{seasonality_str}") * obs_spread
        ).groupby(f"init.{seasonality_str}") + model_member_mean
    init_std_corrected.attrs = hind.attrs
    return init_std_corrected


def _mean_additive_bias_removal_func_cross_validate(hind, bias, dim, how):
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
        how (str): additive or multiplicative bias.

    Returns:
        xr.object: bias removed hind

    Reference:
        * Jolliffe, Ian T., and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley &
          Sons, Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80
    """
    how_operator = sub if how == "additive" else div
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
                init_bias_removed = how_operator(
                    hind.sel(init=[init]),
                    bias.sel(init=hind_drop_init_where_bias)
                    .groupby(
                        bias.sel(init=hind_drop_init_where_bias)
                        .init.dt.isocalendar()
                        .week
                    )
                    .mean(),
                )
            else:  # dayofyear month
                init_bias_removed = how_operator(
                    hind.sel(init=init),
                    bias.sel(init=hind_drop_init_where_bias)
                    .groupby(f"init.{seasonality_str}")
                    .mean(),
                )
        bias_removed_hind.append(init_bias_removed)
    bias_removed_hind = xr.concat(bias_removed_hind, "init")
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind.init.to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, "init", "hindcast")
    return bias_removed_hind


def mean_bias_removal(
    hindcast, alignment, cross_validate=True, how="additive_mean", **metric_kwargs
):
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
        how (str): what kind of bias removal to perform. Select
            from ['additive_mean', 'multiplicative_mean','multiplicative_std']. Defaults to 'additive_mean'.
        cross_validate (bool): Use properly defined mean bias removal function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    """
    if OPTIONS["seasonality"] not in ["month"]:
        warnings.warn(
            "HindcastEnsemble.remove_bias() is still experimental and is only tested "
            "for seasonality in ['month']. Please consider contributing to "
            "https://github.com/pangeo-data/climpred/issues/605"
        )

    # Todo: refactor into metrics
    def additive_mean_bias_func(a, b, **kwargs):
        return a - b

    additive_mean_bias_metric = Metric(
        "additive_bias", additive_mean_bias_func, True, False, 1
    )

    def multiplicative_mean_bias_removal_func(a, b, **kwargs):
        return a / b

    multiplicative_mean_bias_metric = Metric(
        "multiplicative_mean_bias",
        multiplicative_mean_bias_removal_func,
        True,
        False,
        1,
    )

    def multiplicative_std_bias_func(forecast, obs, dim=None, **metric_kwargs):
        """Model spread"""
        return forecast.std(dim, **metric_kwargs)

    multiplicative_std_bias = Metric(
        "multiplicative_std_bias", multiplicative_std_bias_func, True, False, 0
    )

    if "mean" in how:
        bias_metric = (
            additive_mean_bias_metric
            if how == "additive_mean"
            else multiplicative_mean_bias_metric
        )

        # calculate bias lead-time dependent
        bias = hindcast.verify(
            metric=bias_metric,
            comparison="e2o",
            dim=[],  # not used by bias func, therefore best to add [] here
            alignment=alignment,
            **metric_kwargs,
        ).squeeze()
        # todo: squeeze needed? replace with sel(skill='initialized')

    if how == "multiplicative_std":
        bias = hindcast.verify(
            metric=multiplicative_std_bias,
            comparison="m2o",
            dim="member",
            alignment=alignment,
        )  # model spread

    # how to remove bias
    if "mean" in how:
        if cross_validate:  # more correct
            bias_removal_func = _mean_additive_bias_removal_func_cross_validate
        else:  # faster
            bias_removal_func = _mean_additive_bias_removal_func
        bias_removal_func_kwargs = dict(how=how.split("_")[0])
    elif how == "multiplicative_std":
        if cross_validate:
            assert False
        else:
            bias_removal_func = _multiplicative_std_correction_quick
            bias_removal_func_kwargs = dict(obs=hindcast.get_observations())

    bias_removed_hind = bias_removal_func(
        hindcast.get_initialized(), bias, "init", **bias_removal_func_kwargs
    )
    bias_removed_hind = bias_removed_hind.squeeze()
    # remove groupby label from coords
    for c in ["season", "dayofyear", "skill", "week", "month"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind

    return hindcast_bias_removed


from .bias_correction import XBiasCorrection


def _bias_correction(
    hindcast, alignment, cross_validate=True, how="quantile_mapping", **metric_kwargs
):
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
        how (str): what kind of bias removal to perform. Select
            from ['additive_mean', 'multiplicative_mean','multiplicative_std']. Defaults to 'additive_mean'.
        cross_validate (bool): Use properly defined mean bias removal function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    Todo:
    - groupby
    """
    if OPTIONS["seasonality"] not in ["month"]:
        warnings.warn(
            "HindcastEnsemble.remove_bias() is still experimental and is only tested "
            "for seasonality in ['month']. Please consider contributing to "
            "https://github.com/pangeo-data/climpred/issues/605"
        )

    # Todo: refactor into metrics
    def additive_mean_bias_func(a, b, **kwargs):
        return a - b

    def bc_func(forecast, observations, dim=None, method=how, **metric_kwargs):
        """Metrics are used after alignment.
        Todo:
        - nicer way to map to many variables?
        """
        if (
            "member" in forecast.dims
        ):  # broadcast member dim to observations, then stack to use all members for training
            forecast, observations = xr.broadcast(forecast, observations)
            forecast = forecast.stack(time2=["time", "member"])
            observations = observations.stack(time2=["time", "member"])
            dim = "time2"
        else:
            dim = "time"
        reference = observations
        # no cross val
        model = forecast
        data_to_be_corrected = forecast
        # using bias-correction
        bc = XBiasCorrection(reference, model, data_to_be_corrected, dim=dim)
        corrected = bc.correct(method=method)
        return corrected.unstack(dim)

    bc = Metric(
        "bias_correction", bc_func, positive=False, probabilistic=False, unit_power=1
    )

    # calculate bias lead-time dependent
    bias_removed_hind = hindcast.verify(
        metric=bc,
        comparison="m2o" if "member" in hindcast.dims else "e2o",
        dim=[],  # not used by bias func, therefore best to add [] here
        alignment=alignment,
        **metric_kwargs,
    ).squeeze()
    # todo: squeeze needed? replace with sel(skill='initialized')

    # if cross_validate:  # more correct
    bias_removed_hind = bias_removed_hind.squeeze()
    # remove groupby label from coords
    for c in ["season", "dayofyear", "skill", "week", "month"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind

    return hindcast_bias_removed
