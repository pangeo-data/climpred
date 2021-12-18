import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .constants import GROUPBY_SEASONALITIES
from .metrics import Metric
from .options import OPTIONS
from .utils import (
    convert_cftime_to_datetime_coords,
    convert_time_index,
    get_lead_cftime_shift_args,
    shift_cftime_singular,
)

try:
    from bias_correction import XBiasCorrection
except ImportError:
    pass
try:
    from xclim import sdba
except ImportError:
    pass


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def leave_one_out(bias, dim):
    """Leave-one-out creating a new dimension 'sample' and fill with np.NaN.

    See also:
        * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html # noqa: E501
    """
    bias_nan = []
    for i in range(bias[dim].size):
        bias_nan.append(
            bias.drop_isel({dim: i}).reindex({dim: bias[dim]}).rename({dim: "sample"})
        )
    bias_nan = xr.concat(bias_nan, dim).assign_coords({dim: bias[dim]})
    return bias_nan


def leave_one_out_drop(bias, dim):
    """
    Leave-one-out creating a new dimension ``sample``.

    See also: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html. # noqa: E501
    """
    bias_nan = []
    for i in range(bias[dim].size):
        bias_nan.append(
            bias.drop_isel({dim: i}).rename({dim: "sample"}).drop_vars("sample")
        )
    bias_nan = xr.concat(bias_nan, dim).assign_coords(
        {dim: bias[dim], "sample": np.arange(bias[dim].size - 1)}
    )
    return bias_nan


def _mean_bias_removal_func(hind, bias, dim, how):
    """Quick removal of mean bias over all initializations without cross validation.

    Args:
        hind (xr.Dataset): hindcast.
        bias (xr.Dataset): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.Dataset: bias removed hind

    """
    how_operator = sub if how == "additive" else div
    seasonality = OPTIONS["seasonality"]

    with xr.set_options(keep_attrs=True):
        if seasonality == "weekofyear":
            # convert to datetime for weekofyear operations
            hind = convert_cftime_to_datetime_coords(hind, dim)
            bias = convert_cftime_to_datetime_coords(bias, dim)
        hind_groupby = f"{dim}.{seasonality}"

        bias_removed_hind = how_operator(
            hind.groupby(hind_groupby),
            bias.groupby(hind_groupby).mean(),
        )
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind[dim].to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, dim, "hindcast")
    return bias_removed_hind


def _multiplicative_std_correction(hind, spread, dim, obs=None):
    """Quick removal of std bias over all initializations without cross validation.

    Args:
        hind (xr.Dataset): hindcast.
        spread (xr.Dataset): model spread.
        dim (str): Time dimension name in bias.
        obs (xr.Dataset): observations

    Returns:
        xr.Dataset: bias removed hind

    """
    seasonality = OPTIONS["seasonality"]
    if seasonality == "weekofyear":
        # convert to datetime for weekofyear operations
        hind = convert_cftime_to_datetime_coords(hind, "init")
        spread = convert_cftime_to_datetime_coords(spread, "init")
        obs = convert_cftime_to_datetime_coords(obs, "time")

    init_groupby = f"init.{seasonality}"
    obs_groupby = f"time.{seasonality}"

    with xr.set_options(keep_attrs=True):
        model_mean_spread = spread.groupby(init_groupby).mean()
        model_member_mean = hind.mean("member").groupby(init_groupby).mean()
        # assume that no trend here
        obs_spread = obs.groupby(obs_groupby).std()

        # z distr
        init_z = (hind.groupby(init_groupby) - model_member_mean).groupby(
            init_groupby
        ) / model_mean_spread

        # scale with obs_spread and model mean
        init_std_corrected = (init_z.groupby(init_groupby) * obs_spread).groupby(
            init_groupby
        ) + model_member_mean

    init_std_corrected.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(init_std_corrected.init.to_index(), pd.DatetimeIndex):
        init_std_corrected = convert_time_index(init_std_corrected, "init", "hindcast")
    return init_std_corrected


def _std_multiplicative_bias_removal_func_cv(hind, spread, dim, obs, cv="LOO"):
    """Remove std bias from all but the given initialization (cross-validation).

    .. note::
        This method follows Jolliffe 2011. For a given initialization, bias is computed
        over all other initializations, excluding the one in question. This calculated
        bias is removed from the given initialization, and then the process proceeds to
        the following one.

    Args:
        hind (xr.Dataset): hindcast.
        bias (xr.Dataset): bias.
        dim (str): Time dimension name in bias.
        how (str): additive or multiplicative bias.

    Returns:
        xr.Dataset: bias removed hind

    Reference:
        * Jolliffe, Ian T., and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley &
          Sons, Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80
    """
    seasonality = OPTIONS["seasonality"]
    if seasonality == "weekofyear":
        # convert to datetime for weekofyear operations
        hind = convert_cftime_to_datetime_coords(hind, "init")
        spread = convert_cftime_to_datetime_coords(spread, "init")
        obs = convert_cftime_to_datetime_coords(obs, "time")

    bias_removed_hind = []
    for init in hind.init.data:
        hind_drop_init = hind.drop_sel(init=init).init

        with xr.set_options(keep_attrs=True):
            init_groupby = f"init.{seasonality}"
            time_groupby = f"time.{seasonality}"

            model_mean_spread = (
                spread.sel(init=hind_drop_init).groupby(init_groupby).mean()
            )
            model_member_mean = (
                hind.drop_sel(init=init).mean("member").groupby(init_groupby).mean()
            )
            # assume that no trend here
            obs_spread = obs.groupby(time_groupby).std()

            # z distr
            init_z = (
                hind.sel(init=[init]).groupby(init_groupby) - model_member_mean
            ).groupby(init_groupby) / model_mean_spread
            # scale with obs_spread and model mean
            init_std_corrected = (init_z.groupby(init_groupby) * obs_spread).groupby(
                init_groupby
            ) + model_member_mean

        bias_removed_hind.append(init_std_corrected)

    init_std_corrected = xr.concat(bias_removed_hind, "init")

    init_std_corrected.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(init_std_corrected.init.to_index(), pd.DatetimeIndex):
        init_std_corrected = convert_time_index(init_std_corrected, "init", "hindcast")
    return init_std_corrected


def _mean_bias_removal_func_cv(hind, bias, dim, how, cv="LOO"):
    """Remove mean bias from all but the given initialization (cross-validation).

    .. note::
        This method follows Jolliffe 2011. For a given initialization, bias is computed
        over all other initializations, excluding the one in question. This calculated
        bias is removed from the given initialization, and then the process proceeds to
        the following one.

    Args:
        hind (xr.Dataset): hindcast.
        bias (xr.Dataset): bias.
        dim (str): Time dimension name in bias.
        how (str): additive or multiplicative bias.

    Returns:
        xr.Dataset: bias removed hind

    Reference:
        * Jolliffe, Ian T., and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley &
          Sons, Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80
    """
    how_operator = sub if how == "additive" else div
    seasonality = OPTIONS["seasonality"]
    bias = bias.rename({dim: "init"})
    bias_removed_hind = []
    logging.info(f"mean {how} bias removal with seasonality {seasonality}:")
    if seasonality == "weekofyear":
        # convert to datetime for weekofyear operations
        hind = convert_cftime_to_datetime_coords(hind, "init")
        bias = convert_cftime_to_datetime_coords(bias, "init")

    if cv == "LOO":
        for init in hind.init.data:
            hind_drop_init = hind.drop_sel(init=init).init
            with xr.set_options(keep_attrs=True):
                init_groupby = f"init.{seasonality}"
                init_bias_removed = how_operator(
                    hind.sel(init=[init]).groupby(init_groupby),
                    bias.sel(
                        init=hind_drop_init.to_index().intersection(
                            bias.init.to_index()
                        )
                    )
                    .groupby(init_groupby)
                    .mean(),
                )
                if seasonality in init_bias_removed.coords:
                    del init_bias_removed.coords[seasonality]
            bias_removed_hind.append(init_bias_removed)
        bias_removed_hind = xr.concat(bias_removed_hind, "init")
    else:
        raise NotImplementedError(f'try cv="LOO", found {cv}')
    bias_removed_hind.attrs = hind.attrs
    # convert back to CFTimeIndex if needed
    if isinstance(bias_removed_hind.init.to_index(), pd.DatetimeIndex):
        bias_removed_hind = convert_time_index(bias_removed_hind, "init", "hindcast")
    return bias_removed_hind


def gaussian_bias_removal(
    hindcast,
    alignment,
    cv=False,
    how="additive_mean",
    train_test_split="fair",
    train_time=None,
    train_init=None,
    **metric_kwargs,
):
    """Calc bias based on ``OPTIONS['seasonality']`` and remove bias from
    py:class:`~climpred.classes.HindcastEnsemble`.

    Args:
        hindcast (HindcastEnsemble): hindcast.
        alignment (str): which inits or verification times should be aligned?

            - ``maximize``: maximize the degrees of freedom by slicing ``initialized``
                and ``verif`` to a common time frame at each lead.
            - ``same_inits``: slice to a common ``init`` frame prior to computing
                metric. This philosophy follows the thought that each lead should be
                based on the same set of initializations.
            - ``same_verif``: slice to a common/consistent verification time frame prior
                to computing metric. This philosophy follows the thought that each lead
                should be based on the same set of verification dates.

        how (str): what kind of bias removal to perform. Select
            from ``['additive_mean', 'multiplicative_mean','multiplicative_std']``.
            Defaults to ``'additive_mean'``.
        cv (bool or str): Defaults to ``True``.

            - ``True``: Use cross validation in bias removal function.
                This excludes the given initialization from the bias calculation.
            - ``'LOO'``: see ``True``
            - ``False``: include the given initialization in the calculation, which
                is much faster and but yields similar skill with a large N of
                initializations.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    """
    if train_test_split == "fair":
        hindcast_train = hindcast.copy()
        hindcast_test = hindcast.copy()
        if alignment in ["same_inits", "maximize"]:
            hindcast_train._datasets["initialized"] = hindcast.get_initialized().sel(
                init=train_init
            )  # for bias
            hindcast_test._datasets[
                "initialized"
            ] = hindcast.get_initialized().drop_sel(
                init=train_init
            )  # to reduce bias
        if alignment in ["same_verif"]:
            train_time = hindcast.coords["time"].sel(time=train_time).to_index()
            # add inits before lead.max()
            n, freq = get_lead_cftime_shift_args(
                hindcast.coords["lead"].attrs["units"], hindcast.coords["lead"].max()
            )
            train_time_init = train_time.union(train_time.shift(-n, freq)).intersection(
                hindcast.coords["init"].to_index()
            )
            hindcast_train._datasets["initialized"] = hindcast.get_initialized().sel(
                init=train_time_init
            )
            hindcast_test._datasets[
                "initialized"
            ] = hindcast.get_initialized().drop_sel(init=train_time_init)
    else:
        assert train_test_split in ["unfair", "unfair-cv"]
        hindcast_train = hindcast
        hindcast_test = hindcast

    if "mean" in how:
        # calculate bias lead-time dependent
        bias = hindcast_train.verify(
            metric="unconditional_bias" if how == "additive_mean" else "mul_bias",
            comparison="e2o",
            dim=[],  # not used by bias func, therefore best to add [] here
            alignment=alignment,
            **metric_kwargs,
        )

    if how == "multiplicative_std":
        bias = hindcast_train.verify(
            metric="spread",
            comparison="m2o",
            dim="member",
            alignment=alignment,
        )
    bias = bias.drop_vars("skill")

    # how to remove bias
    if "mean" in how:
        if cv in [False, None]:
            bias_removal_func = _mean_bias_removal_func
            bias_removal_func_kwargs = dict(how=how.split("_")[0])
        else:
            bias_removal_func = _mean_bias_removal_func_cv
            bias_removal_func_kwargs = dict(how=how.split("_")[0], cv=cv)

    elif how == "multiplicative_std":
        if cv in [False, None]:
            bias_removal_func = _multiplicative_std_correction
            bias_removal_func_kwargs = dict(
                obs=hindcast.get_observations(),
            )
        else:
            bias_removal_func = _std_multiplicative_bias_removal_func_cv
            bias_removal_func_kwargs = dict(obs=hindcast.get_observations(), cv=cv)

    hind = hindcast_test.get_initialized()
    if OPTIONS["seasonality"] == "weekofyear":
        hind, bias = xr.align(hind, bias)

    bias_removed_hind = bias_removal_func(
        hind, bias, "init", **bias_removal_func_kwargs
    ).squeeze(drop=True)

    # remove groupby label from coords
    for c in GROUPBY_SEASONALITIES + ["skill"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind
    return hindcast_bias_removed


def bias_correction(
    hindcast,
    alignment,
    cv=False,
    how="normal_mapping",
    train_test_split="fair",
    train_time=None,
    train_init=None,
    **metric_kwargs,
):
    """Calc bias based on OPTIONS['seasonality'] and remove bias from
    py:class:`~climpred.classes.HindcastEnsemble`.

    Args:
        hindcast (HindcastEnsemble): hindcast.
        alignment (str): which inits or verification times should be aligned?

            - ``maximize``: maximize the degrees of freedom by slicing ``initialized``
                and ``verif`` to a common time frame at each lead.
            - ``same_inits``: slice to a common ``init`` frame prior to computing
                metric. This philosophy follows the thought that each lead should be
                based on the same set of initializations.
            - ``same_verif``: slice to a common/consistent verification time frame prior
                to computing metric. This philosophy follows the thought that each lead
                should be based on the same set of verification dates.

        how (str): what kind of bias removal to perform. Select
            from ``['additive_mean', 'multiplicative_mean','multiplicative_std']``.
            Defaults to ``'additive_mean'``.
        cv (bool): Use cross validation in bias removal function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to ``True``.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    """

    def bc_func(
        forecast,
        observations,
        dim=None,
        method=how,
        cv=False,
        **metric_kwargs,
    ):
        """Wrapping
        https://github.com/pankajkarman/bias_correction/blob/master/bias_correction.py.

        Functions to perform bias correction of datasets to remove biases across
        datasets. Implemented methods include:
        - quantile_mapping:
            https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2168)
        - modified quantile mapping:
            https://www.sciencedirect.com/science/article/abs/pii/S0034425716302000?via%3Dihub # noqa: E501
        - scaled distribution mapping (Gamma and Normal Corrections):
            https://www.hydrol-earth-syst-sci.net/21/2649/2017/
        """
        corrected = []
        seasonality = OPTIONS["seasonality"]
        dim = "time"
        if seasonality == "weekofyear":
            forecast = convert_cftime_to_datetime_coords(forecast, dim)
            observations = convert_cftime_to_datetime_coords(observations, dim)

        if train_test_split in ["fair"]:
            if alignment in ["same_inits", "maximize"]:
                train_dim = train_init.rename({"init": "time"})
                # shift init to time
                n, freq = get_lead_cftime_shift_args(
                    forecast.lead.attrs["units"], forecast.lead
                )
                train_dim = shift_cftime_singular(train_dim[dim], n, freq)
                data_to_be_corrected = forecast.drop_sel({dim: train_dim})
            elif alignment in ["same_verif"]:
                train_dim = train_time
                intersection = (
                    train_dim[dim].to_index().intersection(forecast[dim].to_index())
                )
                data_to_be_corrected = forecast.drop_sel({dim: intersection})

            intersection = (
                train_dim[dim].to_index().intersection(forecast[dim].to_index())
            )
            forecast = forecast.sel({dim: intersection})
            reference = observations.sel({dim: intersection})
        else:
            model = forecast
            data_to_be_corrected = forecast
            reference = observations

        data_to_be_corrected_ori = data_to_be_corrected.copy()

        for label, group in forecast.groupby(f"{dim}.{seasonality}"):
            reference = observations.sel({dim: group[dim]})
            model = forecast.sel({dim: group[dim]})
            if train_test_split in ["unfair", "unfair-cv"]:
                # take all
                data_to_be_corrected = forecast.sel({dim: group[dim]})
            else:
                group_dim_data_to_be_corrected = (
                    getattr(data_to_be_corrected_ori[dim].dt, seasonality) == label
                )
                data_to_be_corrected = data_to_be_corrected_ori.sel(
                    {dim: group_dim_data_to_be_corrected}
                )

            if cv == "LOO" and train_test_split == "unfair-cv":
                reference = leave_one_out(reference, dim)
                model = leave_one_out(model, dim)
                data_to_be_corrected = leave_one_out(data_to_be_corrected, dim)

            dim2 = "time_member"
            if "member" in model.dims:
                reference = reference.broadcast_like(model)
                data_to_be_corrected = data_to_be_corrected.broadcast_like(model)
                model = model.stack({dim2: ["time", "member"]})
                reference = reference.stack({dim2: ["time", "member"]})
                data_to_be_corrected = data_to_be_corrected.stack(
                    {dim2: ["time", "member"]}
                )
            dim_used = dim2 if "member" in forecast.dims else dim

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bc = XBiasCorrection(
                    reference,
                    model,
                    data_to_be_corrected,
                    dim=dim_used,
                )
                c = bc.correct(method=method, join="outer", **metric_kwargs)
            if dim2 in c.dims:
                c = c.unstack(dim2)
            if cv and dim in c.dims and "sample" in c.dims:
                c = c.mean(dim)
                c = c.rename({"sample": dim})
            # select only where data_to_be_corrected was input
            if dim2 in data_to_be_corrected.dims:
                data_to_be_corrected = data_to_be_corrected.unstack(dim2)
            c = c.sel({dim: data_to_be_corrected[dim]})
            corrected.append(c)
        corrected = xr.concat(corrected, dim).sortby(dim)
        # convert back to CFTimeIndex if needed
        if isinstance(corrected[dim].to_index(), pd.DatetimeIndex):
            corrected = convert_time_index(corrected, dim, "hindcast")
        return corrected

    bc = Metric(
        "bias_correction", bc_func, positive=False, probabilistic=False, unit_power=1
    )

    # calculate bias lead-time dependent
    bias_removed_hind = hindcast.verify(
        metric=bc,
        comparison="m2o" if "member" in hindcast.dims else "e2o",
        dim=[],  # set internally inside bc
        alignment=alignment,
        cv=cv,
        **metric_kwargs,
    ).squeeze(drop=True)

    # remove groupby label from coords
    for c in GROUPBY_SEASONALITIES + ["skill"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # keep attrs
    bias_removed_hind.attrs = hindcast.get_initialized().attrs
    for v in bias_removed_hind.data_vars:
        bias_removed_hind[v].attrs = hindcast.get_initialized()[v].attrs

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind

    return hindcast_bias_removed


def xclim_sdba(
    hindcast,
    alignment,
    cv=False,
    how="DetrendedQuantileMapping",
    train_test_split="fair",
    train_time=None,
    train_init=None,
    **metric_kwargs,
):
    """Calc bias based on ``grouper`` to be passed as ``metric_kwargs`` and remove bias
    from py:class:`~climpred.classes.HindcastEnsemble`.

    See :py:func:`~climpred.constants.XCLIM_BIAS_CORRECTION_METHODS` for implemented
    methods for ``how``.

    Args:
        hindcast (HindcastEnsemble): hindcast.
        alignment (str): which inits or verification times should be aligned?

            - ``maximize``: maximize the degrees of freedom by slicing ``initialized``
                and ``verif`` to a common time frame at each lead.
            - ``same_inits``: slice to a common ``init`` frame prior to computing
                metric. This philosophy follows the thought that each lead should be
                based on the same set of initializations.
            - ``same_verif``: slice to a common/consistent verification time frame prior
                to computing metric. This philosophy follows the thought that each lead
                should be based on the same set of verification dates.

        how (str): methods for bias reduction, see
            :py:func:`~climpred.constants.XCLIM_BIAS_CORRECTION_METHODS`
        cv (bool): Use cross validation in removal function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias removed hindcast.

    """

    def bc_func(
        forecast,
        observations,
        dim=None,
        method=how,
        cv=False,
        **metric_kwargs,
    ):
        """Wrapping
        https://github.com/Ouranosinc/xclim/blob/master/xclim/sdba/adjustment.py.

        Functions to perform bias correction of datasets to remove biases across
        datasets. See :py:func:`~climpred.constants.XCLIM_BIAS_CORRECTION_METHODS`
        for implemented methods.
        """
        seasonality = OPTIONS["seasonality"]
        dim = "time"
        if seasonality == "weekofyear":
            forecast = convert_cftime_to_datetime_coords(forecast, dim)
            observations = convert_cftime_to_datetime_coords(observations, dim)

        if train_test_split in ["fair"]:
            if alignment in ["same_inits", "maximize"]:
                train_dim = train_init.rename({"init": "time"})
                # shift init to time
                n, freq = get_lead_cftime_shift_args(
                    forecast.lead.attrs["units"], forecast.lead
                )
                train_dim = shift_cftime_singular(train_dim[dim], n, freq)
                data_to_be_corrected = forecast.drop_sel({dim: train_dim})
            elif alignment in ["same_verif"]:
                train_dim = train_time
                intersection = (
                    train_dim[dim].to_index().intersection(forecast[dim].to_index())
                )
                data_to_be_corrected = forecast.drop_sel({dim: intersection})

            intersection = (
                train_dim[dim].to_index().intersection(forecast[dim].to_index())
            )
            forecast = forecast.sel({dim: intersection})
            model = forecast
            reference = observations.sel({dim: intersection})
        else:
            model = forecast
            data_to_be_corrected = forecast
            reference = observations

        if train_test_split in ["unfair", "unfair-cv"]:
            # take all
            data_to_be_corrected = forecast

        if cv == "LOO" and train_test_split == "unfair-cv":
            reference = leave_one_out(reference, dim)
            model = leave_one_out(model, dim)
            data_to_be_corrected = leave_one_out(data_to_be_corrected, dim)

        if "group" not in metric_kwargs:
            metric_kwargs["group"] = dim + "." + OPTIONS["seasonality"]
        elif metric_kwargs["group"] is None:
            metric_kwargs["group"] = dim + "." + OPTIONS["seasonality"]

        if "init" in metric_kwargs["group"]:
            metric_kwargs["group"] = metric_kwargs["group"].replace("init", "time")
        if "member" in model.dims:
            metric_kwargs["add_dims"] = ["member"]
            if "member" not in reference.dims:
                reference = reference.expand_dims(member=[model.member[0]])

        adjust_kwargs = {}
        for k in ["interp", "extrapolation", "detrend"]:
            if k in metric_kwargs:
                adjust_kwargs[k] = metric_kwargs.pop(k)

        def adjustment(reference, model, data_to_be_corrected):
            dqm = getattr(sdba.adjustment, method).train(
                reference, model, **metric_kwargs
            )
            data_to_be_corrected = dqm.adjust(data_to_be_corrected, **adjust_kwargs)
            return data_to_be_corrected

        del model.coords["lead"]

        c = xr.Dataset()
        for v in model.data_vars:
            c[v] = adjustment(reference[v], model[v], data_to_be_corrected[v])

        if cv and dim in c.dims and "sample" in c.dims:
            c = c.mean(dim)
            c = c.rename({"sample": dim})
        # select only where data_to_be_corrected was input
        corrected = c.sel({dim: data_to_be_corrected[dim]})
        # convert back to CFTimeIndex if needed
        if isinstance(corrected[dim].to_index(), pd.DatetimeIndex):
            corrected = convert_time_index(corrected, dim, "hindcast")
        return corrected

    bc = Metric(
        "bias_correction", bc_func, positive=False, probabilistic=False, unit_power=1
    )

    # calculate bias lead-time dependent
    bias_removed_hind = hindcast.verify(
        metric=bc,
        comparison="m2o" if "member" in hindcast.dims else "e2o",
        dim=[],  # set internally inside bc
        alignment=alignment,
        cv=cv,
        **metric_kwargs,
    ).squeeze(drop=True)

    # remove groupby label from coords
    for c in GROUPBY_SEASONALITIES + ["skill"]:
        if c in bias_removed_hind.coords and c not in bias_removed_hind.dims:
            del bias_removed_hind.coords[c]

    # keep attrs
    bias_removed_hind.attrs = hindcast.get_initialized().attrs
    for v in bias_removed_hind.data_vars:
        bias_removed_hind[v].attrs = hindcast.get_initialized()[v].attrs

    # replace raw with bias reducted initialized dataset
    hindcast_bias_removed = hindcast.copy()
    hindcast_bias_removed._datasets["initialized"] = bias_removed_hind

    return hindcast_bias_removed
