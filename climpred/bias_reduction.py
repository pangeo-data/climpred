import logging

import xarray as xr

from climpred.metrics import Metric


def _mean_bias_reduction_quick(hind, bias, dim):
    """Quick reduction of mean bias over all initializations.

    Args:
        hind (xr.object): hindcast.
        bias (xr.object): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.object: bias reduced hind

    """
    bias_reduced_hind = (
        hind.groupby(f"{dim}.dayofyear") - bias.groupby(f"{dim}.dayofyear").mean()
    )
    return bias_reduced_hind


def _mean_bias_reduction_cross_validate(hind, bias, dim):
    """Reduce mean bias from all but the given initialization (cross-validation).

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
        xr.object: bias reduced hind

    Reference:
        * Jolliffe, Ian T., and David B. Stephenson. Forecast Verification: A
          Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley &
          Sons, Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80
    """
    bias = bias.rename({dim: "init"})
    bias_reduced_hind = []
    logging.info("mean bias reduction:")
    for init in hind.init.data:
        hind_drop_init = hind.drop_sel(init=init).init
        hind_drop_init_where_bias = hind_drop_init.where(bias.init)
        logging.info(
            f"initialization {init}: remove bias from"
            f"{hind_drop_init_where_bias.min().values}-"
            f"{hind_drop_init_where_bias.max().values}"
        )
        bias_reduced_hind.append(
            hind.sel(init=init)
            - bias.sel(init=hind_drop_init_where_bias).groupby("init.dayofyear").mean()
        )
    bias_reduced_hind = xr.concat(bias_reduced_hind, "init")
    return bias_reduced_hind


def mean_bias_reduction(hindcast, alignment, cross_validate=True, **metric_kwargs):
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
        cross_validate (bool): Use properly defined mean bias reduction function. This
            excludes the given initialization from the bias calculation. With False,
            include the given initialization in the calculation, which is much faster
            but yields similar skill with a large N of initializations.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias reduced hindcast.

    """

    def bias_func(a, b, **kwargs):
        return a - b

    bias_metric = Metric("bias", bias_func, True, False, 1)

    bias = hindcast.verify(
        metric=bias_metric,
        comparison="e2o",
        dim="init",
        alignment=alignment,
        **metric_kwargs,
    ).squeeze()

    if cross_validate:
        mean_bias_func = _mean_bias_reduction_cross_validate
    else:
        mean_bias_func = _mean_bias_reduction_quick

    bias_reduced_hind = mean_bias_func(hindcast._datasets["initialized"], bias, "init")
    bias_reduced_hind = bias_reduced_hind.squeeze()
    if "dayofyear" in bias_reduced_hind.coords:
        del bias_reduced_hind["dayofyear"]
    hindcast_bias_reduced = hindcast.copy()
    hindcast_bias_reduced._datasets["initialized"] = bias_reduced_hind
    return hindcast_bias_reduced
