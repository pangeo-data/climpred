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
    bias_reduced_hind = hind - bias.mean(dim)
    return bias_reduced_hind


def _mean_bias_reduction_properly(hind, bias, dim):
    """Reduce mean bias from all but the given initialization (cross-validation).

    Args:
        hind (xr.object): hindcast.
        bias (xr.object): bias.
        dim (str): Time dimension name in bias.

    Returns:
        xr.object: bias reduced hind

    Reference:
    - Jolliffe, Ian T., and David B. Stephenson. Forecast Verification:
      A Practitioner’s Guide in Atmospheric Science. Chichester, UK: John Wiley & Sons,
      Ltd, 2011. https://doi.org/10.1002/9781119960003., Chapter: 5.3.1, p.80

    """
    bias = bias.rename({dim: 'init'})
    bias_reduced_hind = []
    logging.info('mean bias reduction:')
    for init in hind.init.values:
        hind_drop_init = hind.init.drop_sel(init=init)
        hind_drop_init_where_bias = hind_drop_init.where(bias.init)
        logging.info(
            f'initialization {init}: remove bias from'
            f'{hind_drop_init_where_bias.min().values}-'
            f'{hind_drop_init_where_bias.max().values}'
        )
        bias_reduced_hind.append(
            hind.sel(init=init) - bias.sel(init=hind_drop_init_where_bias).mean('init')
        )
    bias_reduced_hind = xr.concat(bias_reduced_hind, 'init')
    return bias_reduced_hind


def mean_bias_reduction(hindcast, slow=True):
    """Calc and remove bias from HindcastEnsemble.

    Args:
        hindcast (HindcastEnsemble): hindcast.
        slow (bool): Use slow and properly defined mean bias reduction function.
            Defaults to True.

    Returns:
        HindcastEnsemble: bias reduced hindcast.

    """

    def bias_func(a, b, **kwargs):
        return a - b

    bias_metric = Metric('bias', bias_func, True, False, 1)
    comparison = 'e2r'

    bias = hindcast.verify(
        metric=bias_metric, comparison=comparison, dim='member'
    ).squeeze()

    if slow:
        mean_bias_func = _mean_bias_reduction_quick
    else:
        mean_bias_func = _mean_bias_reduction_properly
    bias_reduced_hind_quick = mean_bias_func(
        hindcast._datasets['initialized'], bias, 'time'
    )

    hindcast_bias_reduced_quick = hindcast.copy()
    hindcast_bias_reduced_quick._datasets['initialized'] = bias_reduced_hind_quick
    return hindcast_bias_reduced_quick
