import numpy as np
import xarray as xr

from .checks import has_dims, has_min_len
from .exceptions import DimensionError


def _drop_members(ds, rmd_member=None):
    """
    Drop members by name selection .sel(member=) from ds.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension
        rmd_ensemble (list): list of members to be dropped. Default: [0]

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with less members.

    Raises:
        DimensionError: if list items are not all in ds.member

    """
    if rmd_member is None:
        rmd_member = [0]
    if all(m in ds.member.values for m in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise DimensionError('select available members only')
    return ds.sel(member=member_list)


def _stack_to_supervector(ds, new_dim='svd', stacked_dims=('init', 'member')):
    """
    Stack stacked_dims (default init and member) dimensions
     into one supervector dimension to perform metric over.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        new_dim (str): name of new supervector dimension. Default: 'svd'
        stacked_dims (set): dimensions to be stacked.

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with stacked new_dim
                            dimension.
    """
    return ds.stack({new_dim: stacked_dims})


# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# based on supervector approach
# --------------------------------------------#
def _m2m(ds, supervector_dim='svd', stack_dims=True):
    """
    Create two supervectors to compare all members to all others in turn.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        stack_dims (bool): if True, forecast and reference have member dim and both
                      get stacked to new supervector_dim
                      if False, only forecast has member dim
                      (needed for probabilistic metrics)

    Returns:
        xr.object: forecast, reference.
    """
    supervector_dim2 = 'svd2'
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        # drop the member being reference
        if stack_dims:
            ds_reduced = _drop_members(ds, rmd_member=[m])
        else:
            # TODO: when not stack_dims, m2m create a member vs forecast_member
            # matrix with the diagonal empty if there would be the following line:
            # ds_reduced = _drop_members(ds, rmd_member=[m])
            # if the verification member is not left out (as now), there is one
            # identical comparison, which inflates the skill. To partly fix
            # this there is a m2m correction applied in the end of
            # compute_perfect_model.
            ds_reduced = ds
        reference = ds.sel(member=m).squeeze()
        if stack_dims:
            for m2 in ds_reduced.member:
                reference_list.append(reference)
                forecast_list.append(ds_reduced.sel(member=m2).squeeze())
        else:
            reference_list.append(reference)
            forecast_list.append(ds_reduced)

    if stack_dims:
        reference = xr.concat(reference_list, supervector_dim2).stack(
            svd=(supervector_dim2, 'init')
        )
        forecast = xr.concat(forecast_list, supervector_dim2).stack(
            svd=(supervector_dim2, 'init')
        )
        reference['svd'] = np.arange(1, 1 + reference.svd.size)
        forecast['svd'] = np.arange(1, 1 + forecast.svd.size)
    else:
        supervector_dim = 'forecast_member'
        reference = xr.concat(reference_list, supervector_dim)
        forecast = xr.concat(forecast_list, supervector_dim)
        reference[supervector_dim] = np.arange(1, 1 + reference[supervector_dim].size)
        forecast[supervector_dim] = np.arange(1, 1 + forecast[supervector_dim].size)
    return forecast, reference


def _m2e(ds, supervector_dim='svd', stack_dims=True):
    """
    Create two supervectors to compare all members to ensemble mean while
     leaving out the reference when creating the forecasts.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        stack_dims (bool): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, 'init').rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list, 'init').rename({'init': supervector_dim})
    return forecast, reference


def _m2c(ds, supervector_dim='svd', control_member=None, stack_dims=True):
    """
    Create two supervectors to compare all members to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0
        stack_dims (bool): if True, forecast and reference have member dim and both
                      get stacked to new supervector_dim
                      if False, only forecast has member dim
                      (needed for probabilistic metrics)

    Returns:
        xr.object: forecast, reference.
    """
    if control_member is None:
        control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    if stack_dims:
        forecast, reference = xr.broadcast(ds_dropped, reference)
        forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
        reference = _stack_to_supervector(reference, new_dim=supervector_dim)
    else:
        forecast = ds_dropped
    return forecast, reference


def _e2c(ds, supervector_dim='svd', control_member=None, stack_dims=True):
    """
    Create two supervectors to compare ensemble mean to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0
        stack_dims (bool): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    if control_member is None:
        control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    if 'member' in reference.coords:
        del reference['member']
    reference = reference.rename({'init': supervector_dim})
    # drop the member being reference
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    forecast = ds.mean('member')
    forecast = forecast.rename({'init': supervector_dim})
    return forecast, reference


# --------------------------------------------#
# REFERENCE COMPARISONS
# based on supervector approach
# --------------------------------------------#
def _e2r(ds, reference, stack_dims=True):
    """
    Compare the ensemble mean forecast to a reference in HindcastEnsemble.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.
        stack_dims (bool): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    if 'member' in ds.dims:
        forecast = ds.mean('member')
    else:
        forecast = ds
    return forecast, reference


def _m2r(ds, reference, stack_dims=True):
    """
    Compares each member individually to a reference in HindcastEnsemble.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.
        stack_dims (bool): if True, forecast and reference have member dim and both
                      get stacked to new supervector_dim
                      if False, only forecast has member dim
                      (needed for probabilistic metrics)

    Returns:
        xr.object: forecast, reference.
    """
    # check that this contains more than one member
    has_dims(ds, 'member', 'decadal prediction ensemble')
    has_min_len(ds['member'], 1, 'decadal prediction ensemble member')
    forecast = ds
    if stack_dims:
        reference = reference.expand_dims('member')
        nMember = forecast.member.size
        reference = reference.isel(member=[0] * nMember)
        reference['member'] = forecast['member']
    return forecast, reference
