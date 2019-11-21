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


# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# based on supervector approach
# --------------------------------------------#
def _m2m(ds, stack_dims=True):
    """
    Create two supervectors to compare all members to all others in turn.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        stack_dims (bool): if True, forecast and reference have member dim
                      if False, only forecast has member dim
                      (needed for probabilistic metrics)

    Returns:
        xr.object: forecast, reference.
    """
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        if stack_dims:
            forecast = _drop_members(ds, rmd_member=[m])
        else:
            # TODO: when not stack_dims, m2m create a member vs forecast_member
            # matrix with the diagonal empty if there would be the following line:
            # forecast = _drop_members(ds, rmd_member=[m])
            # if the verification member is not left out (as now), there is one
            # identical comparison, which inflates the skill. To partly fix
            # this there is a m2m correction applied in the end of
            # compute_perfect_model.
            forecast = ds
        reference = ds.sel(member=m).squeeze()
        if stack_dims:
            forecast, reference = xr.broadcast(forecast, reference)
        reference_list.append(reference)
        forecast_list.append(forecast)
    supervector_dim = 'forecast_member'
    reference = xr.concat(reference_list, supervector_dim)
    forecast = xr.concat(forecast_list, supervector_dim)
    reference[supervector_dim] = np.arange(reference[supervector_dim].size)
    forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
    return forecast, reference


def _m2e(ds, stack_dims=True):
    # stack_dims
    """
    Create two supervectors to compare all members to ensemble mean while
     leaving out the reference when creating the forecasts.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        stack_dims (bool): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    reference_list = []
    forecast_list = []
    supervector_dim = 'member'
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, supervector_dim)
    forecast = xr.concat(forecast_list, supervector_dim)
    forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
    reference[supervector_dim] = np.arange(reference[supervector_dim].size)
    return forecast, reference


def _m2c(ds, control_member=None, stack_dims=True):
    """
    Create two supervectors to compare all members to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        control_member: list of the one integer member serving as
                        reference. Default 0
        stack_dims (bool): if True, forecast and reference have member dim
                      if False, only forecast has member dim
                      (needed for probabilistic metrics)

    Returns:
        xr.object: forecast, reference.
    """
    if control_member is None:
        control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    forecast = _drop_members(ds, rmd_member=ds.member.values[control_member])
    if stack_dims:
        forecast, reference = xr.broadcast(forecast, reference)
    return forecast, reference


def _e2c(ds, control_member=None, stack_dims=True):
    """
    Create two supervectors to compare ensemble mean to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        control_member: list of the one integer member serving as
                        reference. Default 0
        stack_dims (bool): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    # stack_dim irrelevant
    if control_member is None:
        control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    if 'member' in reference.coords:
        del reference['member']
    ds = _drop_members(ds, rmd_member=[ds.member.values[control_member]])
    forecast = ds.mean('member')
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
        stack_dims (bool): if True, forecast and reference have member dim and
                           both; if False, only forecast has member dim
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
