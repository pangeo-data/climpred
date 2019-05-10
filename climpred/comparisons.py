import numpy as np
import xarray as xr

from .utils import get_dims


def _drop_members(ds, rmd_member=None):
    """Drop members by name selection .sel(member=) from ds.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member dimension
        rmd_ensemble (list): list of members to be dropped. Default: [0]

    Returns:
        ds (xarray object): xr.Dataset/xr.DataArray with less members.

    Raises:
        ValueError: if list items are not all in ds.member

    """
    if rmd_member is None:
        rmd_member = [0]
    if all(m in ds.member.values for m in rmd_member):
        member_list = list(ds.member.values)
        for ens in rmd_member:
            member_list.remove(ens)
    else:
        raise ValueError('select available members only', rmd_member)
    return ds.sel(member=member_list)


def _stack_to_supervector(ds,
                          new_dim='svd',
                          stacked_dims=('init', 'member')):
    """Stack all stacked_dims (likely init and member) dimensions
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


def get_comparison_function(comparison):
    """Converts a string comparison entry from the user into an actual
       function for the package to interpret.

    PERFECT MODEL:
    m2m: Compare all members to all other members.
    m2c: Compare all members to the control.
    m2e: Compare all members to the ensemble mean.
    e2c: Compare the ensemble mean to the control.

    REFERENCE:
    e2r: Compare the ensemble mean to the reference.
    m2r: Compare each ensemble member to the reference.

    Args:
        comparison (str): name of comparison.

    Returns:
        comparison (function): comparison function.

    """
    if comparison == 'm2m':
        comparison = _m2m
    elif comparison == 'm2c':
        comparison = _m2c
    elif comparison == 'm2e':
        comparison = _m2e
    elif comparison == 'e2c':
        comparison = _e2c
    elif comparison == 'e2r':
        comparison = _e2r
    elif comparison == 'm2r':
        comparison = _m2r
    else:
        raise ValueError("""Please supply a comparison from the following list:
            'm2m'
            'm2c'
            'm2e'
            'e2c'
            'e2r'
            'm2r'
            """)
    return comparison

# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# based on supervector approach
# --------------------------------------------#
def _m2m(ds, supervector_dim='svd'):
    """
    Create two supervectors to compare all members to all other members in
    turn.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    supervector_dim2 = 'svd2'
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        # drop the member being reference
        ds_reduced = _drop_members(ds, rmd_member=[m])
        reference = ds.sel(member=m).squeeze()
        for m2 in ds_reduced.member:
            reference_list.append(reference)
            forecast_list.append(
                ds_reduced.sel(member=m2).squeeze())

    reference = xr.concat(reference_list, supervector_dim2).stack(
        svd=(supervector_dim2, 'init'))
    reference['svd'] = np.arange(1, 1+reference.svd.size)
    forecast = xr.concat(forecast_list, supervector_dim2).stack(
        svd=(supervector_dim2, 'init'))
    forecast['svd'] = np.arange(1, 1+forecast.svd.size)
    return forecast, reference


def _m2e(ds, supervector_dim='svd'):
    """
    Create two supervectors to compare all members to ensemble mean while leaving out the reference when creating the forecasts.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = _drop_members(ds, rmd_member=[m]).mean('member')
        reference = ds.sel(member=m).squeeze()
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list,'init').rename({'init': supervector_dim})
    forecast = xr.concat(forecast_list,'init').rename({'init': supervector_dim})
    return forecast, reference


def _m2c(ds, supervector_dim='svd', control_member=None):
    """
    Create two supervectors to compare all members to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    if control_member is None:
        control_member = [0]
    reference = ds.isel(member=control_member).squeeze()
    # drop the member being reference
    ds_dropped = _drop_members(ds, rmd_member=ds.member.values[control_member])
    forecast, reference = xr.broadcast(ds_dropped, reference)
    forecast = _stack_to_supervector(forecast, new_dim=supervector_dim)
    reference = _stack_to_supervector(reference, new_dim=supervector_dim)
    return forecast, reference


def _e2c(ds, supervector_dim='svd', control_member=None):
    """
    Create two supervectors to compare ensemble mean to control.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        supervector_dim (str): name of new supervector dimension.
                               Default: 'svd'
        control_member: list of the one integer member serving as
                        reference. Default 0

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.
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
def _e2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares the
    ensemble mean prediction to the reference (hindcast, simulation,
    observations).

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    if 'member' in get_dims(ds):
        print("Taking ensemble mean...")
        forecast = ds.mean('member')
    else:
        forecast = ds
    return forecast, reference


def _m2r(ds, reference):
    """
    For a reference-based decadal prediction ensemble. This compares each
    member individually to the reference (hindcast, simulation,
    observations).

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        reference (xarray object): reference xr.Dataset/xr.DataArray.

    Returns:
        forecast (xarray object): forecast.
        reference (xarray object): reference.

    """
    # check that this contains more than one member
    if ('member' not in get_dims(ds)) or (ds.member.size == 1):
        raise ValueError("""Please supply a decadal prediction ensemble with
            more than one member. You might have input the ensemble mean here
            although asking for a member-to-reference comparison.""")
    else:
        forecast = ds
    reference = reference.expand_dims('member')
    nMember = forecast.member.size
    reference = reference.isel(member=[0] * nMember)
    reference['member'] = forecast['member']
    return forecast, reference
