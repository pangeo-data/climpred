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


def _display_comparison_metadata(self):
    summary = '----- Comparison metadata -----\n'
    summary += f'Name: {self.name}\n'
    # probabilistic or only deterministic
    if not self.probabilistic:
        summary += 'Kind: deterministic\n'
    else:
        summary += 'Kind: deterministic and probabilistic\n'
    summary += f'long_name: {self.long_name}\n'
    # doc
    summary += f'Function: {self.function.__doc__}\n'
    return summary


class Comparison:
    """Master class for all comparisons."""

    def __init__(self, name, function, hindcast, probabilistic, long_name=None):
        """Comparison initialization.

        Args:
            name (str): name of comparison.
            function (function): comparison function.
            hindcast (bool): Can comparison be used in `compute_hindcast`?
             `False` means `compute_perfect_model`
            probabilistic (bool): Can this comparison be used for probabilistic
             metrics also? Probabilistic metrics require multiple forecasts.
             `False` means that comparison is only deterministic.
             `True` means that comparison can be used both deterministic and
             probabilistic.
            long_name (str, optional): longname of comparison. Defaults to None.

        Returns:
            comparison: comparison class Comparison.

        """
        self.name = name
        self.function = function
        self.hindcast = hindcast
        self.probabilistic = probabilistic
        self.long_name = long_name

    def __repr__(self):
        """Show metadata of comparison class."""
        return _display_comparison_metadata(self)


# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# --------------------------------------------#


def _m2m(ds, stack_dims=True):
    """
    Compare all members to all others in turn while leaving out the verification member.

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


__m2m = Comparison(
    name='m2m',
    function=_m2m,
    hindcast=False,
    probabilistic=True,
    long_name='Comparison of all forecasts vs. all other members as verification',
)


def _m2e(ds, stack_dims=True):
    """
    Compare all members to ensemble mean while leaving out the reference in
     ensemble mean.

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


__m2e = Comparison(
    name='m2e',
    function=_m2e,
    hindcast=False,
    probabilistic=False,
    long_name='Comparison of all members as verification vs. the ensemble mean'
    'forecast',
)


def _m2c(ds, control_member=None, stack_dims=True):
    """
    Compare all other members forecasts to control member verification.

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


__m2c = Comparison(
    name='m2c',
    function=_m2c,
    hindcast=False,
    probabilistic=True,
    long_name='Comparison of multiple forecasts vs. control verification',
)


def _e2c(ds, control_member=None, stack_dims=True):
    """
    Compare ensemble mean forecast to control member verification.

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


__e2c = Comparison(
    name='e2c',
    function=_e2c,
    hindcast=False,
    probabilistic=False,
    long_name='Comparison of the ensemble mean forecast vs. control as verification',
)


# --------------------------------------------#
# HINDCAST COMPARISONS
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


__e2r = Comparison(
    name='e2r',
    function=_e2r,
    hindcast=True,
    probabilistic=False,
    long_name='Comparison of the ensemble mean vs. reference verification',
)


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


__m2r = Comparison(
    name='m2r',
    function=_m2r,
    hindcast=True,
    probabilistic=True,
    long_name='Comparison of multiple forecasts vs. reference verification',
)


__ALL_COMPARISONS__ = [__m2m, __m2e, __m2c, __e2c, __e2r, __m2r]
