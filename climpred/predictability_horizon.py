import numpy as np
import xarray as xr


def _last_item_cond_true(cond, dim, shift=-1):
    """Get the last item of condition ``cond`` which is True.

    Args:
        cond (xr.DataArray, xr.Dataset): Condition True means predictable. ``cond``
            contains at least the dimension lead and checks for how many leads the
            ``condition`` is true and hence skill is predictable.
        dim (str): Dimension to check for condition == True over.
        shift (int): Description of parameter `shift`. Defaults to 1.

    Returns:
        xr.DataArray, xr.Dataset: ``dim`` value until condition is True.

    """
    reached = cond.idxmin(dim) + shift  # to get the last True lead
    # fix below one
    reached = reached.where(reached >= 1, np.nan)
    # reset that never reach to nan
    # reached = reached.where(reached!=cond[dim].min(),other=np.nan)
    # reset where always true to max dim
    reached = reached.where(~cond.all("lead"), other=cond[dim].max())
    # fix locations where always nan to nan
    mask = cond.notnull().all("lead")  # ~(cond == False).all("lead")
    reached = reached.where(mask, other=np.nan)
    return reached


def predictability_horizon(cond):
    """Calculate the predictability horizon based on a condition ```cond``.

    Args:
        cond (xr.DataArray, xr.Dataset): Condition True means predictable. ``cond``
        contains at least the dimension lead and checks for how many leads the
        ``condition`` is true and hence skill is predictable.

    Returns:
        xr.DataArray, xr.Dataset: predictability horizon reduced by ``lead`` dimension.

    Example:
        >>> skill = pm.verify(metric='acc', comparison='m2e', dim=['init','member'],
                reference=['persistence'])
        >>> predictability_horizon(skill.sel(skill='initialized') >
                skill.sel(skill='persistence'))

        >>> bskill = pm.bootstrap(metric='acc', comparison='m2e', dim=['init','member'],
                reference=['persistence'], iterations=21)
        >>> predictability_horizon(bskill.sel(skill='persistence', results='p') <= 0.05)

    """
    ph = _last_item_cond_true(cond, "lead")
    if isinstance(ph, xr.DataArray):
        ph.attrs["units"] = cond["lead"].attrs["units"]
    elif isinstance(ph, xr.Dataset):
        for v in ph.data_vars:
            ph[v].attrs["units"] = cond["lead"].attrs["units"]
    return ph
