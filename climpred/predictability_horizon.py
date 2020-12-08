import numpy as np
import xarray as xr


def _last_item_cond_true(cond, dim):
    """Return the final item from cond that evaluates to True.

    Args:
        cond (xr.DataArray, xr.Dataset): User-defined boolean array where True means
            the system is predictable at the given lead. E.g., this could be based on
            the dynamical forecast beating a reference forecast, p values, confidence
            intervals, etc. cond should contain the dimension dim at the minimum.
        dim (str): Dimension to check for condition == True over.

    Returns:
        xr.DataArray, xr.Dataset: ``dim`` value until condition is True.

    """
    # force DataArray because isel (when transforming to dim space) requires DataArray
    if isinstance(cond, xr.Dataset):
        was_dataset = True
        cond = cond.to_array()
    else:
        was_dataset = False
    # index last True
    reached = cond.argmin(dim)
    # fix below one
    reached = reached.where(reached >= 1, np.nan)
    # reset where always true to len(lead)
    reached = reached.where(~cond.all("lead"), other=cond[dim].size)
    # fix locations where always nan to nan
    mask = cond.notnull().all("lead")  # ~(cond == False).all("lead")
    reached = reached.where(mask, other=np.nan)
    ## shift back into coordinate space ##
    # problem: cannot convert nan to idx in isel
    # therefore set to dim:0 and mask again afterwards
    reached_notnull = reached.notnull()  # remember where not masked
    reached = reached.where(
        reached.notnull(), other=cond.isel({dim: 0})
    )  # set nan to dim:0
    # take one index before calculated by argmin
    reached_dim_space = cond[dim].isel(
        {dim: reached.astype(int) - 1}
    )  # to not break conversion to dim space
    reached_dim_space = reached_dim_space.where(
        reached_notnull, other=np.nan
    )  # cleanup replace dim:0 with nan again
    if was_dataset:
        reached_dim_space = reached_dim_space.to_dataset(dim="variable")
    return reached_dim_space


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
