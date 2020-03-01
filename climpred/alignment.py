import xarray as xr

from .utils import get_multiple_lead_cftime_shift_args, shift_cftime_index


def return_inits_and_verif_dates(forecast, verif, alignment):
    """Returns initializations and verification dates for an arbitrary number of leads
    per a given alignment strategy.

    Args:
        forecast (``xarray`` object): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        verif (``xarray`` object): Verification data with ``time`` dim.
        alignment (str): Strategy for initialization-verification alignment.
            * 'same_inits':
            * 'same_verifs':
            * 'maximize':

    Returns:
        inits (dict): Keys are the lead time integer, values are an ``xr.DataArray`` of
            initialization dates.
        verif_dates (dict): Keys are the lead time integer, values are an
            ``xr.CFTimeIndex`` of verification dates.

    Raises:
    """
    # Add check that alignment is one of `same_init`, `same_inits`, `same_verif`,
    # `same_verifs`, `maximize`
    if (alignment == 'same_inits') | (alignment == 'same_init'):
        return same_inits_alignment(forecast, verif)
    else:
        raise NotImplementedError('Work in progress.')


def same_inits_alignment(forecast, verif):
    """Returns initializations and verification dates, maintaining the same inits at
    each lead.

    Args:
        forecast (``xarray`` object): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        verif (``xarray`` object): Verification data with ``time`` dim.

    Returns:
        inits (dict): Keys are the lead time integer, values are an ``xr.DataArray`` of
            initialization dates.
        verif_dates (dict): Keys are the lead time integer, values are an
            ``xr.CFTimeIndex`` of verification dates.
    """
    units = forecast['lead'].attrs['units']
    leads = forecast['lead'].values
    # Construct list of `n` offset over all leads.
    n, freq = get_multiple_lead_cftime_shift_args(units, leads)
    # Note that `init` is renamed to `time` in the compute function to compute metrics.
    init_lead_matrix = xr.concat(
        [
            xr.DataArray(
                shift_cftime_index(forecast, 'time', n, freq),
                dims=['time'],
                coords=[forecast['time']],
            )
            for n in n
        ],
        'lead',
    )
    verifies_at_all_leads = init_lead_matrix.isin(verif['time']).all('lead')
    union_with_observations = init_lead_matrix['time'].isin(verif['time'])
    inits = forecast['time'].where(
        verifies_at_all_leads & union_with_observations, drop=True
    )
    inits = {l: inits for l in leads}
    verif_dates = {
        l: shift_cftime_index(inits[l], 'time', n, freq) for (l, n) in zip(leads, n)
    }
    return inits, verif_dates
