import xarray as xr

from .checks import is_in_list
from .constants import VALID_ALIGNMENTS
from .utils import get_multiple_lead_cftime_shift_args, shift_cftime_index


def return_inits_and_verif_dates(forecast, verif, alignment):
    """Returns initializations and verification dates for an arbitrary number of leads
    per a given alignment strategy.

    Args:
        forecast (``xarray`` object): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        verif (``xarray`` object): Verification data with ``time`` dim.
        alignment (str): Strategy for initialization-verification alignment.
            * 'same_inits': Use a common set of initializations that verify
               across all leads. This ensures that there is no bias in the result due
               to the state of the system for the given initializations.
            * 'same_verifs': Use a common verification window across all leads. This
               ensures that there is no bias in the result due to the observational
               period being verified against.
            * 'maximize': Use all available initializations at each lead that verify
               against the observations provided. This changes both the set of
               initializations and the verification window used at each lead.

    Returns:
        inits (dict): Keys are the lead time integer, values are an ``xr.DataArray`` of
            initialization dates.
        verif_dates (dict): Keys are the lead time integer, values are an
            ``xr.CFTimeIndex`` of verification dates.
    """
    is_in_list(alignment, VALID_ALIGNMENTS, 'alignment')
    units = forecast['lead'].attrs['units']
    leads = forecast['lead'].values
    # `init` renamed to `time` in compute functions.
    all_inits = forecast['time']
    all_verifs = verif['time']
    union_with_verifs = all_inits.isin(all_verifs)

    # Construct list of `n` offset over all leads.
    n, freq = get_multiple_lead_cftime_shift_args(units, leads)
    init_lead_matrix = _construct_init_lead_matrix(forecast, n, freq, leads)
    # Currently enforce a union between `inits` and observations in the verification
    # data. This is because persistence forecasts with the verification data need to
    # have the same initializations for consistency. This behavior should be changed
    # as alternative reference forecasts are introduced.
    init_lead_matrix = init_lead_matrix.where(union_with_verifs, drop=True)
    valid_inits = init_lead_matrix['time']

    if 'same_init' in alignment:
        return _same_inits_alignment(
            init_lead_matrix, valid_inits, all_verifs, leads, n, freq
        )
    elif 'same_verif' in alignment:
        return _same_verifs_alignment(
            init_lead_matrix, valid_inits, all_verifs, leads, n, freq
        )
    else:
        raise NotImplementedError('Still need to do maximize.')


def _same_inits_alignment(init_lead_matrix, valid_inits, all_verifs, leads, n, freq):
    """Returns initializations and verification dates, maintaining a common set of inits
    at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    verifies_at_all_leads = init_lead_matrix.isin(all_verifs).all('lead')
    inits = valid_inits.where(verifies_at_all_leads, drop=True)
    inits = {l: inits for l in leads}
    verif_dates = {
        l: shift_cftime_index(inits[l], 'time', n, freq) for (l, n) in zip(leads, n)
    }
    return inits, verif_dates


def _same_verifs_alignment(init_lead_matrix, valid_inits, all_verifs, leads, n, freq):
    """Returns initializations and verification dates, maintaining a common verification
    window at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    common_set_of_verifs = [
        i for i in all_verifs if (i == init_lead_matrix).any('time').all('lead')
    ]
    if not common_set_of_verifs:
        raise ValueError(
            'A common set of verification dates cannot be found for the '
            'initializations and verification data supplied. Change `alignment` to '
            "'same_inits' or 'maximize'."
        )
    # Force to CFTimeIndex for consistency with `same_inits`
    verif_dates = xr.concat(common_set_of_verifs, 'time').to_index()
    inits_that_verify_with_verif_dates = init_lead_matrix.isin(verif_dates)
    inits = {
        l: valid_inits.where(inits_that_verify_with_verif_dates.sel(lead=l), drop=True)
        for l in leads
    }
    verif_dates = {l: verif_dates for l in leads}
    return inits, verif_dates


def _construct_init_lead_matrix(forecast, n, freq, leads):
    """Returns xr.DataArray of "real time" (init + lead) over all inits and leads.

    Arguments:
        forecast (``xarray object``): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        n (tuple of ints): Number of units to shift for ``leads``. ``value`` for
            ``CFTimeIndex.shift(value, str)``.
        freq (str): Pandas frequency alias. ``str`` for
            ``CFTimeIndex.shift(value, str)``.
        leads (list, array, xr.DataArray of ints): Leads to return offset for.

    Returns:
        init_lead_matrix (``xr.DataArray``): DataArray with x=inits and y=lead with
            values corresponding to "real time", or ``init + lead`` over all inits and
            leads.
    """
    # Note that `init` is renamed to `time` in compute functions.
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
    init_lead_matrix['lead'] = leads
    return init_lead_matrix
