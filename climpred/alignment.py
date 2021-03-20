import dask
import numpy as np
import xarray as xr

from .checks import is_in_list
from .constants import VALID_ALIGNMENTS
from .exceptions import CoordinateError
from .utils import get_multiple_lead_cftime_shift_args, shift_cftime_index

ALIGNMENT_ALIASES = {
    "same_init": "same_init",
    "same_verif": "same_verif",
    "same_inits": "same_init",
    "same_verifs": "same_verif",
    "maximize": "maximize",
}

def return_inits_and_verif_dates(forecast, verif, alignment, reference=None, hist=None):
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
    if isinstance(reference, str):
        reference = [reference]
    elif reference is None:
        reference = []

    is_in_list(alignment, VALID_ALIGNMENTS, "alignment")
    alignment = ALIGNMENT_ALIASES.get(alignment)
    print('alignment =',alignment)
    units = forecast["lead"].attrs["units"]
    leads = forecast["lead"].values

    all_inits = forecast["init"]
    all_verifs = verif["time"]

    # If aligning reference='uninitialized', need to account for potential differences
    # in its temporal coverage. Note that the reference='uninitialized' only aligns
    # verification dates and doesn't care about inits. # TODO: should be changed
    if hist is not None and "uninitialzed" not in reference:
        all_verifs = np.sort(list(set(all_verifs.data) & set(hist["time"].data)))
        all_verifs = xr.DataArray(all_verifs, dims=["time"], coords=[all_verifs])

    # Construct list of `n` offset over all leads.
    n, freq = get_multiple_lead_cftime_shift_args(units, leads)

    if "time" not in forecast.coords:  # construct time(init, lead) in compute_hindcast
        init_lead_matrix = _construct_init_lead_matrix(forecast, n, freq, leads)
        init_lead_matrix["lead"].attrs = forecast.lead.attrs
    else:  # use time(init, lead)
        init_lead_matrix = forecast["time"].drop("time").rename(None)

    if dask.is_dask_collection(init_lead_matrix):
        init_lead_matrix = init_lead_matrix.compute()
    # xr.testing.assert_identical(init_lead_matrix, init_lead_matrix_new)
    # A union between `inits` and observations in the verification data is required
    # for persistence, since the persistence forecast is based off a common set of
    # initializations.
    if "persistence" in reference:
        union_with_verifs = all_inits.isin(all_verifs)
        init_lead_matrix = init_lead_matrix.where(union_with_verifs, drop=True)
    valid_inits = init_lead_matrix["init"]

    if "same_init" in alignment:
        return _same_inits_alignment(
            init_lead_matrix, valid_inits, all_verifs, leads, n, freq
        )
    elif "same_verif" in alignment:
        return _same_verifs_alignment(
            init_lead_matrix, valid_inits, all_verifs, leads, n, freq
        )
    elif alignment == "maximize":
        return _maximize_alignment(init_lead_matrix, all_verifs, leads)


def _maximize_alignment(init_lead_matrix, all_verifs, leads):
    """Returns initializations and verification dates, maximizing the degrees of freedom
    at each lead individually.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    # Move row-wise and find all forecasted times that align with verification dates at
    # the given lead.
    verify_with_observations = init_lead_matrix.isin(all_verifs)
    lead_dependent_verif_dates = init_lead_matrix.where(verify_with_observations)
    # Probably a way to do this more efficiently since we're doing essentially
    # the same thing at each step.
    verif_dates = {
        lead: lead_dependent_verif_dates.sel(lead=lead).dropna("init").to_index()
        for lead in leads
    }
    inits = {
        lead: lead_dependent_verif_dates.sel(lead=lead).dropna("init")["init"]
        for lead in leads
    }
    return inits, verif_dates


def _same_inits_alignment(init_lead_matrix, valid_inits, all_verifs, leads, n, freq):
    """Returns initializations and verification dates, maintaining a common set of inits
    at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    verifies_at_all_leads = init_lead_matrix.isin(all_verifs).all("lead")
    inits = valid_inits.where(verifies_at_all_leads, drop=True)
    inits = {lead: inits for lead in leads}
    verif_dates = {
        lead: shift_cftime_index(inits[lead], "init", n, freq)
        for (lead, n) in zip(leads, n)
    }
    return inits, verif_dates


def _same_verifs_alignment(init_lead_matrix, valid_inits, all_verifs, leads, n, freq):
    """Returns initializations and verification dates, maintaining a common verification
    window at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    common_set_of_verifs = [
        i for i in all_verifs if (i == init_lead_matrix).any("init").all("lead")
    ]
    if not common_set_of_verifs:
        raise CoordinateError(
            "A common set of verification dates cannot be found for the "
            "initializations and verification data supplied. Change `alignment` to "
            "'same_inits' or 'maximize'."
        )
    # Force to CFTimeIndex for consistency with `same_inits`
    verif_dates = xr.concat(common_set_of_verifs, "init").to_index()
    inits_that_verify_with_verif_dates = init_lead_matrix.isin(verif_dates)
    inits = {
        lead: valid_inits.where(
            inits_that_verify_with_verif_dates.sel(lead=lead), drop=True
        )
        for lead in leads
    }
    verif_dates = {lead: verif_dates for lead in leads}
    return inits, verif_dates


def _construct_init_lead_matrix(forecast, n, freq, leads):
    """Returns xr.DataArray of "real time" (init + lead) over all inits and leads.
    Identical results as add_time_from_init_lead, which also works for non integer n.
    _construct_init_lead_matrix is only used in compute_hindcast now.

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
    init_lead_matrix = xr.concat(
        [
            xr.DataArray(
                shift_cftime_index(forecast, "init", n, freq),
                dims=["init"],
                coords=[forecast["init"]],
            )
            for n in n
        ],
        "lead",
    )
    init_lead_matrix["lead"] = leads
    return init_lead_matrix
