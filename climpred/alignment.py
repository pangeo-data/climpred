"""Align ``initialized`` ``valid_time=init+lead`` with ``observations`` ``time``."""

from typing import Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import xarray as xr

from .checks import is_in_list
from .constants import VALID_ALIGNMENTS
from .exceptions import CoordinateError
from .utils import get_multiple_lead_cftime_shift_args, shift_cftime_index

returnType = Tuple[Dict[float, xr.DataArray], Dict[float, xr.CFTimeIndex]]


def return_inits_and_verif_dates(
    forecast: xr.Dataset,
    verif: xr.Dataset,
    alignment: str,
    reference: Optional[Union[str, List[str]]] = None,
    hist: Optional[xr.Dataset] = None,
) -> returnType:
    """Return initializations and verification dates per a given alignment strategy.

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

    Return:
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
    units = forecast["lead"].attrs["units"]
    leads = forecast["lead"].values

    # `init` renamed to `time` in compute functions.
    all_inits = forecast["time"]
    all_verifs = verif["time"]

    # If aligning reference='uninitialized', need to account for potential differences
    # in its temporal coverage. Note that the reference='uninitialized' only aligns
    # verification dates and doesn't need to care about inits.
    if hist is not None:
        all_verifs = np.sort(list(set(all_verifs.data) & set(hist["time"].data)))
        all_verifs = xr.DataArray(all_verifs, dims=["time"], coords=[all_verifs])

    # Construct list of `n` offset over all leads.
    n, freq = get_multiple_lead_cftime_shift_args(units, leads)

    if "valid_time" not in forecast.coords:  # old: create init_lead_matrix
        init_lead_matrix = _construct_init_lead_matrix(forecast, n, freq, leads)
    else:  # new: use valid_time(init, lead)
        init_lead_matrix = forecast["valid_time"].drop_vars("valid_time").rename(None)
    if dask.is_dask_collection(init_lead_matrix):
        init_lead_matrix = init_lead_matrix.compute()

    # A union between `inits` and observations in the verification data is required
    # for persistence, since the persistence forecast is based off a common set of
    # initializations.
    if "persistence" in reference:
        union_with_verifs = all_inits.isin(all_verifs)
        init_lead_matrix = init_lead_matrix.where(union_with_verifs, drop=True)
    valid_inits = init_lead_matrix["time"]

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
    else:
        raise ValueError


def _maximize_alignment(
    init_lead_matrix: xr.DataArray, all_verifs: xr.DataArray, leads: xr.DataArray
) -> returnType:
    """Return inits and verif dates, maximizing the samples at each lead individually.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    # Move row-wise and find all forecasted times that align with verification dates at
    # the given lead.
    verify_with_observations = init_lead_matrix.isin(all_verifs)
    lead_dependent_verif_dates = init_lead_matrix.where(verify_with_observations)
    # Probably a way to do this more efficiently since we're doing essentially
    # the same thing at each step.
    verif_dates = {
        lead: lead_dependent_verif_dates.sel(lead=lead).dropna("time").to_index()
        for lead in leads
    }
    inits = {
        lead: lead_dependent_verif_dates.sel(lead=lead).dropna("time")["time"]
        for lead in leads
    }
    return inits, verif_dates


def _same_inits_alignment(
    init_lead_matrix: xr.DataArray,
    valid_inits: xr.DataArray,
    all_verifs: xr.DataArray,
    leads: xr.DataArray,
    n: int,
    freq: str,
) -> returnType:
    """Return inits and verif dates, maintaining a common set of inits at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    verifies_at_all_leads = init_lead_matrix.isin(all_verifs).all("lead")
    inits = valid_inits.where(verifies_at_all_leads, drop=True)
    inits = {lead: inits for lead in leads}
    verif_dates = {
        lead: shift_cftime_index(inits[lead], "time", n, freq)
        for (lead, n) in zip(leads, n)  # type: ignore
    }
    return inits, verif_dates


def _same_verifs_alignment(
    init_lead_matrix: xr.DataArray,
    valid_inits: xr.DataArray,
    all_verifs: xr.DataArray,
    leads: xr.DataArray,
    n: int,
    freq: str,
) -> returnType:
    """Return inits and verifs, maintaining a common verification window at all leads.

    See ``return_inits_and_verif_dates`` for descriptions of expected variables.
    """
    common_set_of_verifs = [
        i for i in all_verifs if (i == init_lead_matrix).any("time").all("lead")
    ]
    if not common_set_of_verifs:
        raise CoordinateError(
            "A common set of verification dates cannot be found for the "
            "initializations and verification data supplied. Change `alignment` to "
            "'same_inits' or 'maximize'."
        )
    # Force to CFTimeIndex for consistency with `same_inits`
    verif_dates = xr.concat(common_set_of_verifs, "time").to_index()
    inits_that_verify_with_verif_dates = init_lead_matrix.isin(verif_dates)
    inits = {
        lead: valid_inits.where(
            inits_that_verify_with_verif_dates.sel(lead=lead), drop=True
        )
        for lead in leads
    }
    verif_dates = {lead: verif_dates for lead in leads}
    return inits, verif_dates


def _construct_init_lead_matrix(
    forecast: xr.Dataset, n: Tuple[int], freq: str, leads: xr.DataArray
) -> xr.DataArray:
    """Return xr.DataArray of "valid time" (init + lead) over all inits and leads.

    Arguments:
        forecast (``xarray object``): Prediction ensemble with ``init`` dim renamed to
            ``time`` and containing ``lead`` dim.
        n (tuple of ints): Number of units to shift for ``leads``. ``value`` for
            ``CFTimeIndex.shift(value, str)``.
        freq (str): Pandas frequency alias. ``str`` for
            ``CFTimeIndex.shift(value, str)``.
        leads (list, array, xr.DataArray of ints): Leads to return offset for.

    Return:
        init_lead_matrix (``xr.DataArray``): DataArray with x=inits and y=lead with
            values corresponding to "real time", or ``init + lead`` over all inits and
            leads.
    """
    # Note that `init` is renamed to `time` in compute functions.
    init_lead_matrix = xr.concat(
        [
            xr.DataArray(
                shift_cftime_index(forecast, "time", n, freq),
                dims=["time"],
                coords=[forecast["time"]],
            )
            for n in n
        ],
        "lead",
    )
    init_lead_matrix["lead"] = leads
    return init_lead_matrix
