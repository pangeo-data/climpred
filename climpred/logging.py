import logging
from datetime import datetime

import xarray as xr


def log_hindcast_verify_header(metric, comparison, dim, alignment, reference) -> None:
    """Add header to the log for a `HindcastEnsemble.verify()` instance."""
    logging.info(
        f"`HindcastEnsemble.verify()` for metric {metric.name}, "
        f"comparison {comparison.name}, dim {dim}, alignment {alignment} and "
        f"reference {reference} at {str(datetime.now())}\n"
        f"++++++++++++++++++++++++++++++++++++++++++++++++"
    )


def log_hindcast_verify_inits_and_verifs(
    dim, lead, inits, verif_dates, reference=None
) -> None:
    """At each lead, log the inits and verification dates being used in computations."""
    if reference is None:
        reference = "initialized"
    time_datetimeindex = (
        True if not isinstance(verif_dates[lead], xr.CFTimeIndex) else False
    )
    if len(inits[lead]) > 1 or not time_datetimeindex:
        logging.info(
            f"{reference} | lead: {str(lead).zfill(2)} | "
            # This is the init-sliced forecast, thus displaying actual
            # initializations.
            f"inits: {inits[lead].min().values}"
            f"-{inits[lead].max().values} | "
            # This is the verification window, thus displaying the
            # verification dates.
            f"verifs: {verif_dates[lead].min()}"
            f"-{verif_dates[lead].max()}"
        )
    init_output_iter = (
        inits[lead].to_index() if time_datetimeindex else inits[lead].values
    )
    init_output = [
        f"{i.year}-{str(i.month).zfill(2)}-{str(i.day).zfill(2)}"
        for i in init_output_iter
    ]
    verif_output_iter = (
        verif_dates[lead] if time_datetimeindex else verif_dates[lead].values
    )
    verif_output = [
        f"{i.year}-{str(i.month).zfill(2)}-{str(i.day).zfill(2)}"
        for i in verif_output_iter
    ]
    logging.debug(f"\ninits: {init_output}" f"\nverifs: {verif_output}")
