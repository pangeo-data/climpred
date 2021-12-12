import logging
from datetime import datetime


def log_compute_hindcast_header(metric, comparison, dim, alignment, reference):
    """Add header to the log for a `compute_hindcast` instance."""
    logging.info(
        f"`compute_hindcast` for metric {metric.name}, "
        f"comparison {comparison.name}, dim {dim}, alignment {alignment} and "
        f"reference {reference} at {str(datetime.now())}\n"
        f"++++++++++++++++++++++++++++++++++++++++++++++++"
    )


def log_compute_hindcast_inits_and_verifs(
    dim, lead, inits, verif_dates, reference=None
):
    """At each lead, log the inits and verification dates being used in computations."""
    if reference is None:
        reference = "initialized"
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
    init_output = [
        f"{i.year}-{str(i.month).zfill(2)}-{str(i.day).zfill(2)}"
        for i in inits[lead].values
    ]
    verif_output = [
        f"{i.year}-{str(i.month).zfill(2)}-{str(i.day).zfill(2)}"
        for i in verif_dates[lead].values
    ]
    logging.debug(f"\ninits: {init_output}" f"\nverifs: {verif_output}")
