import logging
from datetime import datetime


def log_compute_hindcast_header(metric, comparison, dim, alignment):
    """Add header to the log for a `compute_hindcast` instance."""
    logging.info(
        f'`compute_hindcast` for metric "{metric.name}", '
        f'comparison "{comparison.name}", dim "{dim}", and alignment "{alignment}" at '
        f"{str(datetime.now())}\n"
        f"++++++++++++++++++++++++++++++++++++++++++++++++"
    )


def log_compute_hindcast_inits_and_verifs(dim, lead, inits, verif_dates):
    """At each lead, log the inits and verification dates being used in computations."""
    logging.info(
        f"lead: {str(lead).zfill(2)} | "
        # This is the init-sliced forecast, thus displaying actual
        # initializations.
        f"inits: {inits[lead].min().values}"
        f"-{inits[lead].max().values} | "
        # This is the verification window, thus displaying the
        # verification dates.
        f"verifs: {verif_dates[lead].min()}"
        f"-{verif_dates[lead].max()}"
    )
    init_output = [f"{i.year}-{i.month}-{i.day}" for i in inits[lead].values]
    verif_output = [f"{i.year}-{i.month}-{i.day}" for i in verif_dates[lead].values]
    logging.debug(f"\ninits: {init_output}" f"\nverifs: {verif_output}")
