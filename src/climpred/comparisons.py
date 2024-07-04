"""Comparisons: How to compare forecast with verification."""

from typing import Any, Callable, List, Optional, Tuple

import dask
import numpy as np
import xarray as xr

from .checks import has_dims, has_min_len
from .constants import M2M_MEMBER_DIM
from .metrics import Metric


class Comparison:
    """Master class for all comparisons. See :ref:`comparisons`."""

    def __init__(
        self,
        name: str,
        function: Callable[[Any, Any, Any], Tuple[xr.Dataset, xr.Dataset]],
        hindcast: bool,
        probabilistic: bool,
        long_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Comparison initialization See :ref:`comparisons`.

        Args:
            name: name of comparison.
            function: comparison function.
            hindcast: Can comparison be used in
                :py:class:`.HindcastEnsemble`?
                ``False`` means only :py:class:`.PerfectModelEnsemble`
            probabilistic: Can this comparison be used for probabilistic
                metrics also? Probabilistic metrics require multiple forecasts.
                ``False`` means that comparison is only deterministic.
                ``True`` means that comparison can be used both deterministic and
                probabilistic.
            long_name: longname of comparison.
            aliases: Allowed aliases for this comparison.

        """
        self.name = name
        self.function = function
        self.hindcast = hindcast
        self.probabilistic = probabilistic
        self.long_name = long_name
        self.aliases = aliases

    def __repr__(self) -> str:
        """Show metadata of comparison class."""
        summary = "----- Comparison metadata -----\n"
        summary += f"Name: {self.name}\n"
        # probabilistic or only deterministic
        if not self.probabilistic:
            summary += "Kind: deterministic\n"
        else:
            summary += "Kind: deterministic and probabilistic\n"
        summary += f"long_name: {self.long_name}\n"
        # doc
        summary += f"Function: {self.function.__doc__}\n"
        return summary


# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# --------------------------------------------#


def _m2m(
    initialized: xr.Dataset, metric: Metric, verif: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Compare all members to all others in turn while leaving out verification member.

    :ref:`comparisons` for :py:class:`.PerfectModelEnsemble`

    Args:
        initialized: initialized with ``member`` dimension.
        metric:
            If deterministic, forecast and verif have ``member`` dim.
            If probabilistic, only forecast has ``member`` dim.
        verif: not used in :py:class:`.PerfectModelEnsemble`

    Returns:
        forecast, verification
    """
    if verif is not None:
        raise ValueError("`verif` not expected.")

    verif_list = []
    forecast_list = []
    for m in initialized.member.values:
        forecast = initialized.drop_sel(member=m)
        # set incrementing members to avoid nans from broadcasting
        forecast["member"] = np.arange(1, 1 + forecast.member.size)
        verif = initialized.sel(member=m, drop=True)
        # Tiles the singular "verif" member to compare directly to all other members
        if not metric.probabilistic:
            forecast, verif = xr.broadcast(forecast, verif)
        verif_list.append(verif)
        forecast_list.append(forecast)
    verif = xr.concat(verif_list, M2M_MEMBER_DIM)
    forecast = xr.concat(forecast_list, M2M_MEMBER_DIM)
    verif[M2M_MEMBER_DIM] = np.arange(verif[M2M_MEMBER_DIM].size)
    forecast[M2M_MEMBER_DIM] = np.arange(forecast[M2M_MEMBER_DIM].size)
    return forecast, verif


__m2m = Comparison(
    name="m2m",
    function=_m2m,
    hindcast=False,
    probabilistic=True,
    long_name="Comparison of all forecasts vs. all other members as verification",
)


def _m2e(
    initialized: xr.Dataset,
    metric: Optional[Metric] = None,
    verif: Optional[xr.Dataset] = None,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compare all members to ensemble mean while leaving out the verif in ensemble mean.

    :ref:`comparisons` for :py:class:`.PerfectModelEnsemble`

    Args:
        initialized: ``initialized`` with ``member`` dimension.
        metric: needed for probabilistic metrics. Therefore useless in ``m2e``
            comparison, but expected by internal API.
        verif: not used in :py:class:`.PerfectModelEnsemble`

    Returns:
        forecast, verification
    """
    if verif is not None:
        raise ValueError("`verif` not expected.")
    verif_list = []
    forecast_list = []
    M2E_COMPARISON_DIM = "member"
    for m in initialized.member.values:
        forecast = initialized.drop_sel(member=m).mean("member")
        verif = initialized.sel(member=m, drop=True)
        forecast_list.append(forecast)
        verif_list.append(verif)
    verif = xr.concat(verif_list, M2E_COMPARISON_DIM)
    forecast = xr.concat(forecast_list, M2E_COMPARISON_DIM)
    forecast[M2E_COMPARISON_DIM] = np.arange(forecast[M2E_COMPARISON_DIM].size)
    verif[M2E_COMPARISON_DIM] = np.arange(verif[M2E_COMPARISON_DIM].size)
    if dask.is_dask_collection(forecast):
        forecast = forecast.transpose(*initialized.dims).chunk(initialized.chunks)
        verif = verif.transpose(*initialized.dims).chunk(initialized.chunks)
    return forecast, verif


__m2e = Comparison(
    name="m2e",
    function=_m2e,
    hindcast=False,
    probabilistic=False,
    long_name="Comparison of all members as verification vs. the ensemble mean"
    "forecast",
)


def _m2c(
    initialized: xr.Dataset, metric: Metric, verif: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compare all other member forecasts to a single member verification.

    Verification member is the first member.
    If the initialized dataset is concatinated in a way that the first member
    is taken from the control simulation, this compares all other member forecasts
    to the control simulation.

    :ref:`comparisons` for :py:class:`.PerfectModelEnsemble`

    Args:
        initialized: ``initialized`` with ``member`` dimension.
        metric: if deterministic, forecast and verif both have member dim
            if probabilistic, only forecast has ``member`` dim
        verif: not used in :py:class:`.PerfectModelEnsemble`

    Returns:
        forecast, verification
    """
    if verif is not None:
        raise ValueError("`verif` not expected.")
    control_member = initialized.member.values[0]
    verif = initialized.sel(member=control_member, drop=True)
    # drop the member being verif
    forecast = initialized.drop_sel(member=control_member)
    if not metric.probabilistic:
        forecast, verif = xr.broadcast(forecast, verif)
    return forecast, verif


__m2c = Comparison(
    name="m2c",
    function=_m2c,
    hindcast=False,
    probabilistic=True,
    long_name="Comparison of multiple forecasts vs. control verification",
)


def _e2c(
    initialized: xr.Dataset,
    metric: Optional[Metric] = None,
    verif: Optional[xr.Dataset] = None,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Compare ensemble mean forecast to single member verification.

    If the initialized dataset is concatinated in a way that the first member
    is taken from the control simulation, this compares the member mean of all
    other member forecasts to the control simulation.

    :ref:`comparisons` for :py:class:`.PerfectModelEnsemble`

    Args:
        initialized: ``initialized`` with ``member`` dimension.
        metric: needed for probabilistic metrics. Therefore useless in ``e2c``
            comparison, but expected by internal API.
        verif: not used in :py:class:`.PerfectModelEnsemble`

    Returns:
        forecast, verification
    """
    if verif is not None:
        raise ValueError("`verif` not expected.")
    control_member = initialized.member.values[0]
    verif = initialized.sel(member=control_member, drop=True)
    initialized = initialized.drop_sel(member=control_member)
    forecast = initialized.mean("member")
    return forecast, verif


__e2c = Comparison(
    name="e2c",
    function=_e2c,
    hindcast=False,
    probabilistic=False,
    long_name="Comparison of the ensemble mean forecast vs. control as verification",
)


# --------------------------------------------#
# HINDCAST COMPARISONS
# --------------------------------------------#
def _e2o(
    initialized: xr.Dataset, verif: xr.Dataset, metric: Optional[Metric]
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Compare the ensemble mean forecast to the verification data.

    :ref:`comparisons` for :py:class:`.HindcastEnsemble`

    Args:
        initialized: Hindcast with optional ``member`` dimension.
        verif: Verification data.
        metric: needed for probabilistic metrics. Therefore useless in ``e2o``
            comparison, but expected by internal API.

    Returns:
        forecast, verification
    """
    if "member" in initialized.dims:
        forecast = initialized.mean("member")
    else:
        forecast = initialized
    return forecast, verif


__e2o = Comparison(
    name="e2o",
    function=_e2o,
    hindcast=True,
    probabilistic=False,
    long_name="Verify the ensemble mean against the verification data",
    aliases=["e2r"],
)


def _m2o(
    initialized: xr.Dataset, verif: xr.Dataset, metric: Metric
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Compare each ensemble member individually to the verification data.

    :ref:`comparisons` for :py:class:`.HindcastEnsemble`

    Args:
        initialized: ``initialized`` with ``member`` dimension.
        verif: Verification data.
        metric:
            If deterministic, forecast and verif both have ``member`` dim;
            If probabilistic, only forecast has ``member`` dim.

    Returns:
        forecast, verification
    """
    # check that this contains more than one member
    has_dims(initialized, "member", "decadal prediction ensemble")
    has_min_len(initialized["member"], 1, "decadal prediction ensemble member")
    forecast = initialized
    if not metric.probabilistic and "member" not in verif.dims:
        forecast, verif = xr.broadcast(
            forecast, verif, exclude=["time", "init", "lead"]
        )
    return forecast, verif


__m2o = Comparison(
    name="m2o",
    function=_m2o,
    hindcast=True,
    probabilistic=True,
    long_name="Verify each individual forecast member against the verification data.",
    aliases=["m2r"],
)


__ALL_COMPARISONS__ = [__m2m, __m2e, __m2c, __e2c, __e2o, __m2o]

COMPARISON_ALIASES = dict()
for c in __ALL_COMPARISONS__:
    if c.aliases is not None:
        for a in c.aliases:
            COMPARISON_ALIASES[a] = c.name

# Which comparisons work with which set of metrics.
HINDCAST_COMPARISONS = [c.name for c in __ALL_COMPARISONS__ if c.hindcast]
PM_COMPARISONS = [c.name for c in __ALL_COMPARISONS__ if not c.hindcast]
ALL_COMPARISONS = HINDCAST_COMPARISONS + PM_COMPARISONS
PROBABILISTIC_PM_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (not c.hindcast and c.probabilistic)
]
NON_PROBABILISTIC_PM_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (not c.hindcast and not c.probabilistic)
]
PROBABILISTIC_HINDCAST_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (c.hindcast and c.probabilistic)
]
PROBABILISTIC_COMPARISONS = (
    PROBABILISTIC_HINDCAST_COMPARISONS + PROBABILISTIC_PM_COMPARISONS
)
ALL_COMPARISONS = [c.name for c in __ALL_COMPARISONS__]
