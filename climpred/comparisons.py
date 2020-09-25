import dask
import numpy as np
import xarray as xr

from .checks import has_dims, has_min_len
from .constants import M2M_MEMBER_DIM


def _transpose_and_rechunk_to(new_chunk_ds, ori_chunk_ds):
    """Chunk xr.object `new_chunk_ds` as another xr.object `ori_chunk_ds`.
    This is needed after some operations which reduce chunks to size 1.
    First transpose a to ds.dims then apply ds chunking to a."""
    # supposed to be in .utils but circular imports therefore here
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(new_chunk_ds, xr.DataArray) else {}
    )
    return new_chunk_ds.transpose(*ori_chunk_ds.dims, **transpose_kwargs).chunk(
        ori_chunk_ds.chunks
    )


def _display_comparison_metadata(self):
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


class Comparison:
    """Master class for all comparisons."""

    def __init__(
        self, name, function, hindcast, probabilistic, long_name=None, aliases=None,
    ):
        """Comparison initialization.

        Args:
            name (str): name of comparison.
            function (function): comparison function.
            hindcast (bool): Can comparison be used in `compute_hindcast`?
                `False` means `compute_perfect_model`
            probabilistic (bool): Can this comparison be used for probabilistic
                metrics also? Probabilistic metrics require multiple forecasts.
                `False` means that comparison is only deterministic.
                `True` means that comparison can be used both deterministic and
                probabilistic.
            long_name (str, optional): longname of comparison. Defaults to None.
            aliases (list of str, optional): Allowed aliases for this comparison.
                Defaults to ``None``.

        Returns:
            comparison: comparison class Comparison.

        """
        self.name = name
        self.function = function
        self.hindcast = hindcast
        self.probabilistic = probabilistic
        self.long_name = long_name
        self.aliases = aliases

    def __repr__(self):
        """Show metadata of comparison class."""
        return _display_comparison_metadata(self)


# --------------------------------------------#
# PERFECT-MODEL COMPARISONS
# --------------------------------------------#


def _m2m(ds, metric=None):
    """Compare all members to all others in turn while leaving out the verification
    ``member``.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with ``member`` dimension.
        metric (Metric):
            If deterministic, forecast and reference have ``member`` dim.
            If probabilistic, only forecast has ``member`` dim.

    Returns:
        xr.object: forecast, reference.
    """
    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = ds.drop_sel(member=m)
        # set incrementing members to avoid nans from broadcasting
        forecast["member"] = np.arange(1, 1 + forecast.member.size)
        reference = ds.sel(member=m, drop=True)
        # Tiles the singular "reference" member to compare directly to all other members
        if not metric.probabilistic:
            forecast, reference = xr.broadcast(forecast, reference)
        reference_list.append(reference)
        forecast_list.append(forecast)
    reference = xr.concat(reference_list, M2M_MEMBER_DIM)
    forecast = xr.concat(forecast_list, M2M_MEMBER_DIM)
    reference[M2M_MEMBER_DIM] = np.arange(reference[M2M_MEMBER_DIM].size)
    forecast[M2M_MEMBER_DIM] = np.arange(forecast[M2M_MEMBER_DIM].size)
    return forecast, reference


__m2m = Comparison(
    name="m2m",
    function=_m2m,
    hindcast=False,
    probabilistic=True,
    long_name="Comparison of all forecasts vs. all other members as verification",
)


def _m2e(ds, metric=None):
    """
    Compare all members to ensemble mean while leaving out the reference in
     ensemble mean.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        metric (Metric): needed for probabilistic metrics.
                      therefore useless in m2e comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    reference_list = []
    forecast_list = []
    M2E_COMPARISON_DIM = "member"
    for m in ds.member.values:
        forecast = ds.drop_sel(member=m).mean("member")
        reference = ds.sel(member=m, drop=True)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, M2E_COMPARISON_DIM)
    forecast = xr.concat(forecast_list, M2E_COMPARISON_DIM)
    forecast[M2E_COMPARISON_DIM] = np.arange(forecast[M2E_COMPARISON_DIM].size)
    reference[M2E_COMPARISON_DIM] = np.arange(reference[M2E_COMPARISON_DIM].size)
    if dask.is_dask_collection(forecast):
        forecast = _transpose_and_rechunk_to(forecast, ds)
        reference = _transpose_and_rechunk_to(reference, ds)
    return forecast, reference


__m2e = Comparison(
    name="m2e",
    function=_m2e,
    hindcast=False,
    probabilistic=False,
    long_name="Comparison of all members as verification vs. the ensemble mean"
    "forecast",
)


def _m2c(ds, control_member=None, metric=None):
    """
    Compare all other members forecasts to control member verification.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        control_member: list of the one integer member serving as
                        reference. Default 0
        metric (Metric): if deterministic, forecast and reference both have member dim
                      if probabilistic, only forecast has member dim

    Returns:
        xr.object: forecast, reference.
    """
    if control_member is None:
        control_member = ds.member.values[0]
    reference = ds.sel(member=control_member, drop=True)
    # drop the member being reference
    forecast = ds.drop_sel(member=control_member)
    if not metric.probabilistic:
        forecast, reference = xr.broadcast(forecast, reference)
    return forecast, reference


__m2c = Comparison(
    name="m2c",
    function=_m2c,
    hindcast=False,
    probabilistic=True,
    long_name="Comparison of multiple forecasts vs. control verification",
)


def _e2c(ds, control_member=None, metric=None):
    """
    Compare ensemble mean forecast to control member verification.

    Args:
        ds (xarray object): xr.Dataset/xr.DataArray with member and ensemble
                            dimension.
        control_member: list of the one integer member serving as
                        reference. Default 0
        metric (Metric): needed for probabilistic metrics.
                      therefore useless in e2c comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, reference.
    """
    if control_member is None:
        control_member = ds.member.values[0]
    reference = ds.sel(member=control_member, drop=True)
    ds = ds.drop_sel(member=control_member)
    forecast = ds.mean("member")
    return forecast, reference


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
def _e2o(hind, verif, metric=None):
    """Compare the ensemble mean forecast to the verification data for a
    ``HindcastEnsemble`` setup.

    Args:
        hind (xarray object): Hindcast with optional ``member`` dimension.
        verif (xarray object): Verification data.
        metric (Metric): needed for probabilistic metrics.
                      therefore useless in ``e2o`` comparison,
                      but expected by internal API.

    Returns:
        xr.object: forecast, verif.
    """
    if "member" in hind.dims:
        forecast = hind.mean("member")
    else:
        forecast = hind
    return forecast, verif


__e2o = Comparison(
    name="e2o",
    function=_e2o,
    hindcast=True,
    probabilistic=False,
    long_name="Verify the ensemble mean against the verification data",
    aliases=["e2r"],
)


def _m2o(hind, verif, metric=None):
    """Compares each ensemble member individually to the verification data for a
    ``HindcastEnsemble`` setup.

    Args:
        hind (xarray object): Hindcast with ``member`` dimension.
        verif (xarray object): Verification data.
        metric (Metric):
            If deterministic, forecast and verif both have ``member`` dim;
            If probabilistic, only forecast has ``member`` dim.

    Returns:
        xr.object: forecast, verif.
    """
    # check that this contains more than one member
    has_dims(hind, "member", "decadal prediction ensemble")
    has_min_len(hind["member"], 1, "decadal prediction ensemble member")
    forecast = hind
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
# ['e2o', 'm2o']
HINDCAST_COMPARISONS = [c.name for c in __ALL_COMPARISONS__ if c.hindcast]
# ['m2c', 'e2c', 'm2m', 'm2e']
PM_COMPARISONS = [c.name for c in __ALL_COMPARISONS__ if not c.hindcast]
ALL_COMPARISONS = HINDCAST_COMPARISONS + PM_COMPARISONS
# ['m2c', 'm2m']
PROBABILISTIC_PM_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (not c.hindcast and c.probabilistic)
]
NON_PROBABILISTIC_PM_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (not c.hindcast and not c.probabilistic)
]

# ['m2o']
PROBABILISTIC_HINDCAST_COMPARISONS = [
    c.name for c in __ALL_COMPARISONS__ if (c.hindcast and c.probabilistic)
]
PROBABILISTIC_COMPARISONS = (
    PROBABILISTIC_HINDCAST_COMPARISONS + PROBABILISTIC_PM_COMPARISONS
)
ALL_COMPARISONS = [c.name for c in __ALL_COMPARISONS__]
