import warnings

import numpy as np
import xarray as xr
from esmtools.stats import corr
from xrft import power_spectrum

from .checks import is_xarray
from .utils import copy_coords_from_to


@is_xarray(0)
def decorrelation_time(da, r=20, dim="time"):
    """Calculate the decorrelaton time of a time series.

    .. math::
        \\tau_{d} = 1 + 2 * \\sum_{k=1}^{r}(\\alpha_{k})^{k}

    Args:
        da (xarray object): Time series.
        r (optional int): Number of iterations to run the above formula.
        dim (optional str): Time dimension for xarray object.

    Returns:
        Decorrelation time of time series.

    Reference:
        * Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate
          Research. Cambridge ; New York: Cambridge University Press, 1999.,
          p.373

    """
    one = xr.ones_like(da.isel({dim: 0}))
    one = one.where(da.isel({dim: 0}).notnull())
    return one + 2 * xr.concat(
        [corr(da, da, dim=dim, lead=i) ** i for i in range(1, r)], "it"
    ).sum("it")


def dpp(ds, dim="time", m=10, chunk=True):
    """Calculates the Diagnostic Potential Predictability (dpp)

    .. math::

        DPP_{\\mathrm{unbiased}}(m) = \\frac{\\sigma^{2}_{m} -
        \\frac{1}{m}\\cdot\\sigma^{2}}{\\sigma^{2}}

    Note:
        Resplandy et al. 2015 and Seferian et al. 2018 calculate unbiased DPP
        in a slightly different way: chunk=False.

    Args:
        ds (xr.DataArray): control simulation with time dimension as years.
        dim (str): dimension to apply DPP on. Default: time.
        m (optional int): separation time scale in years between predictable
                          low-freq component and high-freq noise.
        chunk (optional boolean): Whether chunking is applied. Default: True.
                    If False, then uses Resplandy 2015 / Seferian 2018 method.

    Returns:
        dpp (xr.DataArray): ds without time dimension.

    References:
        * Boer, G. J. “Long Time-Scale Potential Predictability in an Ensemble of
          Coupled Climate Models.” Climate Dynamics 23, no. 1 (August 1, 2004):
          29–44. https://doi.org/10/csjjbh.
        * Resplandy, L., R. Séférian, and L. Bopp. “Natural Variability of CO2 and
          O2 Fluxes: What Can We Learn from Centuries-Long Climate Models
          Simulations?” Journal of Geophysical Research: Oceans 120, no. 1
          (January 2015): 384–404. https://doi.org/10/f63c3h.
        * Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing the
          Decadal Predictability of Land and Ocean Carbon Uptake.” Geophysical
          Research Letters, March 15, 2018. https://doi.org/10/gdb424.

    """

    def _chunking(ds, dim="time", number_chunks=False, chunk_length=False):
        """
        Separate data into chunks and reshapes chunks in a c dimension.

        Specify either the number chunks or the length of chunks.
        Needed for dpp.

        Args:
            ds (xr.DataArray): control simulation with time dimension as years.
            dim (str): dimension to apply chunking to. Default: time
            chunk_length (int): see dpp(m)
            number_chunks (int): number of chunks in the return data.

        Returns:
            c (xr.DataArray): chunked ds, but with additional dimension c.

        """
        if number_chunks and not chunk_length:
            chunk_length = np.floor(ds[dim].size / number_chunks)
            cmin = int(ds[dim].min())
        elif not number_chunks and chunk_length:
            cmin = int(ds[dim].min())
            number_chunks = int(np.floor(ds[dim].size / chunk_length))
        else:
            raise KeyError("set number_chunks or chunk_length to True")
        c = ds.sel({dim: slice(cmin, cmin + chunk_length - 1)})
        c = c.expand_dims("c")
        c["c"] = [0]
        for i in range(1, number_chunks):
            c2 = ds.sel(
                {dim: slice(cmin + chunk_length * i, cmin + (i + 1) * chunk_length - 1)}
            )
            c2 = c2.expand_dims("c")
            c2["c"] = [i]
            c2[dim] = c[dim]
            c = xr.concat([c, c2], "c")
        return c

    if not chunk:  # Resplandy 2015, Seferian 2018
        s2v = ds.rolling({dim: m}).mean().var(dim)
        s2 = ds.var(dim)

    if chunk:  # Boer 2004 ppvf
        # first chunk
        chunked_means = _chunking(ds, dim=dim, chunk_length=m).mean(dim)
        # sub means in chunks
        chunked_deviations = _chunking(ds, dim=dim, chunk_length=m) - chunked_means
        s2v = chunked_means.var("c")
        s2e = chunked_deviations.var([dim, "c"])
        s2 = s2v + s2e
    dpp = (s2v - s2 / (m)) / s2
    return dpp


@is_xarray(0)
def varweighted_mean_period(da, dim="time", **kwargs):
    """Calculate the variance weighted mean period of time series based on
    xrft.power_spectrum.

    .. math::
        P_{x} = \\frac{\\sum_k V(f_k,x)}{\\sum_k f_k  \\cdot V(f_k,x)}

    Args:
        da (xarray object): input data including dim.
        dim (optional str): Name of time dimension.
        for **kwargs see xrft.power_spectrum

    Reference:
      * Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value
        Decadal Predictability in a CGCM." Journal of Climate 23, no. 23
        (August 27, 2010): 6292-6311. https://doi.org/10/bwq92h.

    See also:
    https://xrft.readthedocs.io/en/latest/api.html#xrft.xrft.power_spectrum
    """
    if isinstance(da, xr.Dataset):
        raise ValueError("require xr.Dataset")
    da = da.fillna(0.0)
    # dim should be list
    if isinstance(dim, str):
        dim = [dim]
    assert isinstance(dim, list)
    ps = power_spectrum(da, dim=dim, **kwargs)
    # take pos
    for d in dim:
        ps = ps.where(ps[f"freq_{d}"] > 0)
    # weighted average
    vwmp = ps
    for d in dim:
        vwmp = vwmp.sum(f"freq_{d}") / ((vwmp * vwmp[f"freq_{d}"]).sum(f"freq_{d}"))
    for d in dim:
        del vwmp[f"freq_{d}_spacing"]
    return vwmp
