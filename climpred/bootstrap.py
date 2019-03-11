import xarray as xr

from climpred.prediction import _pseudo_ens
from climpred.stats import DPP


def DPP_threshold(ds, control, sig=95, bootstrap=500, m=10):
    """Calc DPP from re-sampled dataset.

    Reference:
    * Feng, X., T. DelSole, and P. Houser. “Bootstrap Estimated Seasonal
        Potential Predictability of Global Temperature and Precipitation.”
        Geophysical Research Letters 38, no. 7 (2011).
        https://doi.org/10/ft272w.

    """
    bootstraped_results = []
    for _ in bootstrap:
        smp_ds = _pseudo_ens(ds, control)
        bootstraped_results.append(DPP(smp_ds, control, m=m))
    threshold = xr.concat(bootstraped_results, 'bootstrap').quantile(
        sig / 100, 'bootstrap')
    return threshold
