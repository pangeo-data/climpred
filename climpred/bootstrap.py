import numpy as np
import xarray as xr

from climpred.prediction import _pseudo_ens
from climpred.stats import DPP


def DPP_threshold(control, sig=95, bootstrap=500, **dpp_kwargs):
    """Calc DPP from re-sampled dataset.

    Reference:
    * Feng, X., T. DelSole, and P. Houser. “Bootstrap Estimated Seasonal
        Potential Predictability of Global Temperature and Precipitation.”
        Geophysical Research Letters 38, no. 7 (2011).
        https://doi.org/10/ft272w.

    """
    bootstraped_results = []
    time = control.time.values
    for _ in range(bootstrap):
        smp_time = np.random.choice(time, len(time))
        smp_control = control.sel(time=smp_time)
        smp_control['time'] = time
        bootstraped_results.append(DPP(smp_control, **dpp_kwargs))
    threshold = xr.concat(bootstraped_results, 'bootstrap').quantile(
        sig / 100, 'bootstrap')
    return threshold
