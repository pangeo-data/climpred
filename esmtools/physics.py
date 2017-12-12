"""
Objects dealing with physical conversions. 

Winds and Wind Stress
---------------------
`stress_to_speed` : Convert ocean wind stress to U10 wind speed on the native ocean grid.
"""
import numpy as np
import xarray as xr

def stress_to_speed(x, y):
    """
    This converts ocean wind stress to wind speed at 10m over the ocean so
    that one can use the native ocean grid rather than trying to interpolate between
    ocean and atmosphere grids.

    This is based on the conversion used in Lovenduski et al. (2007), which is related
    to the CESM coupler conversrion:
    http://www.cesm.ucar.edu/models/ccsm3.0/cpl6/users_guide/node20.html

    tau/rho = 0.0027U + 0.000142U2 + 0.0000764U3

    Input
    -----
    x : DataArray of taux or taux**2
    y : DataArray of tauy or tauy**2

    This script expects that tau is in dyn/cm2.

    Return
    ------
    U10 : DataArray of the approximated wind speed.
    """
    tau = (np.sqrt(x**2 + y**2)) / 1.2 * 100**2 / 1e5 # Convert from dyn/cm2 to m2/s2
    U10 = np.zeros(len(tau))
    for t in range(len(tau)):
        c_tau = tau[t]
        p = np.array([0.0000764, 0.000142, 0.0027, -1*c_tau])
        r = np.roots(p)
        i = np.imag(r)
        good = np.where(i == 0)
        U10[t] = np.real(r[good])
    U10 = xr.DataArray(U10, dims=['time'], coords=[tau['time']])
    return U10
