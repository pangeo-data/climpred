"""
Objects dealing with unit conversions.

MPAS
----
`convert_CO2_flux`: convert native units to mol/m2/yr
"""


def convert_CO2_flux(mpas_CO2):
    """
    Convert the native MPAS units (mmol C m^{-3} m s^{-1}) to the more common
    mol C m$^{-2}$ yr$^{-1}$.

    Input
    -----
    mpas_CO2 : array_like
        array of native MPAS values

    Return
    ------
    conv_CO2 : array_like
        array of converted MPAS values
    """
    conv_CO2 = mpas_CO2 * -1 * (60 * 60 * 24 * 365.25) * (1/10**3)
    return conv_CO2
