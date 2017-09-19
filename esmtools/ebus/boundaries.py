"""
Objects dealing with EBUS boundaries for plotting, statistics, etc.

"""

def visual_bounds(EBC, std_lon=False):
    """
    Returns the latitude and longitude bounds for plotting a decently
    large swatch of the EBC.

    Parameters
    ----------
    EBC : str
        Identifier for the EBC.
        'CalCS': California Current
        'HumCS': Humboldt Current
        'CanCS': Canary Current
        'BenCS': Benguela Current
    std_lon : boolean (optional)
        Set to True if you desire -180 to 180 longitude.

    Returns
    -------
    lon1 : int; minimum lon boundary
    lon2 : int; maximum lon boundary
    lat1 : int; minimum lat boundary
    lat2 : int; maximum lat boundary

    Examples
    --------
    import esmtools.ebus as ebus
    x1,x2,y1,y2 = ebus.visual_bounds('CalCS')
    """
    if EBC == 'CalCS':
        lat1 = 25
        lat2 = 45
        lon1 = -133
        lon2 = -110
    elif EBC == 'HumCS':
        lat1 = -20
        lat2 = 0
        lon1 = -85
        lon2 = -70
    elif EBC == 'CanCS':
        lat1 = 15
        lat2 = 35
        lon1 = -25
        lon2 = -5
    elif EBC == 'BenCS':
        lat1 = -35
        lat2 = -15
        lon1 = 5
        lon2 = 20
    else:
        raise ValueError('\n' + 'Must select from the following EBUS strings:' \
                         + '\n' + 'CalCS' + '\n' + 'CanCS' + '\n' + 'BenCS' + \
                         '\n' + 'HumCS')
    if (std_lon == True) & (EBC != 'BenCS'):
        lon1 = lon1 + 360
        lon2 = lon2 + 360
    return lon1,lon2,lat1,lat2

