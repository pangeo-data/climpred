"""
Objects dealing with EBUS boundaries for plotting, statistics, etc.

Functions
---------
- `visual_bounds` : lat/lon bounds for close-up shots of our regions.
- `latitude_bounds` : lat bounds for statistical analysis

To do
-----
- `full_scope_bounds` : regions pulled from Chavez paper for lat/lon to show full
system for validation
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
        lat1 = 32
        lat2 = 45
        lon1 = -135
        lon2 = -115
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
        lat1 = -30
        lat2 = -15
        lon1 = 5
        lon2 = 20
    else:
        raise ValueError('\n' + 'Must select from the following EBUS strings:' \
                         + '\n' + 'CalCS' + '\n' + 'CanCS' + '\n' + 'BenCS' + \
                         '\n' + 'HumCS')
    if (std_lon == False) & (EBC != 'BenCS'):
        lon1 = lon1 + 360
        lon2 = lon2 + 360
    return lon1,lon2,lat1,lat2

def latitude_bounds(EBC):
    """
    Returns the standard 10 degrees of latitude to be analyzed for each system. 
    For the CalCS, HumCS, and BenCS, this comes from the Chavez 2009 EBUS Comparison
    paper. For the CanCS, this comes from the Aristegui 2009 CanCS paper. These
    bounds are used in the EBUS CO2 Flux comparison study to standardize latitude.

    Parameters 
    ----------
    EBC : str
        Identifier for the boundary current.
        'CalCS' : California Current
        'HumCS' : Humboldt Current
        'CanCS' : Canary Current
        'BenCS' : Benguela Current

    Returns
    -------
    lat1 : int
        Minimum latitude bound.
    lat2 : int
        Maximum latitude bound.

    Examples
    --------
    import esmtools.ebus as eb
    y1,y2 = eb.boundaries.latitude_bounds('HumCS')
    """
    if EBC == 'CalCS':
        lat1 = 34
        lat2 = 44
    elif EBC == 'HumCS':
        lat1 = -16
        lat2 = -6
    elif EBC == 'CanCS':
        lat1 = 21
        lat2 = 31
    elif EBC == 'BenCS':
        lat1 = -28
        lat2 = -18
    else:
        raise ValueError('\n' + 'Must select from the following EBUS strings:'
                         + '\n' + 'CalCS' + '\n' + 'CanCS' + '\n' + 'BenCS' +
                         '\n' + 'HumCS')
    return lat1, lat2
