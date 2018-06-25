"""
ESMtools
-------

A package for analyzing, processing, and mapping ESM output with an emphasis
on ocean model output in particular.

Available Modules:
-----------------
1. carbon: Contains definitions related to the carbon cycle and carbonate chemistry.
2. colormaps: Custom colormaps from NCL.
3. ebus: Contains functions specific to working with Eastern Boundary Upwelling
Systems.
4. filtering: Contains definitions that assist in spatial and temporal
filtering of output.
5. physics: Contains definitions related to physical conversions.
6. stats: Contains definitions for computing general statistics on output.
7. unfunc: Contains definitions to use on xarray .apply()
8. vis: Contains definitions for colorbars, color maps, and any sort
of global or regional projections.
9. mpas: Contains definitions for visualization and processing on MPAS output.
"""

from . import filtering
from . import vis 
from . import stats
from . import ebus
from . import ufunc
from . import colormaps
from . import carbon
from . import physics
from . import mpas
