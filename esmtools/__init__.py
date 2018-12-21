"""
ESMtools
-------

A package for analyzing, processing, and mapping ESM output with an emphasis
on ocean model output in particular.

Available Modules:
-----------------
1. carbon: Contains definitions related to the carbon cycle and carbonate chemistry.
2. colormaps: Custom colormaps from NCL.
3. filtering: Contains definitions that assist in spatial and temporal
filtering of output.
4. physics: Contains definitions related to physical conversions.
5. stats: Contains definitions for computing general statistics on output.
6. vis: Contains definitions for colorbars, color maps, and any sort
of global or regional projections.
7. prediction: Contains definitions for decadal climate prediction.
"""

from . import filtering
from . import vis 
from . import stats
from . import colormaps
from . import carbon
from . import physics
from . import prediction
