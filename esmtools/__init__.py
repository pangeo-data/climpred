"""
ESMtools
-------

A package for analyzing, processing, and mapping ESM output with an emphasis
on ocean model output in particular.

Available Modules:
-----------------

1. filtering: Contains definitions that assist in spatial and temporal
filtering of output.
2. vis: Contains definitions for colorbars, color maps, and any sort
of global or regional projections.
3. stats: Contains definitions for computing general statistics on output.

"""

from . import filtering
from . import vis 
from . import stats
import esmtools.ebus 
