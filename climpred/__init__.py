"""
climpred
-------
An xarray wrapper for decadal climate prediction.

Available Modules:
-----------------
1. prediction: Contains definitions related to decadal climate prediction.
2. stats: Contains definitions related to gridded and time series statistics.
3. loadutils: Utilities for loading sample datasets.
"""
from . import prediction
from . import stats
from . import loadutils
from . import relative_entropy
