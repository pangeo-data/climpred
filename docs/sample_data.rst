.. _sample-data:

Sample Data
===========

``climpred`` provides example datasets from the MPI-ESM-LR decadal prediction ensemble and the CESM decadal prediction ensemble. See the example notebooks in the Github repository to see them in use.

You can view the datasets available to be loaded with the ``get_datasets()`` command::
 
  from climpred.loadutils import get_datasets
  get_datasets()
  >>> 'MPI-DP-1D': decadal prediction ensemble area averages of SST/SSS/AMO.
  >>> 'MPI-DP-3D': decadal prediction ensemble lat/lon/time of SST/SSS/AMO.
  >>> 'MPI-control-1D': area averages for the control run of SST/SSS.
  >>> 'MPI-control-3D': lat/lon/time for the control run of SST/SSS.
  >>> 'CESM-DP': decadal prediction ensemble of global mean SSTs.
  >>> 'CESM-LE': uninitialized ensemble of global mean SSTs.
  >>> 'ERSST': observations of global mean SSTs.
  >>> 'CESM-reference': hindcast simulation that initializes CESM-DP.

From here, loading a dataset is easy. Note that you need to be connected to the internet for this to work -- the datasets are being pulled from the ``climpred`` repository. Once loaded, it is cached on your computer so you can reload extremely quickly. These datasets are very small (< 1MB each) so they won't take up much space.::

  from climpred.loadutils import open_dataset
  open_dataset('ERSST')
  >>>  <xarray.Dataset>
  >>> Dimensions:  (year: 61)
  >>> Coordinates:
  >>>  * year     (year) int64 1955 1956 1957 1958 1959 ... 2011 2012 2013 2014 2015
  >>> Data variables:
  >>>    sst      (year) float32 ...


