********************
Initialized Datasets
********************

Probably the hardest part in working with ``climpred`` is getting the ``initialized``
dataset complying to the expectations and data model of ``climpred``.
For names, data types and conventions of :py:class:`xarray.Dataset` dimensions and
coordinates, please refer to `Setting up your Dataset <setting-up-data.html>`_.

Here, we list publicly available initialized datasets and corresponding ``climpred``
examples:

.. list-table:: List of initialized Datasets
   :widths: 25 15 40 40 25 25
   :header-rows: 1

   * - Short Name
     - Community
     - Description
     - Data Source
     - Reference Paper
     - Example
   * - DCPP
     - decadal
     - `Decadal Climate Prediction Project (DCPP) contribution to CMIP6 <https://www.wcrp-climate.org/dcp-overview>`_
     - `ESGF <https://esgf-data.dkrz.de/search/cmip6-dkrz/>`_, `pangeo <https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html#loading-an-esm-collection>`_
     - :cite:t:`Boer2016`
     - `with intake-esm <examples/misc/setup_your_own_data.html#intake-esm-for-cmorized-output>`_, `Anderson <https://github.com/andersy005>`_ at NOAA's 45th CDP Workshop: `slides <https://talks.andersonbanihirwe.dev/climpred-cdpw-2020.html>`_, `Notebook <https://nbviewer.jupyter.org/github/andersy005/talks/blob/gh-pages/notebooks/climpred-demo.ipynb>`_
   * - CESM-DPLE
     - decadal
     - `Decadal Prediction Large Ensemble Project <http://www.cesm.ucar.edu/projects/community-projects/DPLE/>`_
     - `Data <https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.CESM1-CAM5-DP.html>`_
     - :cite:t:`Yeager2018`
     - many standard climpred `examples <quick-start.html>`_
   * - NMME
     - seasonal
     - `The North American Multimodel Ensemble: Phase-1 Seasonal-to-Interannual Prediction <https://www.cpc.ncep.noaa.gov/products/NMME/>`_
     - `IRIDL <http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/>`__
     - :cite:t:`Kirtman2014`
     - `seasonal SubX <examples.html#monthly-and-seasonal>`_
   * - SubX
     - subseasonal
     - `A Multimodel Subseasonal Prediction Experiment <http://cola.gmu.edu/subx/>`_
     - `IRIDL <http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/>`__
     - :cite:t:`Pegion2019`
     - `subseasonal SubX <examples.html#subseasonal>`_
   * - S2S
     - subseasonal
     - `The Subseasonal to Seasonal (S2S) Prediction Project Database <http://wwww.s2sprediction.net/>`_
     - `IRIDL <https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/>`__, `climetlab <https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge>`_
     - :cite:t:`Vitart2017`
     - `IRIDL <examples/subseasonal/daily-S2S-IRIDL.html>`_, `EWC Cloud/climetlab <examples/subseasonal/daily-S2S-ECMWF.html>`_
   * - GEFS
     - weather
     - `Global Ensemble Forecast System (GEFS) <https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-ensemble-forecast-system-gefs>`_
     - `NOAA THREDDS <https://www.ncei.noaa.gov/thredds/catalog/model-gefs-003/catalog.html>`_
     - :cite:t:`Toth1993`
     - `GEFS NWP <examples/NWP/NWP_GEFS_6h_forecasts.html>`_
   * - name
     - weather
     - please add a `Pull Request <contributing.html>`_ for numerical weather prediction
     - dataset
     - appreciated
     - `examples to add <https://github.com/pangeo-data/climpred/issues/602>`_

If you find or use another publicly available initialized datasets, please consider
adding a `Pull Request <contributing.html>`_.

References
##########

.. bibliography::
  :filter: docname in docnames
