********************
Initialized Datasets
********************

Probably the hardest part in working with ``climpred`` is getting the initialized datasets complying to the expectations and data model of ``climpred``. For names, data types and conventions of ``xr.Dataset`` dimensions and coordinates, please refer to `Setting up your Dataset <setting-up-data.html>`_.

Here, we list publicly available initialized datasets and corresponding climpred examples:

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
     - Decadal Climate Prediction Project (DCPP) contribution to CMIP6 `Website <https://www.wcrp-climate.org/dcp-overview>`_
     - `ESGF <https://esgf-data.dkrz.de/search/cmip6-dkrz/>`_, `pangeo <https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html#loading-an-esm-collection>`_
     - [Boer2016]_
     - `with intake-esm <examples/misc/setup_your_own_data.html#intake-esm-for-cmorized-output>`_, `Anderson <https://github.com/andersy005>`_ at NOAA's 45th CDP Workshop: `slides <https://talks.andersonbanihirwe.dev/climpred-cdpw-2020.html>`_, `Notebook <https://nbviewer.jupyter.org/github/andersy005/talks/blob/gh-pages/notebooks/climpred-demo.ipynb>`_
   * - CESM-DPLE
     - decadal
     - Decadal Prediction Large Ensemble Project `Website <http://www.cesm.ucar.edu/projects/community-projects/DPLE/>`_
     - `Data <https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.CESM1-CAM5-DP.html>`_
     - [Yeager2018]_
     - many standard climpred `examples <quick-start.html>`_
   * - NMME
     - seasonal
     - The North American Multimodel Ensemble: Phase-1 Seasonal-to-Interannual Prediction `Website <https://www.cpc.ncep.noaa.gov/products/NMME/>`_
     - `IRIDL <http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/>`_
     - [Kirtman2014]_
     - `seasonal SubX <examples.html#monthly-and-seasonal>`_
   * - SubX
     - subseasonal
     - A Multimodel Subseasonal Prediction Experiment `Website <http://cola.gmu.edu/subx/>`_
     - `IRIDL <http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/>`_
     - [Pegion2019]_
     - `subseasonal SubX <examples.html#subseasonal>`_
   * - S2S
     - subseasonal
     - The Subseasonal to Seasonal (S2S) Prediction Project Database `Website <http://wwww.s2sprediction.net/>`_
     - `IRIDL <https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/>`_, `climetlab <https://github.com/ecmwf-lab/climetlab-s2s-ai-challenge>`_
     - [Vitart2017]_
     - `IRIDL <examples/subseasonal/daily-S2S-IRIDL.html>`_, `EWC Cloud/climetlab <examples/subseasonal/daily-S2S-ECMWF.html>`_
   * - GEFS
     - weather
     - Global Ensemble Forecast System (GEFS), `Website <https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-ensemble-forecast-system-gefs>`_
     - `NOAA THREDDS <https://www.ncei.noaa.gov/thredds/catalog/model-gefs-003/catalog.html>`_
     - add publication
     - `GEFS NWP <examples/NWP/NWP_GEFS_6h_forecasts.html>`_
   * - name
     - weather
     - please add a `Pull Request <contributing.html>`_ for numerical weather prediction
     - dataset
     - appreciated
     - `examples to add <https://github.com/pangeo-data/climpred/issues/602>`_

If you find or use another publicly available initialized datasets, please consider adding a `Pull Request <contributing.html>`_.

References
##########

.. [Kirtman2014] Kirtman, Ben P., et al.: The North American Multimodel Ensemble: Phase-1 seasonal-to-interannual prediction; Phase-2 toward developing intraseasonal prediction. Bull. Amer. Meteor. Soc., 2014, 95, 585–601. doi: http://dx.doi.org/10.1175/BAMS-D-12-00050.1

.. [Boer2016] Boer, G. J., Smith, D. M., Cassou, C., Doblas-Reyes, F., Danabasoglu, G., Kirtman, B., Kushnir, Y., Kimoto, M., Meehl, G. A., Msadek, R., Mueller, W. A., Taylor, K. E., Zwiers, F., Rixen, M., Ruprich-Robert, Y., and Eade, R.: The Decadal Climate Prediction Project (DCPP) contribution to CMIP6, Geosci. Model Dev., 2016, 9, 3751-3777, https://doi.org/10.5194/gmd-9-3751-2016

.. [Vitart2017] Vitart, F., Ardilouze, C., Bonet, A., Brookshaw, A., Chen, M., Codorean, C., Déqué, M., Ferranti, L., Fucile, E., Fuentes, M., Hendon, H., Hodgson, J., Kang, H.-S., Kumar, A., Lin, H., Liu, G., Liu, X., Malguzzi, P., Mallas, I., … Zhang, L.: The Subseasonal to Seasonal (S2S) Prediction Project Database. Bulletin of the American Meteorological Society, 2017, 98(1), 163–173. doi: https://doi.org/10.1175/BAMS-D-16-0017.1

.. [Yeager2018] Yeager, S. G., Danabasoglu, G., Rosenbloom, N., Strand, W., Bates, S., Meehl, G., Karspeck, A., Lindsay, K., Long, M. C., Teng, H., & Lovenduski, N. S.: Predicting near-term changes in the Earth System: A large ensemble of initialized decadal prediction simulations using the Community Earth System Model. Bulletin of the American Meteorological Society, 2018. doi: https://doi.org/10.1175/BAMS-D-17-0098.1

.. [Pegion2019] Pegion, K., Kirtman, B. P., Becker, E., Collins, D. C., LaJoie, E., Burgman, R., Bell, R., DelSole, T., Min, D., Zhu, Y., Li, W., Sinsky, E., Guan, H., Gottschalck, J., Metzger, E. J., Barton, N. P., Achuthavarier, D., Marshak, J., Koster, R. D., … Kim, H.: The Subseasonal Experiment (SubX): A Multimodel Subseasonal Prediction Experiment. Bulletin of the American Meteorological Society, 2019, 100(10), 2043–2060. doi: https://doi.org/10.1175/BAMS-D-18-0270.1
