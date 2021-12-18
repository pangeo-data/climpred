Examples
########

.. note::

    Please use the ``climpred-dev`` environment

    .. code-block:: bash

       conda env create -f ci/requirements/climpred-dev.yml

    to ensure that all dependencies are installed to complete all example
    notebooks listed here.


Numerical Weather Prediction
============================
.. toctree::
  :maxdepth: 1

  examples/NWP/NWP_GEFS_6h_forecasts.ipynb


Subseasonal
===========
.. toctree::
  :maxdepth: 1

  examples/subseasonal/daily-subx-example.ipynb
  examples/subseasonal/daily-S2S-IRIDL.ipynb
  examples/subseasonal/weekly-subx-example.ipynb
  examples/subseasonal/daily-S2S-ECMWF.ipynb


Monthly and Seasonal
====================
.. toctree::
  :maxdepth: 1

  examples/monseas/monthly-enso-subx-example.ipynb
  examples/monseas/seasonal-enso-subx-example.ipynb


Decadal
=======
.. toctree::
  :maxdepth: 1

  examples/decadal/perfect-model-predictability-demo.ipynb
  examples/decadal/tropical-pacific-ssts.ipynb
  examples/decadal/diagnose-potential-predictability.ipynb
  examples/decadal/Significance.ipynb


Misc
====
.. toctree::
  :maxdepth: 1

  examples/misc/efficient_dask.ipynb
  examples/misc/climpred_gpu.ipynb
  examples/misc/setup_your_own_data.ipynb
