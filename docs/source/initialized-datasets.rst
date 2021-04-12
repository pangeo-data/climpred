********************
Initialized Datasets
********************

Probably the hardest part working with ``climpred`` is getting the initialized datasets complying to the expectations and data model of ``climpred``. For names, data types and conventions of ``xr.Dataset`` dimensions and coordinates, please refer to `Setting up your Dataset <setting-up-data.html>`_.

Here, we list publicly available initialized datasets:


+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| Short Name | Community | Description | Data Source | Reference Paper | Example |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| DCPP       | decadal   |             | ESGF | Boer et al. 2016 | `DCPP with intake-esm <examples/preprocessing/setup_your_own_data.html#intake-esm-for-cmorized-output>`_., Presentation by Anderson Banihirwe at NOAA's 45th Climate Diagnostics & Prediction Workshop `slides <https://talks.andersonbanihirwe.dev/climpred-cdpw-2020.html>`_ `Notebook <https://nbviewer.jupyter.org/github/andersy005/talks/blob/gh-pages/notebooks/climpred-demo.ipynb>`_
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| CESM-DPLE | decadal |     | ESGF-Link? | Yeager et al. 2020 | many standard climpred examples
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| NMME | seasonal |     | ESGF-Link? | Kirtman et al. | seasonal example
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| SubX   | subseasonal  |       | `IRIDL <http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/>`_ | Pegion et al. 2019 | many in examples
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| S2S | subseasonal |      | `IRIDL <https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/>`_, `climetlab-s2s-ai-competition <https://github.com/ecmwf-lab/climetlab-s2s-ai-competition>`_ | Vitart, Robertson | `PR <https://github.com/pangeo-data/climpred/pull/593>`_
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
|  | weather | `<Pull Request <contributing.html>`_ | for numerical weather prediction dataset | highly appreciated  |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+

If you find or use another publicly available initialized datasets, please consider adding a `<Pull Request <contributing.html>`_.
