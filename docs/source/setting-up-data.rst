***********************
Setting Up Your Dataset
***********************

``climpred`` relies on a consistent naming system for ``xarray`` dimensions. This allows things to run more easily under-the-hood.

**Prediction ensembles** are expected at the minimum to contain dimensions ``init`` and ``lead``. ``init`` is the initialization dimension, that relays the time steps at which the ensemble was initialized. ``lead`` is the lead time of the forecasts from initialization. Another crucial dimension is ``member``, which holds the various ensemble members. Any additional dimensions will be passed through ``climpred`` without issue: these could be things like ``lat``, ``lon``, ``depth``, etc.

**Control runs, references, and observational products** are expected to contain the ``time`` dimension at the minimum. For best use of ``climpred``, their ``time`` dimension should cover the full length of ``init`` from the accompanying prediction ensemble, if possible. These products can also include additional dimensions, such as ``lat``, ``lon``, ``depth``, etc.

See the below table for a summary of dimensions used in ``climpred``, and data types that ``climpred`` supports for them.

+------------+-----------------------------------+-----------------------------------------------+
| short_name | types                             | long_name                                     |
+------------+-----------------------------------+-----------------------------------------------+
| ``lead``   | ``int``                           | lead timestep after initialization [``init``] |
+------------+-----------------------------------+-----------------------------------------------+
| ``init``   | ``int``                           | initialization: start date of experiment      |
+------------+-----------------------------------+-----------------------------------------------+
| ``member`` | ``int``, ``str``                  | ensemble member                               |
+------------+-----------------------------------+-----------------------------------------------+
