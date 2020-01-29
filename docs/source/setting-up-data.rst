***********************
Setting Up Your Dataset
***********************

``climpred`` relies on a consistent naming system for ``xarray`` dimensions.
This allows things to run more easily under-the-hood.

**Prediction ensembles** are expected at the minimum to contain dimensions
``init`` and ``lead``. ``init`` is the initialization dimension, that relays the time
steps at which the ensemble was initialized. ``init`` must be of type int,
pd.DatetimeIndex, or xr.cftimeIndex.  If ``init`` is of type ``int``, it is assumed to
be annual data.  A user warning is issues when this assumption is made.  ``lead`` is
the lead time of the forecasts from initialization. The units for the ``lead``
dimension must be specified in as an attribute.  Valid options are
``years, seasons, months, weeks, pentads, days``.  Another crucial dimension is
``member``, which holds the various ensemble members. Any additional dimensions will
be passed through ``climpred`` without issue: these could be things like ``lat``,
``lon``, ``depth``, etc.

**Verification products** are expected to contain the ``time`` dimension at the minimum.
For best use of ``climpred``, their ``time`` dimension should cover the full length of
``init`` from the accompanying prediction ensemble, if possible. The ``time`` dimension
must be of type ``int``, ``pd.DatetimeIndex`` or ``xr.cftimeIndex``. ``time`` dimension
of type int is assumed to be annual data.  A user warning is issued when this assumption
is made. These products can also include additional dimensions, such as ``lat``,
``lon``, ``depth``, etc.

See the below table for a summary of dimensions used in ``climpred``, and data types
that ``climpred`` supports for them.

+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| Short Name | Types                                             | Long Name                                      | Attribute(s)                                               |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| ``lead``   | ``int``                                           | lead timestep after initialization, [``init``] | units (str) [years, seasons, months, weeks, pentads, days] |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| ``init``   | ``int``, ``pd.DatetimeIndex``, ``xr.CFTimeIndex`` | initialization: start date of experiment       | None                                                       |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
| ``member`` | ``int``, ``str``                                  | ensemble member                                | None                                                       |
+------------+---------------------------------------------------+------------------------------------------------+------------------------------------------------------------+
