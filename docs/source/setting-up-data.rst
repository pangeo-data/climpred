***********************
Setting Up Your Dataset
***********************

``climpred`` relies on a consistent naming system for ``xarray`` dimensions.
This allows things to run more easily under-the-hood.

**Prediction ensembles** are expected at the minimum to contain dimensions
``init`` and ``lead``. ``init`` is the initialization dimension, that relays the time
steps at which the ensemble was initialized. ``init`` must be of type ``int``,
``pd.DatetimeIndex``, or ``xr.cftimeIndex``. If ``init`` is of type ``int``, it is assumed to
be annual data. A user warning is issues when this assumption is made.

``lead`` is the lead time of the forecasts from initialization. The units for the ``lead``
dimension must be specified in as an attribute. Valid options are
``years, seasons, months, weeks, pentads, days, hours, minutes, seconds``.
If ``lead`` is provided as ``pd.Timedelta`` up to weeks, ``lead`` is converted to
``int`` and a corresponding ``lead.attrs['units']``. For larger ``lead`` as
``pd.Timedelta`` (months or years), there is no conversion possible.

Another crucial dimension is ``member``, which holds the various ensemble members.
Any additional dimensions will
be passed through ``climpred`` without issue: these could be things like ``lat``,
``lon``, ``depth``, etc.

If the expected dimensions are not found, but the matching ``standard_name`` in a
coordinate attribute, the dimension is renamed to the corresponding ``climpred``
ensemble dimension.

Check out the demo to setup a ``climpred``-ready prediction ensemble
`from your own data <examples/misc/setup_your_own_data.html>`_ or via `intake-esm <https://intake-esm.readthedocs.io/>`_ from `CMIP DCPP <examples/misc/setup_your_own_data.html#intake-esm-for-cmorized-output>`_.

**Verification products** are expected to contain the ``time`` dimension at the minimum.
For best use of ``climpred``, their ``time`` dimension should cover the full length of
``init`` and be the same calendar type as the accompanying prediction ensemble, if possible. The ``time`` dimension
must be of type ``int``, ``pd.DatetimeIndex`` or ``xr.cftimeIndex``. ``time`` dimension
of type ``int`` is assumed to be annual data.  A user warning is issued when this assumption
is made. These products can also include additional dimensions, such as ``lat``,
``lon``, ``depth``, etc.

See the below table for a summary of dimensions used in ``climpred``, and data types
that ``climpred`` supports for them.

.. list-table:: List of ``climpred`` dimension and coordinates
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Short Name
     - Types
     - Long name
     - `CF convention <http://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_
     - Attribute(s)
   * - ``lead``
     - ``int`` (or ``pd.Timedelta`` for units up to weeks)
     - lead timestep after initialization ``init``
     - ``forecast_period``
     - units (str) [years, seasons, months, weeks, pentads, days, hours, minutes, seconds]
   * - ``init``
     - ``int``, ``pd.DatetimeIndex``, ``xr.CFTimeIndex``
     - initialization as start date of experiment
     - ``forecast_reference_time``
     - None
   * - ``member``
     - ``int``, ``str``
     - ensemble member
     - ``realization``
     - None
