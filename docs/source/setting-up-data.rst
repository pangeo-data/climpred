***********************
Setting Up Your Dataset
***********************

``climpred`` relies on a consistent naming system for
`xarray <https://xarray.pydata.org/en/stable/>`_ dimensions.
This allows things to run more easily under-the-hood.

:py:class:`.PredictionEnsemble` expects at the minimum to contain dimensions
``init`` and ``lead``.

``init`` is the initialization dimension, that relays the time
steps at which the ensemble was initialized.
``init`` is known as ``forecast_reference_time`` in the `CF convention <http://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_.
``init`` must be of type :py:class:`pandas.DatetimeIndex`, or
:py:class:`xarray.CFTimeIndex`.
If ``init`` is of type ``int``, it is assumed to be annual data starting Jan 1st.
A UserWarning is issues when this assumption is made.

``lead`` is the lead time of the forecasts from initialization.
``lead`` is known as ``forecast_period`` in the `CF convention <http://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_.
``lead`` must be ``int`` or ``float``.
The units for the ``lead`` dimension must be specified in as an attribute.
Valid options are ``["years", "seasons", "months"]`` and
``["weeks", "pentads", "days", "hours", "minutes", "seconds"]``.
If ``lead`` is provided as :py:class:`pandas.Timedelta` up to ``"weeks"``, ``lead``
is converted to ``int`` and a corresponding ``lead.attrs["units"]``.
For larger ``lead`` as :py:class:`pandas.Timedelta`
``["months", "seasons" or "years"]``, no conversion is possible.

``valid_time=init+lead`` will be calculated in :py:class:`.PredictionEnsemble` upon
instantiation.

Another crucial dimension is ``member``, which holds the various ensemble members,
which is only required for probabilistic metrics. ``member`` is known as
``realization`` in the `CF convention <http://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_

Any additional dimensions will be broadcasted: these could be dimensions like ``lat``,
``lon``, ``depth``, etc.

If the expected dimensions are not found, but the matching `CF convention <http://cfconventions.org/Data/cf-standard-names/77/build/cf-standard-name-table.html>`_
``standard_name`` in a coordinate attribute, the dimension is renamed to the
corresponding ``climpred`` ensemble dimension.

Check out the demo to setup a ``climpred``-ready prediction ensemble
`from your own data <examples/misc/setup_your_own_data.html>`_ or via
`intake-esm <https://intake-esm.readthedocs.io/>`_ from `CMIP DCPP <examples/misc/setup_your_own_data.html#intake-esm-for-cmorized-output>`_.

**Verification products** are expected to contain the ``time`` dimension at the minimum.
For best use of ``climpred``, their ``time`` dimension should cover the full length of
``init`` and be the same calendar type as the accompanying prediction ensemble.
The ``time`` dimension must be :py:class:`pandas.DatetimeIndex`, or
:py:class:`xarray.CFTimeIndex`.
``time`` dimension of type ``int`` is assumed to be annual data starting Jan 1st.
A UserWarning is issued when this assumption is made.
These products can also include additional dimensions, such as ``lat``, ``lon``,
``depth``, etc.

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
     - ``int``, ``float`` or :py:class:`pandas.Timedelta` up to ``weeks``
     - lead timestep after initialization ``init``
     - ``forecast_period``
     - units (str) [``years``, ``seasons``, ``months``, ``weeks``, ``pentads``, ``days``, ``hours``, ``minutes``, ``seconds``] or :py:class:`pandas.Timedelta`
   * - ``init``
     -  :py:class:`pandas.DatetimeIndex` or :py:class:`xarray.CFTimeIndex`.
     - initialization as start date of experiment
     - ``forecast_reference_time``
     - None
   * - ``member``
     - ``int``, ``str``
     - ensemble member
     - ``realization``
     - None

Probably the most challenging part is concatenating
(:py:func:`xarray.concat`) raw model output with dimension ``time`` of
multiple simulations to a multi-dimensional :py:class:`xarray.Dataset` containing
dimensions ``init``, (``member``) and ``lead``, where ``time`` becomes
``valid_time=init+lead``. One way of doing it is
:py:func:`climpred.preprocessing.shared.load_hindcast`.
