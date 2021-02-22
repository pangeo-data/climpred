==========
What's New
==========


climpred v2.1.2 (2021-01-22)
============================

This release is the fixed version for our Journal of Open Source Software (JOSS)
article about ``climpred``, see `review
<https://github.com/openjournals/joss-reviews/issues/2781>`_.

New Features
------------
- Function to calculate predictability horizon
  :py:func:`~climpred.predictability_horizon.predictability_horizon` based on condition.
  (:issue:`46`, :pr:`521`) `Aaron Spring`_.

Bug fixes
---------
- :py:meth:`~climpred.classes.PredictionEnsemble.smooth` now carries ``lead.attrs``
  (:issue:`527`, pr:`521`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.PerfectModelEnsemble.verify` now works with ``references``
  also for geospatial inputs, which returned ``NaN`` before.
  (:issue:`522`, pr:`521`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.PredictionEnsemble.plot` now shifts composite lead
  frequencies like ``days``, ``pentads``, ``seasons`` correctly.
  (:issue:`532`, :pr:`533`) `Aaron Spring`_.
- Adapt to ``xesmf>=0.5.2`` for spatial xesmf smoothing. (:issue:`543`, :pr:`548`)
  `Aaron Spring`_.
- :py:meth:`~climpred.classes.HindcastEnsemble.remove_bias` now carries attributes.
  (:issue:`531`, :pr:`551`) `Aaron Spring`_.


climpred v2.1.1 (2020-10-13)
============================

Breaking changes
----------------

This version introduces a lot of breaking changes. We are trying to overhaul
``climpred`` to have an intuitive API that also forces users to think about methodology
choices when running functions. The main breaking changes we introduced are for
:py:meth:`~climpred.classes.HindcastEnsemble.verify` and
:py:meth:`~climpred.classes.PerfectModelEnsemble.verify`. Now, instead of assuming
defaults for most keywords, we require the user to define ``metric``, ``comparison``,
``dim``, and ``alignment`` (for hindcast systems). We also require users to designate
the number of ``iterations`` for bootstrapping.

- User now has to designate number of iterations with ``iterations=...`` in
  :py:meth:`~climpred.classes.HindcastEnsemble.bootstrap` (:issue:`384`, :pr:`436`)
  `Aaron Spring`_ and `Riley X. Brady`_.
- Make ``metric``, ``comparison``, ``dim``, and ``alignment`` required (previous default
  ``None``) arguments for :py:meth:`~climpred.classes.HindcastEnsemble.verify`
  (:issue:`384`, :pr:`436`) `Aaron Spring`_ and `Riley X. Brady`_.
- Metric :py:class:`~climpred.metrics._brier_score` and
  :py:func:`~climpred.metrics._threshold_brier_score` now requires callable keyword
  argument ``logical`` instead of ``func`` (:pr:`388`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.HindcastEnsemble.verify` does not correct ``dim``
  automatically to ``member`` for probabilistic metrics.
  (:issue:`282`, :pr:`407`) `Aaron Spring`_.
- Users can no longer add multiple observations to
  :py:class:`~climpred.classes.HindcastEnsemble`. This will make current and future
  development much easier on maintainers (:issue:`429`, :pr:`453`) `Riley X. Brady`_.
- Standardize the names of the output coordinates for
  :py:meth:`~climpred.classes.PredictionEnsemble.verify` and
  :py:meth:`~climpred.classes.PredictionEnsemble.bootstrap` to ``initialized``,
  ``uninitialized``, and ``persistence``. ``initialized`` showcases the metric result
  after comparing the initialized ensemble to the verification data; ``uninitialized``
  when comparing the uninitialized (historical) ensemble to the verification data;
  ``persistence`` is the evaluation of the persistence forecast
  (:issue:`460`, :pr:`478`, :issue:`476`, :pr:`480`) `Aaron Spring`_.
- ``reference`` keyword in :py:meth:`~climpred.classes.HindcastEnsemble.verify` should
  be choosen from [``uninitialized``, ``persistence``]. ``historical`` no longer works (:issue:`460`, :pr:`478`, :issue:`476`, :pr:`480`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.HindcastEnsemble.verify` returns no ``skill`` dimension
  if ``reference=None``  (:pr:`480`) `Aaron Spring`_.
- ``comparison`` is not applied to uninitialized skill in
  :py:meth:`~climpred.classes.HindcastEnsemble.bootstrap`.
  (:issue:`352`, :pr:`418`) `Aaron Spring`_.

New Features
------------

This release is accompanied by a bunch of new features. Math operations can now be used
with our :py:class:`~climpred.classes.PredictionEnsemble` objects and their variables
can be sub-selected. Users can now quick plot time series forecasts with these objects.
Bootstrapping is available for :py:class:`~climpred.classes.HindcastEnsemble`. Spatial
dimensions can be passed to metrics to do things like pattern correlation. New metrics
have been implemented based on Contingency tables. We now include an early version
of bias removal for :py:class:`~climpred.classes.HindcastEnsemble`.

- Use math operations like ``+-*/`` with :py:class:`~climpred.classes.HindcastEnsemble`
  and :py:class:`~climpred.classes.PerfectModelEnsemble`. See a demo of this
  `here <prediction-ensemble-object.html#Arithmetic-Operations-with-PredictionEnsemble-Objects>`__
  (:pr:`377`) `Aaron Spring`_.
- Subselect data variables from ``PredictionEnsemble`` as from ``xr.Dataset``:
  ``PredictionEnsemble[['var1', 'var3']]`` (:pr:`409`) `Aaron Spring`_.
- Plot all datasets in :py:class:`~climpred.classes.HindcastEnsemble` or
  :py:class:`~climpred.classes.PerfectModelEnsemble` by
  :py:meth:`~climpred.classes.PredictionEnsemble.plot` if no other spatial dimensions
  are present. (:pr:`383`) `Aaron Spring`_.
- Bootstrapping now available for :py:class:`~climpred.classes.HindcastEnsemble` as
  :py:meth:`~climpred.classes.HindcastEnsemble.bootstrap`, which is analogous to
  the :py:class:`~climpred.classes.PerfectModelEnsemble` method (:issue:`257`, :pr:`418`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.HindcastEnsemble.verify` allows all dimensions from
  ``initialized`` ensemble as ``dim``. This allows e.g. spatial dimensions to be used
  for pattern correlation. Make sure to use ``skipna=True`` when using spatial dimensions
  and output has nans (in the case of land, for instance) (:issue:`282`, :pr:`407`) `Aaron Spring`_.
- Allow binary forecasts at when calling :py:meth:`~climpred.classes.HindcastEnsemble.verify`,
  rather than needing to supply binary results beforehand. In other words,
  ``hindcast.verify(metric='brier_score', comparison='m2o', dim='member', logical=logical)``
  is now the same as
  ``hindcast.map(logical).verify(metric='brier_score', comparison='m2o', dim='member'``.
  (:pr:`431`) `Aaron Spring`_.
- Check calendar types when using
  :py:meth:`~climpred.classes.HindcastEnsemble.add_observations`,
  :py:meth:`~climpred.classes.HindcastEnsemble.add_uninitialized`,
  :py:meth:`~climpred.classes.PerfectModelEnsemble.add_control` to ensure that the
  verification data calendars match that of the initialized ensemble.
  (:issue:`300`, :pr:`452`, :issue:`422`, :pr:`462`)
  `Riley X. Brady`_ and `Aaron Spring`_.
- Implement new metrics which have been ported over from
  https://github.com/csiro-dcfp/doppyo/ to ``xskillscore`` by `Dougie Squire`_.
  (:pr:`439`, :pr:`456`) `Aaron Spring`_

    * rank histogram :py:func:`~climpred.metrics._rank_histogram`
    * discrimination :py:func:`~climpred.metrics._discrimination`
    * reliability :py:func:`~climpred.metrics._reliability`
    * ranked probability score :py:func:`~climpred.metrics._rps`
    * contingency table and related scores :py:func:`~climpred.metrics._contingency`

- Perfect Model :py:meth:`~climpred.classes.PerfectModelEnsemble.verify`
  no longer requires ``control`` in :py:class:`~climpred.classes.PerfectModelEnsemble`.
  It is only required when ``reference=['persistence']``. (:pr:`461`) `Aaron Spring`_.
- Implemented bias removal
  :py:class:`~climpred.classes.HindcastEnsemble.remove_bias`.
  ``remove_bias(how='mean')`` removes the mean bias of initialized hindcasts with
  respect to observations. See `example <bias_removal.html>`__.
  (:pr:`389`, :pr:`443`, :pr:`459`) `Aaron Spring`_ and `Riley X. Brady`_.

Deprecated
----------

- ``spatial_smoothing_xrcoarsen`` no longer used for spatial smoothing.
  (:pr:`391`) `Aaron Spring`_.
- ``compute_metric``, ``compute_uninitialized`` and ``compute_persistence`` no longer
  in use for :py:class:`~climpred.classes.PerfectModelEnsemble` in favor of
  :py:meth:`~climpred.classes.PerfectModelEnsemble.verify` with the ``reference``
  keyword instead. (:pr:`436`, :issue:`468`, :pr:`472`) `Aaron Spring`_ and `Riley X. Brady`_.
- ``'historical'`` no longer a valid choice for ``reference``. Use ``'uninitialized'``
  instead. (:pr:`478`) `Aaron Spring`_.

Bug Fixes
---------

- :py:meth:`~climpred.classes.PredictionEnsemble.verify` and
  :py:meth:`~climpred.classes.PredictionEnsemble.bootstrap` now accept ``metric_kwargs``.
  (:pr:`387`) `Aaron Spring`_.
- :py:meth:`~climpred.classes.PerfectModelEnsemble.verify` now accepts ``'uninitialized'``
  as a reference. (:pr:`395`) `Riley X. Brady`_.
- Spatial and temporal smoothing :py:meth:`~climpred.classes.PredictionEnsemble.smooth` now
  work as expected and rename time dimensions after
  :py:meth:`~climpred.classes.PredictionEnsembleEnsemble.verify`. (:pr:`391`) `Aaron Spring`_.
- ``PredictionEnsemble.verify(comparison='m2o', references=['uninitialized',
  'persistence']`` does not fail anymore. (:issue:`385`, :pr:`400`) `Aaron Spring`_.
- Remove bias using ``dayofyear`` in
  :py:meth:`~climpred.classes.HindcastEnsemble.reduce_bias`.
  (:pr:`443`) `Aaron Spring`_.
- ``climpred`` works with ``dask=>2.28``. (:issue:`479`, :pr:`482`) `Aaron Spring`_.

Documentation
-------------
- Updates ``climpred`` tagline to "Verification of weather and climate forecasts."
  (:pr:`420`) `Riley X. Brady`_.
- Adds section on how to use arithmetic with :py:class:`~climpred.classes.HindcastEnsemble`.
  (:pr:`378`) `Riley X. Brady`_.
- Add docs section for similar open-source forecasting packages.
  (:pr:`432`) `Riley X. Brady`_.
- Add all metrics to main API in addition to metrics page.
  (:pr:`438`) `Riley X. Brady`_.
- Add page on bias removal `Aaron Spring`_.

Internals/Minor Fixes
---------------------
- :py:meth:`~climpred.classes.PredictionEnsemble.verify` replaces deprecated
  ``PerfectModelEnsemble.compute_metric()`` and accepts ``reference`` as keyword.
  (:pr:`387`) `Aaron Spring`_.
- Cleared out unnecessary statistics functions from ``climpred`` and migrated them to
  ``esmtools``. Add ``esmtools`` as a required package. (:pr:`395`) `Riley X. Brady`_.
- Remove fixed pandas dependency from ``pandas=0.25`` to stable ``pandas``.
  (:issue:`402`, :pr:`403`) `Aaron Spring`_.
- ``dim`` is expected to be a list of strings in
  :py:func:`~climpred.prediction.compute_perfect_model` and
  :py:func:`~climpred.prediction.compute_hindcast`.
  (:issue:`282`, :pr:`407`) `Aaron Spring`_.
- Update ``cartopy`` requirement to 0.0.18 or greater to release lock on
  ``matplotlib`` version. Update ``xskillscore`` requirement to 0.0.18 to
  cooperate with new ``xarray`` version. (:pr:`451`, :pr:`449`)
  `Riley X. Brady`_
- Switch from Travis CI and Coveralls to Github Actions and CodeCov.
  (:pr:`471`) `Riley X. Brady`_
- Assertion functions added for :py:class:`~climpred.classes.PerfectModelEnsemble`:
  :py:func:`~climpred.testing.assert_PredictionEnsemble`. (:pr:`391`) `Aaron Spring`_.
- Test all metrics against synthetic data. (:pr:`388`) `Aaron Spring`_.


climpred v2.1.0 (2020-06-08)
============================

Breaking Changes
----------------

- Keyword ``bootstrap`` has been replaced with ``iterations``. We feel that this more accurately
  describes the argument, since "bootstrap" is really the process as a whole.
  (:pr:`354`) `Aaron Spring`_.

New Features
------------

- :py:class:`~climpred.classes.HindcastEnsemble` and
  :py:class:`~climpred.classes.PerfectModelEnsemble` now use an HTML representation, following the
  more recent versions of ``xarray``. (:pr:`371`) `Aaron Spring`_.
- ``HindcastEnsemble.verify()`` now takes ``reference=...`` keyword. Current options are
  ``'persistence'`` for a persistence forecast of the observations and
  ``'uninitialized'`` for an uninitialized/historical reference, such as an
  uninitialized/forced run. (:pr:`341`) `Riley X. Brady`_.
- We now only enforce a union of the initialization dates with observations if
  ``reference='persistence'`` for :py:class:`~climpred.classes.HindcastEnsemble`. This is to ensure
  that the same set of initializations is used
  by the observations to construct a persistence forecast. (:pr:`341`) `Riley X. Brady`_.
- :py:func:`~climpred.prediction.compute_perfect_model` now accepts initialization (``init``) as
  ``cftime`` and ``int``. ``cftime`` is now implemented into the bootstrap uninitialized functions
  for the perfect model configuration. (:pr:`332`) `Aaron Spring`_.
- New explicit keywords in bootstrap functions for ``resampling_dim`` and
  ``reference_compute`` (:pr:`320`) `Aaron Spring`_.
- Logging now included for ``compute_hindcast`` which displays the ``inits`` and
  verification dates used at each lead (:pr:`324`) `Aaron Spring`_,
  (:pr:`338`) `Riley X. Brady`_. See (`logging <alignment.html#Logging>`__).
- New explicit keywords added for ``alignment`` of verification dates and
  initializations. (:pr:`324`) `Aaron Spring`_. See (`alignment <alignment.html>`__)

    * ``'maximize'``: Maximize the degrees of freedom by slicing ``hind`` and
      ``verif`` to a common time frame at each lead. (:pr:`338`) `Riley X. Brady`_.
    * ``'same_inits'``: slice to a common init frame prior to computing
      metric. This philosophy follows the thought that each lead should be
      based on the same set of initializations. (:pr:`328`) `Riley X. Brady`_.
    * ``'same_verifs'``: slice to a common/consistent verification time frame prior
      to computing metric. This philosophy follows the thought that each lead
      should be based on the same set of verification dates. (:pr:`331`)
      `Riley X. Brady`_.

Performance
-----------

The major change for this release is a dramatic speedup in bootstrapping functions, led by
`Aaron Spring`_. We focused on scalability with ``dask`` and found many places we could compute
skill simultaneously over all bootstrapped ensemble members rather than at each iteration.

- Bootstrapping uninitialized skill in the perfect model framework is now sped up significantly for
  annual lead resolution. (:pr:`332`) `Aaron Spring`_.
- General speedup in :py:func:`~climpred.bootstrap.bootstrap_hindcast` and
  :py:func:`~climpred.bootstrap.bootstrap_perfect_model`: (:pr:`285`) `Aaron Spring`_.

    * Properly implemented handling for lazy results when inputs are chunked.

    * User gets warned when chunking potentially unnecessarily and/or inefficiently.

Bug Fixes
---------
- Alignment options now account for differences in the historical time series if
  ``reference='historical'``. (:pr:`341`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Added a `Code of Conduct <code_of_conduct.html>`__ (:pr:`285`) `Aaron Spring`_.
- Gather ``pytest.fixture in ``conftest.py``. (:pr:`313`) `Aaron Spring`_.
- Move ``x_METRICS`` and ``COMPARISONS`` to ``metrics.py`` and ``comparisons.py`` in
  order to avoid circular import dependencies. (:pr:`315`) `Aaron Spring`_.
- ``asv`` benchmarks added for ``HindcastEnsemble`` (:pr:`285`) `Aaron Spring`_.
- Ignore irrelevant warnings in ``pytest`` and mark slow tests
  (:pr:`333`) `Aaron Spring`_.
- Default ``CONCAT_KWARGS`` now in all ``xr.concat`` to speed up bootstrapping.
  (:pr:`330`) `Aaron Spring`_.
- Remove ``member`` coords for ``m2c`` comparison for probabilistic metrics.
  (:pr:`330`) `Aaron Spring`_.
- Refactored :py:func:`~climpred.prediction.compute_hindcast` and
  :py:func:`~climpred.prediction.compute_perfect_model`. (:pr:`330`) `Aaron Spring`_.
- Changed lead0 coordinate modifications to be compliant with ``xarray=0.15.1`` in
  :py:func:`~climpred.reference.compute_persistence`. (:pr:`348`) `Aaron Spring`_.
- Exchanged ``my_quantile`` with ``xr.quantile(skipna=False)``. (:pr:`348`) `Aaron Spring`_.
- Remove ``sig`` from
  :py:func:`~climpred.graphics.plot_bootstrapped_skill_over_leadyear`.
  (:pr:`351`) `Aaron Spring`_.
- Require ``xskillscore v0.0.15`` and use their functions for effective sample
  size-based metrics. (:pr: `353`) `Riley X. Brady`_.
- Faster bootstrapping without replacement used in threshold functions of
  ``climpred.stats`` (:pr:`354`) `Aaron Spring`_.
- Require ``cftime v1.1.2``, which modifies their object handling to create 200-400x
  speedups in some basic operations. (:pr:`356`) `Riley X. Brady`_.
- Resample first and then calculate skill in
  :py:func:`~climpred.bootstrap.bootstrap_perfect_model` and
  :py:func:`~climpred.bootstrap.bootstrap_hindcast` (:pr:`355`) `Aaron Spring`_.

Documentation
-------------
- Added demo to setup your own raw model output compliant to ``climpred``
  (:pr:`296`) `Aaron Spring`_. See (`here <examples/preprocessing/setup_your_own_data.html>`__).
- Added demo using ``intake-esm`` with ``climpred`` (:pr:`296`) `Aaron Spring`_.
  See (`here <examples/preprocessing/setup_your_own_data.html#intake-esm-for-cmorized-output>`__).
- Added `Verification Alignment <alignment.html>`_ page explaining how initializations
  are selected and aligned with verification data. (:pr:`328`) `Riley X. Brady`_.
  See (`here <alignment.html>`__).


climpred v2.0.0 (2020-01-22)
============================

New Features
------------
- Add support for ``days``, ``pentads``, ``weeks``, ``months``, ``seasons`` for lead
  time resolution. ``climpred`` now requires a ``lead`` attribute "units" to decipher
  what resolution the predictions are at. (:pr:`294`) `Kathy Pegion`_ and
  `Riley X. Brady`_.

.. code-block:: python

        >>> hind = climpred.tutorial.load_dataset('CESM-DP-SST')
        >>> hind.lead.attrs['units'] = 'years'

- ``HindcastEnsemble`` now has ``.add_observations()`` and ``.get_observations()``
  methods. These are the same as ``.add_reference()`` and ``.get_reference()``, which
  will be deprecated eventually. The name change clears up confusion, since "reference"
  is the appropriate name for a reference forecast, e.g. persistence. (:pr:`310`)
  `Riley X. Brady`_.

- ``HindcastEnsemble`` now has ``.verify()`` function, which duplicates the
  ``.compute_metric()`` function. We feel that ``.verify()`` is more clear and easy
  to write, and follows the terminology of the field. (:pr:`310`) `Riley X. Brady`_.

- ``e2o`` and ``m2o`` are now the preferred keywords for comparing hindcast ensemble
  means and ensemble members to verification data, respectively. (:pr:`310`)
  `Riley X. Brady`_.

Documentation
-------------
- New example pages for subseasonal-to-seasonal prediction using ``climpred``.
  (:pr:`294`) `Kathy Pegion`_

    * Calculate the skill of the MJO index as a function of lead time
      (`link <examples/subseasonal/daily-subx-example.html>`__).

    * Calculate the skill of the MJO index as a function of lead time for weekly data
      (`link <examples/subseasonal/weekly-subx-example.html>`__).

    * Calculate ENSO skill as a function of initial month vs. lead time
      (`link <examples/monseas/monthly-enso-subx-example.html>`__).

    * Calculate Seasonal ENSO skill
      (`link <examples/monseas/seasonal-enso-subx-example.html>`__).

- `Comparisons <comparisons.html>`__ page rewritten for more clarity. (:pr:`310`)
  `Riley X. Brady`_.

Bug Fixes
---------
- Fixed `m2m` broken comparison issue and removed correction (:pr:`290`) `Aaron Spring`_.

Internals/Minor Fixes
---------------------
- Updates to ``xskillscore`` v0.0.12 to get a 30-50% speedup in compute functions that
  rely on metrics from there. (:pr:`309`) `Riley X. Brady`_.
- Stacking dims is handled by ``comparisons``, no need for internal keyword
  ``stack_dims``. Therefore ``comparison`` now takes ``metric`` as argument instead.
  (:pr:`290`) `Aaron Spring`_.
- ``assign_attrs`` now carries `dim` (:pr:`290`) `Aaron Spring`_.
- ``reference`` changed to ``verif`` throughout hindcast compute functions. This is more
  clear, since ``reference`` usually refers to a type of forecast, such as persistence.
  (:pr:`310`) `Riley X. Brady`_.
- ``Comparison`` objects can now have aliases. (:pr:`310`) `Riley X. Brady`_.



climpred v1.2.1 (2020-01-07)
============================

Depreciated
-----------
- ``mad`` no longer a keyword for the median absolute error metric. Users should now
  use ``median_absolute_error``, which is identical to changes in ``xskillscore``
  version 0.0.10. (:pr:`283`) `Riley X. Brady`_
- ``pacc`` no longer a keyword for the p value associated with the Pearson
  product-moment correlation, since it is used by the correlation coefficient.
  (:pr:`283`) `Riley X. Brady`_
- ``msss`` no longer a keyword for the Murphy's MSSS, since it is reserved for the
  standard MSSS. (:pr:`283`) `Riley X. Brady`_

New Features
------------
- Metrics ``pearson_r_eff_p_value`` and ``spearman_r_eff_p_value`` account for
  autocorrelation in computing p values. (:pr:`283`) `Riley X. Brady`_
- Metric ``effective_sample_size`` computes number of independent samples between two
  time series being correlated. (:pr:`283`) `Riley X. Brady`_
- Added keywords for metrics: (:pr:`283`) `Riley X. Brady`_

    * ``'pval'`` for ``pearson_r_p_value``
    * ``['n_eff', 'eff_n']`` for ``effective_sample_size``
    * ``['p_pval_eff', 'pvalue_eff', 'pval_eff']`` for ``pearson_r_eff_p_value``
    * ``['spvalue', 'spval']`` for ``spearman_r_p_value``
    * ``['s_pval_eff', 'spvalue_eff', 'spval_eff']`` for ``spearman_r_eff_p_value``
    * ``'nev'`` for ``nmse``

Internals/Minor Fixes
---------------------
- ``climpred`` now requires ``xarray`` version 0.14.1 so that the ``drop_vars()``
  keyword used in our package does not throw an error. (:pr:`276`) `Riley X. Brady`_
- Update to ``xskillscore`` version 0.0.10 to fix errors in weighted metrics with
  pairwise NaNs. (:pr:`283`) `Riley X. Brady`_
- ``doc8`` added to ``pre-commit`` to have consistent formatting on ``.rst`` files.
  (:pr:`283`) `Riley X. Brady`_
- Remove ``proper`` attribute on ``Metric`` class since it isn't used anywhere.
  (:pr:`283`) `Riley X. Brady`_
- Add testing for effective p values. (:pr:`283`) `Riley X. Brady`_
- Add testing for whether metric aliases are repeated/overwrite each other.
  (:pr:`283`) `Riley X. Brady`_
- ``ppp`` changed to ``msess``, but keywords allow for ``ppp`` and ``msss`` still.
  (:pr:`283`) `Riley X. Brady`_

Documentation
-------------
- Expansion of `metrics documentation <metrics.html>`_ with much more
  detail on how metrics are computed, their keywords, references, min/max/perfect
  scores, etc. (:pr:`283`) `Riley X. Brady`_
- Update `terminology page <terminology.html>`_ with more information on metrics
  terminology. (:pr:`283`) `Riley X. Brady`_


climpred v1.2.0 (2019-12-17)
============================

Depreciated
-----------
- Abbreviation ``pval`` depreciated. Use ``p_pval`` for ``pearson_r_p_value`` instead.
  (:pr:`264`) `Aaron Spring`_.

New Features
------------
- Users can now pass a custom ``metric`` or ``comparison`` to compute functions.
  (:pr:`268`) `Aaron Spring`_.

    * See `user-defined-metrics <metrics.html#user-defined-metrics>`_ and
      `user-defined-comparisons <comparisons.html#user-defined-comparisons>`_.

- New deterministic metrics (see `metrics <metrics.html>`_). (:pr:`264`)
  `Aaron Spring`_.

    * Spearman ranked correlation (spearman_r_)
    * Spearman ranked correlation p-value (spearman_r_p_value_)
    * Mean Absolute Deviation (mad_)
    * Mean Absolute Percent Error (mape_)
    * Symmetric Mean Absolute Percent Error (smape_)

.. _spearman_r: metrics.html#spearman-anomaly-correlation-coefficient-sacc
.. _spearman_r_p_value: metrics.html#spearman-anomaly-correlation-coefficient-sacc
.. _mad: metrics.html#median-absolute-deviation-mad
.. _mape: metrics.html#mean-absolute-percentage-error-mape
.. _smape: metrics.html#symmetric-mean-absolute-percentage-error-smape

- Users can now apply arbitrary ``xarray`` methods to
  :py:class:`~climpred.classes.HindcastEnsemble` and
  :py:class:`~climpred.classes.PerfectModelEnsemble`. (:pr:`243`) `Riley X. Brady`_.

    * See the
      `Prediction Ensemble objects demo page <prediction-ensemble-object.html>`_.

- Add "getter" methods to :py:class:`~climpred.classes.HindcastEnsemble` and
  :py:class:`~climpred.classes.PerfectModelEnsemble` to retrieve ``xarray`` datasets
  from the objects. (:pr:`243`) `Riley X. Brady`_.

    .. code-block:: python

        >>> hind = climpred.tutorial.load_dataset('CESM-DP-SST')
        >>> ref = climpred.tutorial.load_dataset('ERSST')
        >>> hindcast = climpred.HindcastEnsemble(hind)
        >>> hindcast = hindcast.add_reference(ref, 'ERSST')
        >>> print(hindcast)
        <climpred.HindcastEnsemble>
        Initialized Ensemble:
            SST      (init, lead, member) float64 ...
        ERSST:
            SST      (time) float32 ...
        Uninitialized:
            None
        >>> print(hindcast.get_initialized())
        <xarray.Dataset>
        Dimensions:  (init: 64, lead: 10, member: 10)
        Coordinates:
        * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
        * member   (member) int32 1 2 3 4 5 6 7 8 9 10
        * init     (init) float32 1954.0 1955.0 1956.0 1957.0 ... 2015.0 2016.0 2017.0
        Data variables:
            SST      (init, lead, member) float64 ...
        >>> print(hindcast.get_reference('ERSST'))
        <xarray.Dataset>
        Dimensions:  (time: 61)
        Coordinates:
        * time     (time) int64 1955 1956 1957 1958 1959 ... 2011 2012 2013 2014 2015
        Data variables:
            SST      (time) float32 ...

- ``metric_kwargs`` can be passed to :py:class:`~climpred.metrics.Metric`.
  (:pr:`264`) `Aaron Spring`_.

    * See ``metric_kwargs`` under `metrics <metrics.html>`_.

Bug Fixes
---------
- :py:meth:`~climpred.classes.HindcastEnsemble.compute_metric` doesn't drop coordinates
  from the initialized hindcast ensemble anymore. (:pr:`258`) `Aaron Spring`_.
- Metric ``uacc`` does not crash when ``ppp`` negative anymore. (:pr:`264`)
  `Aaron Spring`_.
- Update ``xskillscore`` to version 0.0.9 to fix all-NaN issue with ``pearson_r`` and
  ``pearson_r_p_value`` when there's missing data. (:pr:`269`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Rewrote :py:func:`~climpred.stats.varweighted_mean_period` based on ``xrft``.
  Changed ``time_dim`` to ``dim``. Function no longer drops coordinates. (:pr:`258`)
  `Aaron Spring`_
- Add ``dim='time'`` in :py:func:`~climpred.stats.dpp`. (:pr:`258`) `Aaron Spring`_
- Comparisons ``m2m``, ``m2e`` rewritten to not stack dims into supervector because
  this is now done in ``xskillscore``. (:pr:`264`) `Aaron Spring`_
- Add ``tqdm`` progress bar to :py:func:`~climpred.bootstrap.bootstrap_compute`.
  (:pr:`244`) `Aaron Spring`_
- Remove inplace behavior for :py:class:`~climpred.classes.HindcastEnsemble` and
  :py:class:`~climpred.classes.PerfectModelEnsemble`. (:pr:`243`) `Riley X. Brady`_

    * See `demo page on prediction ensemble objects <prediction-ensemble-object.html>`_

- Added tests for chunking with ``dask``. (:pr:`258`) `Aaron Spring`_
- Fix test issues with esmpy 8.0 by forcing esmpy 7.1 (:pr:`269`). `Riley X. Brady`_
- Rewrote ``metrics`` and ``comparisons`` as classes to accomodate custom metrics and
  comparisons. (:pr:`268`) `Aaron Spring`_

    * See `user-defined-metrics <metrics.html#user-defined-metrics>`_ and
      `user-defined-comparisons <comparisons.html#user-defined-comparisons>`_.

Documentation
-------------
- Add examples notebook for
  `temporal and spatial smoothing <examples/smoothing.html>`_. (:pr:`244`)
  `Aaron Spring`_
- Add documentation for computing a metric over a
  `specified dimension <comparisons.html#compute-over-dimension>`_.
  (:pr:`244`) `Aaron Spring`_
- Update `API <api.html>`_ to be more organized with individual function/class pages.
  (:pr:`243`) `Riley X. Brady`_.
- Add `page <prediction-ensemble-object.html>`_ describing the
  :py:class:`~climpred.classes.HindcastEnsemble` and
  :py:class:`~climpred.classes.PerfectModelEnsemble` objects more clearly.
  (:pr:`243`) `Riley X. Brady`_
- Add page for `publications <publications.html>`_ and
  `helpful links <helpful-links.html>`_. (:pr:`270`) `Riley X. Brady`_.

climpred v1.1.0 (2019-09-23)
============================

Features
--------
- Write information about skill computation to netcdf attributes(:pr:`213`)
  `Aaron Spring`_
- Temporal and spatial smoothing module (:pr:`224`) `Aaron Spring`_
- Add metrics `brier_score`, `threshold_brier_score` and `crpss_es` (:pr:`232`)
  `Aaron Spring`_
- Allow `compute_hindcast` and `compute_perfect_model` to specify which dimension `dim`
  to calculate metric over (:pr:`232`) `Aaron Spring`_

Bug Fixes
---------
- Correct implementation of probabilistic metrics from `xskillscore` in
  `compute_perfect_model`, `bootstrap_perfect_model`, `compute_hindcast` and
  `bootstrap_hindcast`, now requires xskillscore>=0.05 (:pr:`232`) `Aaron Spring`_

Internals/Minor Fixes
---------------------
- Rename .stats.DPP to dpp (:pr:`232`) `Aaron Spring`_
- Add `matplotlib` as a main dependency so that a direct pip installation works
  (:pr:`211`) `Riley X. Brady`_.
- ``climpred`` is now installable from conda-forge (:pr:`212`) `Riley X. Brady`_.
- Fix erroneous descriptions of sample datasets (:pr:`226`) `Riley X. Brady`_.
- Benchmarking time and peak memory of compute functions with `asv` (:pr:`231`)
  `Aaron Spring`_

Documentation
-------------
- Add scope of package to docs for clarity for users and developers. (:pr:`235`)
  `Riley X. Brady`_.

climpred v1.0.1 (2019-07-04)
============================

Bug Fixes
---------
- Accomodate for lead-zero within the ``lead`` dimension (:pr:`196`) `Riley X. Brady`_.
- Fix issue with adding uninitialized ensemble to ``HindcastEnsemble`` object
  (:pr:`199`) `Riley X. Brady`_.
- Allow ``max_dof`` keyword to be passed to ``compute_metric`` and
  ``compute_persistence`` for ``HindcastEnsemble`` (:pr:`199`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Force ``xskillscore`` version 0.0.4 or higher to avoid ``ImportError``
  (:pr:`204`) `Riley X. Brady`_.
- Change ``max_dfs`` keyword to ``max_dof`` (:pr:`199`) `Riley X. Brady`_.
- Add testing for ``HindcastEnsemble`` and ``PerfectModelEnsemble`` (:pr:`199`)
  `Riley X. Brady`_

climpred v1.0.0 (2019-07-03)
============================
``climpred`` v1.0.0 represents the first stable release of the package. It includes
``HindcastEnsemble`` and ``PerfectModelEnsemble`` objects to perform analysis with.
It offers a suite of deterministic and probabilistic metrics that are optimized to be
run on single time series or grids of data (e.g., lat, lon, and depth). Currently,
``climpred`` only supports annual forecasts.

Features
--------
- Bootstrap prediction skill based on resampling with replacement consistently in
  ``ReferenceEnsemble`` and ``PerfectModelEnsemble``. (:pr:`128`) `Aaron Spring`_
- Consistent bootstrap function for ``climpred.stats`` functions via ``bootstrap_func``
  wrapper. (:pr:`167`) `Aaron Spring`_
- many more metrics: ``_msss_murphy``, ``_less`` and probabilistic ``_crps``,
  ``_crpss`` (:pr:`128`) `Aaron Spring`_

Bug Fixes
---------
- ``compute_uninitialized`` now trims input data to the same time window.
  (:pr:`193`) `Riley X. Brady`_
- ``rm_poly`` now properly interpolates/fills NaNs. (:pr:`192`) `Riley X. Brady`_

Internals/Minor Fixes
---------------------
- The ``climpred`` version can be printed. (:pr:`195`) `Riley X. Brady`_
- Constants are made elegant and pushed to a separate module. (:pr:`184`)
  `Andrew Huang`_
- Checks are consolidated to their own module. (:pr:`173`) `Andrew Huang`_

Documentation
-------------
- Documentation built extensively in multiple PRs.


climpred v0.3 (2019-04-27)
==========================

``climpred`` v0.3 really represents the entire development phase leading up to the
version 1 release. This was done in collaboration between `Riley X. Brady`_,
`Aaron Spring`_, and `Andrew Huang`_. Future releases will have less additions.

Features
--------
- Introduces object-oriented system to ``climpred``, with classes
  ``ReferenceEnsemble`` and ``PerfectModelEnsemble``. (:pr:`86`) `Riley X. Brady`_
- Expands bootstrapping module for perfect-module configurations. (:pr:`78`, :pr:`87`)
  `Aaron Spring`_
- Adds functions for computing Relative Entropy (:pr:`73`) `Aaron Spring`_
- Sets more intelligible dimension expectations for ``climpred``
  (:pr:`98`, :pr:`105`) `Riley X. Brady`_ and `Aaron Spring`_:

    -   ``init``:  initialization dates for the prediction ensemble
    -   ``lead``:  retrospective forecasts from prediction ensemble;
        returned dimension for prediction calculations
    -   ``time``:  time dimension for control runs, references, etc.
    -   ``member``:  ensemble member dimension.
- Updates ``open_dataset`` to display available dataset names when no argument is
  passed. (:pr:`123`) `Riley X. Brady`_
- Change ``ReferenceEnsemble`` to ``HindcastEnsemble``. (:pr:`124`) `Riley X. Brady`_
- Add probabilistic metrics to ``climpred``. (:pr:`128`) `Aaron Spring`_
- Consolidate separate perfect-model and hindcast functions into singular functions
  (:pr:`128`) `Aaron Spring`_
- Add option to pass proxy through to ``open_dataset`` for firewalled networks.
  (:pr:`138`) `Riley X. Brady`_

Bug Fixes
---------
- ``xr_rm_poly`` can now operate on Datasets and with multiple variables.
  It also interpolates across NaNs in time series. (:pr:`94`) `Andrew Huang`_
- Travis CI, ``treon``, and ``pytest`` all run for automated testing of new features.
  (:pr:`98`, :pr:`105`, :pr:`106`) `Riley X. Brady`_ and `Aaron Spring`_
- Clean up ``check_xarray`` decorators and make sure that they work. (:pr:`142`)
  `Andrew Huang`_
- Ensures that ``help()`` returns proper docstring even with decorators.
  (:pr:`149`) `Andrew Huang`_
- Fixes bootstrap so p values are correct. (:pr:`170`) `Aaron Spring`_

Internals/Minor Fixes
---------------------
- Adds unit testing for all perfect-model comparisons. (:pr:`107`) `Aaron Spring`_
- Updates CESM-LE uninitialized ensemble sample data to have 34 members.
  (:pr:`113`) `Riley X. Brady`_
- Adds MPI-ESM hindcast, historical, and assimilation sample data.
  (:pr:`119`) `Aaron Spring`_
- Replaces ``check_xarray`` with a decorator for checking that input arguments are
  xarray objects. (:pr:`120`) `Andrew Huang`_
- Add custom exceptions for clearer error reporting. (:pr:`139`) `Riley X. Brady`_
- Remove "xr" prefix from stats module. (:pr:`144`) `Riley X. Brady`_
- Add codecoverage for testing. (:pr:`152`) `Riley X. Brady`_
- Update exception messages for more pretty error reporting. (:pr:`156`) `Andrew Huang`_
- Add ``pre-commit`` and ``flake8``/``black`` check in CI. (:pr:`163`) `Riley X. Brady`_
- Change ``loadutils`` module to ``tutorial`` and ``open_dataset`` to
  ``load_dataset``. (:pr:`164`) `Riley X. Brady`_
- Remove predictability horizon function to revisit for v2. (:pr:`165`)
  `Riley X. Brady`_
- Increase code coverage through more testing. (:pr:`167`) `Aaron Spring`_
- Consolidates checks and constants into modules. (:pr:`173`) `Andrew Huang`_

climpred v0.2 (2019-01-11)
==========================

Name changed to ``climpred``, developed enough for basic decadal prediction tasks on a
perfect-model ensemble and reference-based ensemble.

climpred v0.1 (2018-12-20)
==========================

Collaboration between Riley Brady and Aaron Spring begins.

.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Andrew Huang`: https://github.com/ahuang11
.. _`Kathy Pegion`: https://github.com/kpegion
.. _`Aaron Spring`: https://github.com/aaronspring
.. _`Dougie Squire`: https://github.com/dougiesquire
