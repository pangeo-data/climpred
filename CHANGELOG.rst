==========
What's New
==========

climpred v1.2.0 (2019-12-17)
============================

Depreciated
-----------
- Abbreviation ``pval`` depreciated. Use ``p_pval`` for ``pearson_r_p_value`` instead. (:pr:`264`) `Aaron Spring`_.

New Features
------------

- Users can now pass a custom ``metric`` or ``comparison`` to compute functions. (:pr:`268`) `Aaron Spring`_.

    * See `user-defined-metrics <metrics.html#user-defined-metrics>`_ and `user-defined-comparisons <comparisons.html#user-defined-comparisons>`_.

- New deterministic metrics (see `metrics <metrics.html>`_). (:pr:`264`) `Aaron Spring`_.

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

- Users can now apply arbitrary ``xarray`` methods to :py:class:`~climpred.classes.HindcastEnsemble` and :py:class:`~climpred.classes.PerfectModelEnsemble`. (:pr:`243`) `Riley X. Brady`_.

    * See the `Prediction Ensemble objects demo page <prediction-ensemble-object.html>`_.

- Add "getter" methods to :py:class:`~climpred.classes.HindcastEnsemble` and :py:class:`~climpred.classes.PerfectModelEnsemble` to retrieve ``xarray`` datasets from the objects. (:pr:`243`) `Riley X. Brady`_.

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

- ``metric_kwargs`` can be passed to :py:class:`~climpred.metrics.Metric`. (:pr:`264`) `Aaron Spring`_.

    * See ``metric_kwargs`` under `metrics <metrics.html>`_.

Bug Fixes
---------
- :py:meth:`~climpred.classes.HindcastEnsemble.compute_metric` doesn't drop coordinates from the initialized hindcast ensemble anymore. (:pr:`258`) `Aaron Spring`_.
- Metric ``uacc`` does not crash when ``ppp`` negative anymore. (:pr:`264`) `Aaron Spring`_.
- Update ``xskillscore`` to version 0.0.9 to fix all-NaN issue with ``pearson_r`` and ``pearson_r_p_value`` when there's missing data. (:pr:`269`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Rewrote :py:func:`~climpred.stats.varweighted_mean_period` based on ``xrft``. Changed ``time_dim`` to ``dim``. Function no longer drops coordinates. (:pr:`258`) `Aaron Spring`_
- Add ``dim='time'`` in :py:func:`~climpred.stats.dpp`. (:pr:`258`) `Aaron Spring`_
- Comparisons ``m2m``, ``m2e`` rewritten to not stack dims into supervector because this is now done in ``xskillscore``. (:pr:`264`) `Aaron Spring`_
- Add ``tqdm`` progress bar to :py:func:`~climpred.bootstrap.bootstrap_compute`. (:pr:`244`) `Aaron Spring`_
- Remove inplace behavior for :py:class:`~climpred.classes.HindcastEnsemble` and :py:class:`~climpred.classes.PerfectModelEnsemble`. (:pr:`243`) `Riley X. Brady`_

    * See `demo page on prediction ensemble objects <prediction-ensemble-object.html>`_

- Added tests for chunking with ``dask``. (:pr:`258`) `Aaron Spring`_
- Fix test issues with esmpy 8.0 by forcing esmpy 7.1 (:pr:`269`). `Riley X. Brady`_
- Rewrote ``metrics`` and ``comparisons`` as classes to accomodate custom metrics and comparisons. (:pr:`268`) `Aaron Spring`_

    * See `user-defined-metrics <metrics.html#user-defined-metrics>`_ and `user-defined-comparisons <comparisons.html#user-defined-comparisons>`_.

Documentation
-------------
- Add examples notebook for `temporal and spatial smoothing <examples/smoothing.html>`_. (:pr:`244`) `Aaron Spring`_
- Add documentation for `computing a metric over a specified dimension <comparisons.html#compute-over-dimension>`_. (:pr:`244`) `Aaron Spring`_
- Update `API <api.html>`_ to be more organized with individual function/class pages. (:pr:`243`) `Riley X. Brady`_.
- Add `page <prediction-ensemble-object.html>`_ describing the :py:class:`~climpred.classes.HindcastEnsemble` and :py:class:`~climpred.classes.PerfectModelEnsemble` objects more clearly. (:pr:`243`) `Riley X. Brady`_
- Add page for `publications <publications.html>`_ and `helpful links <helpful-links.html>`_. (:pr:`270`) `Riley X. Brady`_.

climpred v1.1.0 (2019-09-23)
============================

Features
--------
- Write information about skill computation to netcdf attributes(:pr:`213`) `Aaron Spring`_
- Temporal and spatial smoothing module (:pr:`224`) `Aaron Spring`_
- Add metrics `brier_score`, `threshold_brier_score` and `crpss_es` (:pr:`232`) `Aaron Spring`_
- Allow `compute_hindcast` and `compute_perfect_model` to specify which dimension `dim` to calculate metric over (:pr:`232`) `Aaron Spring`_

Bug Fixes
---------
- Correct implementation of probabilistic metrics from `xskillscore` in `compute_perfect_model`, `bootstrap_perfect_model`, `compute_hindcast` and `bootstrap_hindcast`, now requires xskillscore>=0.05 (:pr:`232`) `Aaron Spring`_

Internals/Minor Fixes
---------------------
- Rename .stats.DPP to dpp (:pr:`232`) `Aaron Spring`_
- Add `matplotlib` as a main dependency so that a direct pip installation works (:pr:`211`) `Riley X. Brady`_.
- ``climpred`` is now installable from conda-forge (:pr:`212`) `Riley X. Brady`_.
- Fix erroneous descriptions of sample datasets (:pr:`226`) `Riley X. Brady`_.
- Benchmarking time and peak memory of compute functions with `asv` (:pr:`231`) `Aaron Spring`_

Documentation
-------------
- Add scope of package to docs for clarity for users and developers. (:pr:`235`) `Riley X. Brady`_.

climpred v1.0.1 (2019-07-04)
============================

Bug Fixes
---------
- Accomodate for lead-zero within the ``lead`` dimension (:pr:`196`) `Riley X. Brady`_.
- Fix issue with adding uninitialized ensemble to ``HindcastEnsemble`` object (:pr:`199`) `Riley X. Brady`_.
- Allow ``max_dof`` keyword to be passed to ``compute_metric`` and ``compute_persistence`` for ``HindcastEnsemble`` (:pr:`199`) `Riley X. Brady`_.

Internals/Minor Fixes
---------------------
- Force ``xskillscore`` version 0.0.4 or higher to avoid ``ImportError`` (:pr:`204`) `Riley X. Brady`_.
- Change ``max_dfs`` keyword to ``max_dof`` (:pr:`199`) `Riley X. Brady`_.
- Add testing for ``HindcastEnsemble`` and ``PerfectModelEnsemble`` (:pr:`199`) `Riley X. Brady`_

climpred v1.0.0 (2019-07-03)
============================
``climpred`` v1.0.0 represents the first stable release of the package. It includes ``HindcastEnsemble`` and ``PerfectModelEnsemble`` objects to perform analysis with. It offers a suite of deterministic and probabilistic metrics that are optimized to be run on single time series or grids of data (e.g., lat, lon, and depth). Currently, ``climpred`` only supports annual forecasts.

Features
--------
- Bootstrap prediction skill based on resampling with replacement consistently in ``ReferenceEnsemble`` and ``PerfectModelEnsemble``. (:pr:`128`) `Aaron Spring`_
- Consistent bootstrap function for ``climpred.stats`` functions via ``bootstrap_func`` wrapper. (:pr:`167`) `Aaron Spring`_
- many more metrics: ``_msss_murphy``, ``_less`` and probabilistic ``_crps``, ``_crpss`` (:pr:`128`) `Aaron Spring`_

Bug Fixes
---------
- ``compute_uninitialized`` now trims input data to the same time window. (:pr:`193`) `Riley X. Brady`_
- ``rm_poly`` now properly interpolates/fills NaNs. (:pr:`192`) `Riley X. Brady`_

Internals/Minor Fixes
---------------------
- The ``climpred`` version can be printed. (:pr:`195`) `Riley X. Brady`_
- Constants are made elegant and pushed to a separate module. (:pr:`184`) `Andrew Huang`_
- Checks are consolidated to their own module. (:pr:`173`) `Andrew Huang`_

Documentation
-------------
- Documentation built extensively in multiple PRs.


climpred v0.3 (2019-04-27)
==========================

``climpred`` v0.3 really represents the entire development phase leading up to the version 1 release. This was done in collaboration between `Riley X. Brady`_, `Aaron Spring`_, and `Andrew Huang`_. Future releases will have less additions.

Features
--------
- Introduces object-oriented system to ``climpred``, with classes ``ReferenceEnsemble`` and ``PerfectModelEnsemble``. (:pr:`86`) `Riley X. Brady`_
- Expands bootstrapping module for perfect-module configurations. (:pr:`78`, :pr:`87`) `Aaron Spring`_
- Adds functions for computing Relative Entropy (:pr:`73`) `Aaron Spring`_
- Sets more intelligible dimension expectations for ``climpred`` (:pr:`98`, :pr:`105`) `Riley X. Brady`_ and `Aaron Spring`_:

    -   ``init``:  initialization dates for the prediction ensemble
    -   ``lead``:  retrospective forecasts from prediction ensemble; returned dimension for prediction calculations
    -   ``time``:  time dimension for control runs, references, etc.
    -   ``member``:  ensemble member dimension.
- Updates ``open_dataset`` to display available dataset names when no argument is passed. (:pr:`123`) `Riley X. Brady`_
- Change ``ReferenceEnsemble`` to ``HindcastEnsemble``. (:pr:`124`) `Riley X. Brady`_
- Add probabilistic metrics to ``climpred``. (:pr:`128`) `Aaron Spring`_
- Consolidate separate perfect-model and hindcast functions into singular functions. (:pr:`128`) `Aaron Spring`_
- Add option to pass proxy through to ``open_dataset`` for firewalled networks. (:pr:`138`) `Riley X. Brady`_


Bug Fixes
---------
- ``xr_rm_poly`` can now operate on Datasets and with multiple variables. It also interpolates across NaNs in time series. (:pr:`94`) `Andrew Huang`_
- Travis CI, ``treon``, and ``pytest`` all run for automated testing of new features. (:pr:`98`, :pr:`105`, :pr:`106`) `Riley X. Brady`_ and `Aaron Spring`_
- Clean up ``check_xarray`` decorators and make sure that they work. (:pr:`142`) `Andrew Huang`_
- Ensures that ``help()`` returns proper docstring even with decorators. (:pr:`149`) `Andrew Huang`_
- Fixes bootstrap so p values are correct. (:pr:`170`) `Aaron Spring`_

Internals/Minor Fixes
---------------------
- Adds unit testing for all perfect-model comparisons. (:pr:`107`) `Aaron Spring`_
- Updates CESM-LE uninitialized ensemble sample data to have 34 members. (:pr:`113`) `Riley X. Brady`_
- Adds MPI-ESM hindcast, historical, and assimilation sample data. (:pr:`119`) `Aaron Spring`_
- Replaces ``check_xarray`` with a decorator for checking that input arguments are xarray objects. (:pr:`120`) `Andrew Huang`_
- Add custom exceptions for clearer error reporting. (:pr:`139`) `Riley X. Brady`_
- Remove "xr" prefix from stats module. (:pr:`144`) `Riley X. Brady`_
- Add codecoverage for testing. (:pr:`152`) `Riley X. Brady`_
- Update exception messages for more pretty error reporting. (:pr:`156`) `Andrew Huang`_
- Add ``pre-commit`` and ``flake8``/``black`` check in CI. (:pr:`163`) `Riley X. Brady`_
- Change ``loadutils`` module to ``tutorial`` and ``open_dataset`` to ``load_dataset``. (:pr:`164`) `Riley X. Brady`_
- Remove predictability horizon function to revisit for v2. (:pr:`165`) `Riley X. Brady`_
- Increase code coverage through more testing. (:pr:`167`) `Aaron Spring`_
- Consolidates checks and constants into modules. (:pr:`173`) `Andrew Huang`_

climpred v0.2 (2019-01-11)
==========================

Name changed to ``climpred``, developed enough for basic decadal prediction tasks on a perfect-model ensemble and reference-based ensemble.

climpred v0.1 (2018-12-20)
==========================

Collaboration between Riley Brady and Aaron Spring begins.

.. _`Riley X. Brady`: https://github.com/bradyrx
.. _`Aaron Spring`: https://github.com/aaronspring
.. _`Andrew Huang`: https://github.com/ahuang11
