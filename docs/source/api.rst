API Reference
=============

This page provides an auto-generated summary of climpred's API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

High-Level Classes
------------------

.. currentmodule:: climpred.classes


A primary feature of ``climpred`` is our prediction ensemble objects,
:py:class:`~climpred.classes.HindcastEnsemble` and
:py:class:`~climpred.classes.PerfectModelEnsemble`. Users can append their initialized
ensemble to these classes, as well as an arbitrary number of verification products (assimilations,
reconstructions, observations), control runs, and uninitialized ensembles.

HindcastEnsemble
~~~~~~~~~~~~~~~~

A ``HindcastEnsemble`` is a prediction ensemble that is initialized off of some form of
observations (an assimilation, renanalysis, etc.). Thus, it is anticipated that forecasts are
verified against observation-like products. Read more about the terminology
`here <terminology.html>`_.

.. autosummary::
    :toctree: api/

    HindcastEnsemble

-------------------------
Add and Retrieve Datasets
-------------------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.__init__
    HindcastEnsemble.add_observations
    HindcastEnsemble.add_uninitialized
    HindcastEnsemble.get_initialized
    HindcastEnsemble.get_observations
    HindcastEnsemble.get_uninitialized

------------------
Analysis Functions
------------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.verify
    HindcastEnsemble.bootstrap

--------------
Pre-Processing
--------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.smooth
    HindcastEnsemble.remove_bias

-------------
Visualization
-------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.plot


PerfectModelEnsemble
~~~~~~~~~~~~~~~~~~~~

A ``PerfectModelEnsemble`` is a prediction ensemble that is initialized off of a control simulation
for a number of randomly chosen initialization dates. Thus, forecasts cannot be verified against
real-world observations. Instead, they are `compared <comparisons.html>`_ to one another and to the
original control run. Read more about the terminology `here <terminology.html>`_.

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble

-------------------------
Add and Retrieve Datasets
-------------------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.__init__
    PerfectModelEnsemble.add_control
    PerfectModelEnsemble.get_initialized
    PerfectModelEnsemble.get_control
    PerfectModelEnsemble.get_uninitialized

------------------
Analysis Functions
------------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.verify
    PerfectModelEnsemble.bootstrap

-------------
Generate Data
-------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.generate_uninitialized

--------------
Pre-Processing
--------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.smooth

-------------
Visualization
-------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.plot


Direct Function Calls
---------------------

A user can directly call functions in ``climpred``. This requires entering more arguments, e.g.
the initialized ensemble
:py:class:`~xarray.core.dataset.Dataset`/:py:class:`xarray.core.dataarray.DataArray` directly as
well as a verification product. Our object
:py:class:`~climpred.classes.HindcastEnsemble` and
:py:class:`~climpred.classes.PerfectModelEnsemble` wrap most of these functions, making the
analysis process much simpler. Once we have wrapped all of the functions in their entirety, we will
likely deprecate the ability to call them directly.

Bootstrap
~~~~~~~~~
.. currentmodule:: climpred.bootstrap

.. autosummary::
    :toctree: api/

    bootstrap_compute
    bootstrap_hindcast
    bootstrap_perfect_model
    bootstrap_uninit_pm_ensemble_from_control_cftime
    bootstrap_uninitialized_ensemble
    dpp_threshold
    varweighted_mean_period_threshold

Prediction
~~~~~~~~~~
.. currentmodule:: climpred.prediction

.. autosummary::
    :toctree: api/

    compute_hindcast
    compute_perfect_model


Reference
~~~~~~~~~
.. currentmodule:: climpred.reference

.. autosummary::
    :toctree: api/

    compute_persistence
    compute_uninitialized

Horizon
~~~~~~~
.. currentmodule:: climpred.horizon

.. autosummary::
    :toctree: api/

    horizon

Statistics
~~~~~~~~~~
.. currentmodule:: climpred.stats

.. autosummary::
    :toctree: api/

    decorrelation_time
    dpp
    varweighted_mean_period

Tutorial
~~~~~~~~
.. currentmodule:: climpred.tutorial

.. autosummary::
    :toctree: api/

    load_dataset

Preprocessing
~~~~~~~~~~~~~
.. currentmodule:: climpred.preprocessing.shared

.. autosummary::
    :toctree: api/

    load_hindcast
    rename_to_climpred_dims
    rename_SLM_to_climpred_dims

.. currentmodule:: climpred.preprocessing.mpi

.. autosummary::
    :toctree: api/

    get_path

Smoothing
~~~~~~~~~
.. currentmodule:: climpred.smoothing

.. autosummary::
    :toctree: api/

    temporal_smoothing
    spatial_smoothing_xesmf

Metrics
-------

For a thorough look at our metrics library, please see the
`metrics </metrics.html>`_ page.

.. currentmodule:: climpred.metrics

.. autosummary::
    :toctree: api/

    Metric
    _get_norm_factor
    _pearson_r
    _pearson_r_p_value
    _effective_sample_size
    _pearson_r_eff_p_value
    _spearman_r
    _spearman_r_p_value
    _spearman_r_eff_p_value
    _mse
    _rmse
    _mae
    _median_absolute_error
    _nmse
    _nmae
    _nrmse
    _msess
    _mape
    _smape
    _uacc
    _std_ratio
    _conditional_bias
    _unconditional_bias
    _bias_slope
    _msess_murphy
    _crps
    _crpss
    _crpss_es
    _brier_score
    _threshold_brier_score
    _rps
    _discrimination
    _reliability
    _rank_histogram
    _contingency

Comparisons
-----------

For a thorough look at our metrics library, please see the
`comparisons </comparisons.html>`_ page.

.. currentmodule:: climpred.comparisons

.. autosummary::
    :toctree: api/

    Comparison
    _e2o
    _m2o
    _m2m
    _m2e
    _m2c
    _e2c
