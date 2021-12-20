API Reference
=============

This page provides an auto-generated summary of ``climpred``'s API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

High-Level Classes
------------------

.. currentmodule:: climpred.classes


A primary feature of ``climpred`` is our prediction ensemble objects,
:py:class:`.HindcastEnsemble` and :py:class:`.PerfectModelEnsemble`.
Users can add their initialized ensemble to these classes, as well as verification
products (assimilations, reconstructions, observations), control runs, and uninitialized
ensembles.

PredictionEnsemble
~~~~~~~~~~~~~~~~~~

:py:class:`.PredictionEnsemble` is the base class for :py:class:`.HindcastEnsemble` and
:py:class:`.PerfectModelEnsemble`. :py:class:`.PredictionEnsemble` cannot be called
directly, but :py:class:`.HindcastEnsemble` and :py:class:`.PerfectModelEnsemble`
inherit the common base functionality.

.. autosummary::
    :toctree: api/

    PredictionEnsemble
    PredictionEnsemble.__init__

-------
Builtin
-------

.. autosummary::
    :toctree: api/

    PredictionEnsemble.__len__
    PredictionEnsemble.__iter__
    PredictionEnsemble.__delitem__
    PredictionEnsemble.__contains__
    PredictionEnsemble.__add__
    PredictionEnsemble.__sub__
    PredictionEnsemble.__mul__
    PredictionEnsemble.__truediv__
    PredictionEnsemble.__getitem__
    PredictionEnsemble.__getattr__

----------
Properties
----------

.. autosummary::
    :toctree: api/

    PredictionEnsemble.coords
    PredictionEnsemble.nbytes
    PredictionEnsemble.sizes
    PredictionEnsemble.dims
    PredictionEnsemble.chunks
    PredictionEnsemble.chunksizes
    PredictionEnsemble.data_vars
    PredictionEnsemble.equals
    PredictionEnsemble.identical


HindcastEnsemble
~~~~~~~~~~~~~~~~

A :py:class:`.HindcastEnsemble` is a prediction ensemble that is
initialized off of some form of observations (an assimilation, reanalysis, etc.). Thus,
it is anticipated that forecasts are verified against observation-like products. Read
more about the terminology `here <terminology.html>`_.

.. autosummary::
    :toctree: api/

    HindcastEnsemble
    HindcastEnsemble.__init__

-------------------------
Add and Retrieve Datasets
-------------------------

.. autosummary::
    :toctree: api/

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

-------------
Generate Data
-------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.generate_uninitialized

--------------
Pre-Processing
--------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.smooth
    HindcastEnsemble.remove_bias
    HindcastEnsemble.remove_seasonality

-------------
Visualization
-------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.plot
    HindcastEnsemble.plot_alignment


PerfectModelEnsemble
~~~~~~~~~~~~~~~~~~~~

A ``PerfectModelEnsemble`` is a prediction ensemble that is initialized off of a
control simulation for a number of randomly chosen initialization dates. Thus,
forecasts cannot be verified against real-world observations.
Instead, they are `compared <comparisons.html>`_ to one another and to the
original control run. Read more about the terminology `here <terminology.html>`_.

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble
    PerfectModelEnsemble.__init__

-------------------------
Add and Retrieve Datasets
-------------------------

.. autosummary::
    :toctree: api/

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
    PerfectModelEnsemble.remove_seasonality

-------------
Visualization
-------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.plot


Direct Function Calls
---------------------

While not encouraged anymore, a user can directly call functions in ``climpred``.
This requires entering more arguments, e.g. the initialized ensemble directly as
well as a verification product. Our object
:py:class:`.HindcastEnsemble` and
:py:class:`.PerfectModelEnsemble` wrap most of these functions, making
the analysis process much simpler.

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
    compute_persistence_from_first_lead
    compute_uninitialized
    compute_climatology

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
    rm_poly
    rm_trend

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
    set_integer_time_axis

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

Visualization
~~~~~~~~~~~~~
.. currentmodule:: climpred.graphics

.. autosummary::
    :toctree: api/

    plot_bootstrapped_skill_over_leadyear
    plot_ensemble_perfect_model
    plot_lead_timeseries_hindcast


Metrics
-------

For a thorough look at our metrics library, please see the
`metrics </metrics.html>`_ page.

.. currentmodule:: climpred.metrics

.. autosummary::
    :toctree: api/

    Metric
    Metric.__init__
    Metric.__repr__
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
    _roc
    _spread
    _mul_bias
    _less

Comparisons
-----------

For a thorough look at our metrics library, please see the
`comparisons </comparisons.html>`_ page.

.. currentmodule:: climpred.comparisons

.. autosummary::
    :toctree: api/

    Comparison
    Comparison.__init__
    Comparison.__repr__
    _e2o
    _m2o
    _m2m
    _m2e
    _m2c
    _e2c

Config
------
Set options analogous to
`xarray <http://xarray.pydata.org/en/stable/generated/xarray.set_options.html>`_.

.. currentmodule:: climpred.options

.. autosummary::
    :toctree: api/

    set_options
