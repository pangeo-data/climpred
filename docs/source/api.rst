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
    HindcastEnsemble.compute_persistence
    HindcastEnsemble.compute_uninitialized

--------------
Pre-Processing
--------------

.. autosummary::
    :toctree: api/

    HindcastEnsemble.smooth

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

    PerfectModelEnsemble.bootstrap
    PerfectModelEnsemble.compute_metric
    PerfectModelEnsemble.compute_persistence
    PerfectModelEnsemble.compute_uninitialized

-------------
Generate Data
-------------

.. autosummary::
    :toctree: api/

    PerfectModelEnsemble.generate_uninitialized


Direct Function Calls
---------------------

A user can directly call functions in ``climpred``. This requires entering more arguments, e.g.
the initialized ensemble
:py:class:`~xarray.core.dataset.Dataset`/:py:class:`xarray.core.dataarray.DataArray` directly as
well as a verification product. Our object
:py:class:`~climpred.classes.HindcastEnsemble` and
:py:class:`~climpred.classes.PerfectModelEnsemble` wrap most of these functions, making the
analysis process much simpler. Once we have wrapped all of the functions in their entirety, we will
likely depricate the ability to call them directly.

Bootstrap
~~~~~~~~~
.. currentmodule:: climpred.bootstrap

.. autosummary::
    :toctree: api/

    bootstrap_compute
    bootstrap_hindcast
    bootstrap_perfect_model
    bootstrap_uninit_pm_ensemble_from_control
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
    compute_persistence
    compute_uninitialized

Metrics
~~~~~~~
.. currentmodule:: climpred.metrics

.. autosummary::
    :toctree: api/

    Metric
    _get_norm_factor

Comparisons
~~~~~~~~~~~
.. currentmodule:: climpred.comparisons

.. autosummary::
    :toctree: api/

    Comparison

Statistics
~~~~~~~~~~
.. currentmodule:: climpred.stats

.. autosummary::
    :toctree: api/

    autocorr
    corr
    decorrelation_time
    dpp
    rm_poly
    rm_trend
    varweighted_mean_period

Tutorial
~~~~~~~~
.. currentmodule:: climpred.tutorial

.. autosummary::
    :toctree: api/

    load_dataset
