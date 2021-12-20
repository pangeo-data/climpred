.. currentmodule:: climpred.metrics

.. ipython:: python
    :suppress:

    from climpred.metrics import __ALL_METRICS__ as all_metrics

    metric_aliases = {}
    for m in all_metrics:
        if m.aliases is not None:
            metric_list = [m.name] + m.aliases
        else:
            metric_list = [m.name]
        metric_aliases[m.name] = metric_list

#######
Metrics
#######

All high-level functions like :py:meth:`.HindcastEnsemble.verify`,
:py:meth:`.HindcastEnsemble.bootstrap`, :py:meth:`.PerfectModelEnsemble.verify` and
:py:meth:`.PerfectModelEnsemble.bootstrap` have a ``metric`` argument
that has to be called to determine which metric is used in computing predictability.

.. note::

    We use the term 'observations' ``o`` here to refer to the 'truth' data to which
    we compare the forecast ``f``. These metrics can also be applied relative
    to a control simulation, reconstruction, observations, etc. This would just
    change the resulting score from quantifying skill to quantifying potential
    predictability.

Internally, all metric functions require ``forecast`` and ``observations`` as inputs.
The dimension ``dim`` has to be set to specify over which dimensions
the ``metric`` is applied and are hence reduced.
See :ref:`comparisons` for more on the ``dim`` argument.

*************
Deterministic
*************

Deterministic metrics assess the forecast as a definite prediction of the future, rather
than in terms of probabilities. Another way to look at deterministic metrics is that
they are a special case of probabilistic metrics where a value of one is assigned to
one category and zero to all others :cite:p:`Jolliffe2011`.

Correlation Metrics
===================

The below metrics rely fundamentally on correlations in their computation. In the
literature, correlation metrics are typically referred to as the Anomaly Correlation
Coefficient (ACC). This implies that anomalies in the forecast and observations
are being correlated. Typically, this is computed using the linear
`Pearson Product-Moment Correlation <#pearson-product-moment-correlation-coefficient>`_.
However, ``climpred`` also offers the
`Spearman's Rank Correlation <#spearman-s-rank-correlation-coefficient>`_.

Note that the p value associated with these correlations is computed via a separate
metric. Use :py:func:`~climpred.metrics._pearson_r_p_value` or
:py:func:`~climpred.metrics._spearman_r_p_value` to compute p values assuming
that all samples in the correlated time series are independent. Use
:py:func:`~climpred.metrics._pearson_r_eff_p_value` or
:py:func:`~climpred.metrics._spearman_r_eff_p_value` to account for autocorrelation
in the time series by calculating the
:py:func:`~climpred.metrics._effective_sample_size`.

Pearson Product-Moment Correlation Coefficient
----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['pearson_r']}")

.. autofunction:: _pearson_r

Pearson Correlation p value
---------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['pearson_r_p_value']}")

.. autofunction:: _pearson_r_p_value

Effective Sample Size
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['effective_sample_size']}")

.. autofunction:: _effective_sample_size

Pearson Correlation Effective p value
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['pearson_r_eff_p_value']}")

.. autofunction:: _pearson_r_eff_p_value


Spearman's Rank Correlation Coefficient
---------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['spearman_r']}")

.. autofunction:: _spearman_r


Spearman's Rank Correlation Coefficient p value
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['spearman_r_p_value']}")

.. autofunction:: _spearman_r_p_value

Spearman's Rank Correlation Effective p value
---------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['spearman_r_eff_p_value']}")

.. autofunction:: _spearman_r_eff_p_value

Distance Metrics
================

This class of metrics simply measures the distance (or difference) between forecasted
values and observed values.

Mean Squared Error (MSE)
------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['mse']}")

.. autofunction:: _mse


Root Mean Square Error (RMSE)
-----------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['rmse']}")

.. autofunction:: _rmse


Mean Absolute Error (MAE)
-------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['mae']}")

.. autofunction:: _mae


Median Absolute Error
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['median_absolute_error']}")

.. autofunction:: _median_absolute_error


Spread
------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['spread']}")

.. autofunction:: _spread


Multiplicative bias
-------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['mul_bias']}")

.. autofunction:: _mul_bias



Normalized Distance Metrics
===========================

Distance metrics like ``mse`` can be normalized to 1. The normalization factor
depends on the comparison type choosen. For example, the distance between an ensemble
member and the ensemble mean is half the distance of an ensemble member with other
ensemble members. See :py:func:`~climpred.metrics._get_norm_factor`.

Normalized Mean Square Error (NMSE)
-----------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['nmse']}")

.. autofunction:: _nmse


Normalized Mean Absolute Error (NMAE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['nmae']}")

.. autofunction:: _nmae


Normalized Root Mean Square Error (NRMSE)
-----------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['nrmse']}")

.. autofunction:: _nrmse


Mean Square Error Skill Score (MSESS)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['msess']}")

.. autofunction:: _msess


Mean Absolute Percentage Error (MAPE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['mape']}")

.. autofunction:: _mape

Symmetric Mean Absolute Percentage Error (sMAPE)
------------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['smape']}")

.. autofunction:: _smape


Unbiased Anomaly Correlation Coefficient (uACC)
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['uacc']}")

.. autofunction:: _uacc


Murphy Decomposition Metrics
============================

Metrics derived in :cite:p:`Murphy1988` which decompose the ``MSESS`` into a correlation term,
a conditional bias term, and an unconditional bias term. See
https://www-miklip.dkrz.de/about/murcss/ for a walk through of the decomposition.

Standard Ratio
--------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['std_ratio']}")

.. autofunction:: _std_ratio

Conditional Bias
----------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['conditional_bias']}")

.. autofunction:: _conditional_bias

Unconditional Bias
------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['unconditional_bias']}")

Simple bias of the forecast minus the observations.

.. autofunction:: _unconditional_bias

Bias Slope
----------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['bias_slope']}")

.. autofunction:: _bias_slope

Murphy's Mean Square Error Skill Score
--------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['msess_murphy']}")

.. autofunction:: _msess_murphy

*************
Probabilistic
*************

Probabilistic metrics include the spread of the ensemble simulations in their
calculations and assign a probability value between 0 and 1 to their forecasts
:cite:p:`Jolliffe2011`.

Continuous Ranked Probability Score (CRPS)
==========================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['crps']}")

.. autofunction:: _crps

Continuous Ranked Probability Skill Score (CRPSS)
=================================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['crpss']}")

.. autofunction:: _crpss

Continuous Ranked Probability Skill Score Ensemble Spread
=========================================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['crpss_es']}")

.. autofunction:: _crpss_es

Brier Score
===========

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['brier_score']}")

.. autofunction:: _brier_score

Threshold Brier Score
=====================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['threshold_brier_score']}")

.. autofunction:: _threshold_brier_score

Ranked Probability Score
========================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['rps']}")

.. autofunction:: _rps

Reliability
===========

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['reliability']}")

.. autofunction:: _reliability

Discrimination
==============

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['discrimination']}")

.. autofunction:: _discrimination

Rank Histogram
==============

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['rank_histogram']}")

.. autofunction:: _rank_histogram

Logarithmic Ensemble Spread Score
=================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['less']}")

.. autofunction:: _less

*************************
Contingency-based metrics
*************************

Contingency
===========

A number of metrics can be derived from a `contingency table <https://www.cawcr.gov.au/projects/verification/#Contingency_table>`_. To use this in ``climpred``, run ``.verify(metric='contingency', score=...)`` where score can be chosen from `xskillscore <https://xskillscore.readthedocs.io/en/stable/api.html#contingency-based-metrics>`_.

.. autofunction:: _contingency

Receiver Operating Characteristic
=================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"Keywords: {metric_aliases['roc']}")

.. autofunction:: _roc


********************
User-defined metrics
********************

You can also construct your own metrics via the :py:class:`climpred.metrics.Metric`
class.

.. autosummary:: Metric

First, write your own metric function, similar to the existing ones with required
arguments ``forecast``, ``observations``, ``dim=None``, and ``**metric_kwargs``::

  from climpred.metrics import Metric

  def _my_msle(forecast, observations, dim=None, **metric_kwargs):
      """Mean squared logarithmic error (MSLE).
      https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error."""
      # function
      return ( (np.log(forecast + 1) + np.log(observations + 1) ) ** 2).mean(dim)

Then initialize this metric function with :py:class:`climpred.metrics.Metric`::

  _my_msle = Metric(
      name='my_msle',
      function=_my_msle,
      probabilistic=False,
      positive=False,
      unit_power=0,
      )

Finally, compute skill based on your own metric::

  skill = hindcast.verify(metric=_my_msle, comparison='e2o', alignment='same_verif', dim='init')

Once you come up with an useful metric for your problem, consider contributing
this metric to `climpred`, so all users can benefit from your metric, see
`contributing <contributing.html>`_.

**********
References
**********

.. bibliography::
  :filter: docname in docnames
