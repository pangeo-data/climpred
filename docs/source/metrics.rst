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

All high-level functions have an optional ``metric`` argument that can be called to
determine which metric is used in computing predictability.

.. note::

    We use the phrase 'observations' ``o`` here to refer to the 'truth' data to which
    we compare the forecast ``f``. These metrics can also be applied relative
    to a control simulation, reconstruction, observations, etc. This would just
    change the resulting score from quantifying skill to quantifying potential
    predictability.

Internally, all metric functions require ``forecast`` and ``observations`` as inputs.
The dimension ``dim`` is set internally by
:py:func:`~climpred.prediction.compute_hindcast` or
:py:func:`~climpred.prediction.compute_perfect_model` to specify over which dimensions
the ``metric`` is applied. See :ref:`comparisons` for more on the ``dim`` argument.

*************
Deterministic
*************

Deterministic metrics assess the forecast as a definite prediction of the future, rather
than in terms of probabilities. Another way to look at deterministic metrics is that
they are a special case of probabilistic metrics where a value of one is assigned to
one category and zero to all others [Jolliffe2011]_.

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
metric. Use ``pearson_r_p_value`` or ``spearman_r_p_value`` to compute p values assuming
that all samples in the correlated time series are independent. Use
``pearson_r_eff_p_value`` or ``spearman_r_eff_p_value`` to account for autocorrelation
in the time series by calculating the ``effective_sample_size``.

Pearson Product-Moment Correlation Coefficient
----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['pearson_r']}")

.. autofunction:: _pearson_r

Pearson Correlation p value
---------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['pearson_r_p_value']}")

.. autofunction:: _pearson_r_p_value

Effective Sample Size
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['effective_sample_size']}")

.. autofunction:: _effective_sample_size

Pearson Correlation Effective p value
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['pearson_r_eff_p_value']}")

.. autofunction:: _pearson_r_eff_p_value


Spearman's Rank Correlation Coefficient
---------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r']}")

.. autofunction:: _spearman_r


Spearman's Rank Correlation Coefficient p value
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r_p_value']}")

.. autofunction:: _spearman_r_p_value

Spearman's Rank Correlation Effective p value
---------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r_eff_p_value']}")

.. autofunction:: _spearman_r_eff_p_value

Distance Metrics
================

This class of metrics simply measures the distance (or difference) between forecasted
values and observed values.

Mean Squared Error (MSE)
------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['mse']}")

.. autofunction:: _mse


Root Mean Square Error (RMSE)
-----------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['rmse']}")

.. autofunction:: _rmse


Mean Absolute Error (MAE)
-------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['mae']}")

.. autofunction:: _mae


Median Absolute Error
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['median_absolute_error']}")

.. autofunction:: _median_absolute_error

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
    print(f"\n\nKeywords: {metric_aliases['nmse']}")

.. autofunction:: _nmse


Normalized Mean Absolute Error (NMAE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['nmae']}")

.. autofunction:: _nmae


Normalized Root Mean Square Error (NRMSE)
-----------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['nrmse']}")

.. autofunction:: _nrmse


Mean Square Error Skill Score (MSESS)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['msess']}")

.. autofunction:: _msess


Mean Absolute Percentage Error (MAPE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['mape']}")

.. autofunction:: _mape

Symmetric Mean Absolute Percentage Error (sMAPE)
------------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['smape']}")

.. autofunction:: _smape


Unbiased Anomaly Correlation Coefficient (uACC)
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['uacc']}")

.. autofunction:: _uacc


Murphy Decomposition Metrics
============================

Metrics derived in [Murphy1988]_ which decompose the ``MSESS`` into a correlation term,
a conditional bias term, and an unconditional bias term. See
https://www-miklip.dkrz.de/about/murcss/ for a walk through of the decomposition.

Standard Ratio
--------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['std_ratio']}")

.. autofunction:: _std_ratio

Conditional Bias
----------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['conditional_bias']}")

.. autofunction:: _conditional_bias

Unconditional Bias
------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['unconditional_bias']}")

Simple bias of the forecast minus the observations.

.. autofunction:: _unconditional_bias

Bias Slope
----------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['bias_slope']}")

.. autofunction:: _bias_slope

Murphy's Mean Square Error Skill Score
--------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['msess_murphy']}")

.. autofunction:: _msess_murphy

*************
Probabilistic
*************

Probabilistic metrics include the spread of the ensemble simulations in their
calculations and assign a probability value between 0 and 1 to their forecasts
[Jolliffe2011]_.

Continuous Ranked Probability Score (CRPS)
==========================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['crps']}")

.. autofunction:: _crps

Continuous Ranked Probability Skill Score (CRPSS)
=================================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['crpss']}")

.. autofunction:: _crpss

Continuous Ranked Probability Skill Score Ensemble Spread
=========================================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['crpss_es']}")

.. autofunction:: _crpss_es

Brier Score
===========

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['brier_score']}")

.. autofunction:: _brier_score

Threshold Brier Score
=====================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['threshold_brier_score']}")

.. autofunction:: _threshold_brier_score

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

  skill = compute_perfect_model(ds, control, metric=_my_msle)

Once you come up with an useful metric for your problem, consider contributing
this metric to `climpred`, so all users can benefit from your metric, see
:ref:`contributing`.

**********
References
**********

.. [Jolliffe2011] Ian T. Jolliffe and David B. Stephenson. Forecast Verification: A Practitioner’s Guide in Atmospheric Science. John Wiley & Sons, Ltd, Chichester, UK, December 2011. ISBN 978-1-119-96000-3 978-0-470-66071-3. URL: http://doi.wiley.com/10.1002/9781119960003.

.. [Murphy1988] Allan H. Murphy. Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. Monthly Weather Review, 116(12):2417–2424, December 1988. https://doi.org/10/fc7mxd.
