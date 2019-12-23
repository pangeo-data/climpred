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
    we compare the forecast ``f``. These metrics can also be applied in reference
    to a control simulation, reconstruction, observations, etc. This would just
    change the resulting score from referencing skill to referencing potential
    predictability.

Internally, all metric functions require ``forecast`` and ``reference`` as inputs.
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

The below metrics rely fundamentally on correlations in their computation. The most
common correlation-based metric in forecasting is the Anomaly Correlation Coefficient,
or ACC. The ACC isn't a special type of correlation, but rather the correlation of
forecasted anomalies (rather than raw values). ``climpred`` offers the linear
`Pearson Product-Moment Correlation <#pearson-product-moment-correlation-coefficient>`_
and `Spearman's Rank Correlation <#spearman-s-rank-correlation-coefficient>`_.

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

A measure of the linear association between the forecast and observations that is
independent of the mean and variance of the individual distributions [Jolliffe2011]_.
This is also known as the Anomaly Correlation Coefficient (ACC) when comparing
anomalies.

.. autofunction:: _pearson_r

Pearson Correlation p value
---------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['pearson_r_p_value']}")

Two-tailed p value associated with the Pearson product-moment correlation coefficient
(``pearson_r``), assuming that all samples are independent (use
``pearson_r_eff_p_value`` to account for autocorrelation in the forecast and
observations).

.. autofunction:: _pearson_r_p_value

Effective Sample Size
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['effective_sample_size']}")

Number of independent samples when autocorrelation in the forecast and observations are
taken into account. This is used in computing the effective p value
(``pearson_r_eff_p_value`` or ``spearman_r_eff_p_value``) for correlations.

.. autofunction:: _effective_sample_size

Pearson Correlation Effective p value
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['pearson_r_eff_p_value']}")

p value associated with the Pearson product-moment correlation coefficient
(``pearson_r``) when autocorrelation is taken into account via the effective sample
size (``effective_sample_size``).

.. autofunction:: _pearson_r_eff_p_value


Spearman's Rank Correlation Coefficient
---------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r']}")

A measure of how well the relationship between two variables can be described using a
monotonic function. This is also known as the anomaly correlation coefficient (ACC)
when comparing anomalies.

.. autofunction:: _spearman_r


Spearman's Rank Correlation Coefficient p value
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r_p_value']}")

p value associated with the Spearman's product-moment correlation coefficient
(``spearman_r``), assuming that all samples are independent (use
``spearman_r_eff_p_value`` to account for autocorrelation in the forecast and
observations).

.. autofunction:: _spearman_r_p_value

Spearman's Rank Correlation Effective p value
---------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['spearman_r_eff_p_value']}")

p value associated with the Spearman's Rank Correlation Coefficient (``spearman_r``)
when autocorrelation is taken into account via the effective sample size
(``effective_sample_size``).

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

The average of the squared difference between forecasts and observations. This
incorporates both the variance and bias of the estimator. Because the error is squared,
it is more sensitive to large forecast errors than ``mae``, and thus a more conservative
metric. For example, a single error of 2°C counts the same as two 1°C errors when using
``mae``. On the other hand, the 2°C error counts double for ``mse`` [Jolliffe2011]_.

.. autofunction:: _mse


Root Mean Square Error (RMSE)
-----------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['rmse']}")

The square root of the average of the squared differences between forecasts and
observations.

.. autofunction:: _rmse


Mean Absolute Error (MAE)
-------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['mae']}")

The average of the absolute differences between forecasts and observations. A more
robust measure of forecast accuracy than ``mse`` which is sensitive to large outlier
forecast errors [EOS]_; [Jolliffe2011]_.

.. autofunction:: _mae


Median Absolute Error
---------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['median_absolute_error']}")

The median of the absolute differences between forecasts and observations. Applying
the median function to absolute error makes it more robust to outliers.

.. autofunction:: _median_absolute_error

Normalized Distance Metrics
===========================

Distance metrics like ``mse`` can be normalized to 1. The normalization factor
depends on the comparison type choosen. For example, the distance between an ensemble
member and the ensemble mean is half the distance of an ensemble member with other
ensemble members. (see :py:func:`~climpred.metrics._get_norm_factor`).

Normalized Mean Square Error (NMSE)
-----------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['nmse']}")

Mean Square Error (``mse``) normalized by the variance of the observations.

.. autofunction:: _nmse


Normalized Mean Absolute Error (NMAE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['nmae']}")

Mean Absolute Error (``mae``) normalized by the standard deviation of the observations.


.. autofunction:: _nmae


Normalized Root Mean Square Error (NRMSE)
-----------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['nrmse']}")

Root Mean Square Error (``rmse``) normalized by the standard deviation of the
observations.

.. autofunction:: _nrmse


Mean Square Skill Score (MSSS)
------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['ppp']}")

One minus the ratio of the squared error of the forecasts to the variance of the
observations.

.. autofunction:: _ppp


Mean Absolute Percentage Error (MAPE)
-------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['mape']}")

Mean absolute error (``mae``) expressed as a percentage error relative to the
observations.

.. autofunction:: _mape

Symmetric Mean Absolute Percentage Error (sMAPE)
------------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['smape']}")

Similar to the Mean Absolute Percentage Error (``mape``), but sums the forecast and
observation mean in the denominator.

.. autofunction:: _smape


Unbiased Anomaly Correlation Coefficient (uACC)
-----------------------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['uacc']}")

This is typically used in perfect model studies. Because the perfect model Anomaly
Correlation Coefficient (ACC) is strongly state dependent, a standard ACC (e.g. one
computed using ``pearson_r``) will be highly sensitive to the set of start dates chosen
for the perfect model study. The Mean Square Skill Score (``MSSS``) can be related
directly to the ACC as ``MSSS = ACC^(2)`` (see [Murphy1988]_ and [Bushuk2019]_), so the
unbiased ACC can be derived as ``uACC = sqrt(MSSS)``.

.. autofunction:: _uacc


Murphy Decomposition Metrics
============================

Various decomposition metrics from [Murphy1988]_ which relates the ``MSSS`` to the
``ACC`` and unconditional bias.

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

Murphy's Mean Square Skill Score
--------------------------------

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['msss_murphy']}")

.. autofunction:: _msss_murphy

*************
Probabilistic
*************

Probabilistic metrics include the spread of the ensemble simulations in their
calculations.

Continuous Ranked Probability Score (CRPS)
==========================================

.. ipython:: python

    # Enter any of the below keywords in ``metric=...`` for the compute functions.
    print(f"\n\nKeywords: {metric_aliases['crps']}")

The CRPS can also be considered as the probabilistic Mean Absolute Error (``mae``). It
compares the empirical distribution of an ensemble forecast to a scalar observation.
Smaller scores indicate better skill.

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

The Mean Square Error (``mse``) of probabilistic two-category forecasts where the
observations are either 0 (no occurrence) or 1 (occurrence) and forecast probability
may be arbitrarily distributed between occurrence and non-occurrence. The Brier Score
equals zero for perfect (single-valued) forecasts and one for forecasts that are always
incorrect. [NOAA Glossary of Forecast Verification Metrics]_

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
arguments ``forecast``, ``reference``, ``dim=None``, and ``**metric_kwargs``::

  from climpred.metrics import Metric

  def _my_msle(forecast, reference, dim=None, **metric_kwargs):
      """Mean squared logarithmic error (MSLE).
      https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error."""
      return ( (np.log(forecast + 1) + np.log(reference + 1) ) ** 2).mean(dim)

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

.. [EOS] https://eos.org/opinions/climate-and-other-models-may-be-more-accurate-than-reported

.. [Jolliffe2011] Ian T. Jolliffe and David B. Stephenson. Forecast Verification: A Practitioner’s Guide in Atmospheric Science. John Wiley & Sons, Ltd, Chichester, UK, December 2011. ISBN 978-1-119-96000-3 978-0-470-66071-3. URL: http://doi.wiley.com/10.1002/9781119960003.

.. [Murphy1988] Allan H. Murphy. Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. Monthly Weather Review, 116(12):2417–2424, December 1988. https://doi.org/10/fc7mxd.

.. [Bushuk2019] Bushuk, Mitchell, et al. "Regional Arctic sea–ice prediction: potential versus operational seasonal forecast skill." Climate Dynamics 52.5-6 (2019): 2721-2743. https://doi.org/10.1007/s00382-018-4288-y.

.. [NOAA Glossary of Forecast Verification Metrics] https://www.nws.noaa.gov/oh/rfcdev/docs/Glossary_Forecast_Verification_Metrics.pdf
