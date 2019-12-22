.. currentmodule:: climpred.metrics

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
The dimension ``dim`` is set by :py:func:`~climpred.prediction.compute_hindcast` or
:py:func:`~climpred.prediction.compute_perfect_model` to specify over which dimensions
the ``metric`` is applied. See :ref:`comparisons`.

*************
Deterministic
*************

Deterministic metrics quantify the level to which the forecast predicts the
observations. These metrics are just a special case of probabilistic metrics where a
value of 100% is assigned to the forecasted value [Jolliffe2011]_.

Correlation Metrics
===================

Pearson Product-Moment Correlation Coefficient
----------------------------------------------

``keyword: 'pearson_r', 'pr', 'pacc', 'acc'``

A measure of the linear association between the forecast and observations that is
independent of the mean and variance of the individual distributions [Jolliffe2011]_.
This is also known as the anomaly correlation coefficient (ACC) when comparing
anomalies.

.. autofunction:: _pearson_r

Pearson Correlation p-value
---------------------------

``keyword: 'pearson_r_p_value', 'p_pval', 'pvalue', 'pval'``

p-value associated with the Pearson product-moment correlation coefficient
(``pearson_r``), assuming that all samples are independent (use
``pearson_r_eff_p_value`` to account for autocorrelation in the forecast and
observations).

.. autofunction:: _pearson_r_p_value

Effective Sample Size
---------------------

``keyword: 'effective_sample_size', 'eff_n', 'n_eff'``

Number of independent samples when autocorrelation in the forecast and observations are
taken into account. This is used in computing the effective p-value
(``pearson_r_eff_p_value``) for correlations.

.. autofunction:: _effective_sample_size

Pearson Correlation Effective p-value
-------------------------------------

``keyword: 'pearson_r_eff_p_value', 'p_pval_eff', 'pvalue_eff', 'pval_eff'``

p-value associated with the Pearson product-moment correlation coefficient
(``pearson_r``) when autocorrelation is taken into account via the effective sample
size (``effective_sample_size``).

.. autofunction:: _pearson_r_eff_p_value


Spearman's Rank Correlation Coefficient
---------------------------------------

``keyword: 'spearman_r', 'sacc', 'sr'``

A measure of how well the relationship between two variables can be described using a
monotonic function. This is also known as the anomaly correlation coefficient (ACC)
when comparing anomalies.

.. autofunction:: _spearman_r


Spearman's Rank Correlation Coefficient p-value
-----------------------------------------------

``keyword: 'spearman_r_p_value', 's_pval', 'spvalue', 'spval'``

p-value associated with the Spearman's product-moment correlation coefficient
(``spearman_r``), assuming that all samples are independent (use
``spearman_r_eff_p_value`` to account for autocorrelation in the forecast and
observations).

.. autofunction:: _spearman_r_p_value

Distance Metrics
================

Mean Squared Error (MSE)
------------------------

``keyword: 'mse'``

The average of the squared difference between forecasts and observations. This
incorporates both the variance and bias of the estimator.

.. autofunction:: _mse


Root Mean Square Error (RMSE)
-----------------------------

``keyword: 'rmse'``

The square root of the average of the squared differences between forecasts and
observations [Jolliffe2011]_. It puts a greater influence on large errors than small
errors, which makes this a good choice if large errors are undesirable or one wants to
be a more conservative forecaster.

.. autofunction:: _rmse


Mean Absolute Error (MAE)
-------------------------

``keyword: 'mae'``

The average of the absolute differences between forecasts and observations
[Jolliffe2011]_. A more robust measure of forecast accuracy than root mean square error
or mean square error which is sensitive to large outlier forecast errors [EOS]_.

.. autofunction:: _mae


Median Absolute Error
---------------------

``keyword: 'median_absolute_error'``

The median of the absolute differences between forecasts and observations.

.. autofunction:: _median_absolute_error

Normalized Distance Metrics
===========================

Distance metrics like ``mse`` can be normalized to 1. The normalization factor
depends on the comparison type choosen. For example, the distance between an ensemble
member and the ensemble mean is half the distance of an ensemble member with other
ensemble members. (see :py:func:`~climpred.metrics._get_norm_factor`).

Normalized Mean Square Error (NMSE)
-----------------------------------

``keyword: 'nmse', 'nev'``

Mean Square Error (``mse``) normalized by the variance of the observations.

.. autofunction:: _nmse


Normalized Mean Absolute Error (NMAE)
-------------------------------------

``keyword: 'nmae'``

Mean Absolute Error (``mae``) normalized by the standard deviation of the observations.

.. autofunction:: _nmae


Normalized Root Mean Square Error (NRMSE)
-----------------------------------------

``keyword: 'nrmse'``

Root Mean Square Error (``rmse``) normalized by the standard deviation of the
observations.

.. autofunction:: _nrmse


Mean Square Skill Score (MSSS)
------------------------------

``keyword: 'msss', 'ppp'``

One minus the ratio of the squared error of the forecasts to the variance of the
observations.

.. autofunction:: _ppp


Mean Absolute Percentage Error (MAPE)
-------------------------------------

``keyword: 'mape'``

Mean absolute error (``mae``) expressed as a percentage error relative to the
observations.

.. autofunction:: _mape

Symmetric Mean Absolute Percentage Error (sMAPE)
------------------------------------------------

``keyword: 'smape'``

Similar to the Mean Absolute Percentage Error (``mape``), but sums the forecast and
observation mean in the denominator.

.. autofunction:: _smape


Unbiased Anomaly Correlation Coefficient (uACC)
-----------------------------------------------

``keyword: 'uacc'``

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

``keyword: 'std_ratio'``

.. autofunction:: _std_ratio

Conditional Bias
----------------

``keyword: 'conditional_bias', c_b'``

.. autofunction:: _conditional_bias

Unconditional Bias
------------------

``keyword: 'unconditional_bias', 'bias', 'u_b'``

Simple bias of the forecast minus the observations.

.. autofunction:: _unconditional_bias

Bias Slope
----------

``keyword: 'bias_slope'``

.. autofunction:: _bias_slope

Murphy's Mean Square Skill Score
--------------------------------

``keyword: 'msss_murphy'``

.. autofunction:: _msss_murphy

*************
Probabilistic
*************

Probabilistic metrics include the spread of the ensemble simulations in their
calculations.

Continuous Ranked Probability Score (CRPS)
==========================================

``keyword: 'crps'``

The CRPS can also be considered as the probabilistic Mean Absolute Error (``mae``). It
compares the empirical distribution of an ensemble forecast to a scalar observation.
Smaller scores indicate better skill.

.. autofunction:: _crps

Continuous Ranked Probability Skill Score (CRPSS)
=================================================

``keyword: 'crpss'``

.. autofunction:: _crpss

Continuous Ranked Probability Skill Score Ensemble Spread
=========================================================

``keyword: 'crpss_es'``

.. autofunction:: _crpss_es

Brier Score
===========

``keyword: 'brier_score', 'brier', 'bs'``

The Mean Square Error (``mse``) of probabilistic two-category forecasts where the
observations are either 0 (no occurrence) or 1 (occurrence) and forecast probability
may be arbitrarily distributed between occurrence and non-occurrence. The Brier Score
equals zero for perfect (single-valued) forecasts and one for forecasts that are always
incorrect. [NOAA Glossary of Forecast Verification Metrics]_

.. autofunction:: _brier_score

Threshold Brier Score
=====================

``keyword: 'threshold_brier_score', 'tbs'``

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
