*******
Metrics
*******

All high-level functions have an optional ``metric`` argument that can be called to determine which metric is used in computing predictability (potential predictability or prediction skill).

.. note::

    We use the phrase 'observations' ``o`` here to refer to the 'truth' data to which
    we compare the forecast ``f``. These metrics can also be applied in reference
    to a control simulation, reconstruction, observations, etc. This would just
    change the resulting score from referencing skill to referencing potential
    predictability.

.. currentmodule:: climpred.metrics

Internally, all metric functions require ``forecast`` and ``reference`` as inputs. The dimension ``dim`` is set by :py:func:`climpred.prediction.compute_hindcast` or :py:func:`climpred.prediction.compute_perfect_model` to specify over which dimensions the ``metric`` is applied. See :ref:`comparisons`.


Deterministic
#############

Deterministic metrics quantify the level to which the forecast predicts the observations. These metrics are just a special case of probabilistic metrics where a value of 100% is assigned to the forecasted value [Jolliffe2011]_.

Core Metrics
============

Anomaly Correlation Coefficient (ACC)
-------------------------------------

``keyword: 'pearson_r','pr','acc'``

A measure of the linear association between the forecast and observations that is independent of the mean and variance of the individual distributions [Jolliffe2011]_. ``climpred`` uses the Pearson correlation coefficient.

.. autofunction:: _pearson_r

.. note:: The p-value associated with the ACC can be computed using keywords ``pearson_r_p_value``, ``pval``, or ``pvalue``.


Mean Squared Error (MSE)
------------------------

``keyword: 'mse'``

The average of the squared difference between forecasts and observations. This incorporates both the variance and bias of the estimator.


.. autofunction:: _mse

Root Mean Square Error (RMSE)
-----------------------------

``keyword: 'rmse'``

The square root of the average of the squared differences between forecasts and observations [Jolliffe2011]_.
It puts a greater influence on large errors than small errors, which makes this a good choice if large errors are undesirable or one wants to be a more conservative forecaster.

.. autofunction:: _rmse


Mean Absolute Error (MAE)
-------------------------

``keyword: 'mae'``

The average of the absolute differences between forecasts and observations [Jolliffe2011]_. A more robust measure of forecast accuracy than root mean square error or mean square error which is sensitive to large outlier forecast errors [EOS]_.

.. autofunction:: _mae



Derived Metrics
===============

Distance-based metrics like ``mse`` can be normalized to 1. The normalization factor depends on the comparison type choosen, eg. the distance between an ensemble member and the ensemble mean is half the distance of an ensemble member with other ensemble members. (see :py:func:`climpred.metrics._get_norm_factor`).


Normalized Mean Square Error (NMSE)
-----------------------------------

``keyword: 'nmse','nev'``

.. autofunction:: _nmse


Normalized Mean Absolute Error (NMAE)
-------------------------------------

``keyword: 'nmae'``

.. autofunction:: _nmae


Normalized Root Mean Square Error (NRMSE)
-----------------------------------------

``keyword: 'nrmse'``

.. autofunction:: _nrmse


Mean Square Skill Score (MSSS)
------------------------------

``keyword: 'msss','ppp'``

.. autofunction:: _ppp


Murphy decomposition metrics
============================

[Murphy1988]_ relates the MSSS with ACC and unconditional bias.

.. autofunction:: _std_ratio
.. autofunction:: _bias
.. autofunction:: _bias_slope
.. autofunction:: _conditional_bias
.. autofunction:: _msss_murphy


Other metrics
=============

.. autofunction:: _less
.. autofunction:: _uacc


Probabilistic
#############

.. autofunction:: _crps
.. autofunction:: _crpss


References
##########

.. [EOS] https://eos.org/opinions/climate-and-other-models-may-be-more-accurate-than-reported

.. [Jolliffe2011] Ian T. Jolliffe and David B. Stephenson. Forecast Verification: A Practitioner’s Guide in Atmospheric Science. John Wiley & Sons, Ltd, Chichester, UK, December 2011. ISBN 978-1-119-96000-3 978-0-470-66071-3. URL: http://doi.wiley.com/10.1002/9781119960003.

.. [Murphy1988] Allan H. Murphy. Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient. Monthly Weather Review, 116(12):2417–2424, December 1988. https://doi.org/10/fc7mxd.
