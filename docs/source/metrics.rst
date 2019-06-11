*******
Metrics
*******

All high-level functions have an optional "metric" argument that can be called to determine which metric is used in computing predictability (potential predictability or prediction skill).

.. note::

    We use the phrase 'observations' here to refer to the 'truth' data to which
    we compare the forecast. These metrics can also be applied in reference
    to a control simulation, reconstruction, observations, etc. This would just
    change the resulting score from referencing skill to referencing potential
    predictability.

.. currentmodule:: climpred.metrics

Internally, all metric functions require ``forecast`` and ``reference`` as inputs. The dimension ``dim`` is set by ``compute_hindcast`` or ``compute_perfect_model`` to specify over which dimensions the ``metric`` is applied. See ``comparison.page``, ``compute_philosophy``
# ToDo: add links between pages


Deterministic
#############

Deterministic metrics quantify the level to which the forecast predicts the observations. These metrics are just a special case of probabilistic metrics where a value of 100% is assigned to the forecasted value [1]_. :cite:`Jolliffe2011`

Core Metrics
============

Anomaly Correlation Coefficient (ACC)
-------------------------------------

``keyword: 'pearson_r'``

``perfect score: 1``

A measure of the linear association between the forecast and observations that is independent of the mean and variance of the individual distributions [2]_. ``climpred`` uses the Pearson correlation coefficient.

.. autofunction:: _pearson_r

Mean Squared Error (MSE)
------------------------

``keyword: 'mse'``

``perfect score: 0``

The average of the squared difference between forecasts and observations. This incorporates both the variance and bias of the estimator [4]_.

.. math::
    MSE = \overline{(f - o)^{2}}

.. autofunction:: _mse

Root Mean Square Error (RMSE)
-----------------------------

``keyword: 'rmse'``

``perfect score: 0``

The square root of the average of the squared differences between forecasts and observations [2]_.
It puts a greater influence on large errors than small errors, which makes this a good choice if large errors are undesirable or one wants to be a more conservative forecaster.

.. math::
    RMSE = \sqrt{\overline{(f - o)^{2}}}

.. autofunction:: _rmse

Mean Absolute Error (MAE)
-------------------------

``keyword: 'mae'``

``perfect score: 0``

The average of the absolute differences between forecasts and observations [2]_. A more robust measure of forecast accuracy than root mean square error or mean square error which is sensitive to large outlier forecast errors [3]_.

.. math::
    MAE = (\overline{\vert f - o \vert})
.. autofunction:: _mae


Derived Metrics
===============

Normalization based on comparison type. Comparison versus mean requires different normalization than comparison against every member (see ``climpred.metrics._get_norm_factor``).

.. autofunction:: _ppp
.. autofunction:: _nmse
.. autofunction:: _nmae
.. autofunction:: _nrmse


Murphy decomposition metrics
============================

:cite:`Murphy1988`

.. autofunction:: _std_ratio
.. autofunction:: _bias
.. autofunction:: _bias_slope
.. autofunction:: _conditional_bias
.. autofunction:: _msss_murphy


Other metrics
=============

.. autofunction:: _less


Probabilistic
#############

.. autofunction:: _crps
.. autofunction:: _crpss


.. bibliography:: refs.bib

below old: to be transfered to refs.bib or deleted
References
##########

.. [2] http://www.nws.noaa.gov/oh/rfcdev/docs/Glossary_Verification_Metrics.pdf

.. [3] https://eos.org/opinions/climate-and-other-models-may-be-more-accurate-than-reported

.. [4] https://en.wikipedia.org/wiki/Mean_squared_error
