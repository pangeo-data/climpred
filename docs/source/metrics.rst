*******
Metrics
*******

.. msss_murphy
.. conditional_bias
.. crps
.. crpss
.. less
.. nmae
.. nrmse
.. nmse
.. ppp
.. uacc

All high-level functions have an optional "metric" flag that can be called to determine which metric is used in computing predictability (potential predictability or skill).


.. note::

    We use the phrase 'observations' here to refer to the 'truth' data to which
    we compare the forecast. These metrics can also be applied in reference
    to a control simulation, reconstruction, observations, etc. This would just change the resulting
    score from referencing skill to referencing potential predictability.

Deterministic
#############

Deterministic metrics quantify the level to which the forecast predicts the observations. These metrics are just a special case of probabilistic metrics where a value of 100% is assigned to the forecasted value [1]_.

Anomaly Correlation Coefficient (ACC)
-------------------------------------

``keyword: 'pearson_r', 'pr', 'acc'``

``perfect score: 1``

A measure of the linear association between the forecast and observations that is independent of the mean and variance of the individual distributions [2]_. ``climpred`` uses the Pearson correlation coefficient.

To compute the p-value associated with the correlation coefficient, pass ``pearson_r_p_value``.

Mean Absolute Error (MAE)
-------------------------

``keyword: 'mae'``

``perfect score: 0``

The average of the absolute differences between forecasts and observations [2]_. A more robust measure of forecast accuracy than root mean square error or mean square error which is sensitive to large outlier forecast errors [3]_.

.. math::
    MAE = (\overline{\vert f - o \vert})

Mean Squared Error (MSE)
------------------------

``keyword: 'mse'``

``perfect score: 0``

The average of the squared difference between forecasts and observations. This incorporates both the variance and bias of the estimator [4]_.

.. math::
    MSE = \overline{(f - o)^{2}}

Root Mean Square Error (RMSE)
-----------------------------

``keyword: 'rmse'``

``perfect score: 0``

The square root of the average of the squared differences between forecasts and observations [2]_.
It puts a greater influence on large errors than small errors, which makes this a good choice if large errors are undesirable or one wants to be a more conservative forecaster.

.. math::
    RMSE = \sqrt{\overline{(f - o)^{2}}}

Diagnostics
###########


Bias Slope
----------

``keyword: 'bias_slope'``

Slope of the linear regression between the forecast and observations [5]_.

.. math::
    m = \frac{\sigma_{o}}{\sigma_{f}} \cdot \mathrm{ACC}

Standard Ratio
--------------

``keyword: 'std_ratio'``

Ratio of the standard deviations of the forecast and observations [5]_.

.. math::
    \frac{\sigma_{o}}{\sigma_{f}}

Unconditional Bias
------------------

``keyword: 'unconditional_bias', 'u_b', 'bias'``

``perfect score: 0``

The difference between the mean of the forecasts and the mean of the observations. Also known as the overall bias, systematic bias, or unconditional forecasts [2]_.

.. math::
   \mathrm{bias} = \overline{f} - \overline{o}

Probabilistic
#############

References
##########

.. [1] Jolliffe, Ian T., and David B. Stephenson, eds. Forecast verification: a practitioner's guide in atmospheric science. John Wiley & Sons, 2003.

.. [2] http://www.nws.noaa.gov/oh/rfcdev/docs/Glossary_Verification_Metrics.pdf

.. [3] https://eos.org/opinions/climate-and-other-models-may-be-more-accurate-than-reported

.. [4] https://en.wikipedia.org/wiki/Mean_squared_error

.. [5] https://www-miklip.dkrz.de/about/murcss/
