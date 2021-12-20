####################
Significance Testing
####################

Significance testing is important for assessing whether a given initialized prediction
system is skillful. Some questions that significance testing can answer are:

    - Is the correlation coefficient of a lead time series significantly different from
      zero?

    - What is the probability that the retrospective forecast is more valuable than a
      historical/uninitialized simulation?

    - Are correlation coefficients statistically significant despite temporal and
      spatial autocorrelation?

All of these questions deal with statistical significance. See below on how to use
``climpred`` to address these questions.
Please also have a look at the
`significance testing example <examples/decadal/significance.html>`__.

p value for temporal correlations
#################################

For the correlation `metrics <metrics.html>`__, like
:py:func:`~climpred.metrics._pearson_r` and :py:func:`~climpred.metrics._spearman_r`,
``climpred`` also hosts the associated p-value, like
:py:func:`~climpred.metrics._pearson_r_p_value`,
that this correlation is significantly different from zero.
:py:func:`~climpred.metrics._pearson_r_eff_p_value` also incorporates the reduced
degrees of freedom due to temporal autocorrelation. See
`example <examples/decadal/significance.html#p-value-for-temporal-correlations>`__.

Bootstrapping with replacement
##############################

Testing statistical significance through bootstrapping is commonly used in the field of
climate prediction. Bootstrapping relies on
resampling the underlying data with replacement for a large number of ``iterations``, as
proposed by the decadal prediction framework :cite:p:`Goddard2013,Boer2016`.
This means that the ``initialized`` ensemble is resampled with replacement along a
dimension (``init`` or ``member``) and then that resampled ensemble is verified against
the observations. This leads to a distribution of ``initialized`` skill. Further, a
``reference`` forecast uses the resampled ``initialized`` ensemble, which creates a
``reference`` skill distribution. Lastly, an ``uninitialized`` skill distribution is
created from the underlying historical members or the control simulation.

The probability or p value is the fraction of these resampled ``initialized`` metrics
beaten by the ``uninitialized`` or resampled reference metrics calculated from their
respective distributions. Confidence intervals using these distributions are also
calculated.

This behavior is incorporated by :py:meth:`.HindcastEnsemble.bootstrap` and
:py:meth:`.PerfectModelEnsemble.bootstrap`, see
`example <examples/decadal/significance.html#Bootstrapping-with-replacement>`__.


Field significance
##################

Please use :py:func:`esmtools.testing.multipletests` to control the false discovery
rate (FDR) in geospatial data from the above obtained p-values :cite:p:`Wilks2016`.
See the `FDR example <examples/decadal/significance.html#Field-significance>`__.


Sign test
#########

Use DelSole's sign test relying on the statistics of a random walk to decide whether
one forecast is significantly better than another forecast
:cite:p:`Benjamini1994,DelSole2016`, see :py:func:`xskillscore.sign_test` and
`sign test example <examples/decadal/significance.html#sign-test>`__.

References
##########

.. bibliography::
  :filter: docname in docnames
