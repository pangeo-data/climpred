############
Significance
############

How to access whether initialized ensembles are skillful? Is correlation of a lead
timeseries different from zero? What is a probability that a retrospective forecast was
more valueable than a historical simulation? How to correct for spatial autocorrelation?
All these questions deal with significance. Here is how you can use ``climpred`` to
calculate this. Please also have a look at the `significance example <examples/decadal/significance.html>`__.

Metric-based p-value
####################

For the correlation `metrics <metrics.html>`__, like
:py:func:`~climpred.metrics._pearson_r`, ``climpred`` also hosts the associated p-value, like
:py:func:`~climpred.metrics._pearson_r_p_value`, that this correlation is significantly
different to zero incorporating reduced degrees of freedom due to temporal
autocorrelation.

Bootstrapping with replacement
##############################

Bootstrapping significance relies of resampling the underlying data with replacement of
large number of iterations as proposed by the decadal prediction framework of Goddard
et al. 2013 [Goddard2013]_. This means that the initialized ensemble is resampled with
replacement along a dimension (``init`` or ``member``) and then skill is computed. This
leads to a distribution of initialized skill.
Also, the baseline skill uses these resampled initialized ensembles, e.g. typically
:py:func:`~climpred.prediction.compute_persistence`, which also creates a baseline skill
distribution.
Lastly, uninitialized skill is resampled from the underlying historical members or
the control simulation.
The probability or p-value is the fraction of these resampled initialized skills
beaten by the uninitialized or resampled baseline skills calculated from the respective
distributions. Also confidence intervals using these distributions are calculated.
This is behaviour is incorporated into ``climpred`` by the base function
:py:func:`~climpred.bootstrap.bootstrap_compute`, which is wrapped by
:py:func:`~climpred.bootstrap.bootstrap_hindcast` and
:py:func:`~climpred.bootstrap.bootstrap_perfect_model` for the respective prediction
simulation type.

Field significance
##################

Please use :py:func:`~esmtools.testing.multipletests` to control the false discovery
rate (FDR) from the above obtained p-values [Wilks2016]_.


References
##########

.. [Goddard2013]  Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P. Gonzalez, V.
    Kharin, et al. “A Verification Framework for Interannual-to-Decadal Predictions
    Experiments.” Climate Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
    https://doi.org/10/f4jjvf.


.. [Wilks2016]  Wilks, D. S. “‘The Stippling Shows Statistically Significant Grid
    Points’: How Research Results Are Routinely Overstated and Overinterpreted, and
    What to Do about It.” Bulletin of the American Meteorological Society 97, no. 12
    (March 9, 2016): 2263–73. https://doi.org/10/f9mvth.
