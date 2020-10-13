*******************
Reference Forecasts
*******************

To quantify the quality of an initialized forecast, it is useful to judge it against some simple
reference forecast. ``climpred`` currently supports a persistence forecast, but future releases
will allow computation of other reference forecasts. Consider opening a
`Pull Request <contributing.html>`_ to get it implemented more quickly.

**Persistence Forecast**: Whatever is observed at the time of initialization is forecasted to
persist into the forecast period [Jolliffe2012]_. You can compute this by passing
``reference='persistence'`` into the ``.verify()`` method for
:py:class:`~climpred.classes.HindcastEnsemble` and
:py:class:`~climpred.classes.PerfectModelEnsemble` objects.

**Damped Persistence Forecast**: (*Not Implemented*) The amplitudes of the anomalies reduce in time
exponentially at a time scale of the local autocorrelation [Yuan2016]_.

.. math::

    v_{dp}(t) = v(0)e^{-\alpha t}

**Climatology**: (*Not Implemented*) The average values at the temporal forecast resolution
(e.g., annual, monthly) over some long period, which is usually 30 years [Jolliffe2012]_.

**Random Mechanism**: (*Not Implemented*) A probability distribution is assigned to the possible
range of the variable being forecasted, and a sequence of forecasts is produced by taking a sequence
of independent values from that distribution [Jolliffe2012]_. This would be similar to computing an
uninitialized forecast, using ``reference='uninitialized'`` in
:py:class:`~climpred.classes.HindcastEnsemble` and
:py:class:`~climpred.classes.PerfectModelEnsemble` objects. For ``HindcastEnsemble`` objects, an
uninitialized ensemble has to be added through ``.add_uninitialized(...)``. This could be, for
example, output from a Large Ensemble. For ``PerfectModelEnsemble`` objects, one can run
``.generate_uninitialized()`` which uses a bootstrapping approach to create an uninitialized
equivalent.

References
##########

.. [Jolliffe2012] Jolliffe, Ian T., and David B. Stephenson, eds. Forecast verification:
   a practitioner's guide in atmospheric science. John Wiley & Sons, 2012.

.. [Yuan2016] Yuan, Xiaojun, et al. "Arctic sea ice seasonal prediction by a linear Markov model."
   Journal of Climate 29.22 (2016): 8151-8173.
