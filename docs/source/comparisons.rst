***********
Comparisons
***********

Forecasts have to be verified against some product to evaluate their performance.
However, when verifying against a product, there are many different ways one can compare
the ensemble of forecasts. Here we cover the comparison options for both hindcast and
perfect model ensembles. See `terminology <terminology.html>`__ for clarification on
the differences between these two experimental setups.

Note that all compute functions (:py:func:`~climpred.prediction.compute_hindcast`,
:py:func:`~climpred.prediction.compute_perfect_model`,
:py:func:`~climpred.prediction.compute_hindcast`,
:py:func:`~climpred.bootstrap.bootstrap_hindcast`,
:py:func:`~climpred.bootstrap.bootstrap_perfect_model`) take an optional
``comparison=''`` keyword to select the comparison style. See below for a detailed
description on the differences between these comparisons.

Hindcast Ensembles
##################

In hindcast ensembles, the ensemble mean forecast (``comparison='e2o'``) is expected to
perform better than individual ensemble members (``comparison='m2o'``) as the chaotic
component of forecasts is expected to be suppressed by this averaging, while the memory
of the system sustains. [Boer2016]_

``keyword: 'e2o', 'e2r'``

**This is the default option.**

.. currentmodule:: climpred.comparisons

.. autosummary:: _e2o


``keyword: 'm2o', 'm2r'``

.. autosummary:: _m2o

Perfect Model Ensembles
#######################

In perfect-model frameworks, there are many more ways of verifying forecasts.
[Seferian2018]_ uses a comparison of all ensemble members against the
control run (``comparison='m2c'``) and all ensemble members against all other ensemble
members (``comparison='m2m'``). Furthermore, the ensemble mean forecast can be verified
against one control member (``comparison='e2c'``) or all members (``comparison='m2e'``)
as done in [Griffies1997]_.

``keyword: 'm2e'``

**This is the default option.**

.. autosummary:: _m2e

``keyword: 'm2c'``

.. autosummary:: _m2c

``keyword: 'm2m'``

.. autosummary:: _m2m

``keyword: 'e2c'``

.. autosummary:: _e2c


Normalization
#############

The goal of a normalized distance metric is to get a constant or comparable value of
typically 1 (or 0 for metrics defined as 1 - metric) when the metric saturates and the
predictability horizon is reached (see `metrics <metrics.html>`__).

A factor is added in the normalized metric formula (see [Seferian2018]_) to accomodate
different comparison styles. For example, ``nrmse`` gets smalled in comparison ``m2e``
than ``m2m`` by design, since the ensembe mean is always closer to individual members
than the ensemble members to each other. In turn, the normalization factor is ``2`` for
comparisons ``m2c``, ``m2m``, and ``m2o``. It is 1 for ``m2e``, ``e2c``, and ``e2o``.

Interpretation of Results
#########################

While ``HindcastEnsemble`` skill is computed over all initializations ``init`` of the
hindcast, the resulting skill is a mean forecast skill over all initializations.

``PerfectModelEnsemble`` skill is computed over a supervector comprised of all
initializations and members, which allows the computation of the ACC-based skill
[Bushuk2018]_, but also returns a mean forecast skill over all initializations.

The supervector approach shown in [Bushuk2018]_ and just calculating a distance-based
metric like ``rmse`` over the member dimension as in [Griffies1997]_ yield very similar
results.

Compute over dimension
######################

The optional argument ``dim`` defines over which dimension a metric is computed. We can
apply a metric over ``dim`` from [``'init'``, ``'member'``, ``['member', 'init']``] in
:py:func:`~climpred.prediction.compute_perfect_model` and [``'init'``, ``'member'``]
in :py:func:`~climpred.prediction.compute_hindcast`. The resulting skill is then
reduced by this ``dim``. Therefore, applying a metric over ``dim='member'`` creates a
skill for all initializations individually. This can show the initial conditions
dependence of skill. Likewise when computing skill over ``'init'``, we get skill for
each member. This ``dim`` argument is different from the ``comparison`` argument which
just specifies how ``forecast`` and ``observations`` are defined.

However, this above logic applies to deterministic metrics. Probabilistic metrics need
to be applied to the ``member`` dimension and ``comparison`` from [``'m2c'``, ``'m2m'``]
in :py:func:`~climpred.prediction.compute_perfect_model` and ``'m2o'`` comparison in
:py:func:`~climpred.prediction.compute_hindcast`. Using a probabilistic metric
automatically switches internally to using ``dim='member'``.


User-defined comparisons
########################

You can also construct your own comparisons via the
:py:class:`~climpred.comparisons.Comparison` class.

.. autosummary:: Comparison

First, write your own comparison function, similar to the existing ones. If a
comparison should also be used for probabilistic metrics, make sure that
``metric.probabilistic`` returns ``forecast`` with ``member`` dimension and
``observations`` without. For deterministic metrics, return ``forecast`` and
``observations`` with identical dimensions but without an identical comparison::

  from climpred.comparisons import Comparison, _drop_members

  def _my_m2median_comparison(ds, metric=None):
      """Identical to m2e but median."""
      observations_list = []
      forecast_list = []
      supervector_dim = 'member'
      for m in ds.member.values:
          forecast = _drop_members(ds, rmd_member=[m]).median('member')
          observations = ds.sel(member=m).squeeze()
          forecast_list.append(forecast)
          observations_list.append(observations)
      observations = xr.concat(observations_list, supervector_dim)
      forecast = xr.concat(forecast_list, supervector_dim)
      forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
      observations[supervector_dim] = np.arange(observations[supervector_dim].size)
      return forecast, observations

Then initialize this comparison function with
:py:class:`~climpred.comparisons.Comparison`::

  __my_m2median_comparison = Comparison(
      name='m2me',
      function=_my_m2median_comparison,
      probabilistic=False,
      hindcast=False)

Finally, compute skill based on your own comparison::

  skill = compute_perfect_model(ds, control,
                                metric='rmse',
                                comparison=__my_m2median_comparison)

Once you come up with an useful comparison for your problem, consider contributing this
comparison to ``climpred``, so all users can benefit from your comparison, see
`contributing <contributing.html>`_.


References
##########

.. [Boer2016] Boer, G. J., D. M. Smith, C. Cassou, F. Doblas-Reyes, G. Danabasoglu, B. Kirtman, Y. Kushnir, et al. “The Decadal Climate Prediction Project (DCPP) Contribution to CMIP6.” Geosci. Model Dev. 9, no. 10 (October 25, 2016): 3751–77. https://doi.org/10/f89qdf.

.. [Bushuk2018] Mitchell Bushuk, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. Regional Arctic sea–ice prediction: potential versus operational seasonal forecast skill. Climate Dynamics, June 2018. https://doi.org/10/gd7hfq.

.. [Griffies1997] S. M. Griffies and K. Bryan. A predictability study of simulated North Atlantic multidecadal variability. Climate Dynamics, 13(7-8):459–487, August 1997. https://doi.org/10/ch4kc4.

.. [Seferian2018] Roland Séférian, Sarah Berthet, and Matthieu Chevallier. Assessing the Decadal Predictability of Land and Ocean Carbon Uptake. Geophysical Research Letters, March 2018. https://doi.org/10/gdb424.
