***********
Comparisons
***********

Forecast skill is always evaluated against a reference for verification. In ESM-based predictions, it is common to compare the ensemble mean forecast against the reference.

In hindcast ensembles :py:func:`~climpred.prediction.compute_hindcast`, this ensemble mean forecast (``comparison='e2r'``) is expected to perform better than individual ensemble members (``comparison='m2r'``) as the chaotic component of forecasts is expected to be suppressed by this averaging, while the memory of the system sustains. [Boer2016]_
:py:class:`~climpred.classes.HindcastEnsemble` skill is computed by default as the ensemble mean forecast against the reference (``comparison='e2r'``).

In perfect-model frameworks :py:func:`~climpred.prediction.compute_perfect_model`, there are even more ways of comparisons. [Seferian2018]_ shows comparison of the ensemble members against the control run (``comparison='m2c'``) and ensemble members against all other ensemble members (``comparison='m2m'``). Furthermore, using the ensemble mean forecast can be also verified against one control member (``comparison='e2c'``) or all members (``comparison='m2e'``) as done in [Griffies1997]_.
Perfect-model framework comparison defaults to the ensemble mean forecast verified against each member in turns (``comparison='m2e'``).

These different comparisons demand for a normalization factor to arrive at a normalized skill of 1, when skill saturation is reached (ref: metrics).

While HindcastEnsemble skill is computed over all initializations ``init`` of the hindcast, the resulting skill is a mean forecast skill over all initializations.
PerfectModelEnsemble skill is computed over a supervector comprised of all initializations and members, which allows the computation of the ACC-based skill [Bushuk2018]_, but also returns a mean forecast skill over all initializations.
The supervector approach shown in [Bushuk2018]_ and just calculating a distance-based metric like ``rmse`` over the member dimension as in [Griffies1997]_ yield very similar results.


Compute over dimension
######################

The optional argument ``dim`` defines over which dimension a metric is computed. We can apply a metric over ``dim`` from [``'init'``, ``'member'``, ``['member', 'init']``] in :py:func:`~climpred.prediction.compute_perfect_model` and [``'init'``, ``'member'``] in :py:func:`~climpred.prediction.compute_hindcast`. The resulting skill is then reduced by this ``dim``. Therefore, applying a metric over ``dim='member'`` creates a skill for all initializations individually. This can show the initial conditions dependence of skill. Likewise when computing skill over ``'init'``, we get skill for each member. This ``dim`` argument is different from the ``comparison`` argument which just specifies how ``forecast`` and ``reference`` are defined.
However, this above logic applies to deterministic metrics. Probabilistic metrics need to be applied to the ``member`` dimension and ``comparison`` from [``'m2c'``, ``'m2m'``] in :py:func:`~climpred.prediction.compute_perfect_model` and ``'m2r'`` comparison in :py:func:`~climpred.prediction.compute_hindcast`. Using a probabilistic metric automatically switches internally to using ``dim='member'``.

HindcastEnsemble
################

``keyword: 'e2r'``

.. currentmodule:: climpred.comparisons

.. autosummary:: _e2r


``keyword: 'm2r'``

.. autosummary:: _m2r


PerfectModelEnsemble
####################

``keyword: 'm2e'``

.. autosummary:: _m2e

``keyword: 'm2c'``

.. autosummary:: _m2c

``keyword: 'm2m'``

.. autosummary:: _m2m

``keyword: 'e2c'``

.. autosummary:: _e2c


User-defined comparisons
########################

You can also construct your own comparisons via the :py:class:`~climpred.comparisons.Comparison` class.

.. autosummary:: Comparison

First, write your own comparison function, similar to the existing ones. If a comparison should also be used for probabilistic metrics, use ``stack_dims`` to return ``forecast`` with ``member`` dimension and ``reference`` without. For deterministic metric, return ``forecast`` and ``reference`` with identical dimensions::

  from climpred.comparisons import Comparison, _drop_members

  def _my_m2median_comparison(ds, stack_dims=True):
      """Identical to m2e but median."""
      reference_list = []
      forecast_list = []
      supervector_dim = 'member'
      for m in ds.member.values:
          forecast = _drop_members(ds, rmd_member=[m]).median('member')
          reference = ds.sel(member=m).squeeze()
          forecast_list.append(forecast)
          reference_list.append(reference)
      reference = xr.concat(reference_list, supervector_dim)
      forecast = xr.concat(forecast_list, supervector_dim)
      forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
      reference[supervector_dim] = np.arange(reference[supervector_dim].size)
      return forecast, reference

Then initialize this comparison function with :py:class:`~climpred.comparisons.Comparison`::

  __my_m2median_comparison = Comparison(
      name='m2me',
      function=_my_m2median_comparison,
      probabilistic=False,
      hindcast=False)

Finally, compute skill based on your own comparison::

  skill = compute_perfect_model(ds, control, metric='rmse', comparison=__my_m2median_comparison)

Once you come up with an useful comparison for your problem, consider contributing this comparison to ``climpred``, so all users can benefit from your comparison, see `contributing <contributing.html>`_.


References
##########

.. [Boer2016] Boer, G. J., D. M. Smith, C. Cassou, F. Doblas-Reyes, G. Danabasoglu, B. Kirtman, Y. Kushnir, et al. “The Decadal Climate Prediction Project (DCPP) Contribution to CMIP6.” Geosci. Model Dev. 9, no. 10 (October 25, 2016): 3751–77. https://doi.org/10/f89qdf.

.. [Bushuk2018] Mitchell Bushuk, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. Regional Arctic sea–ice prediction: potential versus operational seasonal forecast skill. Climate Dynamics, June 2018. https://doi.org/10/gd7hfq.

.. [Griffies1997] S. M. Griffies and K. Bryan. A predictability study of simulated North Atlantic multidecadal variability. Climate Dynamics, 13(7-8):459–487, August 1997. https://doi.org/10/ch4kc4.

.. [Seferian2018] Roland Séférian, Sarah Berthet, and Matthieu Chevallier. Assessing the Decadal Predictability of Land and Ocean Carbon Uptake. Geophysical Research Letters, March 2018. https://doi.org/10/gdb424.
