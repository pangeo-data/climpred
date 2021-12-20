***********
Comparisons
***********

Forecasts have to be verified against some product to evaluate their performance.
However, when verifying against a product, there are many different ways one can
compare the ensemble of forecasts. Here, we cover the comparison options for both
:py:class:`.HindcastEnsemble` and
:py:class:`.PerfectModelEnsemble`.
See `terminology <terminology.html>`__ for clarification on the differences between
these two experimental setups.

All high-level functions like :py:meth:`.HindcastEnsemble.verify`,
:py:meth:`.HindcastEnsemble.bootstrap`, :py:meth:`.PerfectModelEnsemble.verify` and
:py:meth:`.PerfectModelEnsemble.bootstrap` take a ``comparison`` keyword to select the
comparison style. See below for a detailed description on the differences between these
comparisons.

Hindcast Ensembles
##################

In :py:class:`.HindcastEnsemble`, the ensemble mean forecast
(``comparison="e2o"``) is expected to perform better than individual ensemble members
(``comparison="m2o"``) as the chaotic component of forecasts is expected to be
suppressed by this averaging, while the memory of the system sustains. :cite:p:`Boer2016`

.. currentmodule:: climpred.comparisons

``keyword: "e2o", "e2r"``

.. autosummary:: _e2o

``keyword: "m2o", "m2r"``

.. autosummary:: _m2o


Perfect Model Ensembles
#######################

In :py:class:`.PerfectModelEnsemble`, there are many more ways of
verifying forecasts. :cite:t:`Seferian2018` uses a comparison of all ensemble members against
the control run (``comparison="m2c"``) and all ensemble members against all other
ensemble members (``comparison="m2m"``). Furthermore, the ensemble mean forecast can
be verified against one control member (``comparison="e2c"``) or all members
(``comparison="m2e"``) as done in :cite:t:`Griffies1997`.

``keyword: "m2e"``

.. autosummary:: _m2e

``keyword: "m2c"``

.. autosummary:: _m2c

``keyword: "m2m"``

.. autosummary:: _m2m

``keyword: "e2c"``

.. autosummary:: _e2c


Normalization
#############

The goal of a normalized distance metric is to get a constant or comparable value of
typically ``1`` (or ``0`` for metrics defined as ``1 - metric``) when the metric
saturates and the predictability horizon is reached (see `metrics <metrics.html>`__).

A factor is added in the normalized metric formula :cite:p:`Seferian2018` to accomodate
different comparison styles. For example, ``metric="nrmse"`` gets smaller in
comparison ``"m2e"``.
than ``"m2m"`` by design, since the ensembe mean is always closer to individual members
than the ensemble members to each other. In turn, the normalization factor is ``2`` for
comparisons ``"m2c"``, ``"m2m"``, and ``"m2o"``. It is 1 for ``"m2e"``, ``"e2c"``, and
``"e2o"``.

Interpretation of Results
#########################

When :py:class:`.HindcastEnsemble` skill is computed over all
initializations ``dim="init"`` of the hindcast, the resulting skill is a mean forecast
skill over all initializations.

:py:class:`.PerfectModelEnsemble` skill is computed over a
supervector comprised of all
initializations and members, which allows the computation of the ACC-based skill
:cite:p:`Bushuk2018`, but also returns a mean forecast skill over all initializations.

The supervector approach shown in :cite:t:`Bushuk2018` and just calculating a distance-based
metric like ``rmse`` over the member dimension as in :cite:t:`Griffies1997` yield very similar
results.

Compute over dimension
######################

The argument ``dim`` defines over which dimension a metric is computed. We can
apply a metric over all dimensions from the ``initialized`` dataset expect ``lead``.
The resulting skill is then
reduced by this ``dim``. Therefore, applying a metric over ``dim="member"`` or
``dim=[]`` creates a skill for all initializations individually.
This can show the initial conditions dependence of skill.
Likewise when computing skill over ``"init"``, we get skill for each member.
This ``dim`` argument is different from the ``comparison`` argument which
just specifies how ``forecast`` and ``observations`` are defined.

However, this above logic applies to deterministic metrics. Probabilistic metrics need
to be applied to the ``member`` dimension and ``comparison`` from
``["m2c", "m2m"]`` in :py:meth:`.PerfectModelEnsemble.verify` and ``"m2o"`` comparison
in :py:meth:`.HindcastEnsemble.verify`.

``dim`` should not contain ``member`` when the comparison already computes ensemble
means as in ``["e2o", "e2c"]``.


User-defined comparisons
########################

You can also construct your own comparisons via the
:py:class:`climpred.comparisons.Comparison` class.

.. autosummary:: Comparison

First, write your own comparison function, similar to the existing ones. If a
comparison should also be used for probabilistic metrics, make sure that probabilistic
metrics returns ``forecast`` with ``member`` dimension and
``observations`` without. For deterministic metrics, return ``forecast`` and
``observations`` with identical dimensions but without an identical comparison::

  from climpred.comparisons import Comparison, M2M_MEMBER_DIM

  def _my_m2median_comparison(initialized, metric=None):
      """Identical to m2e but median."""
      observations_list = []
      forecast_list = []
      supervector_dim = "member"
      for m in initialized.member.values:
          forecast = initialized.drop_sel(member=m).median("member")
          observations = initialized.sel(member=m).squeeze()
          forecast_list.append(forecast)
          observations_list.append(observations)
      observations = xr.concat(observations_list, M2M_MEMBER_DIM)
      forecast = xr.concat(forecast_list, M2M_MEMBER_DIM)
      forecast[M2M_MEMBER_DIM] = np.arange(forecast[M2M_MEMBER_DIM].size)
      observations[M2M_MEMBER_DIM] = np.arange(observations[M2M_MEMBER_DIM].size)
      return forecast, observations

Then initialize this comparison function with
:py:class:`climpred.comparisons.Comparison`::

  __my_m2median_comparison = Comparison(
      name="m2me",
      function=_my_m2median_comparison,
      probabilistic=False,
      hindcast=False)

Finally, compute skill based on your own comparison::

  PerfectModelEnsemble.verify(
    metric="rmse",
    comparison=__my_m2median_comparison,
    dim=[],
  )

Once you come up with an useful comparison for your problem, consider contributing this
comparison to ``climpred``, so all users can benefit from your comparison, see
`contributing <contributing.html>`_.


References
##########

.. bibliography::
  :filter: docname in docnames
