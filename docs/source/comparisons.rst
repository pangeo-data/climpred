***********
Comparisons
***********

Forecast skill is always evaluated against a reference for verification. In ESM-based predictions, it is common to compare the ensemble mean forecast against the reference. This ensemble mean forecast (``comparison=e2r``) is expected to perform better than individual ensemble members (``comparison=m2r``) as the chaotic component of forecasts is expected to be suppressed by this averaging, while the memory of the system sustains. Todo: add citations to this
HindcastEnsemble skill is computed by default as the ensemble mean forecast against the reference (``comparison=e2r``).

In perfect-model frameworks, there are even more ways of comparisons. :cite:`Seferian2018` shows comparison of the ensemble members against the control run (``comparison=m2c``) and ensemble members against all other ensemble members (``comparison=m2m``). Furthermore, using the ensemble mean forecast can be also verified against one control member (``comparison=e2c``) or all members (``comparison=m2e``) as done in :cite:`Griffies1997`.
Perfect-model framework comparison defaults to the ensemble mean forecast verified against each member in turns (``comparison=m2e``).

These different comparisons demand for a normalization factor to arrive at a normalized skill of 1, when skill saturation is reached (ref: metrics).

While HindcastEnsemble skill is computed over all `initializations` ``init`` of the hindcast, the resulting skill is a mean forecast skill over all initializations.
PerfectModelEnsemble skill is computed over a supervector comprised of all initializations and members, which allows the computation of the ACC-based skill :cite:`Bushuk2018`, but also returns a mean forecast skill over all initializations.
The supervector approach shown in :cite:`Bushuk2018` and just calculating a distance-based metric like ``rmse`` over the member dimension as in :cite:`Griffies1997` yield very similar results.

HindcastEnsemble
################

.. currentmodule:: climpred.comparisons

.. autofunction:: _e2r
.. autofunction:: _m2r


PerfectModelEnsemble
####################

.. autofunction:: _m2e
.. autofunction:: _m2c
.. autofunction:: _m2m
.. autofunction:: _e2c

.. bibliography:: refs.bib
