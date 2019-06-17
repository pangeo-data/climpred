***********
Comparisons
***********

Forecast skill is always evaluated against a reference for verification. In ESM-based predictions, it is common to compare the ensemble mean forecast against the reference.

In hindcast ensembles :py:func:`climpred.prediction.compute_hindcast`, this ensemble mean forecast (``comparison='e2r'``) is expected to perform better than individual ensemble members (``comparison='m2r'``) as the chaotic component of forecasts is expected to be suppressed by this averaging, while the memory of the system sustains. [Boer2016]_
HindcastEnsemble skill is computed by default as the ensemble mean forecast against the reference (``comparison='e2r'``).

In perfect-model frameworks :py:func:`climpred.prediction.compute_perfect_model`, there are even more ways of comparisons. [Seferian2018]_ shows comparison of the ensemble members against the control run (``comparison='m2c'``) and ensemble members against all other ensemble members (``comparison='m2m'``). Furthermore, using the ensemble mean forecast can be also verified against one control member (``comparison='e2c'``) or all members (``comparison='m2e'``) as done in [Griffies1997]_.
Perfect-model framework comparison defaults to the ensemble mean forecast verified against each member in turns (``comparison='m2e'``).

These different comparisons demand for a normalization factor to arrive at a normalized skill of 1, when skill saturation is reached (ref: metrics).

While HindcastEnsemble skill is computed over all initializations ``init`` of the hindcast, the resulting skill is a mean forecast skill over all initializations.
PerfectModelEnsemble skill is computed over a supervector comprised of all initializations and members, which allows the computation of the ACC-based skill [Bushuk2018]_, but also returns a mean forecast skill over all initializations.
The supervector approach shown in [Bushuk2018]_ and just calculating a distance-based metric like ``rmse`` over the member dimension as in [Griffies1997]_ yield very similar results.

HindcastEnsemble
################

.. currentmodule:: climpred.comparisons

.. autosummary:: _e2r
.. autosummary:: _m2r


PerfectModelEnsemble
####################

.. autosummary:: _m2e
.. autosummary:: _m2c
.. autosummary:: _m2m
.. autosummary:: _e2c


References
##########

.. [Boer2016] Boer, G. J., D. M. Smith, C. Cassou, F. Doblas-Reyes, G. Danabasoglu, B. Kirtman, Y. Kushnir, et al. “The Decadal Climate Prediction Project (DCPP) Contribution to CMIP6.” Geosci. Model Dev. 9, no. 10 (October 25, 2016): 3751–77. https://doi.org/10/f89qdf.

.. [Bushuk2018] Mitchell Bushuk, Rym Msadek, Michael Winton, Gabriel Vecchi, Xiaosong Yang, Anthony Rosati, and Rich Gudgel. Regional Arctic sea–ice prediction: potential versus operational seasonal forecast skill. Climate Dynamics, June 2018. https://doi.org/10/gd7hfq.

.. [Griffies1997] S. M. Griffies and K. Bryan. A predictability study of simulated North Atlantic multidecadal variability. Climate Dynamics, 13(7-8):459–487, August 1997. https://doi.org/10/ch4kc4.

.. [Seferian2018] Roland Séférian, Sarah Berthet, and Matthieu Chevallier. Assessing the Decadal Predictability of Land and Ocean Carbon Uptake. Geophysical Research Letters, March 2018. https://doi.org/10/gdb424.
