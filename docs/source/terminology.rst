**********************
Prediction Terminology
**********************

Terminology is often confusing and highly variable amongst those that make predictions in the geoscience community. Here we define some common terms in climate prediction and how we use them in ``climpred``.

Simulation Design
#################

Initialized Ensemble:
=====================

*Perfect Model Experiment*: ``m`` ensemble members are initialized from a control simulation at ``n`` randomly chosen initialization dates and integrated for ``l`` lead years [Griffies1997]_ (:py:class:`PerfectModelEnsemble`).

*Hindcast Ensemble*: ``m`` ensemble members are initialized from a reference simulation (generally a reconstruction from reanalysis) at ``n`` initialization dates and integrated for ``l`` lead years [Boer2016]_ (:py:class:`HindcastEnsemble`).


Uninitialized Ensemble:
=======================

*Uninitialized Ensemble*: Ensemble members do not have common initial conditions:

- many members of historical simulations like in CMIP6 for :py:class:`HindcastEnsemble`.
- MPI-ESM Grand Ensemble or CESM Large Ensemble for :py:class:`HindcastEnsemble`.
- randomly concatenated time chunks of control simulation for :py:class:`PerfectModelEnsemble`.



Reconstruction:
===============

*Reconstruction/Assimilation*: Nudging an ESM simulation towards reanalysis.


Predictability vs. Prediction skill
###################################

*(Potential) Predictability*: This characterizes the "ability to be predicted" rather than the current "ability to predict." One acquires this by computing a metric (like the anomaly correlation coefficient (ACC)) between the prediction ensemble and a verification member (in a perfect-model setup) or the reconstruction that initialized it (in a hindcast setup) [Meehl2013]_.

*(Prediction) Skill*: This characterizes the current ability of the ensemble forecasting system to predict the real world. This is derived by computing the ACC between the prediction ensemble and observations of the real world [Meehl2013]_.

Forecasting
###########

*Hindcast*: Retrospective forecasts of the past initialized from a reconstruction integrated under external forcing [Boer2016]_.

*Prediction*: Forecasts initialized from a reconstruction integrated into the future with external forcing [Boer2016]_.

*Projection* An estimate of the future climate that is dependent on the externally forced climate response, such as anthropogenic greenhouse gases, aerosols, and volcanic eruptions [Meehl2013]_.


References
##########

.. [Griffies1997] Griffies, S. M., and K. Bryan. “A Predictability Study of Simulated North Atlantic Multidecadal Variability.” Climate Dynamics 13, no. 7–8 (August 1, 1997): 459–87. https://doi.org/10/ch4kc4

.. [Boer2016] Boer, G. J., Smith, D. M., Cassou, C., Doblas-Reyes, F., Danabasoglu, G., Kirtman, B., Kushnir, Y., Kimoto, M., Meehl, G. A., Msadek, R., Mueller, W. A., Taylor, K. E., Zwiers, F., Rixen, M., Ruprich-Robert, Y., and Eade, R.: The Decadal Climate Prediction Project (DCPP) contribution to CMIP6, Geosci. Model Dev., 9, 3751-3777, https://doi.org/10.5194/gmd-9-3751-2016, 2016.

.. [Meehl2013] Meehl, G. A., Goddard, L., Boer, G., Burgman, R., Branstator, G., Cassou, C., ... & Karspeck, A. (2014). Decadal climate prediction: an update from the trenches. Bulletin of the American Meteorological Society, 95(2), 243-267. https://doi.org/10.1175/BAMS-D-12-00241.1.
