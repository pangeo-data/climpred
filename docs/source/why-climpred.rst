Overview: Why climpred?
=======================

There are many packages out there related to computing metrics on initialized geoscience predictions. However, we didn't find any one package that unified all our needs.

``climpred`` is built on top of ``xarray`` and introduces easy-to-use ``PredictionEnsemble`` objects. These objects track the prediction ensemble and all associated products with it (e.g., a control simulation, a reconstruction, observations). After attaching these products to the ``PredictionEnsemble``, one can compute a given metric across many variables and products at once: ``PredictionEnsemble.compute(metric='mae')``.
