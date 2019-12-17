Overview: Why climpred?
=======================

There are many packages out there related to computing metrics on initialized geoscience predictions. However, we didn't find any one package that unified all our needs.

Output from decadal climate prediction experiments is difficult to work with. A typical output file could contain the dimensions ``initialization``, ``lead time``, ``ensemble member``, ``latitude``, ``longitude``, ``depth``. ``climpred`` leverages the labeled dimensions of ``xarray`` to handle the headache of bookkeeping for you. We offer :py:class:`~climpred.classes.HindcastEnsemble` and :py:class:`~climpred.classes.PerfectModelEnsemble` objects that carry references (e.g., control runs, reconstructions, uninitialized ensembles) along with your decadal prediction output.

When computing lead-dependent skill scores, ``climpred`` handles all of the lag-correlating for you. We offer a suite of vectorized deterministic and probabilistic metrics that can be applied to time series and grids. It's as easy as adding your decadal prediction output to an object and running compute: ``HindcastEnsemble.compute_metric(metric='rmse')``.
