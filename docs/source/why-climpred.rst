Overview: Why climpred?
=======================

There are many packages out there related to computing metrics on initialized
geoscience predictions. However, we didn't find any one package that unified all our
needs.

Output from earth system prediction hindcast (also called re-forecast) experiments is
difficult to work with. A typical output file could contain the dimensions
``initialization``, ``lead time``, ``ensemble member``, ``latitude``, ``longitude``,
``depth``. ``climpred`` leverages the labeled dimensions of ``xarray`` to handle the
headache of bookkeeping for you. We offer :py:class:`.HindcastEnsemble` and
:py:class:`.PerfectModelEnsemble` objects that carry products to verify against (e.g.,
control runs, reconstructions, uninitialized ensembles) along with your initialized
prediction output.

When computing lead-dependent skill scores, ``climpred`` handles all of the
``init+lead-valid_time``-matching for you, properly aligning the multiple
``time`` dimensions between the hindcast and verification datasets.
We offer a suite of vectorized `deterministic <metrics.html#deterministic>`_
and `probabilistic <metrics.html#probabilistic>`_ metrics that can be applied to time
series and grids. It's as easy as concatenating your initialized prediction output into
one :py:class:`xarray.Dataset` and running the :py:meth:`.HindcastEnsemble.verify`
command:

.. :: python

>>> HindcastEnsemble.verify(
...     metric="rmse", comparison="e2o", dim="init", alignment="maximize"
... )
