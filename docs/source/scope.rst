Scope of ``climpred``
=====================

``climpred`` aims to be the primary package used to analyze output from initialized
dynamical forecast models, ranging from short-term weather forecasts to decadal climate
forecasts. The code base is driven by the geoscientific prediction community through
open source development. It leverages `xarray <http://xarray.pydata.org/en/stable/>`_
to keep track of core prediction ensemble dimensions (e.g., ensemble member,
initialization date, and lead time) and `dask <https://dask.org/>`_ to perform
out-of-memory computations on large datasets.

The primary goal of ``climpred`` is to offer a comprehensive set of analysis tools for
assessing the forecasts relative to a validation product (e.g., observations,
reanalysis products, control simulations, baseline forecasts). This ranges from simple
deterministic and probabilistic verification `metrics <metrics.html>`_ — such as, e.g.
mean absolute error or rank histogram — to more advanced contingency table-derived
metrics. ``climpred`` expects users to handle their domain-specific post-processing of
model output, so that the package can focus on the actual analysis of forecasts.

Finally, the ``climpred`` documentation will serve as a repository of unified analysis
methods through `jupyter <https://jupyter.org/>`_ notebook `examples <examples.html>`_,
and collects relevant references and literature.
