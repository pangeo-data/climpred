Scope of ``climpred``
=====================

``climpred`` aims to be the primary package used to analyze output from initialized dynamical
forecast models, ranging from short-term weather forecasts to decadal climate forecasts. The code
base will be driven entirely by the geoscientific prediction community through open source
development. It leverages ``xarray`` to keep track of core prediction ensemble dimensions
(e.g., ensemble member, initialization date, and lead time) and ``dask`` to perform out-of-memory
computations on large datasets.

The primary goal of ``climpred`` is to offer a comprehensive set of analysis tools for assessing
the forecasts relative to a validation product (e.g., observations, reanalysis products, control
runs, baseline forecasts). This will range from simple deterministic and probabilistic verification
metrics—such as mean absolute error and various skill scores—to more advanced analysis methods,
such as relative entropy and mutual information. ``climpred`` expects users to handle their
domain-specific post-processing of model output, so that the package can focus on the actual
analysis of forecasts.

Finally, the ``climpred`` documentation will serve as a repository of unified analysis methods
through jupyter notebook examples, and will also collect relevant references and literature.
