climpred: verification of weather and climate forecasts
=======================================================

..
    Table version of badges inspired by pySTEPS.

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs| |joss|
    * - tests
      - |ci| |requires| |codecov|
    * - package
      - |conda| |pypi|
    * - license
      - |license|
    * - community
      - |gitter| |contributors| |downloads|
    * - tutorials
      - |gallery| |workshop| |cloud|

.. |docs| image:: https://img.shields.io/readthedocs/climpred/stable.svg?style=flat
    :target: https://climpred.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. |joss| image:: https://joss.theoj.org/papers/246d440e3fcb19025a3b0e56e1af54ef/status.svg
    :target: https://joss.theoj.org/papers/246d440e3fcb19025a3b0e56e1af54ef

.. |ci|  image:: https://github.com/pangeo-data/climpred/workflows/climpred%20testing/badge.svg

.. |requires| image:: https://requires.io/github/pangeo-data/climpred/requirements.svg?branch=master
     :target: https://requires.io/github/pangeo-data/climpred/requirements/?branch=master
     :alt: Requirements Status

.. |codecov| image:: https://codecov.io/gh/pangeo-data/climpred/branch/master/graph/badge.svg
      :target: https://codecov.io/gh/pangeo-data/climpred

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/climpred.svg
    :target: https://anaconda.org/conda-forge/climpred
    :alt: Conda Version

.. |pypi| image:: https://img.shields.io/pypi/v/climpred.svg
   :target: https://pypi.python.org/pypi/climpred/

.. |license| image:: https://img.shields.io/github/license/pangeo-data/climpred.svg
    :alt: license
    :target: ../../LICENSE.txt

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/climpred

.. |contributors| image:: https://img.shields.io/github/contributors/pangeo-data/climpred
    :alt: GitHub contributors
    :target: https://github.com/pangeo-data/climpred/graphs/contributors

.. |downloads| image:: https://img.shields.io/conda/dn/conda-forge/climpred
    :alt: Conda downloads
    :target: https://anaconda.org/conda-forge/climpred

.. |gallery| image:: https://img.shields.io/badge/climpred-example_gallery-ed7b0e.svg
    :alt: climpred gallery
    :target: https://climpred.readthedocs.io/en/stable/examples.html

.. |workshop| image:: https://img.shields.io/badge/climpred-workshop-f5a252
    :alt: climpred workshop
    :target: https://mybinder.org/v2/gh/bradyrx/climpred_workshop/master

.. |cloud| image:: https://img.shields.io/badge/climpred-cloud_demo-f9c99a
    :alt: climpred cloud demo
    :target: https://github.com/aaronspring/climpred-cloud-demo

Version 2.1.1 Release
=====================

*October 13th, 2020*

The most recent release adds a few new features along with a few deprecations. We want
users to think about methodology with every call of
:py:meth:`~climpred.classes.HindcastEnsemble.verify`, so we now require explicit
keywords for ``metric``, ``comparison``, ``dim``, and ``alignment``. We also require
the explicit definition of ``iterations`` for
:py:meth:`~climpred.classes.HindcastEnsemble.bootstrap`.

We've added a few new features as well (see key additions below). For a complete list,
please see the `changelog <changelog.html>`__.

* An early implementation of `bias correction <bias_removal.html>`__.
* Spatial dimensions can now be used in metric calls, e.g. for pattern correlation.
* New metrics have been added from ``xskillscore``, which are mostly based on the `Contingency
  table <metrics.html#contingency-based-metrics>`__. We have also
  added additional `probability metrics <metrics.html#probabilistic>`__:
  the ranked probability score, reliability, discrimination, and ranked histogram.
* Math operations can be used between :py:class:`~climpred.classes.PredictionEnsemble` objects
  (see `example here <prediction-ensemble-object.html#Arithmetic-Operations-with-PredictionEnsemble-Objects>`__).
* Users can now quick plot their prediction system (if there are no spatial dimensions) with
  :py:meth:`~climpred.classes.HindcastEnsemble.plot`. See an example of this in the
  `quick start <quick-start.html>`__.

Installation
============

You can install the latest release of ``climpred`` using ``pip`` or ``conda``:

.. code-block:: bash

    pip install climpred

.. code-block:: bash

    conda install -c conda-forge climpred

You can also install the bleeding edge (pre-release versions) by cloning this
repository and running ``pip install . --upgrade`` in the main directory

**Getting Started**


* :doc:`why-climpred`
* :doc:`scope`
* :doc:`quick-start`
* :doc:`examples`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    why-climpred
    scope
    quick-start.ipynb
    examples

**User Guide**

* :doc:`setting-up-data`
* :doc:`prediction-ensemble-object`
* :doc:`alignment`
* :doc:`metrics`
* :doc:`comparisons`
* :doc:`significance`
* :doc:`bias_removal`
* :doc:`smoothing`
* :doc:`terminology`
* :doc:`reference_forecast`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    setting-up-data
    prediction-ensemble-object.ipynb
    alignment.ipynb
    metrics
    comparisons
    significance
    bias_removal.ipynb
    smoothing
    terminology
    reference_forecast

**Help & Reference**

* :doc:`api`
* :doc:`changelog`
* :doc:`code_of_conduct`
* :doc:`contributing`
* :doc:`contributors`
* :doc:`helpful-links`
* :doc:`publications`
* :doc:`related-packages`
* :doc:`release_procedure`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & Reference

    api
    changelog
    code_of_conduct
    contributing
    contributors
    helpful-links
    publications
    related-packages
    release_procedure
