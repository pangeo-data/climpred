climpred: verification of weather and climate forecasts
=======================================================

..
    Table version of badges inspired by pySTEPS.

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs|
    * - tests
      - |travis| |requires| |codecov|
    * - package
      - |conda| |pypi|
    * - license
      - |license|
    * - community
      - |gitter| |contributors| |downloads|
    * - tutorials
      - |gallery| |tutorial|

.. |docs| image:: https://img.shields.io/readthedocs/climpred/stable.svg?style=flat
    :target: https://climpred.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/pangeo-data/climpred.svg?branch=master
    :target: https://travis-ci.org/pangeo-data/climpred

.. |requires| image:: https://requires.io/github/pangeo-data/climpred/requirements.svg?branch=master
     :target: https://requires.io/github/pangeo-data/climpred/requirements/?branch=master
     :alt: Requirements Status

.. |codecov| image:: https://codecov.io/gh/pangeo-data/climpred/branch/master/graph/badge.svg?token=e53kXaaOqS
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

.. |gallery| image:: https://img.shields.io/badge/climpred-example_gallery-F5A252.svg
    :alt: climpred gallery
    :target: https://climpred.readthedocs.io/en/stable/examples.html

.. |tutorial| image:: https://img.shields.io/badge/climpred-tutorial-f5a252
    :alt: climpred workshop
    :target: https://mybinder.org/v2/gh/bradyrx/climpred_workshop/master

Version 2.0.0 Release
=====================

**We now support sub-annual (e.g., seasonal, monthly, weekly, daily) forecasts**.
We provide a host of deterministic and probabilistic metrics_. We support both
perfect-model and hindcast-based prediction ensembles, and provide
:py:class:`~climpred.classes.PerfectModelEnsemble` and
:py:class:`~climpred.classes.HindcastEnsemble` classes to make analysis easier.

See `quick start <quick-start.html>`_ and our `examples <examples.html>`_ to get started.

.. _metrics: metrics.html

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
