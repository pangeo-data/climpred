climpred: analysis of ensemble forecast models for climate prediction
=====================================================================

.. image:: https://travis-ci.org/bradyrx/climpred.svg?branch=master
    :target: https://travis-ci.org/bradyrx/climpred

.. image:: https://img.shields.io/pypi/v/climpred.svg
   :target: https://pypi.python.org/pypi/climpred/

.. image:: https://img.shields.io/conda/vn/conda-forge/climpred.svg
    :target: https://anaconda.org/conda-forge/climpred
    :alt: Conda Version

.. image:: https://coveralls.io/repos/github/bradyrx/climpred/badge.svg?branch=master
    :target: https://coveralls.io/github/bradyrx/climpred?branch=master

.. image:: https://img.shields.io/readthedocs/climpred/stable.svg?style=flat
    :target: https://climpred.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/climpred

.. image:: https://img.shields.io/github/license/bradyrx/climpred.svg
    :alt: license
    :target: ../../LICENSE.txt

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
* :doc:`comparisons`
* :doc:`metrics`
* :doc:`terminology`
* :doc:`baselines`

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    setting-up-data
    prediction-ensemble-object.ipynb
    metrics
    comparisons
    terminology
    baselines

**Help & Reference**

* :doc:`api`
* :doc:`changelog`
* :doc:`helpful-links`
* :doc:`publications`
* :doc:`contributing`
* :doc:`release_procedure`
* :doc:`contributors`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & Reference

    api
    changelog
    helpful-links
    publications
    contributing
    release_procedure
    contributors
