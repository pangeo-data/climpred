climpred: analysis of ensemble forecast models for climate prediction
=====================================================================

.. image:: https://travis-ci.org/bradyrx/climpred.svg?branch=master)](https://travis-ci.org/bradyrx/climpred
    :target: https://travis-ci.org/bradyrx/climpred

.. image:: https://api.codacy.com/project/badge/Grade/a532752e9e814c6e895694463f307cd9
    :target: https://www.codacy.com/app/bradyrx/climpred?utm_source=github.com&utm_medium=referral&utm_content=bradyrx/climpred&utm_campaign=Badge_Grade

.. image:: https://coveralls.io/repos/github/bradyrx/climpred/badge.svg?branch=master
    :target: https://coveralls.io/github/bradyrx/climpred?branch=master

.. image:: https://img.shields.io/readthedocs/climpred/latest.svg?style=flat
    :target: https://climpred.readthedocs.io/en/latest/?badge=latest

.. image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/climpred

Release
=======

We are diligently working on our v1 release. The goal for v1 is to have ``climpred`` ready to take in subseasonal-to-seasonal and seasonal-to-decadal prediction ensembles and to perform determinstic skill metrics on them. We will have early-stage objects (``HindcastEnsemble`` and ``PerfectModelEnsemble``) to make computations easier on the user. We will also have documentation ready so the capabilities of ``climpred`` are more clear.

In the meantime, you can install the package following the steps below and reference the notebooks for guidance. Please raise any issues if you encounter any bugs or have any ideas; you can also raise a PR to add new features. Feel free to contact us if you have questions.

Installation
============

.. code-block:: bash

    pip install git+https://github.com/bradyrx/climpred

**Getting Started**


* :doc:`why-climpred`
* :doc:`quick-start`
* :doc:`sample-data`
* :doc:`examples`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    why-climpred
    quick-start
    sample-data.ipynb
    examples

**User Guide**

* :doc:`setting-up-data`
* :doc:`comparisons`
* :doc:`metrics`
* :doc:`terminology`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User Guide

    setting-up-data
    metrics
    comparisons
    terminology

**Help & Reference**

* :doc:`api`
* :doc:`contributing`
* :doc:`changelog`
* :doc:`contributors`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & Reference

    api
    contributing
    changelog
    contributors
