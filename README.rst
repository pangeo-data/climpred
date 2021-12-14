.. image:: https://i.imgur.com/HPOdOsR.png

Verification of weather and climate forecasts.

..
    Table version of badges inspired by pySTEPS.

.. list-table::
    :stub-columns: 1
    :widths: 10 90

    * - docs
      - |docs| |joss| |doi|
    * - tests
      - |ci| |upstream| |codecov|
    * - package
      - |conda| |conda downloads| |pypi| |pypi downloads|
    * - license
      - |license|
    * - community
      - |gitter| |contributors| |forks| |stars| |issues| |PRs|
    * - tutorials
      - |gallery| |workshop| |cloud|

.. |docs| image:: https://img.shields.io/readthedocs/climpred/stable.svg?style=flat
    :target: https://climpred.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

.. |joss| image:: https://joss.theoj.org/papers/246d440e3fcb19025a3b0e56e1af54ef/status.svg
    :target: https://joss.theoj.org/papers/246d440e3fcb19025a3b0e56e1af54ef
    :alt: JOSS paper

.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4556085.svg
    :target: https://doi.org/10.5281/zenodo.4556085
    :alt: DOI

.. |ci| image:: https://github.com/pangeo-data/climpred/workflows/climpred%20testing/badge.svg
    :target: https://github.com/pangeo-data/climpred/actions/workflows/climpred_testing.yml
    :alt: CI

.. |upstream| image:: https://github.com/pangeo-data/climpred/actions/workflows/upstream-dev-ci.yml/badge.svg
    :target: https://github.com/pangeo-data/climpred/actions/workflows/upstream-dev-ci.yml
    :alt: CI upstream

.. |codecov| image:: https://codecov.io/gh/pangeo-data/climpred/branch/main/graph/badge.svg
      :target: https://codecov.io/gh/pangeo-data/climpred
      :alt: coverage

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/climpred.svg
    :target: https://anaconda.org/conda-forge/climpred
    :alt: Conda Version

.. |pypi| image:: https://img.shields.io/pypi/v/climpred.svg
   :target: https://pypi.python.org/pypi/climpred/
   :alt: pypi Version

.. |license| image:: https://img.shields.io/github/license/pangeo-data/climpred.svg
    :alt: license
    :target: LICENSE.txt

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
    :target: https://gitter.im/climpred
    :alt: gitter chat

.. |contributors| image:: https://img.shields.io/github/contributors/pangeo-data/climpred
    :alt: GitHub contributors
    :target: https://github.com/pangeo-data/climpred/graphs/contributors

.. |conda downloads| image:: https://img.shields.io/conda/dn/conda-forge/climpred
    :alt: Conda downloads
    :target: https://anaconda.org/conda-forge/climpred

.. |pypi downloads| image:: https://pepy.tech/badge/climpred
    :alt: pypi downloads
    :target: https://pepy.tech/project/climpred

.. |gallery| image:: https://img.shields.io/badge/climpred-examples-ed7b0e.svg
    :alt: climpred gallery
    :target: https://mybinder.org/v2/gh/pangeo-data/climpred/main?urlpath=lab%2Ftree%2Fdocs%2Fsource%2Fquick-start.ipynb

.. |workshop| image:: https://img.shields.io/badge/climpred-workshop-f5a252
    :alt: climpred workshop
    :target: https://mybinder.org/v2/gh/bradyrx/climpred_workshop/master

.. |cloud| image:: https://img.shields.io/badge/climpred-cloud_demo-f9c99a
    :alt: climpred cloud demo
    :target: https://github.com/aaronspring/climpred-cloud-demo

.. |forks| image:: https://img.shields.io/github/forks/pangeo-data/climpred
    :alt: GitHub forks
    :target: https://github.com/pangeo-data/climpred/network/members

.. |stars| image:: https://img.shields.io/github/stars/pangeo-data/climpred
    :alt: GitHub stars
    :target: https://github.com/pangeo-data/climpred/stargazers

.. |issues| image:: https://img.shields.io/github/issues/pangeo-data/climpred
    :alt: GitHub issues
    :target: https://github.com/pangeo-data/climpred/issues

.. |PRs| image:: https://img.shields.io/github/issues-pr/pangeo-data/climpred
    :alt: GitHub PRs
    :target: https://github.com/pangeo-data/climpred/pulls

..

.. note::

    We are actively looking for new contributors for climpred! Riley moved to McKinsey's
    Climate Analytics team. Aaron is finishing his PhD, but will stay in academia.
    We especially hope for python enthusiasts from seasonal, subseasonal or weather
    prediction community. In our past coding journey, collaborative coding, feedbacking
    issues and pull requests advanced our code and thinking about forecast verification
    more than we could have ever expected.
    `Aaron <https://github.com/aaronspring/>`_ can provide guidance on
    implementing new features into ``climpred``. Feel free to implement
    your own new feature or take a look at the
    `good first issue <https://github.com/pangeo-data/climpred/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
    tag in the issues. Please reach out to us via `gitter <https://gitter.im/climpred>`_.


Installation
============

You can install the latest release of ``climpred`` using ``pip`` or ``conda``:

.. code-block:: bash

    pip install climpred[complete]

.. code-block:: bash

    conda install -c conda-forge climpred

You can also install the bleeding edge (pre-release versions) by cloning this
repository or installing directly from GitHub:

.. code-block:: bash

    git clone https://github.com/pangeo-data/climpred.git
    cd climpred
    pip install . --upgrade

.. code-block:: bash

    pip install git+https://github.com/pangeo-data/climpred.git


Documentation
=============

Documentation is in development and can be found on readthedocs_.

.. _readthedocs: https://climpred.readthedocs.io/en/latest/
