=====================
Contribution Guide
=====================

Contributions are highly welcomed and appreciated.  Every little help counts,
so do not hesitate! You can make a high impact on ``climpred`` just by using it and
reporting `issues <https://github.com/pangeo-data/climpred/issues>`__.

The following sections cover some general guidelines
regarding development in ``climpred`` for maintainers and contributors.

Please also review our `Code of Conduct <code_of_conduct.html>`__.

Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.



.. contents:: Contribution links
   :depth: 2



.. _submitfeedback:

Feature requests and feedback
-----------------------------

We are eager to hear about your requests for new features and any suggestions about the
API, infrastructure, and so on. Feel free to submit these as
`issues <https://github.com/pangeo-data/climpred/issues/new>`__ with the label "feature request."

Please make sure to explain in detail how the feature should work and keep the scope as
narrow as possible. This will make it easier to implement in small PRs.


.. _reportbugs:

Report bugs
-----------

Report bugs for ``climpred`` in the `issue tracker <https://github.com/pangeo-data/climpred/issues>`_
with the label "bug".

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and ``climpred``
  version.
* Detailed steps to reproduce the bug.

If you can write a demonstration test that currently fails but should pass,
that is a very useful commit to make as well, even if you cannot fix the bug itself.


.. _fixbugs:

Fix bugs
--------

Look through the `GitHub issues for bugs <https://github.com/pangeo-data/climpred/labels/bug>`_.

Talk to developers to find out how you can fix specific bugs.


Write documentation
-------------------

``climpred`` could always use more documentation.  What exactly is needed?

* More complementary documentation.  Have you perhaps found something unclear?
* Docstrings.  There can never be too many of them.
* Example notebooks with different Earth System Models, lead times, etc. -- they're all very
  appreciated.

You can also edit documentation files directly in the GitHub web interface,
without using a local copy.  This can be convenient for small fixes.

Our documentation is written in reStructuredText. You can follow our conventions in already written
documents. Some helpful guides are located
`here <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`__ and
`here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`__.

.. note::
    Build the documentation locally with the following command:

    .. code:: bash

        $ conda env update -f ci/requirements/climpred-dev.yml
        $ cd docs
        $ make html

    The built documentation should be available in the ``docs/build/``.

If you need to add new functions to the API, run ``sphinx-autogen -o api api.rst`` from the
``docs/source`` directory and add the functions to ``api.rst``.

 .. _`pull requests`:
 .. _pull-requests:

Preparing Pull Requests
-----------------------


#. Fork the
   `climpred GitHub repository <https://github.com/pangeo-data/climpred>`__.  It's
   fine to use ``climpred`` as your fork repository name because it will live
   under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_, connect your repository
   to the upstream (main project), and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/climpred.git
    $ cd climpred
    $ git remote add upstream git@github.com:pangeo-data/climpred.git

    # now, to fix a bug or add feature create your own branch off "master":

    $ git checkout -b your-bugfix-feature-branch-name master

   If you need some help with Git, follow this quick start
   guide: https://git.wiki.kernel.org/index.php/QuickStart

#. Install dependencies into a new conda environment::

    $ conda env update -f ci/requirements/climpred-dev.yml
    $ conda activate climpred-dev

#. Make an editable install of climpred by running::

    $ pip install -e .

#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the ``climpred`` repo::

     $ pip install --user pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit.

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit
   hooks to ensure code-style and code formatting is consistent.

    Now you have an environment called ``climpred-dev`` that you can work in.
    You’ll need to make sure to activate that environment next time you want
    to use it after closing the terminal or your system.

    You can now edit your local working copy and run/add tests as necessary. Please follow
    PEP-8 for naming. When committing, ``pre-commit`` will modify the files as needed, or
    will generally be quite clear about what you need to do to pass the commit test.

#. Break your edits up into reasonably sized commits::

    $ git commit -a -m "<commit message>"
    $ git push -u

#. Run all the tests

   Now running tests is as simple as issuing this command::

    $ pytest climpred

   Check that your contribution is covered by tests and therefore increases the overall test coverage::

    $ coverage run --source climpred -m py.test
    $ coverage report
    $ coveralls

  Please stick to `xarray <http://xarray.pydata.org/en/stable/contributing.html>`_'s testing recommendations.

#. Running the performance test suite

Performance matters and it is worth considering whether your code has introduced
performance regressions. `climpred` is starting to write a suite of benchmarking tests
using `asv <https://asv.readthedocs.io/en/stable/>`_
to enable easy monitoring of the performance of critical `climpred` operations.
These benchmarks are all found in the ``asv_bench`` directory.

If you need to run a benchmark, change your directory to ``asv_bench/`` and run::

    $ asv continuous -f 1.1 upstream/master HEAD

You can replace ``HEAD`` with the name of the branch you are working on,
and report benchmarks that changed by more than 10%.
The command uses ``conda`` by default for creating the benchmark
environments.

Running the full benchmark suite can take up to half an hour and use up a few GBs of
RAM. Usually it is sufficient to paste only a subset of the results into the pull
request to show that the committed changes do not cause unexpected performance
regressions.  You can run specific benchmarks using the ``-b`` flag, which
takes a regular expression.  For example, this will only run tests from a
``asv_bench/benchmarks/benchmarks_perfect_model.py`` file::

    $ asv continuous -f 1.1 upstream/master HEAD -b ^benchmarks_perfect_model

If you want to only run a specific group of tests from a file, you can do it
using ``.`` as a separator. For example::

    $ asv continuous -f 1.1 upstream/master HEAD -b benchmarks_perfect_model.Compute.time_bootstrap_perfect_model

will only run the ``time_bootstrap_perfect_model`` benchmark of class ``Compute``
defined in ``benchmarks_perfect_model.py``.

#. Create a new changelog entry in ``CHANGELOG.rst``:

   - The entry should be entered as:

    <description> (``:pr:`#<pull request number>```) ```<author's names>`_``

    where ``<description>`` is the description of the PR related to the change and
    ``<pull request number>`` is the pull request number and ``<author's names>`` are your first
    and last names.

   - Add yourself to list of authors at the end of ``CHANGELOG.rst`` file if not there yet, in
     alphabetical order.

 #. Add yourself to the
    `contributors <https://climpred.readthedocs.io/en/latest/contributors.html>`_
    list via ``docs/source/contributors.rst``.

#. Finally, submit a pull request through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/climpred
    compare: your-branch-name

    base-fork: pangeo-data/climpred
    base: master

Note that you can create the Pull Request while you're working on this. The PR will update
as you add more commits. ``climpred`` developers and contributors can then review your code
and offer suggestions.
