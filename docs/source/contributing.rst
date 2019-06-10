=====================
Contribution Guide
=====================

Contributions are highly welcomed and appreciated.  Every little help counts,
so do not hesitate!

The following sections cover some general guidelines
regarding development in ``climpred`` for maintainers and contributors.
Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.



.. contents:: Contribution links
   :depth: 2



.. _submitfeedback:

Feature requests and feedback
-----------------------------


We'd also like to hear about your propositions and suggestions.  Feel free to
`submit them as issues <https://github.com/bradyrx/climpred>`_ and:

* Explain in detail how they should work.
* Keep the scope as narrow as possible.  This will make it easier to implement.


.. _reportbugs:

Report bugs
-----------

Report bugs for ``climpred`` in the `issue tracker <https://github.com/bradyrx/climpred>`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and esmlab
  version.
* Detailed steps to reproduce the bug.

If you can write a demonstration test that currently fails but should pass
(xfail), that is a very useful commit to make as well, even if you cannot
fix the bug itself.


.. _fixbugs:

Fix bugs
--------

Look through the `GitHub issues for bugs <https://github.com/bradyrx/climpred/labels/bug>`_.

Talk to developers to find out how you can fix specific bugs.


Write documentation
-------------------

``climpred`` could always use more documentation.  What exactly is needed?

* More complementary documentation.  Have you perhaps found something unclear?
* Docstrings.  There can never be too many of them.
* Blog posts, articles and such -- they're all very appreciated.

You can also edit documentation files directly in the GitHub web interface,
without using a local copy.  This can be convenient for small fixes.

Our documentation is written in reStructuredText. You can follow our conventions in already written documents. Some helpful guides are located `here <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_ and `here <https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_. 

.. note::
    Build the documentation locally with the following command:

    .. code:: bash

        $ conda env update -f ci/environment-dev-3.6.yml
        $ cd docs
        $ make html

    The built documentation should be available in the ``docs/build/``.


 .. _`pull requests`:
.. _pull-requests:

Preparing Pull Requests
-----------------------


#. Fork the
   `esmlab GitHub repository <https://github.com/bradyrx/climpred>`__.  It's
   fine to use ``climpred`` as your fork repository name because it will live
   under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_, connect your repository
   to the upstream (main project), and create a branch::

    $ git clone git@github.com:YOUR_GITHUB_USERNAME/climpred.git
    $ cd climpred 
    $ git remote add upstream git@github.com:bradyrx/climpred.git

    # now, to fix a bug or add feature create your own branch off "master":

    $ git checkout -b your-bugfix-feature-branch-name master

   If you need some help with Git, follow this quick start
   guide: https://git.wiki.kernel.org/index.php/QuickStart

#. Install dependencies into a new conda environment::

    $ conda env update -f ci/environment-dev-3.7.yml
    $ conda activate climpred-dev

#. Make an editable install of esmlab by running::

    $ pip install -e .



#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the ``climpred`` repo::

     $ pip install --user pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit.

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit hooks
   to ensure code-style and code formatting is consistent.

    Now you have an environment called ``climpred-dev`` that you can work in.
    You’ll need to make sure to activate that environment next time you want
    to use it after closing the terminal or your system.


#. Run all the tests

   Now running tests is as simple as issuing this command::

    $ coverage run --source climpred -m py.test


   This command will run tests via the "pytest" tool against Python 3.6.



#. Create a new changelog entry in ``CHANGELOG.rst``:

   - The entry should be entered as:

    <description> (``:pr:`#<pull request number>```) ```<author's names>`_``

    where ``<description>`` is the description of the PR related to the change and ``<pull request number>`` is
    the pull request number and ``<author's names>`` are your first and last names.

   - Add yourself to list of authors at the end of ``CHANGELOG.rst`` file if not there yet, in alphabetical order.


#. You can now edit your local working copy and run the tests again as necessary. Please follow PEP-8 for naming.

   When committing, ``pre-commit`` will re-format the files if necessary.

#. Commit and push once your tests pass and you are happy with your change(s)::

    $ git commit -a -m "<commit message>"
    $ git push -u

#. Finally, submit a pull request through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/esmlab
    compare: your-branch-name

    base-fork: NCAR/esmlab
    base: master          # if it's a bugfix or feature
