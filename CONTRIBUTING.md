# Contribution Guide

Contributions are highly welcomed and appreciated. Every little bit helps,
so do not hesitate! You can make a high impact on `climpred` just by using it and
reporting [issues](https://github.com/bradyrx/climpred/issues).

The following sections cover some general guidelines
regarding development in `climpred` for maintainers and contributors.

Please also review our [Code of Conduct](code_of_conduct.html).

* [Feature Requests and Feedback](#Feature-Requests-and-Feedback)
* [Bug Reports](#Bug-Reports)
* [Fix Bugs](#Fix-Bugs)
* [Write Documentation](#Write-Documentation)
* [Preparing Pull Requests](#Preparing-Pull-Requests)

## Feature Requests and Feedback

We are eager to hear about your requests for new features and any suggestions about the
API, infrastructure, and so on. Feel free to submit these as
[issues](https://github.com/bradyrx/climpred/issues/new?assignees=&labels=feature+request&template=feature_request.md) with the label "feature request."

Please make sure to explain in detail how the feature should work and keep the scope as
narrow as possible. This will make it easier to implement in small PRs.

## Bug Reports

Report bugs for `climpred` in the [issue tracker](https://github.com/bradyrx/climpred/issues/new?assignees=&labels=bug&template=bug_report.md)
with the label "bug report."

If you are reporting a bug, please include:

* A _minimal_ example that reproduces the error. I.e., generate some dummy data that can be used by a developer on their local machine to reproduce the issue.
* Any details about your local setup that might be helpful in troubleshooting,
  specifically the python interpreter version, installed libraries, and `climpred`
  version.

If you can write a demonstration test that currently fails but should pass,
that is a very useful commit to make as well, even if you cannot fix the bug itself.

## Fix Bugs

Look through the [GitHub issues for bugs](https://github.com/bradyrx/climpred/labels/bug),
or talk to the developers who can suggest specific bugs for you to help out with.

## Write Documentation

`climpred` could always use more documentation.  What exactly is needed?

* Additions and modifications to existing documentation. Have you perhaps found something unclear?
* Detailed docstrings with examples for all functions.
* Example notebooks using different Earth System Models, lead times, etc. -- they're all very
  appreciated. We want to show people `climpred` in action.

You can edit documentation files directly in the GitHub web interface,
without using a local copy.  This can be convenient for small fixes.

Our documentation is written in reStructuredText. You can follow our conventions in already written
documents. Some helpful guides are located
[here](http://docutils.sourceforge.net/docs/user/rst/quickref.html) and
[here](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst). Example analysis
code can be added as jupyter notebooks under `docs/source/examples`.

After modifying the documentation, you can build it locally by doing the following:

```bash
$ conda env update -f ci/environment-dev-3.6.yml
$ cd docs
$ make html
```

The built documentation will be available under `docs/build`.

If you need to add new functions to the API, run `sphinx-autogen -o api api.rst` from the
`docs/source` directory and add the functions to `api.rst`.

## Preparing Pull Requests

1. Fork the [climpred GitHub repository](https://github.com/bradyrx/climpred).  It's fine to use `climpred` as your fork repository name because it will live under your user.

2. Clone your fork locally using [git](https://git-scm.com/), connect your repository to the upstream (main project), and create a branch.

    ```shell
    $ git clone git@github.com:YOUR_GITHUB_USERNAME/climpred.git
    $ cd climpred
    $ git remote add upstream git@github.com:bradyrx/climpred.git
    ```

3. Now, to fix a bug or to add a feature, create your own branch off `master`

    ```shell
    $ git checkout -b your-bugfix-feature-branch-name master
    ```

    If you need some help with Git, follow this quick start guide: https://git.wiki.kernel.org/index.php/QuickStart.

4. Install dependencies into a new conda environment

    ```shell
    $ conda env update -f ci/environment-dev-3.7.yml
    $ conda activate climpred-dev
    ```

5. Make an editable install of climpred by running

    ```shell
    $ pip install --no-deps -e .
    ```

6. Install [pre-commit](https://pre-commit.com) and its hook on the `climpred` repo

    ```shell
     $ pip install --user pre-commit
     $ pre-commit install
    ```

    Afterwards `pre-commit` will run whenever you commit. `pre-commit` is a framework for managing and maintaining multi-language pre-commit hooks to ensure code-style and code formatting is consistent.

    Now you have an environment called `climpred-dev` that you can work in. You’ll need to make sure to activate that environment after the first time you install it and whenever you reopen terminal.

    ```shell
    $ conda activate climpred-dev
    ```

    You can now edit your local working copy of `climpred` and run/add tests as necessary. Please follow PEP-8 for naming. When committing, `pre-commit` will modify the files as needed, or will generally be quite clear about what you need to do to pass the commit test.

7. Break your edits up into reasonably sized commits

    ```shell
    $ git commit -a -m "<commit message>"
    $ git push -u
    ```

8. Run all the tests. Running tests is as simple as issuing this command

    ```shell
    $ pytest climpred
    ```

    Check that your contribution is covered by tests and therefore increases the overall test coverage

    ```shell
    $ coverage run --source climpred -m py.test
    $ coverage report
    $ coveralls
    ```

    **Note that the Github Pull Request will also automatically run tests**, but you can test specific testing files locally (e.g. through `pytest test_file_name.py`) so you don't have to wait for the entire testing suite to be run through Github.

9. Running the performance test suite.

    Performance matters and it is worth considering whether your code has introduced performance regressions. `climpred` is starting to write a suite of benchmarking tests using [asv](https://asv.readthedocs.io/en/stable/) to enable easy monitoring of the performance of critical `climpred` operations. These benchmarks are all found in the `asv_bench` directory.

    If you need to run a benchmark, change your directory to `asv_bench/` and run

    ```shell
    $ asv continuous -f 1.1 upstream/master HEAD
    ```

    You can replace `HEAD` with the name of the branch you are working on, and report benchmarks that changed by more than 10%. The command uses `conda` by default for creating the benchmark environments.

    Running the full benchmark suite can take up to half an hour and use up a few GBs of RAM. Usually it is sufficient to paste only a subset of the results into the pull request to show that the committed changes do not cause unexpected performance regressions.  You can run specific benchmarks using the `-b` flag, which takes a regular expression.  For example, this will only run tests from a `asv_bench/benchmarks/benchmarks_perfect_model.py` file:

    ```shell
    $ asv continuous -f 1.1 upstream/master HEAD -b ^benchmarks_perfect_model
    ```

    If you want to only run a specific group of tests from a file, you can do it using `.` as a separator. For example,

    ```shell
    $ asv continuous -f 1.1 upstream/master HEAD -b benchmarks_perfect_model.Compute.time_bootstrap_perfect_model
    ```

    will only run the `time_bootstrap_perfect_model` benchmark of class `Compute` defined in `benchmarks_perfect_model.py`.

10. Create a new changelog entry in `CHANGELOG.rst`. The entry should be entered as

    ```shell
    <description> (``:pr:`#<pull request number>```) ```<author's names>`_``
    ```

    where `<description>` is the description of the PR related to the change and `<pull request number>` is the pull request number and `<author's names>` are your first and last names.

    Add yourself to list of authors at the end of `CHANGELOG.rst` file if not there yet, in alphabetical order.

11. Add yourself to the [contributors](contributors.html) list via `docs/source/contributors.rst`.

12. Finally, submit a pull request through the GitHub website using this data

    ```shell
    head-fork: YOUR_GITHUB_USERNAME/climpred
    compare: your-branch-name

    base-fork: bradyrx/climpred
    base: master
    ```

    Note that you can create the Pull Request while you're working on this. The PR will update as you add more commits. `climpred` developers and contributors can then review your code and offer suggestions.
