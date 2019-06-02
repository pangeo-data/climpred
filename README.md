![](https://i.imgur.com/HPOdOsR.png)

An xarray wrapper for analysis of ensemble forecast models for climate prediction.

[![Build Status](https://travis-ci.org/bradyrx/climpred.svg?branch=master)](https://travis-ci.org/bradyrx/climpred)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/a532752e9e814c6e895694463f307cd9)](https://www.codacy.com/app/bradyrx/climpred?utm_source=github.com&utm_medium=referral&utm_content=bradyrx/climpred&utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/bradyrx/climpred/badge.svg?branch=master)](https://coveralls.io/github/bradyrx/climpred?branch=master)


## Release

We are diligently working on our v1 release (see kanban board [here](https://github.com/bradyrx/climpred/projects/2]). The goal for v1 is to have `climpred` ready to take in subseasonal-to-seasonal and seasonal-to-decadal prediction ensembles and to perform determinstic skill metrics on them. We will have early-stage objects (`HindcastEnsemble` and `PerfectModelEnsemble`) to make computations easier on the user. We will also have documentation ready so the capabilities of `climpred` are more clear.

In the meantime, you can install the package following the steps below and reference the notebooks for guidance. Please raise any issues if you encounter any bugs or have any ideas; you can also raise a PR to add new features. Feel free to contact us if you have questions.

## Installation

```shell
pip install git+https://github.com/bradyrx/climpred
```

## Documentation

Documentation can be found on this Github's [wiki](https://github.com/bradyrx/climpred/wiki). It is currently in development and will eventually move to a `sphinx` page.

## Contribution Guide

We are eager for anyone to contribute to `climpred` via Pull Requests. By following these steps, it makes it easy for everyone involved to review code and get new features in as smoothly as possible. This guide is modeled by `esmlab`'s great contribution system.

1. Fork the `climpred` GitHub repository. It's fine to use `climpred` as your fork repository name since it lives under your user.
2. Clone your fork locally using git, connect your repository to the upstream (main project), and create a branch:
```bash
$ git clone git@github.com:YOUR_GITHUB_USERNAME/climpred.git
$ cd climpred
$ git remote add upstream git@github.com:bradyrx/climpred.git

# To fix a bug or add a feature, create your own branch off of master:
$ git checkout -b your-bugfix-feature-branch-name master
```
3. Install dependencies into a new conda environment.
```bash
$ conda env update -f ci/requirements-py36.yml
$ conda activate climpred-dev
```
4. Make an editable install of `climpred` by running the following the main directory:
```bash
$ pip install -e .
```
5. Install `pre-commit` and its hook on the `climpred` repo:
```bash
$ pip install --user pre-commit
$ pre-commit install
```

Now that you have an environment called `climpred-dev`, you need to activate it while editing the code base.

6. Make your edits to the repository, and `pre-commit` will run every time you add a new commit. It will automatically style your code to `climpred`'s style and will run `flake8` on your code to check that there are no egregious PEP8 errors. If `pre-commit` breaks due to `flake8`, you'll have to fix your code manually in some cases to make it pass. If `pre-commit` breaks due to `black` styling issues, it will reformat your code automatically. Just add and re-commit the file afterwards, and it should commit fine.

7. (Optional) Run our testing suite by calling `pytest`. Our continuous integration system will do this automatically on your PR to check if things break, but it's sometimes helpful to do this beforehand so you don't wait around for Travis CI to run a bunch of times.

8. Create a new changelog entry in `CHANGELOG.md`, detailing your addition.

9. Create a Pull Request on `climpred`'s repository. We'll review your code and will be eager to add your new features/bugfix to the package!
