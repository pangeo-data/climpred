Release Procedure
-----------------

We follow semantic versioning, e.g., ``v1.0.0``. A major version causes incompatible API
changes, a minor version adds functionality, and a patch covers bug fixes.

#. Create a new branch ``release-v1.0.0`` with the version for the release.

 * Update `CHANGELOG.rst <CHANGELOG.html>`_.
 * Make sure all new changes and features are reflected in the documentation.

#. Open a new pull request for this branch targeting ``main``

#. After all tests pass and the PR has been approved, merge the PR into ``main``

#. Tag a release and push it to GitHub::

    $ git tag -a v1.0.0 -m "Version 1.0.0"
    $ git push upstream main --tags

#. We use Github Actions to automate the new release being published to TestPyPI (staging) and PyPI (production). When a tag is pushed to the repository, the following happens:
    - A new release is drafted on Github
    - The library is built and a workflow is staged for publishing to TestPyPI
        - A maintainer must manually approve the workflow to publish to TestPyPI
    - If everything clears, maintainers can then finalize the release on GitHub, triggering an upload to PyPI

    If you wish to circumvent the GitHub Actions for whatever reason, you can manually do it by running the following::

    $ git clean -xfd  # remove any files not checked into git
    $ python -m build  # build package
    $ python -m twine upload --repository-url https://test.pypi.org/legacy dist/*  # register and push to testpypi
    $ python -m twine upload dist/*  # register and push to pypi

#. Next, update the stable branch with ``main``. This will trigger a stable build
   for ReadTheDocs::

    $ git checkout stable
    $ git rebase main
    $ git push -f upstream stable
    $ git checkout main

#. Go to https://readthedocs.org and add the new version to ``"Active Versions"`` under the version tab.
   Force-build ``"stable"`` if it isn't already building.

#. Update climpred conda-forge feedstock

 * Fork `climpred-feedstock repository <https://github.com/conda-forge/climpred-feedstock>`_
 * Clone this fork and edit recipe::

        $ git clone git@github.com:username/climpred-feedstock.git
        $ cd climpred-feedstock
        $ cd recipe
        $ # edit meta.yaml

 * Update version
 * Get ``sha256`` from pypi.org for `climpred <https://pypi.org/project/climpred/#files>`_
 * Check that ``requirements.txt`` from the main ``climpred`` repo is accounted for
   in ``meta.yaml`` from the feedstock.
 * Fill in the rest of information as described
   `here <https://github.com/conda-forge/climpred-feedstock#updating-climpred-feedstock>`_
 * Commit and submit a PR
