from setuptools import find_packages, setup

DISTNAME = 'climpred'
VERSION = '0.2'
AUTHOR = 'Riley X. Brady'
AUTHOR_EMAIL = 'riley.brady@colorado.edu'
DESCRIPTION = 'An xarray wrapper for analysis of ensemble forecast models for climate prediction.'
URL = 'https://github.com/bradyrx/climpred'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['numpy', 'pandas',
                    'xarray', 'scipy', 'xskillscore', 'eofs', 'cftime', 'properscoring']
TESTS_REQUIRE = ['pytest']
PYTHON_REQUIRE = '>=3.6'

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md').read(),
      url=URL,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      packages=find_packages(),
      # NOTE: This will be deprecated, so either need to move away from non-pypi packages or find another solution.
      # Needed for dependencies.
      install_requires=INSTALL_REQUIRES,
      python_requires=PYTHON_REQUIRE,
      tests_require=TESTS_REQUIRE,
      )
