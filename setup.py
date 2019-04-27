from setuptools import find_packages, setup

DISTNAME = 'climpred'
VERSION = '0.3'
AUTHOR = 'Riley X. Brady'
AUTHOR_EMAIL = 'riley.brady@colorado.edu'
DESCRIPTION = 'An xarray wrapper for analysis of ensemble forecast models for climate prediction.'
URL = 'https://github.com/bradyrx/climpred'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['numpy', 'pandas',
                    'xarray', 'scipy',
                    'xskillscore', 'eofs',
                    'cftime']
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
      install_requires=INSTALL_REQUIRES,
      python_requires=PYTHON_REQUIRE,
      tests_require=TESTS_REQUIRE,
      )
