from setuptools import find_packages, setup

DISTNAME = 'esmtools'
VERSION = '0.1'
AUTHOR = 'Riley X. Brady'
AUTHOR_EMAIL = 'riley.brady@colorado.edu'
DESCRIPTION = 'Tools for analyzing model output from ESMs, with a focus on ocean models and decadal prediction.'
URL = 'https://github.com/bradyrx/esmtools'
LICENSE = 'MIT'
INSTALL_REQUIRES = ['xarray', 'pandas', 'numpy', 'matplotlib', 'pyfinance', 'xskillscore==0.0.2']
DEPENDENCY_LINKS = ['https://github.com/raybellwaves/xskillscore/tarball/master#egg=xskillscore-0.0.2']
# TODO: Add testing
# TESTS_REQUIRE = ['pytest']
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
      dependency_links=DEPENDENCY_LINKS,
      # Needed for dependencies. Currently do not like the pyfinance or xskillscore dependency.
      install_requires=INSTALL_REQUIRES,
      python_requires=PYTHON_REQUIRE
     )
