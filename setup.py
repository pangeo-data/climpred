from setuptools import setup

setup(name='esmtools',
      version='0.1',
      description='Toolbox for analyzing ESM output, with an emphasis on ocean model output',
      url='http://github.com/bradyrx/esmtools',
      author='Riley X. Brady',
      author_email='riley.brady@colorado.edu',
      license='MIT',
      packages=['esmtools'],
      # Needed for dependencies. Currently do not like the pyfinance dependency.
      install_requires = ['xarray', 'numpy', 'pyfinance'],
      zip_safe=False
     )
