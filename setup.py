from setuptools import setup

setup(name='esmtools',
      version='0.1',
      description='Toolbox for analyzing ESM output, with an emphasis on ocean model output',
      url='http://github.com/bradyrx/esmtools',
      author='Riley X. Brady',
      author_email='riley.brady@colorado.edu',
      license='MIT',
      packages=['esmtools'],
      # Needed for dependencies. Currently do not like the pyfinance or xskillscore dependency.
      install_requires = ['xarray', 'numpy', 'pyfinance', 'xskillscore'],
      zip_safe=False
     )
