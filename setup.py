from setuptools import find_packages, setup

setup(name='esmtools',
      version='0.1',
      description='Toolbox for analyzing ESM output, with an emphasis on ocean model output',
      url='http://github.com/bradyrx/esmtools',
      author='Riley X. Brady',
      author_email='riley.brady@colorado.edu',
      license='MIT',
      packages=find_packages(),
      dependency_links=['https://github.com/raybellwaves/xskillscore'],
      # Needed for dependencies. Currently do not like the pyfinance or xskillscore dependency.
      install_requires = ['xarray',
                          'pandas',
                          'numpy', 
                          'pyfinance'],
      zip_safe=False
     )
