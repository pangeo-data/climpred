from os.path import exists

from setuptools import find_packages, setup

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

extras_require = {
    "accel": ["numba>=0.52", "bottleneck"],
    "bias-correction": ["xclim!=0.30.0", "bias-correction>=0.4"],
    "viz": ["matplotlib", "nc-time-axis>=1.3.1"],
    "io": ["netcdf4"],
    "regridding": ["xesmf"],  # for installation see https://pangeo-xesmf.readthedocs.io/en/latest/installation.html
    "relative_entropy": ["eofs"],
    "vwmp": ["xrft"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})
del extras_require["complete"]["regridding"]
# after complete is set, add in test
extras_require["test"] = [
    "pytest",
    "pytest-cov",
    "pytest-lazyfixures",
    "pytest-xdist",
    "pre-commit",
    "netcdf4",
]
extras_require["docs"] = extras_require["complete"] + [
    "importlib_metadata",
    "nbsphinx",
    "nbstripout",
    "pygments==2.6.1",
    "sphinx",
    "sphinxcontrib-napoleon",
    "sphinx_rtd_theme",
    "sphinx-copybutton",
]

setup(
    maintainer="Aaron Spring",
    maintainer_email="aaron.spring@mpimet.mpg.de",
    description="Verification of weather and climate forecasts." + " prediction.",
    install_requires=install_requires,
    python_requires=">=3.7",
    license="MIT",
    long_description=long_description,
    classifiers=CLASSIFIERS,
    name="climpred",
    packages=find_packages(),
    test_suite="climpred/tests",
    tests_require=["pytest"],
    url="https://github.com/pangeo-data/climpred",
    use_scm_version={"version_scheme": "post-release", "local_scheme": "dirty-tag"},
    zip_safe=False,
    extras_require=extras_require,
)
