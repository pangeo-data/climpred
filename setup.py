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
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]

extras_require = {
    "accel": ["numba>=0.52", "bottleneck"],
    "bias-correction": ["xclim>=0.46", "bias-correction>=0.4"],
    "viz": ["matplotlib", "nc-time-axis>=1.4.0"],
    "io": ["netcdf4"],  # use h5netcdf when encountering seg faults as in GHA CI
    "regridding": [
        "xesmf"
    ],  # for installation see https://pangeo-xesmf.readthedocs.io/
    "relative_entropy": ["eofs"],
    "vwmp": ["xrft"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})
extras_require["complete"].remove(
    "xesmf"
)  # circumventing esmpy/xesmf installation issues; use conda
# after complete is set, add in test
extras_require["test"] = [
    "netcdf4",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "pytest-lazy-fixture",
    "pytest-xdist",
]
extras_require["docs"] = extras_require["complete"] + [
    "myst_nb",
    "sphinx",
    "sphinx-copybutton",
    "sphinx_book_theme",
    "sphinxcontrib-bibtex",
    "sphinxcontrib-napoleon",
]

setup(
    maintainer="Aaron Spring",
    maintainer_email="aaron.spring@mpimet.mpg.de",
    description="Verification of weather and climate forecasts." + " prediction.",
    install_requires=install_requires,
    python_requires=">=3.8",
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
