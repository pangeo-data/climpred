#!/usr/bin/env bash

conda uninstall -y --force \
    bias_correction \
    cftime \
    dask \
    matplotlib \
    nc-time-axis \
    numpy \
    pandas \
    xarray \
    xclim \
    xskillscore \
    climpred

# to limit the runtime of Upstream CI
# numpy and pandas nightlies live in scientific-python-nightly-wheels (the same
# index used for matplotlib below). The old scipy-wheels-nightly channel is no
# longer updated, so resolving against it pulled a numpy far older than the
# `numpy >=2.0` the conda environment was solved with. Replacing numpy under
# conda-built extensions that were compiled against 2.x segfaults the suite.
python -m pip install \
    --index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --extra-index-url https://pypi.org/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    pandas
python -m pip install \
    --upgrade \
    --pre \
    --index-url https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --extra-index-url https://pypi.org/simple \
    matplotlib
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/pydata/xarray  \
    git+https://github.com/xarray-contrib/xskillscore \
    git+https://github.com/xgcm/xrft \
    git+https://github.com/pankajkarman/bias_correction
python -m pip install --upgrade git+https://github.com/Ouranosinc/xclim
