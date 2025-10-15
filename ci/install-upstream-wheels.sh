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
python -m pip install \
    -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
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
    git+https://github.com/xarray-contrib/xskillscore@pr-437 \
    git+https://github.com/xgcm/xrft \
    git+https://github.com/pankajkarman/bias_correction
python -m pip install --upgrade git+https://github.com/Ouranosinc/xclim
