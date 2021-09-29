#!/usr/bin/env bash

conda uninstall -y --force \
    numpy \
    pandas \
    matplotlib \
    dask \
    cftime \
    nc-time-axis \
    bottleneck \
    xarray \
    xskillscore \
    xclim \
    bias_correction \
    climpred


# to limit the runtime of Upstream CI
python -m pip install pytest-timeout
python -m pip install \
    -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    pandas
python -m pip install \
    -f https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com \
    --no-deps \
    --pre \
    --upgrade \
    matplotlib
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/pydata/xarray  \
    git+https://github.com/pydata/bottleneck  \
    git+https://github.com/xarray-contrib/xskillscore \
    git+https://github.com/xgcm/xrft \
    git+https://github.com/pankajkarman/bias_correction
python -m pip install --upgrade git+https://github.com/Ouranosinc/xclim
