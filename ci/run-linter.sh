#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate climpred-dev

echo "[flake8]"
flake8 climpred --exclude=__init__.py

echo "[black]"
black --check -S climpred

echo "[isort]"
isort --recursive --check-only climpred

echo "[doc8]"
doc8 docs/source --ignore-path docs/source/setting-up-data.rst
doc8 *.rst
